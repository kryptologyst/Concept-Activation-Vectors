"""Core Concept Activation Vector (CAV) implementation.

This module implements the TCAV (Testing with Concept Activation Vectors) methodology
for concept-based interpretability in neural networks.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class ConceptDataset:
    """Dataset for concept-positive and concept-negative examples.
    
    Attributes:
        positive_examples: Examples that contain the concept
        negative_examples: Examples that don't contain the concept
        concept_name: Name of the concept being tested
    """
    
    def __init__(
        self,
        positive_examples: torch.Tensor,
        negative_examples: torch.Tensor,
        concept_name: str,
    ) -> None:
        """Initialize concept dataset.
        
        Args:
            positive_examples: Tensor of concept-positive examples
            negative_examples: Tensor of concept-negative examples  
            concept_name: Name of the concept
        """
        self.positive_examples = positive_examples
        self.negative_examples = negative_examples
        self.concept_name = concept_name
        
        # Validate inputs
        if len(positive_examples) == 0 or len(negative_examples) == 0:
            raise ValueError("Both positive and negative examples must be non-empty")
            
        if positive_examples.shape[1:] != negative_examples.shape[1:]:
            raise ValueError("Positive and negative examples must have same feature dimensions")
    
    @property
    def total_examples(self) -> int:
        """Total number of examples in the dataset."""
        return len(self.positive_examples) + len(self.negative_examples)
    
    def get_balanced_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get balanced dataset with equal positive and negative examples.
        
        Returns:
            Tuple of (features, labels) where labels are 1 for positive, 0 for negative
        """
        min_size = min(len(self.positive_examples), len(self.negative_examples))
        
        # Sample equal numbers of positive and negative examples
        pos_indices = torch.randperm(len(self.positive_examples))[:min_size]
        neg_indices = torch.randperm(len(self.negative_examples))[:min_size]
        
        pos_samples = self.positive_examples[pos_indices]
        neg_samples = self.negative_examples[neg_indices]
        
        # Combine and create labels
        features = torch.cat([pos_samples, neg_samples], dim=0)
        labels = torch.cat([
            torch.ones(min_size, dtype=torch.long),
            torch.zeros(min_size, dtype=torch.long)
        ])
        
        # Shuffle
        indices = torch.randperm(len(features))
        return features[indices], labels[indices]


class ConceptActivationVector:
    """Concept Activation Vector (CAV) for testing concept sensitivity.
    
    A CAV is a vector in the activation space of a neural network layer that
    represents the direction corresponding to a human-interpretable concept.
    """
    
    def __init__(
        self,
        concept_dataset: ConceptDataset,
        layer_name: str,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize CAV.
        
        Args:
            concept_dataset: Dataset containing concept-positive and negative examples
            layer_name: Name of the layer to extract activations from
            random_state: Random seed for reproducibility
        """
        self.concept_dataset = concept_dataset
        self.layer_name = layer_name
        self.random_state = random_state
        
        # Set random seeds
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
            torch.manual_seed(random_state)
        
        self.cav_vector: Optional[torch.Tensor] = None
        self.cav_accuracy: Optional[float] = None
        self.concept_sensitivity: Optional[float] = None
        
    def train_cav(
        self,
        model: nn.Module,
        device: torch.device,
        regularization: float = 0.01,
    ) -> None:
        """Train the CAV by learning a linear classifier in activation space.
        
        Args:
            model: Neural network model to extract activations from
            device: Device to run computations on
            regularization: L2 regularization strength for logistic regression
        """
        logger.info(f"Training CAV for concept '{self.concept_dataset.concept_name}' "
                   f"at layer '{self.layer_name}'")
        
        # Get balanced dataset
        features, labels = self.concept_dataset.get_balanced_dataset()
        
        # Extract activations
        activations = self._extract_activations(model, features, device)
        
        # Train linear classifier
        classifier = LogisticRegression(
            C=1.0/regularization,
            random_state=self.random_state,
            max_iter=1000,
        )
        
        classifier.fit(activations.cpu().numpy(), labels.cpu().numpy())
        
        # Store CAV vector (normalized weights)
        self.cav_vector = torch.tensor(
            classifier.coef_[0], 
            dtype=torch.float32, 
            device=device
        )
        self.cav_vector = self.cav_vector / torch.norm(self.cav_vector)
        
        # Calculate accuracy
        predictions = classifier.predict(activations.cpu().numpy())
        self.cav_accuracy = accuracy_score(labels.cpu().numpy(), predictions)
        
        logger.info(f"CAV training completed. Accuracy: {self.cav_accuracy:.4f}")
    
    def _extract_activations(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Extract activations from the specified layer.
        
        Args:
            model: Neural network model
            inputs: Input tensors
            device: Device to run computations on
            
        Returns:
            Activations from the specified layer
        """
        model.eval()
        activations = []
        
        with torch.no_grad():
            inputs = inputs.to(device)
            
            # Hook to capture activations
            def hook_fn(module, input, output):
                activations.append(output.cpu())
            
            # Register hook
            layer = self._get_layer_by_name(model, self.layer_name)
            if layer is None:
                raise ValueError(f"Layer '{self.layer_name}' not found in model")
            
            hook = layer.register_forward_hook(hook_fn)
            
            try:
                # Forward pass
                _ = model(inputs)
            finally:
                hook.remove()
        
        if not activations:
            raise RuntimeError("No activations captured")
        
        # Concatenate all activations
        activations_tensor = torch.cat(activations, dim=0)
        
        # Flatten spatial dimensions if needed
        if len(activations_tensor.shape) > 2:
            activations_tensor = activations_tensor.view(
                activations_tensor.size(0), -1
            )
        
        return activations_tensor
    
    def _get_layer_by_name(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Get layer by name from the model.
        
        Args:
            model: Neural network model
            layer_name: Name of the layer
            
        Returns:
            The layer module if found, None otherwise
        """
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        return None
    
    def compute_concept_sensitivity(
        self,
        model: nn.Module,
        test_inputs: torch.Tensor,
        device: torch.device,
    ) -> float:
        """Compute concept sensitivity score.
        
        Args:
            model: Neural network model
            test_inputs: Test inputs to evaluate
            device: Device to run computations on
            
        Returns:
            Concept sensitivity score (higher = more sensitive to concept)
        """
        if self.cav_vector is None:
            raise ValueError("CAV must be trained before computing sensitivity")
        
        # Extract activations for test inputs
        activations = self._extract_activations(model, test_inputs, device)
        
        # Compute directional derivatives (concept sensitivity)
        activations.requires_grad_(True)
        
        # Forward pass through model
        model.eval()
        outputs = model(test_inputs.to(device))
        
        # Compute gradients of outputs w.r.t. activations
        gradients = torch.autograd.grad(
            outputs=outputs.sum(),
            inputs=activations,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Project gradients onto CAV direction
        concept_sensitivity = torch.sum(gradients * self.cav_vector.to(device), dim=1)
        
        # Return mean absolute sensitivity
        return torch.mean(torch.abs(concept_sensitivity)).item()
    
    def get_concept_direction(self) -> torch.Tensor:
        """Get the CAV direction vector.
        
        Returns:
            Normalized CAV direction vector
        """
        if self.cav_vector is None:
            raise ValueError("CAV must be trained before getting direction")
        return self.cav_vector
    
    def get_stats(self) -> Dict[str, Any]:
        """Get CAV statistics.
        
        Returns:
            Dictionary containing CAV statistics
        """
        return {
            "concept_name": self.concept_dataset.concept_name,
            "layer_name": self.layer_name,
            "cav_accuracy": self.cav_accuracy,
            "concept_sensitivity": self.concept_sensitivity,
            "total_examples": self.concept_dataset.total_examples,
            "positive_examples": len(self.concept_dataset.positive_examples),
            "negative_examples": len(self.concept_dataset.negative_examples),
        }


class TCAVTester:
    """Testing with Concept Activation Vectors (TCAV) implementation.
    
    TCAV is a method for testing whether a neural network has learned
    human-interpretable concepts by measuring directional derivatives.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize TCAV tester.
        
        Args:
            model: Neural network model to test
            device: Device to run computations on
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.device = device
        self.random_state = random_state
        self.cavs: Dict[str, ConceptActivationVector] = {}
        
    def add_concept(
        self,
        concept_dataset: ConceptDataset,
        layer_name: str,
    ) -> ConceptActivationVector:
        """Add a concept to test.
        
        Args:
            concept_dataset: Dataset for the concept
            layer_name: Layer to test the concept at
            
        Returns:
            Trained CAV for the concept
        """
        cav = ConceptActivationVector(
            concept_dataset=concept_dataset,
            layer_name=layer_name,
            random_state=self.random_state,
        )
        
        cav.train_cav(self.model, self.device)
        self.cavs[f"{concept_dataset.concept_name}_{layer_name}"] = cav
        
        return cav
    
    def test_concept_sensitivity(
        self,
        test_inputs: torch.Tensor,
        concept_name: str,
        layer_name: str,
    ) -> float:
        """Test concept sensitivity on given inputs.
        
        Args:
            test_inputs: Inputs to test concept sensitivity on
            concept_name: Name of the concept to test
            layer_name: Layer to test at
            
        Returns:
            Concept sensitivity score
        """
        cav_key = f"{concept_name}_{layer_name}"
        if cav_key not in self.cavs:
            raise ValueError(f"CAV for concept '{concept_name}' at layer '{layer_name}' not found")
        
        cav = self.cavs[cav_key]
        sensitivity = cav.compute_concept_sensitivity(self.model, test_inputs, self.device)
        cav.concept_sensitivity = sensitivity
        
        return sensitivity
    
    def get_all_results(self) -> Dict[str, Dict[str, Any]]:
        """Get results for all tested concepts.
        
        Returns:
            Dictionary mapping concept names to their statistics
        """
        return {key: cav.get_stats() for key, cav in self.cavs.items()}
