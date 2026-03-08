"""Evaluation metrics for Concept Activation Vectors (CAVs)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torchmetrics import Accuracy, Precision, Recall, F1Score

logger = logging.getLogger(__name__)


class CAVEvaluator:
    """Evaluator for Concept Activation Vector quality and statistical significance."""
    
    def __init__(self, random_state: Optional[int] = None) -> None:
        """Initialize CAV evaluator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def evaluate_cav_quality(
        self,
        cav_vector: torch.Tensor,
        concept_dataset: Any,  # ConceptDataset type
        model: torch.nn.Module,
        device: torch.device,
        layer_name: str,
    ) -> Dict[str, float]:
        """Evaluate CAV quality using multiple metrics.
        
        Args:
            cav_vector: CAV direction vector
            concept_dataset: Dataset containing concept examples
            model: Neural network model
            device: Device to run computations on
            layer_name: Name of the layer
        
        Returns:
            Dictionary containing quality metrics
        """
        metrics = {}
        
        # 1. Concept completeness
        metrics["concept_completeness"] = self._compute_concept_completeness(
            cav_vector, concept_dataset, model, device, layer_name
        )
        
        # 2. Concept sensitivity
        metrics["concept_sensitivity"] = self._compute_concept_sensitivity(
            cav_vector, concept_dataset, model, device, layer_name
        )
        
        # 3. CAV accuracy
        metrics["cav_accuracy"] = self._compute_cav_accuracy(
            cav_vector, concept_dataset, model, device, layer_name
        )
        
        # 4. Statistical significance
        metrics["statistical_significance"] = self._compute_statistical_significance(
            cav_vector, concept_dataset, model, device, layer_name
        )
        
        return metrics
    
    def _compute_concept_completeness(
        self,
        cav_vector: torch.Tensor,
        concept_dataset: Any,
        model: torch.nn.Module,
        device: torch.device,
        layer_name: str,
    ) -> float:
        """Compute concept completeness score.
        
        Concept completeness measures how well the CAV captures the concept
        by measuring the alignment between CAV direction and concept gradient.
        
        Args:
            cav_vector: CAV direction vector
            concept_dataset: Dataset containing concept examples
            model: Neural network model
            device: Device to run computations on
            layer_name: Name of the layer
            
        Returns:
            Concept completeness score (0-1, higher is better)
        """
        model.eval()
        
        # Get concept-positive examples
        concept_positive = concept_dataset.positive_examples.to(device)
        
        # Extract activations
        activations = self._extract_activations(model, concept_positive, layer_name)
        
        # Compute gradients
        activations.requires_grad_(True)
        outputs = model(concept_positive)
        
        # Compute gradients w.r.t. activations
        gradients = torch.autograd.grad(
            outputs=outputs.sum(),
            inputs=activations,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Compute alignment with CAV
        cav_alignment = torch.sum(gradients * cav_vector.to(device), dim=1)
        
        # Concept completeness is the mean positive alignment
        completeness = torch.mean(torch.relu(cav_alignment)).item()
        
        return completeness
    
    def _compute_concept_sensitivity(
        self,
        cav_vector: torch.Tensor,
        concept_dataset: Any,
        model: torch.nn.Module,
        device: torch.device,
        layer_name: str,
    ) -> float:
        """Compute concept sensitivity score.
        
        Concept sensitivity measures how much the model's predictions change
        when moving along the CAV direction.
        
        Args:
            cav_vector: CAV direction vector
            concept_dataset: Dataset containing concept examples
            model: Neural network model
            device: Device to run computations on
            layer_name: Name of the layer
            
        Returns:
            Concept sensitivity score (higher is better)
        """
        model.eval()
        
        # Get balanced dataset
        features, labels = concept_dataset.get_balanced_dataset()
        features = features.to(device)
        
        # Extract activations
        activations = self._extract_activations(model, features, layer_name)
        
        # Compute directional derivatives
        activations.requires_grad_(True)
        outputs = model(features)
        
        gradients = torch.autograd.grad(
            outputs=outputs.sum(),
            inputs=activations,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Project gradients onto CAV direction
        sensitivity = torch.sum(gradients * cav_vector.to(device), dim=1)
        
        # Return mean absolute sensitivity
        return torch.mean(torch.abs(sensitivity)).item()
    
    def _compute_cav_accuracy(
        self,
        cav_vector: torch.Tensor,
        concept_dataset: Any,
        model: torch.nn.Module,
        device: torch.device,
        layer_name: str,
    ) -> float:
        """Compute CAV classification accuracy.
        
        Args:
            cav_vector: CAV direction vector
            concept_dataset: Dataset containing concept examples
            model: Neural network model
            device: Device to run computations on
            layer_name: Name of the layer
            
        Returns:
            CAV classification accuracy (0-1, higher is better)
        """
        # Get balanced dataset
        features, labels = concept_dataset.get_balanced_dataset()
        features = features.to(device)
        
        # Extract activations
        activations = self._extract_activations(model, features, layer_name)
        
        # Project activations onto CAV direction
        projections = torch.sum(activations * cav_vector.to(device), dim=1)
        
        # Use median as threshold
        threshold = torch.median(projections)
        predictions = (projections > threshold).long()
        
        # Compute accuracy
        accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        
        return accuracy
    
    def _compute_statistical_significance(
        self,
        cav_vector: torch.Tensor,
        concept_dataset: Any,
        model: torch.nn.Module,
        device: torch.device,
        layer_name: str,
        n_permutations: int = 1000,
    ) -> float:
        """Compute statistical significance using permutation testing.
        
        Args:
            cav_vector: CAV direction vector
            concept_dataset: Dataset containing concept examples
            model: Neural network model
            device: Device to run computations on
            layer_name: Name of the layer
            n_permutations: Number of permutations for testing
            
        Returns:
            P-value for statistical significance (lower is better)
        """
        # Get balanced dataset
        features, labels = concept_dataset.get_balanced_dataset()
        features = features.to(device)
        
        # Extract activations
        activations = self._extract_activations(model, features, layer_name)
        
        # Compute original sensitivity
        original_sensitivity = self._compute_concept_sensitivity(
            cav_vector, concept_dataset, model, device, layer_name
        )
        
        # Permutation test
        null_sensitivities = []
        
        for _ in range(n_permutations):
            # Randomly permute labels
            permuted_labels = labels[torch.randperm(len(labels))]
            
            # Create temporary concept dataset with permuted labels
            pos_mask = permuted_labels == 1
            neg_mask = permuted_labels == 0
            
            temp_concept_dataset = type(concept_dataset)(
                positive_examples=features[pos_mask],
                negative_examples=features[neg_mask],
                concept_name=concept_dataset.concept_name,
            )
            
            # Compute sensitivity with permuted labels
            perm_sensitivity = self._compute_concept_sensitivity(
                cav_vector, temp_concept_dataset, model, device, layer_name
            )
            null_sensitivities.append(perm_sensitivity)
        
        # Compute p-value
        null_sensitivities = np.array(null_sensitivities)
        p_value = np.mean(null_sensitivities >= original_sensitivity)
        
        return p_value
    
    def _extract_activations(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        """Extract activations from the specified layer.
        
        Args:
            model: Neural network model
            inputs: Input tensors
            layer_name: Name of the layer
            
        Returns:
            Activations from the specified layer
        """
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(output.cpu())
        
        # Register hook
        layer = self._get_layer_by_name(model, layer_name)
        if layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model")
        
        hook = layer.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
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
        
        return activations_tensor.to(inputs.device)
    
    def _get_layer_by_name(self, model: torch.nn.Module, layer_name: str) -> Optional[torch.nn.Module]:
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


class ConceptEvaluationMetrics:
    """Comprehensive evaluation metrics for concept-based interpretability."""
    
    def __init__(self) -> None:
        """Initialize metrics."""
        self.accuracy = Accuracy(task="multiclass", num_classes=3)
        self.precision = Precision(task="multiclass", num_classes=3, average="macro")
        self.recall = Recall(task="multiclass", num_classes=3, average="macro")
        self.f1 = F1Score(task="multiclass", num_classes=3, average="macro")
    
    def compute_all_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute all evaluation metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics["accuracy"] = self.accuracy(predictions, targets).item()
        metrics["precision"] = self.precision(predictions, targets).item()
        metrics["recall"] = self.recall(predictions, targets).item()
        metrics["f1_score"] = self.f1(predictions, targets).item()
        
        return metrics
    
    def compute_concept_metrics(
        self,
        concept_predictions: torch.Tensor,
        concept_targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute concept-specific metrics.
        
        Args:
            concept_predictions: Concept predictions
            concept_targets: Concept ground truth
            
        Returns:
            Dictionary containing concept metrics
        """
        metrics = {}
        
        # Convert to binary for concept classification
        concept_pred_binary = (concept_predictions > 0.5).float()
        concept_target_binary = concept_targets.float()
        
        # Concept accuracy
        concept_acc = (concept_pred_binary == concept_target_binary).float().mean()
        metrics["concept_accuracy"] = concept_acc.item()
        
        # Concept precision and recall
        tp = ((concept_pred_binary == 1) & (concept_target_binary == 1)).sum()
        fp = ((concept_pred_binary == 1) & (concept_target_binary == 0)).sum()
        fn = ((concept_pred_binary == 0) & (concept_target_binary == 1)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        metrics["concept_precision"] = precision.item()
        metrics["concept_recall"] = recall.item()
        metrics["concept_f1"] = f1.item()
        
        return metrics


def compute_faithfulness_metrics(
    model: torch.nn.Module,
    test_inputs: torch.Tensor,
    test_targets: torch.Tensor,
    cav_vector: torch.Tensor,
    layer_name: str,
    device: torch.device,
) -> Dict[str, float]:
    """Compute faithfulness metrics for CAV.
    
    Args:
        model: Neural network model
        test_inputs: Test inputs
        test_targets: Test targets
        cav_vector: CAV direction vector
        layer_name: Name of the layer
        device: Device to run computations on
        
    Returns:
        Dictionary containing faithfulness metrics
    """
    model.eval()
    
    # Extract activations
    activations = []
    
    def hook_fn(module, input, output):
        activations.append(output.cpu())
    
    layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            layer = module
            break
    
    if layer is None:
        raise ValueError(f"Layer '{layer_name}' not found in model")
    
    hook = layer.register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            _ = model(test_inputs.to(device))
    finally:
        hook.remove()
    
    activations_tensor = torch.cat(activations, dim=0).to(device)
    
    if len(activations_tensor.shape) > 2:
        activations_tensor = activations_tensor.view(activations_tensor.size(0), -1)
    
    # Compute concept scores
    concept_scores = torch.sum(activations_tensor * cav_vector.to(device), dim=1)
    
    # Compute faithfulness metrics
    metrics = {}
    
    # 1. Concept-prediction correlation
    with torch.no_grad():
        predictions = model(test_inputs.to(device))
        if isinstance(predictions, tuple):
            predictions = predictions[1]  # Handle CBM case
        
        pred_probs = torch.softmax(predictions, dim=1)
        max_probs = torch.max(pred_probs, dim=1)[0]
        
        correlation = torch.corrcoef(torch.stack([concept_scores, max_probs]))[0, 1]
        metrics["concept_prediction_correlation"] = correlation.item()
    
    # 2. Concept sensitivity (how much predictions change with concept)
    concept_sensitivity = torch.std(concept_scores).item()
    metrics["concept_sensitivity"] = concept_sensitivity
    
    return metrics
