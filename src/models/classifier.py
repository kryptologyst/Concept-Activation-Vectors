"""Neural network models for CAV testing."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleClassifier(nn.Module):
    """Simple neural network classifier for tabular data."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [64, 32],
        num_classes: int = 3,
        dropout_rate: float = 0.2,
    ) -> None:
        """Initialize simple classifier.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        return self.network(x)
    
    def get_layer_output(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Get output from a specific layer.
        
        Args:
            x: Input tensor
            layer_name: Name of the layer
            
        Returns:
            Layer output
        """
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output
            return hook
        
        # Register hooks for all layers
        hooks = []
        for name, module in self.named_modules():
            if name:  # Skip empty name
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        try:
            _ = self.forward(x)
            return activations[layer_name]
        finally:
            for hook in hooks:
                hook.remove()


class ConceptBottleneckModel(nn.Module):
    """Concept Bottleneck Model (CBM) for concept-based interpretability."""
    
    def __init__(
        self,
        input_dim: int,
        concept_dim: int,
        num_classes: int,
        hidden_dims: list[int] = [64, 32],
        dropout_rate: float = 0.2,
    ) -> None:
        """Initialize CBM.
        
        Args:
            input_dim: Input feature dimension
            concept_dim: Concept dimension
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.concept_dim = concept_dim
        self.num_classes = num_classes
        
        # Feature to concept mapping
        concept_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            concept_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim
        
        concept_layers.append(nn.Linear(prev_dim, concept_dim))
        self.feature_to_concept = nn.Sequential(*concept_layers)
        
        # Concept to prediction mapping
        self.concept_to_prediction = nn.Linear(concept_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (concepts, predictions)
        """
        concepts = self.feature_to_concept(x)
        predictions = self.concept_to_prediction(concepts)
        
        return concepts, predictions
    
    def forward_concepts(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get concepts only.
        
        Args:
            x: Input tensor
            
        Returns:
            Concept activations
        """
        return self.feature_to_concept(x)
    
    def forward_from_concepts(self, concepts: torch.Tensor) -> torch.Tensor:
        """Forward pass from concepts to predictions.
        
        Args:
            concepts: Concept activations
            
        Returns:
            Predictions
        """
        return self.concept_to_prediction(concepts)


class ModelTrainer:
    """Utility class for training neural network models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
    ) -> None:
        """Initialize trainer.
        
        Args:
            model: Neural network model
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        batch_size: int = 32,
    ) -> float:
        """Train for one epoch.
        
        Args:
            X_train: Training features
            y_train: Training labels
            batch_size: Batch size
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            
            # Handle different model types
            if isinstance(outputs, tuple):
                outputs = outputs[1]  # Get predictions from CBM
            
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(
        self,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        batch_size: int = 32,
    ) -> tuple[float, float]:
        """Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            batch_size: Batch size
            
        Returns:
            Tuple of (accuracy, loss)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_test, y_test)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_X)
                
                # Handle different model types
                if isinstance(outputs, tuple):
                    outputs = outputs[1]  # Get predictions from CBM
                
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        
        return accuracy, avg_loss
    
    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train model for multiple epochs.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing training history
        """
        history = {
            "train_loss": [],
            "test_loss": [],
            "test_accuracy": [],
        }
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(X_train, y_train, batch_size)
            
            # Evaluate
            test_accuracy, test_loss = self.evaluate(X_test, y_test, batch_size)
            
            # Store history
            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["test_accuracy"].append(test_accuracy)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Test Loss: {test_loss:.4f}, "
                    f"Test Accuracy: {test_accuracy:.4f}"
                )
        
        return history
