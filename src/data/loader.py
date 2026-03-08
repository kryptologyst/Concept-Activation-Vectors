"""Data generation and loading utilities for CAV testing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SyntheticConceptDataset:
    """Generate synthetic datasets for concept testing."""
    
    @staticmethod
    def generate_tabular_concept_data(
        n_samples: int = 1000,
        n_features: int = 4,
        n_classes: int = 3,
        concept_strength: float = 0.5,
        random_state: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate synthetic tabular data with a concept.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features
            n_classes: Number of classes
            concept_strength: Strength of the concept signal
            random_state: Random seed
            
        Returns:
            Tuple of (X, y, concept_positive, concept_negative)
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Generate base classification data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_redundant=0,
            n_informative=n_features,
            random_state=random_state,
        )
        
        # Create concept based on first two features
        # Concept: "high values in first two features"
        concept_scores = X[:, 0] + X[:, 1]
        concept_threshold = np.percentile(concept_scores, 70)
        
        concept_positive_mask = concept_scores > concept_threshold
        concept_negative_mask = concept_scores <= concept_threshold
        
        # Add noise to make it more realistic
        noise = np.random.normal(0, 0.1, X.shape)
        X_noisy = X + noise * concept_strength
        
        # Convert to tensors
        X_tensor = torch.tensor(X_noisy, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        concept_positive = X_tensor[concept_positive_mask]
        concept_negative = X_tensor[concept_negative_mask]
        
        return X_tensor, y_tensor, concept_positive, concept_negative
    
    @staticmethod
    def generate_iris_concept_data(
        concept_feature: str = "petal_length",
        concept_threshold: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate concept data from Iris dataset.
        
        Args:
            concept_feature: Feature to base concept on
            concept_threshold: Threshold for concept (if None, uses median)
            random_state: Random seed
            
        Returns:
            Tuple of (X, y, concept_positive, concept_negative)
        """
        # Load Iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target
        feature_names = iris.feature_names
        
        # Find feature index
        try:
            feature_idx = feature_names.index(concept_feature)
        except ValueError:
            raise ValueError(f"Feature '{concept_feature}' not found in Iris dataset")
        
        # Set threshold
        if concept_threshold is None:
            concept_threshold = np.median(X[:, feature_idx])
        
        # Create concept masks
        concept_positive_mask = X[:, feature_idx] > concept_threshold
        concept_negative_mask = X[:, feature_idx] <= concept_threshold
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        concept_positive = X_tensor[concept_positive_mask]
        concept_negative = X_tensor[concept_negative_mask]
        
        return X_tensor, y_tensor, concept_positive, concept_negative


class DataLoader:
    """Data loading and preprocessing utilities."""
    
    def __init__(self, data_dir: Optional[Path] = None) -> None:
        """Initialize data loader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir or Path("data")
        self.data_dir.mkdir(exist_ok=True)
    
    def load_dataset(
        self,
        dataset_name: str,
        test_size: float = 0.3,
        random_state: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load and split dataset.
        
        Args:
            dataset_name: Name of dataset to load
            test_size: Fraction of data for testing
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if dataset_name == "iris":
            iris = load_iris()
            X, y = iris.data, iris.target
        elif dataset_name == "synthetic":
            X, y, _, _ = SyntheticConceptDataset.generate_tabular_concept_data(
                random_state=random_state
            )
            X = X.numpy()
            y = y.numpy()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Normalize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
        return X_train, X_test, y_train, y_test
    
    def save_dataset_metadata(
        self,
        dataset_name: str,
        feature_names: List[str],
        class_names: List[str],
        sensitive_attributes: Optional[List[str]] = None,
    ) -> None:
        """Save dataset metadata.
        
        Args:
            dataset_name: Name of the dataset
            feature_names: List of feature names
            class_names: List of class names
            sensitive_attributes: List of sensitive attribute names
        """
        metadata = {
            "dataset_name": dataset_name,
            "feature_names": feature_names,
            "class_names": class_names,
            "sensitive_attributes": sensitive_attributes or [],
            "n_features": len(feature_names),
            "n_classes": len(class_names),
        }
        
        metadata_path = self.data_dir / f"{dataset_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def load_dataset_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """Load dataset metadata.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing metadata
        """
        metadata_path = self.data_dir / f"{dataset_name}_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, "r") as f:
            return json.load(f)


def create_concept_datasets_from_labels(
    X: torch.Tensor,
    y: torch.Tensor,
    concept_name: str,
    concept_classes: List[int],
    random_state: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create concept datasets from class labels.
    
    Args:
        X: Input features
        y: Class labels
        concept_name: Name of the concept
        concept_classes: Classes that represent the concept
        random_state: Random seed
        
    Returns:
        Tuple of (concept_positive, concept_negative)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Create concept masks
    concept_positive_mask = torch.isin(y, torch.tensor(concept_classes))
    concept_negative_mask = ~concept_positive_mask
    
    # Get examples
    concept_positive = X[concept_positive_mask]
    concept_negative = X[concept_negative_mask]
    
    return concept_positive, concept_negative
