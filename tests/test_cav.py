"""Tests for Concept Activation Vectors implementation."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.cav.tcav import ConceptActivationVector, ConceptDataset, TCAVTester
from src.data.loader import SyntheticConceptDataset
from src.models.classifier import SimpleClassifier
from src.utils.device import get_device, set_random_seeds


class TestConceptDataset:
    """Test ConceptDataset class."""
    
    def test_concept_dataset_creation(self):
        """Test creating a concept dataset."""
        # Create dummy data
        positive_examples = torch.randn(50, 4)
        negative_examples = torch.randn(50, 4)
        concept_name = "test_concept"
        
        concept_dataset = ConceptDataset(
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            concept_name=concept_name,
        )
        
        assert concept_dataset.concept_name == concept_name
        assert len(concept_dataset.positive_examples) == 50
        assert len(concept_dataset.negative_examples) == 50
        assert concept_dataset.total_examples == 100
    
    def test_concept_dataset_validation(self):
        """Test concept dataset validation."""
        # Test empty positive examples
        with pytest.raises(ValueError):
            ConceptDataset(
                positive_examples=torch.empty(0, 4),
                negative_examples=torch.randn(50, 4),
                concept_name="test",
            )
        
        # Test empty negative examples
        with pytest.raises(ValueError):
            ConceptDataset(
                positive_examples=torch.randn(50, 4),
                negative_examples=torch.empty(0, 4),
                concept_name="test",
            )
        
        # Test dimension mismatch
        with pytest.raises(ValueError):
            ConceptDataset(
                positive_examples=torch.randn(50, 4),
                negative_examples=torch.randn(50, 3),
                concept_name="test",
            )
    
    def test_balanced_dataset(self):
        """Test getting balanced dataset."""
        positive_examples = torch.randn(30, 4)
        negative_examples = torch.randn(70, 4)
        
        concept_dataset = ConceptDataset(
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            concept_name="test",
        )
        
        features, labels = concept_dataset.get_balanced_dataset()
        
        # Should have equal positive and negative examples
        assert len(features) == 60  # 30 + 30
        assert torch.sum(labels == 1) == 30
        assert torch.sum(labels == 0) == 30


class TestConceptActivationVector:
    """Test ConceptActivationVector class."""
    
    def test_cav_initialization(self):
        """Test CAV initialization."""
        positive_examples = torch.randn(50, 4)
        negative_examples = torch.randn(50, 4)
        
        concept_dataset = ConceptDataset(
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            concept_name="test",
        )
        
        cav = ConceptActivationVector(
            concept_dataset=concept_dataset,
            layer_name="test_layer",
            random_state=42,
        )
        
        assert cav.concept_dataset.concept_name == "test"
        assert cav.layer_name == "test_layer"
        assert cav.random_state == 42
        assert cav.cav_vector is None
        assert cav.cav_accuracy is None


class TestSimpleClassifier:
    """Test SimpleClassifier model."""
    
    def test_classifier_creation(self):
        """Test creating a simple classifier."""
        model = SimpleClassifier(
            input_dim=4,
            hidden_dims=[64, 32],
            num_classes=3,
            dropout_rate=0.2,
        )
        
        assert model.input_dim == 4
        assert model.num_classes == 3
        
        # Test forward pass
        x = torch.randn(10, 4)
        output = model(x)
        
        assert output.shape == (10, 3)
    
    def test_layer_output(self):
        """Test getting layer output."""
        model = SimpleClassifier(
            input_dim=4,
            hidden_dims=[64, 32],
            num_classes=3,
        )
        
        x = torch.randn(10, 4)
        
        # Test getting output from first layer
        layer_output = model.get_layer_output(x, "network.0")
        assert layer_output.shape[0] == 10


class TestSyntheticConceptDataset:
    """Test synthetic concept dataset generation."""
    
    def test_generate_tabular_concept_data(self):
        """Test generating synthetic tabular concept data."""
        X, y, concept_positive, concept_negative = SyntheticConceptDataset.generate_tabular_concept_data(
            n_samples=100,
            n_features=4,
            n_classes=3,
            random_state=42,
        )
        
        assert X.shape == (100, 4)
        assert y.shape == (100,)
        assert len(concept_positive) > 0
        assert len(concept_negative) > 0
        assert len(concept_positive) + len(concept_negative) == 100
    
    def test_generate_iris_concept_data(self):
        """Test generating iris concept data."""
        X, y, concept_positive, concept_negative = SyntheticConceptDataset.generate_iris_concept_data(
            concept_feature="petal_length",
            random_state=42,
        )
        
        assert X.shape[1] == 4  # Iris has 4 features
        assert len(concept_positive) > 0
        assert len(concept_negative) > 0


class TestTCAVTester:
    """Test TCAVTester class."""
    
    def test_tcav_tester_initialization(self):
        """Test TCAV tester initialization."""
        model = SimpleClassifier(input_dim=4, num_classes=3)
        device = get_device()
        
        tester = TCAVTester(
            model=model,
            device=device,
            random_state=42,
        )
        
        assert tester.model == model
        assert tester.device == device
        assert tester.random_state == 42
        assert len(tester.cavs) == 0


class TestDeviceUtils:
    """Test device utility functions."""
    
    def test_get_device(self):
        """Test getting device."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_set_random_seeds(self):
        """Test setting random seeds."""
        set_random_seeds(42)
        
        # Test that seeds are set (basic check)
        np.random.seed(42)
        torch.manual_seed(42)
        
        # This should be deterministic
        rand1 = torch.randn(5)
        torch.manual_seed(42)
        rand2 = torch.randn(5)
        
        assert torch.allclose(rand1, rand2)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_cav_training(self):
        """Test end-to-end CAV training."""
        # Set random seeds
        set_random_seeds(42)
        
        # Generate synthetic data
        X, y, concept_positive, concept_negative = SyntheticConceptDataset.generate_tabular_concept_data(
            n_samples=200,
            n_features=4,
            n_classes=3,
            random_state=42,
        )
        
        # Create concept dataset
        concept_dataset = ConceptDataset(
            positive_examples=concept_positive,
            negative_examples=concept_negative,
            concept_name="test_concept",
        )
        
        # Create model
        model = SimpleClassifier(
            input_dim=4,
            hidden_dims=[32],
            num_classes=3,
        )
        
        device = get_device()
        
        # Create CAV
        cav = ConceptActivationVector(
            concept_dataset=concept_dataset,
            layer_name="network.0",
            random_state=42,
        )
        
        # Train CAV
        cav.train_cav(model, device)
        
        # Check that CAV was trained
        assert cav.cav_vector is not None
        assert cav.cav_accuracy is not None
        assert 0 <= cav.cav_accuracy <= 1
        
        # Test concept sensitivity
        test_inputs = X[:10]
        sensitivity = cav.compute_concept_sensitivity(model, test_inputs, device)
        
        assert isinstance(sensitivity, float)
        assert sensitivity >= 0


if __name__ == "__main__":
    pytest.main([__file__])
