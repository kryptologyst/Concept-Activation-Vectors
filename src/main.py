"""Main example script for Concept Activation Vectors testing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.cav.tcav import ConceptActivationVector, ConceptDataset, TCAVTester
from src.data.loader import DataLoader, SyntheticConceptDataset, create_concept_datasets_from_labels
from src.eval.metrics import CAVEvaluator, ConceptEvaluationMetrics
from src.models.classifier import ModelTrainer, SimpleClassifier
from src.utils.device import get_device, get_device_info, set_random_seeds
from src.viz.plots import CAVVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(config: DictConfig) -> None:
    """Main function to run CAV testing.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Starting Concept Activation Vectors (CAV) testing")
    
    # Set random seeds for reproducibility
    set_random_seeds(config.random_seed)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Device info: {get_device_info()}")
    
    # Load data
    data_loader = DataLoader()
    X_train, X_test, y_train, y_test = data_loader.load_dataset(
        dataset_name=config.dataset.name,
        test_size=config.dataset.test_size,
        random_state=config.random_seed,
    )
    
    logger.info(f"Dataset loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    # Create model
    model = SimpleClassifier(
        input_dim=X_train.shape[1],
        hidden_dims=config.model.hidden_dims,
        num_classes=len(torch.unique(y_train)),
        dropout_rate=config.model.dropout_rate,
    )
    
    logger.info(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    logger.info("Training model...")
    history = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        verbose=True,
    )
    
    # Evaluate model
    test_accuracy, test_loss = trainer.evaluate(X_test, y_test)
    logger.info(f"Final test accuracy: {test_accuracy:.4f}, test loss: {test_loss:.4f}")
    
    # Create concept datasets
    logger.info("Creating concept datasets...")
    concept_datasets = create_concept_datasets(
        X_train=X_train,
        y_train=y_train,
        config=config.concepts,
    )
    
    # Initialize TCAV tester
    tcav_tester = TCAVTester(
        model=model,
        device=device,
        random_state=config.random_seed,
    )
    
    # Train CAVs for each concept
    logger.info("Training Concept Activation Vectors...")
    cav_results = {}
    
    for concept_name, concept_dataset in concept_datasets.items():
        logger.info(f"Training CAV for concept: {concept_name}")
        
        # Add concept to tester
        cav = tcav_tester.add_concept(
            concept_dataset=concept_dataset,
            layer_name=config.cav.layer_name,
        )
        
        # Test concept sensitivity
        sensitivity = tcav_tester.test_concept_sensitivity(
            test_inputs=X_test,
            concept_name=concept_name,
            layer_name=config.cav.layer_name,
        )
        
        cav_results[concept_name] = {
            "cav": cav,
            "sensitivity": sensitivity,
        }
        
        logger.info(f"Concept '{concept_name}' sensitivity: {sensitivity:.4f}")
    
    # Comprehensive evaluation
    logger.info("Running comprehensive CAV evaluation...")
    evaluator = CAVEvaluator(random_state=config.random_seed)
    evaluation_results = {}
    
    for concept_name, result in cav_results.items():
        logger.info(f"Evaluating concept: {concept_name}")
        
        metrics = evaluator.evaluate_cav_quality(
            cav_vector=result["cav"].get_concept_direction(),
            concept_dataset=concept_datasets[concept_name],
            model=model,
            device=device,
            layer_name=config.cav.layer_name,
        )
        
        evaluation_results[concept_name] = metrics
        
        logger.info(f"Concept '{concept_name}' evaluation:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    # Visualization
    logger.info("Creating visualizations...")
    visualizer = CAVVisualizer()
    
    # Plot CAV directions
    for concept_name, result in cav_results.items():
        visualizer.plot_cav_direction(
            cav_vector=result["cav"].get_concept_direction(),
            feature_names=config.dataset.feature_names,
            concept_name=concept_name,
        )
    
    # Plot concept sensitivity
    sensitivity_scores = {name: result["sensitivity"] for name, result in cav_results.items()}
    visualizer.plot_concept_sensitivity(sensitivity_scores)
    
    # Plot evaluation metrics
    visualizer.plot_cav_evaluation_metrics(evaluation_results)
    
    # Plot model performance
    visualizer.plot_model_performance(history)
    
    # Plot concept comparison
    visualizer.plot_concept_comparison(evaluation_results)
    
    # Create interactive dashboard
    interactive_fig = visualizer.create_interactive_dashboard(
        evaluation_results=evaluation_results,
        concept_names=list(concept_datasets.keys()),
    )
    
    # Save interactive dashboard
    interactive_fig.write_html(str(visualizer.output_dir / "cav_dashboard.html"))
    
    # Print summary
    print_summary(cav_results, evaluation_results)
    
    logger.info("CAV testing completed successfully!")


def create_concept_datasets(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    config: DictConfig,
) -> Dict[str, ConceptDataset]:
    """Create concept datasets for testing.
    
    Args:
        X_train: Training features
        y_train: Training labels
        config: Configuration for concepts
        
    Returns:
        Dictionary mapping concept names to ConceptDataset objects
    """
    concept_datasets = {}
    
    # Create concepts based on class labels
    for concept_config in config.concept_definitions:
        concept_name = concept_config.name
        concept_classes = concept_config.classes
        
        # Create concept datasets from class labels
        concept_positive, concept_negative = create_concept_datasets_from_labels(
            X=X_train,
            y=y_train,
            concept_name=concept_name,
            concept_classes=concept_classes,
            random_state=config.random_seed,
        )
        
        concept_dataset = ConceptDataset(
            positive_examples=concept_positive,
            negative_examples=concept_negative,
            concept_name=concept_name,
        )
        
        concept_datasets[concept_name] = concept_dataset
        
        logger.info(f"Created concept '{concept_name}': "
                   f"{len(concept_positive)} positive, {len(concept_negative)} negative examples")
    
    return concept_datasets


def print_summary(
    cav_results: Dict[str, Dict],
    evaluation_results: Dict[str, Dict[str, float]],
) -> None:
    """Print summary of results.
    
    Args:
        cav_results: CAV results
        evaluation_results: Evaluation results
    """
    print("\n" + "="*80)
    print("CONCEPT ACTIVATION VECTORS (CAV) TESTING SUMMARY")
    print("="*80)
    
    print(f"\nNumber of concepts tested: {len(cav_results)}")
    
    print("\nConcept Sensitivity Scores:")
    print("-" * 40)
    for concept_name, result in cav_results.items():
        sensitivity = result["sensitivity"]
        accuracy = result["cav"].cav_accuracy
        print(f"{concept_name:20s}: Sensitivity = {sensitivity:.4f}, CAV Accuracy = {accuracy:.4f}")
    
    print("\nComprehensive Evaluation Metrics:")
    print("-" * 40)
    for concept_name, metrics in evaluation_results.items():
        print(f"\n{concept_name}:")
        for metric, value in metrics.items():
            print(f"  {metric:25s}: {value:.4f}")
    
    print("\n" + "="*80)
    print("IMPORTANT DISCLAIMER:")
    print("CAV results may be unstable or misleading.")
    print("Always combine with domain expertise and human judgment.")
    print("Not suitable for regulated decisions without human review.")
    print("="*80)


if __name__ == "__main__":
    # Load configuration
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        config = OmegaConf.load(config_path)
    else:
        # Default configuration
        config = OmegaConf.create({
            "random_seed": 42,
            "dataset": {
                "name": "iris",
                "test_size": 0.3,
                "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            },
            "model": {
                "hidden_dims": [64, 32],
                "dropout_rate": 0.2,
            },
            "training": {
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
                "epochs": 100,
                "batch_size": 32,
            },
            "cav": {
                "layer_name": "network.0",  # First hidden layer
            },
            "concepts": {
                "random_seed": 42,
                "concept_definitions": [
                    {
                        "name": "setosa_concept",
                        "classes": [0],  # Setosa class
                    },
                    {
                        "name": "versicolor_concept", 
                        "classes": [1],  # Versicolor class
                    },
                    {
                        "name": "virginica_concept",
                        "classes": [2],  # Virginica class
                    },
                ],
            },
        })
    
    main(config)
