"""Scripts for running CAV experiments."""

#!/usr/bin/env python3
"""Run CAV analysis with different configurations."""

import argparse
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.main import main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_config(
    dataset: str = "iris",
    epochs: int = 100,
    learning_rate: float = 0.001,
    hidden_dims: list = None,
    random_seed: int = 42,
) -> OmegaConf:
    """Create configuration for CAV analysis.
    
    Args:
        dataset: Dataset name
        epochs: Number of training epochs
        learning_rate: Learning rate
        hidden_dims: Hidden layer dimensions
        random_seed: Random seed
        
    Returns:
        Configuration object
    """
    if hidden_dims is None:
        hidden_dims = [64, 32]
    
    config = OmegaConf.create({
        "random_seed": random_seed,
        "dataset": {
            "name": dataset,
            "test_size": 0.3,
            "feature_names": ["feature_1", "feature_2", "feature_3", "feature_4"],
        },
        "model": {
            "hidden_dims": hidden_dims,
            "dropout_rate": 0.2,
        },
        "training": {
            "learning_rate": learning_rate,
            "weight_decay": 1e-4,
            "epochs": epochs,
            "batch_size": 32,
        },
        "cav": {
            "layer_name": "network.0",
        },
        "concepts": {
            "random_seed": random_seed,
            "concept_definitions": [
                {"name": "class_0_concept", "classes": [0]},
                {"name": "class_1_concept", "classes": [1]},
                {"name": "class_2_concept", "classes": [2]},
            ],
        },
    })
    
    return config


def main_cli():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Run CAV analysis")
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="iris",
        choices=["iris", "synthetic"],
        help="Dataset to use"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[64, 32],
        help="Hidden layer dimensions"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_config(
        dataset=args.dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dims=args.hidden_dims,
        random_seed=args.random_seed,
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Starting CAV analysis with config: {config}")
    logger.info(f"Output directory: {output_dir}")
    
    # Run analysis
    try:
        main(config)
        logger.info("CAV analysis completed successfully!")
    except Exception as e:
        logger.error(f"CAV analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
