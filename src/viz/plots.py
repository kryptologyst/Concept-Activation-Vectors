"""Visualization utilities for Concept Activation Vectors."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import torch
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class CAVVisualizer:
    """Visualizer for Concept Activation Vector results."""
    
    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir or Path("assets/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")
    
    def plot_cav_direction(
        self,
        cav_vector: torch.Tensor,
        feature_names: List[str],
        concept_name: str,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot CAV direction vector as a bar chart.
        
        Args:
            cav_vector: CAV direction vector
            feature_names: Names of features
            concept_name: Name of the concept
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert to numpy and sort by absolute value
        cav_np = cav_vector.cpu().numpy()
        abs_values = np.abs(cav_np)
        sorted_indices = np.argsort(abs_values)[::-1]
        
        # Plot bars
        colors = ['red' if x < 0 else 'blue' for x in cav_np[sorted_indices]]
        bars = ax.bar(
            range(len(cav_np)),
            cav_np[sorted_indices],
            color=colors,
            alpha=0.7,
        )
        
        # Customize plot
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("CAV Direction Value")
        ax.set_title(f"CAV Direction for Concept: {concept_name}")
        ax.grid(True, alpha=0.3)
        
        # Add feature names if provided
        if feature_names:
            ax.set_xticks(range(len(feature_names)))
            ax.set_xticklabels([feature_names[i] for i in sorted_indices], rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.output_dir / f"cav_direction_{concept_name}.png", dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def plot_concept_sensitivity(
        self,
        sensitivity_scores: Dict[str, float],
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot concept sensitivity scores.
        
        Args:
            sensitivity_scores: Dictionary mapping concept names to sensitivity scores
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        concepts = list(sensitivity_scores.keys())
        scores = list(sensitivity_scores.values())
        
        bars = ax.bar(concepts, scores, alpha=0.7, color='skyblue')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{score:.3f}',
                ha='center', va='bottom'
            )
        
        ax.set_xlabel("Concepts")
        ax.set_ylabel("Sensitivity Score")
        ax.set_title("Concept Sensitivity Scores")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.output_dir / "concept_sensitivity.png", dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def plot_cav_evaluation_metrics(
        self,
        evaluation_results: Dict[str, Dict[str, float]],
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot comprehensive CAV evaluation metrics.
        
        Args:
            evaluation_results: Dictionary containing evaluation results
            save_path: Path to save the plot
        """
        # Extract data
        concepts = list(evaluation_results.keys())
        metrics = ["concept_completeness", "concept_sensitivity", "cav_accuracy", "statistical_significance"]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = [evaluation_results[concept].get(metric, 0) for concept in concepts]
            
            bars = ax.bar(concepts, values, alpha=0.7)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.01,
                    f'{value:.3f}',
                    ha='center', va='bottom'
                )
            
            ax.set_title(metric.replace("_", " ").title())
            ax.set_ylabel("Score")
            ax.grid(True, alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.output_dir / "cav_evaluation_metrics.png", dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def plot_concept_activations(
        self,
        activations: torch.Tensor,
        concept_scores: torch.Tensor,
        concept_name: str,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot concept activations vs concept scores.
        
        Args:
            activations: Activation values
            concept_scores: Concept scores
            concept_name: Name of the concept
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Concept scores distribution
        ax1.hist(concept_scores.cpu().numpy(), bins=30, alpha=0.7, color='skyblue')
        ax1.set_xlabel("Concept Score")
        ax1.set_ylabel("Frequency")
        ax1.set_title(f"Concept Score Distribution: {concept_name}")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Activations vs concept scores
        activations_np = activations.cpu().numpy()
        concept_scores_np = concept_scores.cpu().numpy()
        
        scatter = ax2.scatter(
            activations_np[:, 0], 
            activations_np[:, 1], 
            c=concept_scores_np, 
            cmap='viridis',
            alpha=0.6
        )
        ax2.set_xlabel("Activation Dimension 1")
        ax2.set_ylabel("Activation Dimension 2")
        ax2.set_title(f"Activations Colored by Concept Score: {concept_name}")
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax2, label="Concept Score")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.output_dir / f"concept_activations_{concept_name}.png", dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def create_interactive_dashboard(
        self,
        evaluation_results: Dict[str, Dict[str, float]],
        concept_names: List[str],
    ) -> go.Figure:
        """Create interactive dashboard using Plotly.
        
        Args:
            evaluation_results: Dictionary containing evaluation results
            concept_names: List of concept names
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Concept Completeness", "Concept Sensitivity", 
                          "CAV Accuracy", "Statistical Significance"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        metrics = ["concept_completeness", "concept_sensitivity", "cav_accuracy", "statistical_significance"]
        
        for i, metric in enumerate(metrics):
            row = i // 2 + 1
            col = i % 2 + 1
            
            values = [evaluation_results[concept].get(metric, 0) for concept in concept_names]
            
            fig.add_trace(
                go.Bar(
                    x=concept_names,
                    y=values,
                    name=metric.replace("_", " ").title(),
                    showlegend=False,
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title_text="CAV Evaluation Dashboard",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def plot_model_performance(
        self,
        train_history: Dict[str, List[float]],
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot model training performance.
        
        Args:
            train_history: Training history dictionary
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = range(1, len(train_history["train_loss"]) + 1)
        
        # Plot loss
        ax1.plot(epochs, train_history["train_loss"], label="Train Loss", color='blue')
        ax1.plot(epochs, train_history["test_loss"], label="Test Loss", color='red')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Test Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(epochs, train_history["test_accuracy"], label="Test Accuracy", color='green')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Test Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.output_dir / "model_performance.png", dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def plot_concept_comparison(
        self,
        concept_results: Dict[str, Dict[str, float]],
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot comparison of different concepts.
        
        Args:
            concept_results: Results for different concepts
            save_path: Path to save the plot
        """
        # Create heatmap
        concepts = list(concept_results.keys())
        metrics = ["concept_completeness", "concept_sensitivity", "cav_accuracy"]
        
        data = []
        for concept in concepts:
            row = [concept_results[concept].get(metric, 0) for metric in metrics]
            data.append(row)
        
        data = np.array(data)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(concepts)))
        ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
        ax.set_yticklabels(concepts)
        
        # Add text annotations
        for i in range(len(concepts)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data[i, j]:.3f}',
                             ha="center", va="center", color="black")
        
        ax.set_title("Concept Comparison Heatmap")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Score", rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.output_dir / "concept_comparison.png", dpi=300, bbox_inches="tight")
        
        plt.show()
