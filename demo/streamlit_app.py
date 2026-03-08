"""Streamlit demo for Concept Activation Vectors (CAV) testing."""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.cav.tcav import ConceptActivationVector, ConceptDataset, TCAVTester
from src.data.loader import DataLoader, SyntheticConceptDataset, create_concept_datasets_from_labels
from src.eval.metrics import CAVEvaluator
from src.models.classifier import ModelTrainer, SimpleClassifier
from src.utils.device import get_device, get_device_info, set_random_seeds
from src.viz.plots import CAVVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Concept Activation Vectors (CAV) Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def main() -> None:
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">Concept Activation Vectors (CAV) Demo</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        <h4>⚠️ Important Disclaimer</h4>
        <p><strong>This demo is for research and educational purposes only.</strong></p>
        <ul>
            <li>CAV results may be unstable or misleading</li>
            <li>Not a substitute for human judgment</li>
            <li>Should not be used for regulated decisions without human review</li>
            <li>Results may contain biases</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Dataset selection
    dataset_name = st.sidebar.selectbox(
        "Dataset",
        ["iris", "synthetic"],
        help="Choose the dataset for CAV testing"
    )
    
    # Model configuration
    st.sidebar.subheader("Model Configuration")
    hidden_dims = st.sidebar.multiselect(
        "Hidden Layer Dimensions",
        [32, 64, 128, 256],
        default=[64, 32],
        help="Select hidden layer dimensions"
    )
    dropout_rate = st.sidebar.slider(
        "Dropout Rate",
        0.0, 0.5, 0.2, 0.05,
        help="Dropout rate for regularization"
    )
    
    # Training configuration
    st.sidebar.subheader("Training Configuration")
    epochs = st.sidebar.slider("Epochs", 10, 200, 100, 10)
    learning_rate = st.sidebar.selectbox(
        "Learning Rate",
        [0.001, 0.01, 0.1],
        index=0
    )
    
    # CAV configuration
    st.sidebar.subheader("CAV Configuration")
    layer_name = st.sidebar.selectbox(
        "Layer for CAV",
        ["network.0", "network.2", "network.4"],
        help="Layer to extract activations from"
    )
    
    # Random seed
    random_seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=1000,
        value=42,
        help="Random seed for reproducibility"
    )
    
    # Run button
    if st.sidebar.button("🚀 Run CAV Analysis", type="primary"):
        run_cav_analysis(
            dataset_name=dataset_name,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            epochs=epochs,
            learning_rate=learning_rate,
            layer_name=layer_name,
            random_seed=random_seed,
        )


def run_cav_analysis(
    dataset_name: str,
    hidden_dims: List[int],
    dropout_rate: float,
    epochs: int,
    learning_rate: float,
    layer_name: str,
    random_seed: int,
) -> None:
    """Run CAV analysis with given parameters.
    
    Args:
        dataset_name: Name of the dataset
        hidden_dims: Hidden layer dimensions
        dropout_rate: Dropout rate
        epochs: Number of training epochs
        learning_rate: Learning rate
        layer_name: Layer name for CAV
        random_seed: Random seed
    """
    
    # Set random seeds
    set_random_seeds(random_seed)
    
    # Get device info
    device = get_device()
    device_info = get_device_info()
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Load data
    status_text.text("Loading dataset...")
    progress_bar.progress(10)
    
    data_loader = DataLoader()
    X_train, X_test, y_train, y_test = data_loader.load_dataset(
        dataset_name=dataset_name,
        test_size=0.3,
        random_state=random_seed,
    )
    
    # Step 2: Create model
    status_text.text("Creating model...")
    progress_bar.progress(20)
    
    model = SimpleClassifier(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        num_classes=len(torch.unique(y_train)),
        dropout_rate=dropout_rate,
    )
    
    # Step 3: Train model
    status_text.text("Training model...")
    progress_bar.progress(30)
    
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
    )
    
    # Train model
    history = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=epochs,
        verbose=False,
    )
    
    # Step 4: Create concept datasets
    status_text.text("Creating concept datasets...")
    progress_bar.progress(50)
    
    concept_datasets = create_concept_datasets(X_train, y_train, dataset_name)
    
    # Step 5: Train CAVs
    status_text.text("Training Concept Activation Vectors...")
    progress_bar.progress(70)
    
    tcav_tester = TCAVTester(
        model=model,
        device=device,
        random_state=random_seed,
    )
    
    cav_results = {}
    for concept_name, concept_dataset in concept_datasets.items():
        cav = tcav_tester.add_concept(
            concept_dataset=concept_dataset,
            layer_name=layer_name,
        )
        
        sensitivity = tcav_tester.test_concept_sensitivity(
            test_inputs=X_test,
            concept_name=concept_name,
            layer_name=layer_name,
        )
        
        cav_results[concept_name] = {
            "cav": cav,
            "sensitivity": sensitivity,
        }
    
    # Step 6: Evaluation
    status_text.text("Running evaluation...")
    progress_bar.progress(90)
    
    evaluator = CAVEvaluator(random_state=random_seed)
    evaluation_results = {}
    
    for concept_name, result in cav_results.items():
        metrics = evaluator.evaluate_cav_quality(
            cav_vector=result["cav"].get_concept_direction(),
            concept_dataset=concept_datasets[concept_name],
            model=model,
            device=device,
            layer_name=layer_name,
        )
        evaluation_results[concept_name] = metrics
    
    # Complete
    status_text.text("Analysis complete!")
    progress_bar.progress(100)
    
    # Display results
    display_results(
        cav_results=cav_results,
        evaluation_results=evaluation_results,
        history=history,
        device_info=device_info,
        dataset_name=dataset_name,
    )


def create_concept_datasets(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    dataset_name: str,
) -> Dict[str, ConceptDataset]:
    """Create concept datasets.
    
    Args:
        X_train: Training features
        y_train: Training labels
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary of concept datasets
    """
    concept_datasets = {}
    
    if dataset_name == "iris":
        # Create concepts based on iris classes
        class_names = ["Setosa", "Versicolor", "Virginica"]
        for i, class_name in enumerate(class_names):
            concept_positive, concept_negative = create_concept_datasets_from_labels(
                X=X_train,
                y=y_train,
                concept_name=f"{class_name.lower()}_concept",
                concept_classes=[i],
                random_state=42,
            )
            
            concept_dataset = ConceptDataset(
                positive_examples=concept_positive,
                negative_examples=concept_negative,
                concept_name=f"{class_name.lower()}_concept",
            )
            
            concept_datasets[f"{class_name.lower()}_concept"] = concept_dataset
    
    elif dataset_name == "synthetic":
        # Create synthetic concepts
        X, y, _, _ = SyntheticConceptDataset.generate_tabular_concept_data(
            n_samples=1000,
            random_state=42,
        )
        
        # Create concepts based on synthetic data
        for i in range(3):  # 3 classes
            concept_positive, concept_negative = create_concept_datasets_from_labels(
                X=X_train,
                y=y_train,
                concept_name=f"class_{i}_concept",
                concept_classes=[i],
                random_state=42,
            )
            
            concept_dataset = ConceptDataset(
                positive_examples=concept_positive,
                negative_examples=concept_negative,
                concept_name=f"class_{i}_concept",
            )
            
            concept_datasets[f"class_{i}_concept"] = concept_dataset
    
    return concept_datasets


def display_results(
    cav_results: Dict[str, Dict],
    evaluation_results: Dict[str, Dict[str, float]],
    history: Dict[str, List[float]],
    device_info: Dict[str, str],
    dataset_name: str,
) -> None:
    """Display CAV analysis results.
    
    Args:
        cav_results: CAV results
        evaluation_results: Evaluation results
        history: Training history
        device_info: Device information
        dataset_name: Name of the dataset
    """
    
    # Device info
    st.subheader("🔧 System Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Device", device_info["device"])
    
    with col2:
        if "gpu_name" in device_info:
            st.metric("GPU", device_info["gpu_name"])
        else:
            st.metric("CPU Cores", device_info.get("cpu_cores", "N/A"))
    
    with col3:
        st.metric("Dataset", dataset_name.title())
    
    # Model performance
    st.subheader("📊 Model Performance")
    
    # Create performance plots
    fig_perf = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Training Loss", "Test Accuracy"),
    )
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig_perf.add_trace(
        go.Scatter(x=epochs, y=history["train_loss"], name="Train Loss", line=dict(color="blue")),
        row=1, col=1
    )
    fig_perf.add_trace(
        go.Scatter(x=epochs, y=history["test_loss"], name="Test Loss", line=dict(color="red")),
        row=1, col=1
    )
    fig_perf.add_trace(
        go.Scatter(x=epochs, y=history["test_accuracy"], name="Test Accuracy", line=dict(color="green")),
        row=1, col=2
    )
    
    fig_perf.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig_perf, use_container_width=True)
    
    # CAV Results
    st.subheader("🧠 Concept Activation Vector Results")
    
    # Create results dataframe
    results_data = []
    for concept_name, result in cav_results.items():
        row = {
            "Concept": concept_name.replace("_concept", "").title(),
            "Sensitivity": result["sensitivity"],
            "CAV Accuracy": result["cav"].cav_accuracy,
            "Completeness": evaluation_results[concept_name]["concept_completeness"],
            "Statistical Significance": evaluation_results[concept_name]["statistical_significance"],
        }
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Concepts Tested", len(cav_results))
    
    with col2:
        avg_sensitivity = results_df["Sensitivity"].mean()
        st.metric("Avg Sensitivity", f"{avg_sensitivity:.3f}")
    
    with col3:
        avg_accuracy = results_df["CAV Accuracy"].mean()
        st.metric("Avg CAV Accuracy", f"{avg_accuracy:.3f}")
    
    with col4:
        avg_completeness = results_df["Completeness"].mean()
        st.metric("Avg Completeness", f"{avg_completeness:.3f}")
    
    # Results table
    st.dataframe(results_df, use_container_width=True)
    
    # CAV Direction Visualization
    st.subheader("🎯 CAV Direction Analysis")
    
    # Select concept for detailed analysis
    selected_concept = st.selectbox(
        "Select concept for detailed analysis",
        list(cav_results.keys()),
        format_func=lambda x: x.replace("_concept", "").title()
    )
    
    # Get CAV direction
    cav_vector = cav_results[selected_concept]["cav"].get_concept_direction()
    
    # Create CAV direction plot
    fig_cav = go.Figure()
    
    cav_np = cav_vector.cpu().numpy()
    feature_names = [f"Feature {i}" for i in range(len(cav_np))]
    
    colors = ['red' if x < 0 else 'blue' for x in cav_np]
    
    fig_cav.add_trace(go.Bar(
        x=feature_names,
        y=cav_np,
        marker_color=colors,
        name="CAV Direction"
    ))
    
    fig_cav.update_layout(
        title=f"CAV Direction for {selected_concept.replace('_concept', '').title()}",
        xaxis_title="Feature Index",
        yaxis_title="CAV Direction Value",
        height=400
    )
    
    st.plotly_chart(fig_cav, use_container_width=True)
    
    # Evaluation Metrics Heatmap
    st.subheader("📈 Evaluation Metrics Heatmap")
    
    # Prepare data for heatmap
    concepts = list(evaluation_results.keys())
    metrics = ["concept_completeness", "concept_sensitivity", "cav_accuracy", "statistical_significance"]
    
    heatmap_data = []
    for concept in concepts:
        row = [evaluation_results[concept].get(metric, 0) for metric in metrics]
        heatmap_data.append(row)
    
    heatmap_data = np.array(heatmap_data)
    
    # Create heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[m.replace("_", " ").title() for m in metrics],
        y=[c.replace("_concept", "").title() for c in concepts],
        colorscale='YlOrRd',
        text=np.round(heatmap_data, 3),
        texttemplate="%{text}",
        textfont={"size": 12},
    ))
    
    fig_heatmap.update_layout(
        title="CAV Evaluation Metrics Heatmap",
        height=400
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Download results
    st.subheader("💾 Download Results")
    
    # Create downloadable CSV
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name=f"cav_results_{dataset_name}.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
