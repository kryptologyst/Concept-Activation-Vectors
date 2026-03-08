# Concept Activation Vectors (CAV)

A comprehensive implementation of Concept Activation Vectors (CAVs) for interpretable AI, based on the TCAV (Testing with Concept Activation Vectors) methodology.

## ⚠️ Important Disclaimer

**This project is for research and educational purposes only.**

- CAV results may be unstable or misleading
- Not a substitute for human judgment  
- Should not be used for regulated decisions without human review
- Results may contain biases
- Always combine with domain expertise and critical thinking

See [DISCLAIMER.md](DISCLAIMER.md) for complete details.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Concept-Activation-Vectors.git
cd Concept-Activation-Vectors

# Install dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.main import main
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load("configs/config.yaml")

# Run CAV analysis
main(config)
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/streamlit_app.py
```

## 📁 Project Structure

```
├── src/                    # Source code
│   ├── cav/               # CAV implementation
│   │   └── tcav.py        # TCAV methodology
│   ├── data/              # Data handling
│   │   └── loader.py      # Data loading utilities
│   ├── models/            # Neural network models
│   │   └── classifier.py  # Model definitions
│   ├── eval/              # Evaluation metrics
│   │   └── metrics.py     # CAV evaluation
│   ├── viz/               # Visualization
│   │   └── plots.py       # Plotting utilities
│   ├── utils/             # Utilities
│   │   └── device.py      # Device management
│   └── main.py            # Main script
├── configs/               # Configuration files
│   └── config.yaml        # Default configuration
├── scripts/               # Utility scripts
│   └── run_cav.py         # CLI runner
├── tests/                 # Test suite
│   └── test_cav.py        # Unit tests
├── demo/                  # Interactive demos
│   └── streamlit_app.py   # Streamlit demo
├── assets/                # Generated assets
│   └── plots/             # Saved plots
├── data/                  # Data directory
├── DISCLAIMER.md          # Important disclaimers
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## What are Concept Activation Vectors?

Concept Activation Vectors (CAVs) are vectors in the activation space of neural networks that represent human-interpretable concepts. They allow us to:

- **Test concept sensitivity**: Measure how much a model's predictions change when a concept is activated
- **Understand model behavior**: Identify which concepts the model has learned
- **Provide explanations**: Generate human-understandable explanations for model decisions

### Key Components

1. **Concept Dataset**: Examples that contain (positive) or don't contain (negative) a concept
2. **CAV Training**: Learn a linear classifier in activation space to separate concept-positive from concept-negative examples
3. **Concept Sensitivity**: Measure directional derivatives to quantify concept influence
4. **Statistical Testing**: Use permutation tests to assess significance

## 🔧 Configuration

The project uses YAML configuration files. Key parameters:

```yaml
# Dataset configuration
dataset:
  name: "iris"  # Options: "iris", "synthetic"
  test_size: 0.3

# Model configuration  
model:
  hidden_dims: [64, 32]
  dropout_rate: 0.2

# Training configuration
training:
  learning_rate: 0.001
  epochs: 100
  batch_size: 32

# CAV configuration
cav:
  layer_name: "network.0"  # Layer to extract activations from
  regularization: 0.01

# Concept definitions
concepts:
  concept_definitions:
    - name: "setosa_concept"
      classes: [0]  # Iris setosa class
```

## Evaluation Metrics

The implementation provides comprehensive evaluation metrics:

### CAV Quality Metrics

- **Concept Completeness**: How well the CAV captures the concept
- **Concept Sensitivity**: How much predictions change with concept activation
- **CAV Accuracy**: Classification accuracy of the CAV classifier
- **Statistical Significance**: P-value from permutation testing

### Faithfulness Metrics

- **Concept-Prediction Correlation**: Correlation between concept scores and model confidence
- **Concept Sensitivity**: Standard deviation of concept scores

## Usage Examples

### Basic CAV Testing

```python
from src.cav.tcav import ConceptActivationVector, ConceptDataset, TCAVTester
from src.models.classifier import SimpleClassifier
from src.utils.device import get_device

# Create model
model = SimpleClassifier(input_dim=4, num_classes=3)
device = get_device()

# Create concept dataset
concept_positive = torch.randn(50, 4)
concept_negative = torch.randn(50, 4)
concept_dataset = ConceptDataset(
    positive_examples=concept_positive,
    negative_examples=concept_negative,
    concept_name="test_concept"
)

# Train CAV
cav = ConceptActivationVector(
    concept_dataset=concept_dataset,
    layer_name="network.0"
)
cav.train_cav(model, device)

# Test concept sensitivity
test_inputs = torch.randn(10, 4)
sensitivity = cav.compute_concept_sensitivity(model, test_inputs, device)
print(f"Concept sensitivity: {sensitivity:.4f}")
```

### Comprehensive Evaluation

```python
from src.eval.metrics import CAVEvaluator

evaluator = CAVEvaluator()
metrics = evaluator.evaluate_cav_quality(
    cav_vector=cav.get_concept_direction(),
    concept_dataset=concept_dataset,
    model=model,
    device=device,
    layer_name="network.0"
)

print("CAV Quality Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")
```

### Visualization

```python
from src.viz.plots import CAVVisualizer

visualizer = CAVVisualizer()

# Plot CAV direction
visualizer.plot_cav_direction(
    cav_vector=cav.get_concept_direction(),
    feature_names=["feature_1", "feature_2", "feature_3", "feature_4"],
    concept_name="test_concept"
)

# Plot evaluation metrics
visualizer.plot_cav_evaluation_metrics(metrics)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test
pytest tests/test_cav.py::TestConceptDataset::test_concept_dataset_creation
```

## Interactive Demo

The Streamlit demo provides an interactive interface for:

- **Dataset Selection**: Choose between Iris and synthetic datasets
- **Model Configuration**: Adjust architecture and training parameters
- **CAV Analysis**: Train and evaluate CAVs for different concepts
- **Visualization**: Interactive plots and dashboards
- **Results Export**: Download results as CSV

Launch the demo:

```bash
streamlit run demo/streamlit_app.py
```

## Research Applications

This implementation is suitable for:

- **Concept-based interpretability research**
- **Model debugging and analysis**
- **Educational purposes in XAI**
- **Prototyping interpretability methods**

### Limitations

- CAV quality depends on concept dataset quality
- Results may vary across different model architectures
- Statistical significance testing has limitations
- Not suitable for production use without validation

## 🛠️ Development

### Code Quality

The project uses modern Python development practices:

- **Type hints**: Full type annotation coverage
- **Documentation**: Google/NumPy style docstrings
- **Formatting**: Black code formatting
- **Linting**: Ruff for code quality
- **Testing**: Comprehensive test suite

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Adding New Features

1. **New CAV Methods**: Extend `ConceptActivationVector` class
2. **New Metrics**: Add to `CAVEvaluator` class
3. **New Visualizations**: Extend `CAVVisualizer` class
4. **New Datasets**: Add to `DataLoader` class

## References

- Kim, B., et al. "Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV)." ICML 2018.
- Concept Activation Vectors: https://github.com/tensorflow/tcav
- Interpretable AI: https://interpretable-ml-book.com/

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original TCAV implementation by Google Research
- PyTorch and Captum communities
- Streamlit for the interactive demo framework

---

**Remember**: This tool is for research and education. Always validate results with domain experts and use responsibly.
# Concept-Activation-Vectors
