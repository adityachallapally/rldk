# Enhanced Data Lineage & Reproducibility Tracking System

## Overview

This comprehensive tracking system provides enhanced data lineage and reproducibility for machine learning experiments. It captures and tracks all critical components needed to reproduce experiments, including datasets, models, environment state, random seeds, and Git repository information.

## Features

### 🔍 **Dataset Versioning & Checksums**
- Automatic checksum computation for datasets
- Support for various data types (NumPy arrays, Pandas DataFrames, PyTorch datasets, Hugging Face datasets, files)
- Efficient handling of large datasets with sampling for checksum computation
- Dataset metadata tracking (size, type, preprocessing steps)

### 🧠 **Model Architecture Fingerprinting**
- Model architecture checksum computation
- Parameter count and structure tracking
- Support for PyTorch models, Hugging Face models, and custom models
- Model metadata capture (architecture type, hyperparameters, etc.)

### 🌍 **Environment State Capture**
- Complete environment snapshot (Python version, system info, dependencies)
- Conda environment capture
- Pip freeze output
- ML framework versions (PyTorch, NumPy, Transformers, etc.)
- System information (CPU, memory, disk usage)

### 🎲 **Random Seed Tracking**
- Comprehensive seed management across all components
- Python, NumPy, PyTorch, and CUDA seed tracking
- Reproducible environment creation
- Seed state save/load functionality

### 📝 **Git Integration**
- Git commit hash capture
- Repository state tracking
- Modified files detection
- Branch and tag information

## Installation

The tracking system is part of the RLDK package. Install the required dependencies:

```bash
pip install torch numpy pandas transformers datasets scikit-learn pyyaml
```

## Quick Start

### Basic Usage

```python
from rldk.tracking import ExperimentTracker, TrackingConfig

# Create configuration
config = TrackingConfig(
    experiment_name="my_experiment",
    output_dir="./runs",
    tags=["classification", "pytorch"]
)

# Initialize tracker
tracker = ExperimentTracker(config)

# Start experiment (captures environment, git, seeds)
tracker.start_experiment()

# Track datasets
import numpy as np
X_train = np.random.randn(1000, 10)
tracker.track_dataset(X_train, "X_train", {"split": "train"})

# Track models
import torch.nn as nn
model = nn.Linear(10, 1)
tracker.track_model(model, "classifier", {"task": "regression"})

# Set seeds for reproducibility
tracker.set_seeds(42)

# Add metadata
tracker.add_metadata("learning_rate", 0.001)
tracker.add_tag("experiment_v1")

# Finish experiment
summary = tracker.finish_experiment()
```

### Advanced Configuration

```python
config = TrackingConfig(
    experiment_name="advanced_experiment",
    output_dir="./runs",
    
    # Enable/disable specific tracking components
    enable_dataset_tracking=True,
    enable_model_tracking=True,
    enable_environment_tracking=True,
    enable_seed_tracking=True,
    enable_git_tracking=True,
    
    # Dataset tracking options
    dataset_checksum_algorithm="sha256",
    
    # Model tracking options
    save_model_architecture=True,
    save_model_weights=False,  # Usually too large
    
    # Environment tracking options
    capture_conda_env=True,
    capture_pip_freeze=True,
    capture_system_info=True,
    
    # Output options
    save_to_json=True,
    save_to_yaml=True,
    save_to_wandb=False,  # Set to True for Weights & Biases integration
    
    # Metadata
    tags=["advanced", "ml", "research"],
    notes="Advanced experiment with comprehensive tracking",
    metadata={"project": "research_project", "version": "1.0"}
)
```

## API Reference

### TrackingConfig

Configuration class for experiment tracking.

**Parameters:**
- `experiment_name` (str): Name of the experiment
- `experiment_id` (str, optional): Unique experiment ID (auto-generated if not provided)
- `output_dir` (Path): Directory to save tracking data
- `enable_dataset_tracking` (bool): Enable dataset tracking (default: True)
- `enable_model_tracking` (bool): Enable model tracking (default: True)
- `enable_environment_tracking` (bool): Enable environment tracking (default: True)
- `enable_seed_tracking` (bool): Enable seed tracking (default: True)
- `enable_git_tracking` (bool): Enable Git tracking (default: True)
- `save_to_json` (bool): Save tracking data to JSON (default: True)
- `save_to_yaml` (bool): Save tracking data to YAML (default: True)
- `save_to_wandb` (bool): Save to Weights & Biases (default: False)
- `tags` (List[str]): List of tags for the experiment
- `notes` (str): Notes about the experiment
- `metadata` (Dict[str, Any]): Additional metadata

### ExperimentTracker

Main experiment tracker class.

**Methods:**
- `start_experiment()`: Start the experiment and capture initial state
- `track_dataset(dataset, name, metadata=None)`: Track a dataset
- `track_model(model, name, metadata=None)`: Track a model
- `track_tokenizer(tokenizer, name, metadata=None)`: Track a tokenizer
- `set_seeds(seed)`: Set seeds for reproducibility
- `create_reproducible_environment(seed)`: Create fully reproducible environment
- `add_metadata(key, value)`: Add custom metadata
- `add_tag(tag)`: Add a tag to the experiment
- `set_notes(notes)`: Set notes for the experiment
- `get_tracking_summary()`: Get summary of tracking data
- `finish_experiment()`: Finish the experiment and save final state

## Examples

### Complete ML Pipeline Tracking

```python
from rldk.tracking import ExperimentTracker, TrackingConfig
import torch
import torch.nn as nn
import numpy as np

# Configuration
config = TrackingConfig(
    experiment_name="ml_pipeline_demo",
    output_dir="./runs",
    tags=["classification", "pytorch", "demo"]
)

tracker = ExperimentTracker(config)

# Start experiment
tracker.start_experiment()

# Create reproducible environment
tracker.create_reproducible_environment(42)

# Track training data
X_train = np.random.randn(1000, 20)
y_train = np.random.randint(0, 2, 1000)
tracker.track_dataset(X_train, "X_train", {"split": "train", "features": 20})
tracker.track_dataset(y_train, "y_train", {"split": "train", "classes": 2})

# Track test data
X_test = np.random.randn(200, 20)
y_test = np.random.randint(0, 2, 200)
tracker.track_dataset(X_test, "X_test", {"split": "test", "features": 20})
tracker.track_dataset(y_test, "y_test", {"split": "test", "classes": 2})

# Create and track model
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

tracker.track_model(model, "classifier", {
    "task": "binary_classification",
    "architecture": "feedforward",
    "layers": 7
})

# Add training metadata
tracker.add_metadata("optimizer", "Adam")
tracker.add_metadata("loss_function", "BCELoss")
tracker.add_metadata("learning_rate", 0.001)
tracker.add_metadata("batch_size", 32)
tracker.add_metadata("epochs", 10)

# Add tags
tracker.add_tag("binary_classification")
tracker.add_tag("pytorch")
tracker.add_tag("test_run")

# Set notes
tracker.set_notes("ML classification demo with comprehensive tracking")

# Finish experiment
summary = tracker.finish_experiment()
```

### Large Model Tracking

```python
# For large models, disable weight saving to save space
config = TrackingConfig(
    experiment_name="large_model_experiment",
    output_dir="./runs",
    save_model_weights=False,  # Don't save weights for large models
    save_model_architecture=True
)

tracker = ExperimentTracker(config)
tracker.start_experiment()

# Track large model (e.g., BERT, GPT, etc.)
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")

tracker.track_model(model, "bert_model", {
    "source": "huggingface",
    "model_name": "bert-base-uncased",
    "parameters": sum(p.numel() for p in model.parameters())
})

# Track tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tracker.track_tokenizer(tokenizer, "bert_tokenizer")

tracker.finish_experiment()
```

## Output Files

The tracking system creates several output files:

### JSON Files
- `{experiment_name}_{timestamp}.json`: Timestamped tracking data
- `{experiment_name}_latest.json`: Latest tracking data

### YAML Files (if enabled)
- `{experiment_name}_{timestamp}.yaml`: Timestamped tracking data in YAML format
- `{experiment_name}_latest.yaml`: Latest tracking data in YAML format

### Model Architecture Files (if enabled)
- `{model_name}_architecture.txt`: Human-readable model architecture

### Model Weight Files (if enabled)
- `{model_name}_weights.pt`: PyTorch model weights

## Tracking Data Structure

The tracking data includes:

```json
{
  "experiment_id": "uuid",
  "experiment_name": "experiment_name",
  "timestamp": "2023-01-01T12:00:00",
  "config": {...},
  "datasets": {
    "dataset_name": {
      "name": "dataset_name",
      "type": "ndarray",
      "checksum": "sha256_hash",
      "metadata": {...}
    }
  },
  "models": {
    "model_name": {
      "name": "model_name",
      "type": "Sequential",
      "architecture_checksum": "sha256_hash",
      "num_parameters": 1000,
      "metadata": {...}
    }
  },
  "environment": {
    "python_version": "3.8.10",
    "system": {...},
    "pip": {...},
    "ml_frameworks": {...},
    "environment_checksum": "sha256_hash"
  },
  "seeds": {
    "set_seed": 42,
    "seeds": {...},
    "seed_checksum": "sha256_hash"
  },
  "git": {
    "commit": {...},
    "diff": {...},
    "git_checksum": "sha256_hash"
  },
  "metadata": {...},
  "tags": [...],
  "notes": "..."
}
```

## Reproducibility

To reproduce an experiment:

1. **Use the same seed**: Check the `seeds.set_seed` value in the tracking data
2. **Verify environment**: Compare environment checksums
3. **Check Git state**: Verify Git commit hash and repository state
4. **Use same model architecture**: Verify architecture checksum
5. **Verify datasets**: Compare dataset checksums

## Performance

The tracking system is designed to be efficient:

- **Large datasets**: Uses sampling for checksum computation on datasets > 1M elements
- **Large models**: Efficiently handles models with millions of parameters
- **Checksum computation**: Optimized for speed while maintaining accuracy
- **File I/O**: Minimal overhead with efficient JSON/YAML serialization

## Testing

Run the comprehensive test suite:

```bash
# Basic functionality tests
python3 test_tracking_standalone.py

# Large model tests
python3 test_large_model_standalone.py

# Demo script
python3 examples/tracking_demo_simple.py
```

## Integration with Weights & Biases

To enable Weights & Biases integration:

```python
config = TrackingConfig(
    experiment_name="wandb_experiment",
    save_to_wandb=True,
    wandb_project="my_project"
)
```

## Best Practices

1. **Always set seeds**: Use `create_reproducible_environment()` for full reproducibility
2. **Track early**: Start tracking at the beginning of your experiment
3. **Use meaningful names**: Give descriptive names to datasets and models
4. **Add metadata**: Include relevant hyperparameters and configuration
5. **Use tags**: Organize experiments with meaningful tags
6. **Save architecture**: Enable architecture saving for model fingerprinting
7. **Disable weight saving**: For large models, disable weight saving to save space

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all required dependencies are installed
2. **Large file sizes**: Disable weight saving for large models
3. **Git errors**: Ensure you're in a Git repository or disable Git tracking
4. **Permission errors**: Ensure write permissions to the output directory

### Performance Tips

1. **Sample large datasets**: The system automatically samples large datasets for checksum computation
2. **Use efficient data types**: Prefer NumPy arrays over Python lists for large datasets
3. **Disable unnecessary tracking**: Turn off tracking components you don't need
4. **Use SSD storage**: Store tracking data on fast storage for better performance

## Contributing

To contribute to the tracking system:

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Ensure backward compatibility

## License

This tracking system is part of the RLDK project and follows the same license terms.