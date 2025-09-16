# Experiment Tracking

RLDK provides comprehensive experiment tracking to ensure complete reproducibility of your RL training runs.

## Overview

The tracking system captures:
- **Dataset versioning** with SHA-256 checksums
- **Model fingerprinting** with architecture tracking
- **Environment state** including dependencies and system info
- **Random seed management** across all frameworks
- **Git integration** with commit hashes and repository state

## Basic Usage

```python
from rldk.tracking import ExperimentTracker, TrackingConfig

# Configure what to track
config = TrackingConfig(
    experiment_name="ppo_training",
    enable_dataset_tracking=True,
    enable_model_tracking=True,
    enable_environment_tracking=True,
    enable_seed_tracking=True,
    enable_git_tracking=True
)

tracker = ExperimentTracker(config)
tracker.start_experiment()

# Track datasets with automatic checksums
tracker.track_dataset(training_data, "training_data")
tracker.track_dataset(eval_data, "eval_data")

# Track models with architecture fingerprinting
tracker.track_model(model, "gpt2_policy")
tracker.track_tokenizer(tokenizer, "gpt2_tokenizer")

# Set reproducible seeds
tracker.set_seeds(42)

# Add custom metadata
tracker.add_metadata("learning_rate", 1e-5)
tracker.add_metadata("batch_size", 32)
tracker.add_metadata("optimizer", "AdamW")

# Finish and save experiment
tracker.finish_experiment()
```

## Advanced Configuration

```python
config = TrackingConfig(
    experiment_name="advanced_experiment",
    experiment_id="exp_001",  # Custom ID
    output_dir="./experiments",  # Custom output directory
    
    # Enable/disable specific tracking
    enable_dataset_tracking=True,
    enable_model_tracking=True,
    enable_environment_tracking=True,
    enable_seed_tracking=True,
    enable_git_tracking=True,
    
    # Output formats
    save_to_json=True,
    save_to_yaml=True,
    save_to_wandb=False,  # Requires wandb setup
    
    # Metadata
    tags=["research", "ppo", "gpt2"],
    notes="Experiment with improved reward model",
    
    # Performance options
    dataset_sample_size=10000,  # For large datasets
    save_model_weights=False,   # For large models
)
```

## Dataset Tracking

RLDK automatically computes checksums for various data types:

```python
# NumPy arrays
tracker.track_dataset(np.array([1, 2, 3]), "numpy_data")

# Pandas DataFrames
tracker.track_dataset(df, "dataframe")

# PyTorch datasets
tracker.track_dataset(torch_dataset, "torch_data")

# Hugging Face datasets
tracker.track_dataset(hf_dataset, "hf_data")

# Files and directories
tracker.track_dataset("./data/train.jsonl", "training_file")
tracker.track_dataset("./data/", "data_directory")
```

## Model Tracking

Track model architectures and optionally weights:

```python
# PyTorch models
tracker.track_model(pytorch_model, "policy_model")

# Hugging Face models
tracker.track_model(hf_model, "reward_model")
tracker.track_tokenizer(tokenizer, "tokenizer")

# Custom models with metadata
tracker.track_model(
    model, 
    "custom_model",
    metadata={
        "architecture": "transformer",
        "layers": 12,
        "hidden_size": 768
    }
)
```

## Seed Management

Ensure reproducibility across all frameworks:

```python
# Set all seeds at once
tracker.set_seeds(42)

# Or set individually
tracker.set_python_seed(42)
tracker.set_numpy_seed(42)
tracker.set_torch_seed(42)
tracker.set_cuda_seed(42)

# Enable deterministic operations
tracker.enable_deterministic_mode()
```

## Output Files

The tracker generates several files:

```
experiments/
└── my_experiment_20240101_120000/
    ├── experiment.json          # Complete experiment data
    ├── experiment.yaml          # Human-readable format
    ├── datasets/               # Dataset checksums and metadata
    ├── models/                 # Model architectures and metadata
    ├── environment.json        # Environment snapshot
    ├── seeds.json             # Random seed state
    └── git.json               # Git repository state
```

## Integration with Training

```python
# Example with TRL training
from trl import PPOTrainer
from rldk.tracking import ExperimentTracker, TrackingConfig

# Setup tracking
config = TrackingConfig(experiment_name="ppo_training")
tracker = ExperimentTracker(config)
tracker.start_experiment()

# Track training components
tracker.track_model(model, "policy_model")
tracker.track_model(ref_model, "reference_model")
tracker.track_dataset(dataset, "training_data")
tracker.set_seeds(42)

# Train with PPO
trainer = PPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    # ... other config
)

# Track training metadata
tracker.add_metadata("ppo_config", trainer.config.to_dict())
tracker.add_metadata("model_name", model.config.name_or_path)

# Train
trainer.train()

# Finish tracking
tracker.finish_experiment()
```

## Best Practices

1. **Start tracking early**: Call `start_experiment()` before any training setup
2. **Track everything**: Include datasets, models, configs, and hyperparameters
3. **Use descriptive names**: Give meaningful names to tracked components
4. **Add metadata**: Include relevant hyperparameters and configuration
5. **Set seeds consistently**: Use the tracker's seed management for reproducibility
6. **Finish properly**: Always call `finish_experiment()` to save data

## Troubleshooting

### Common Issues

1. **Large datasets**: Use `dataset_sample_size` to limit checksum computation
2. **Large models**: Set `save_model_weights=False` to save only architecture
3. **Missing git**: Git tracking will be skipped if not in a repository
4. **Permission errors**: Ensure write access to output directory

### Performance Tips

- Use sampling for datasets > 1M elements
- Disable weight saving for models > 1B parameters
- Use JSON format for faster loading
- Consider disabling environment tracking for faster startup

For more details, see the [API Reference](../reference/api.md).
