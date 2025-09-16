# RLDK Best Practices

This guide provides best practices for using RLDK effectively in research and production environments.

## Experiment Design

### 1. Planning Your Experiment

**Before Starting:**
- Define clear success metrics
- Plan your evaluation strategy
- Set up proper baselines
- Design your data collection

```python
# Example experiment planning
from rldk.tracking import TrackingConfig

config = TrackingConfig(
    experiment_name="ppo_baseline_v1",
    tags=["baseline", "ppo", "gpt2"],
    notes="Baseline PPO experiment with standard hyperparameters",
    
    # Plan what to track
    enable_dataset_tracking=True,
    enable_model_tracking=True,
    enable_environment_tracking=True,
    enable_seed_tracking=True,
    enable_git_tracking=True
)
```

### 2. Reproducibility First

**Always Set Seeds Early:**
```python
from rldk.utils.seed import set_global_seed

# Set seeds before any random operations
set_global_seed(42, deterministic=True)

# Verify determinism
from rldk.determinism import check
report = check(
    cmd="python train.py --seed 42",
    compare=["loss", "reward_mean"],
    replicas=3
)
```

**Track Everything:**
```python
# Track all components for reproducibility
tracker.track_dataset(train_data, "training_data")
tracker.track_dataset(val_data, "validation_data")
tracker.track_model(model, "policy_model")
tracker.track_model(ref_model, "reference_model")
tracker.track_tokenizer(tokenizer, "tokenizer")

# Track hyperparameters
tracker.add_metadata("learning_rate", 1e-5)
tracker.add_metadata("batch_size", 32)
tracker.add_metadata("ppo_epochs", 4)
```

### 3. Incremental Development

**Start Small:**
```python
# Begin with minimal viable experiments
config = TrackingConfig(
    experiment_name="debug_run",
    tags=["debug", "small"],
    notes="Quick test with minimal data"
)

# Use small datasets and models for initial testing
debug_dataset = dataset.select(range(100))  # Small subset
```

**Scale Gradually:**
```python
# Gradually increase complexity
experiments = [
    {"name": "small_test", "size": 100, "steps": 50},
    {"name": "medium_test", "size": 1000, "steps": 200},
    {"name": "full_run", "size": 10000, "steps": 1000}
]

for exp in experiments:
    # Run experiment with increasing scale
    run_experiment(exp)
```

## Training Monitoring

### 1. Real-Time Health Monitoring

**Set Up Comprehensive Forensics:**
```python
from rldk.forensics import ComprehensivePPOForensics

forensics = ComprehensivePPOForensics(
    kl_target=0.1,
    enable_kl_schedule_tracking=True,
    enable_gradient_norms_analysis=True,
    enable_advantage_statistics=True
)

# Monitor every step
for step in training_loop:
    metrics = forensics.update(
        step=step,
        kl=kl_divergence,
        kl_coef=kl_coefficient,
        entropy=entropy,
        reward_mean=reward_mean,
        reward_std=reward_std,
        policy_grad_norm=policy_grad_norm,
        value_grad_norm=value_grad_norm,
        advantage_mean=advantage_mean,
        advantage_std=advantage_std
    )
    
    # Act on anomalies immediately
    if metrics.has_anomalies:
        handle_anomalies(metrics.anomalies)
```

**Implement Early Stopping:**
```python
def should_stop_training(metrics):
    """Decide whether to stop training based on anomalies."""
    critical_anomalies = [
        a for a in metrics.anomalies 
        if a.severity == "critical"
    ]
    
    # Stop on critical issues
    if len(critical_anomalies) > 3:
        return True
        
    # Stop on persistent issues
    if metrics.health_score < 0.3:
        return True
        
    return False

# Use in training loop
if should_stop_training(metrics):
    print("Stopping training due to critical issues")
    break
```

### 2. Logging Strategy

**Structured Logging:**
```python
import json
import os

# Set up structured logging
metrics_path = os.environ.get('RLDK_METRICS_PATH', 'metrics.jsonl')

def log_metrics(step, metrics_dict):
    """Log metrics in structured format."""
    log_entry = {
        "step": step,
        "timestamp": time.time(),
        "seed": current_seed,
        **metrics_dict
    }
    
    with open(metrics_path, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

# Log consistently
log_metrics(step, {
    "reward_mean": reward_mean,
    "kl_divergence": kl_divergence,
    "policy_loss": policy_loss,
    "value_loss": value_loss
})
```

**Multi-Level Logging:**
```python
# Different logging frequencies for different metrics
if step % 1 == 0:  # Every step
    log_metrics(step, core_metrics)
    
if step % 10 == 0:  # Every 10 steps
    log_metrics(step, detailed_metrics)
    
if step % 100 == 0:  # Every 100 steps
    log_metrics(step, expensive_metrics)
```

## Data Management

### 1. Dataset Versioning

**Track Dataset Changes:**
```python
# Version your datasets
tracker.track_dataset(dataset_v1, "training_data_v1")
tracker.track_dataset(dataset_v2, "training_data_v2")

# Track preprocessing steps
tracker.add_metadata("preprocessing_steps", [
    "tokenization",
    "filtering_length",
    "deduplication"
])

# Track data statistics
tracker.add_metadata("dataset_stats", {
    "total_samples": len(dataset),
    "avg_length": np.mean(lengths),
    "vocab_size": tokenizer.vocab_size
})
```

**Data Quality Monitoring:**
```python
from rldk.ingest import run_quality_checks

# Regular data quality checks
quality_report = run_quality_checks(dataset)

if quality_report.quality_score < 0.8:
    print("Data quality issues detected!")
    print(f"Missing values: {quality_report.missing_values}")
    print(f"Duplicates: {quality_report.duplicates}")
```

### 2. Efficient Data Handling

**Large Dataset Strategies:**
```python
# Use sampling for very large datasets
config = TrackingConfig(
    dataset_sample_size=10000,  # Sample for checksum computation
    save_model_weights=False    # Don't save large model weights
)

# Stream large datasets
def stream_dataset(dataset_path, batch_size=1000):
    """Stream large datasets in chunks."""
    for chunk in pd.read_json(dataset_path, lines=True, chunksize=batch_size):
        yield chunk

# Process in chunks
for chunk in stream_dataset("large_dataset.jsonl"):
    process_chunk(chunk)
```

## Model Management

### 1. Model Versioning

**Track Model Evolution:**
```python
# Track different model versions
tracker.track_model(model_v1, "policy_model_v1")
tracker.track_model(model_v2, "policy_model_v2")

# Track model metadata
tracker.add_metadata("model_architecture", {
    "type": "transformer",
    "layers": 12,
    "hidden_size": 768,
    "attention_heads": 12,
    "parameters": count_parameters(model)
})

# Track training configuration
tracker.add_metadata("training_config", {
    "optimizer": "AdamW",
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "warmup_steps": 1000
})
```

**Model Checkpointing:**
```python
# Regular checkpointing with RLDK integration
def save_checkpoint(model, step, tracker):
    """Save checkpoint with RLDK tracking."""
    checkpoint_path = f"checkpoint_{step}.pt"
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'step': step,
        'experiment_id': tracker.experiment_id
    }, checkpoint_path)
    
    # Track checkpoint
    tracker.add_metadata(f"checkpoint_{step}", {
        "path": checkpoint_path,
        "step": step,
        "model_hash": compute_model_hash(model)
    })

# Save checkpoints regularly
if step % checkpoint_frequency == 0:
    save_checkpoint(model, step, tracker)
```

### 2. Model Evaluation

**Comprehensive Evaluation:**
```python
from rldk.evals import get_eval_suite, run_evaluation

# Regular model evaluation
eval_suite = get_eval_suite("comprehensive")

def evaluate_model(model, tokenizer, eval_data, step):
    """Comprehensive model evaluation."""
    result = run_evaluation(
        data=eval_data,
        suite=eval_suite,
        model=model,
        tokenizer=tokenizer
    )
    
    # Track evaluation results
    tracker.add_metadata(f"eval_step_{step}", {
        "overall_score": result.overall_score,
        "individual_metrics": result.metrics,
        "evaluation_time": result.evaluation_time
    })
    
    return result

# Evaluate regularly
if step % eval_frequency == 0:
    eval_result = evaluate_model(model, tokenizer, eval_data, step)
    
    # Early stopping based on evaluation
    if eval_result.overall_score < min_score_threshold:
        print("Evaluation score too low, stopping training")
        break
```

## Performance Optimization

### 1. Memory Management

**Efficient Memory Usage:**
```python
# Use gradient checkpointing for large models
model.gradient_checkpointing_enable()

# Clear cache regularly
if step % 100 == 0:
    torch.cuda.empty_cache()

# Monitor memory usage
from rldk.utils.progress import ProgressTracker

tracker = ProgressTracker(total=num_steps, monitor_memory=True)
for step in range(num_steps):
    # Training step
    train_step()
    
    # Update progress and check memory
    tracker.update(1)
    if tracker.memory_usage > 0.9:
        print("High memory usage detected!")
        torch.cuda.empty_cache()
```

**Batch Size Optimization:**
```python
# Find optimal batch size
def find_optimal_batch_size(model, tokenizer, dataset):
    """Find the largest batch size that fits in memory."""
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    for batch_size in batch_sizes:
        try:
            # Test batch
            batch = dataset.select(range(batch_size))
            inputs = tokenizer(batch['text'], return_tensors='pt', padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            print(f"Batch size {batch_size}: OK")
            optimal_batch_size = batch_size
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size}: OOM")
                break
            else:
                raise e
    
    return optimal_batch_size

# Use optimal batch size
optimal_batch_size = find_optimal_batch_size(model, tokenizer, dataset)
```

### 2. Training Efficiency

**Efficient Training Loops:**
```python
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for step, batch in enumerate(dataloader):
    with autocast():
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
    
    # Backward pass with scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    # RLDK monitoring (minimal overhead)
    if step % log_frequency == 0:
        log_metrics(step, {"loss": loss.item()})
```

**Parallel Processing:**
```python
# Use DataLoader with multiple workers
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    shuffle=True
)

# Parallel evaluation
from rldk.evals import run_evaluation

result = run_evaluation(
    data=eval_data,
    suite=eval_suite,
    parallel=True,  # Parallel evaluation
    max_workers=4
)
```

## Production Deployment

### 1. Model Validation

**Pre-Deployment Checks:**
```python
def validate_model_for_deployment(model, tokenizer, test_data):
    """Comprehensive model validation before deployment."""
    
    # Safety evaluation
    safety_suite = get_eval_suite("safety")
    safety_result = run_evaluation(test_data, safety_suite)
    
    if safety_result.overall_score < 0.9:
        raise ValueError("Model failed safety evaluation")
    
    # Performance evaluation
    perf_suite = get_eval_suite("comprehensive")
    perf_result = run_evaluation(test_data, perf_suite)
    
    if perf_result.overall_score < 0.8:
        raise ValueError("Model failed performance evaluation")
    
    # Determinism check
    from rldk.determinism import check
    det_report = check(
        cmd="python inference.py --model model.pt",
        compare=["output"],
        replicas=3
    )
    
    if not det_report.passed:
        raise ValueError("Model failed determinism check")
    
    return True

# Validate before deployment
validate_model_for_deployment(model, tokenizer, test_data)
```

### 2. Monitoring in Production

**Production Monitoring:**
```python
# Set up production monitoring
production_tracker = ExperimentTracker(
    TrackingConfig(
        experiment_name="production_monitoring",
        enable_model_tracking=False,  # Don't re-track model
        enable_environment_tracking=True,
        tags=["production", "monitoring"]
    )
)

# Monitor inference performance
def monitor_inference(model, inputs):
    """Monitor model inference in production."""
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    inference_time = time.time() - start_time
    
    # Log performance metrics
    production_tracker.add_metadata("inference_metrics", {
        "inference_time": inference_time,
        "input_length": inputs['input_ids'].shape[1],
        "output_length": outputs.logits.shape[1],
        "timestamp": time.time()
    })
    
    return outputs
```

## Collaboration and Sharing

### 1. Experiment Sharing

**Share Reproducible Experiments:**
```python
# Create shareable experiment package
def create_experiment_package(experiment_path):
    """Create a package for sharing experiments."""
    package = {
        "experiment_data": load_experiment_data(experiment_path),
        "model_architecture": load_model_architecture(experiment_path),
        "training_config": load_training_config(experiment_path),
        "evaluation_results": load_evaluation_results(experiment_path),
        "reproduction_script": generate_reproduction_script(experiment_path)
    }
    
    return package

# Share with team
package = create_experiment_package(experiment_path)
save_package(package, "shared_experiment.zip")
```

### 2. Documentation

**Document Everything:**
```python
# Comprehensive experiment documentation
tracker.add_metadata("experiment_description", {
    "objective": "Improve reward model performance",
    "hypothesis": "Larger batch size will improve stability",
    "methodology": "PPO training with increased batch size",
    "expected_outcome": "Higher reward, lower variance",
    "success_criteria": "Reward > 0.8, variance < 0.1"
})

# Document decisions
tracker.add_metadata("design_decisions", {
    "learning_rate": "Reduced from 1e-4 to 1e-5 due to instability",
    "batch_size": "Increased from 16 to 32 for better gradients",
    "kl_target": "Set to 0.1 based on previous experiments"
})
```

## Troubleshooting

### 1. Common Issues

**Debug Training Issues:**
```python
# Comprehensive debugging
def debug_training_issue(model, data, config):
    """Debug common training issues."""
    
    # Check data quality
    quality_report = run_quality_checks(data)
    print(f"Data quality score: {quality_report.quality_score}")
    
    # Check model initialization
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN in parameter: {name}")
    
    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            if grad_norm > 10.0:
                print(f"Large gradient in: {name}, norm: {grad_norm}")
    
    # Run forensics
    forensics = ComprehensivePPOForensics()
    # ... forensics analysis
```

### 2. Performance Issues

**Optimize Performance:**
```python
# Profile training performance
import cProfile
import pstats

def profile_training_step():
    """Profile a single training step."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Single training step
    train_step()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions

# Profile periodically
if step % 1000 == 0:
    profile_training_step()
```

## Summary

Following these best practices will help you:

1. **Maintain Reproducibility**: Through comprehensive tracking and determinism checking
2. **Catch Issues Early**: With real-time monitoring and forensics
3. **Optimize Performance**: Through efficient resource management
4. **Enable Collaboration**: With proper documentation and sharing
5. **Ensure Quality**: Through comprehensive evaluation and validation

Remember: The key to successful RL training is continuous monitoring, early detection of issues, and systematic experimentation. RLDK provides the tools - these practices show you how to use them effectively.

For more specific guidance, see the [User Guide](../user-guide/tracking.md) and [Failure Patterns](failure-patterns.md).
