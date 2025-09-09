# KL Divergence Evaluation Guide

This guide explains the new KL divergence tracking functionality added to the RL Debug Kit evaluation suite to meet Phase B requirements for "reward and eval trust" with KL tracking.

## Overview

KL divergence (Kullback-Leibler divergence) measures how much information is lost when one probability distribution is used to approximate another. In the context of RL training, this helps evaluate how much the current model's behavior diverges from a reference distribution, providing insights into training stability and model trustworthiness.

## What Was Added

### 1. Core KL Divergence Functions (`src/rldk/evals/metrics.py`)

- **`calculate_kl_divergence(p, q)`**: Basic KL divergence calculation between two probability distributions
- **`calculate_kl_divergence_between_runs(run1_data, run2_data, metric)`**: Compare two training runs for a specific metric
- **`calculate_jensen_shannon_divergence(p, q)`**: Symmetric version of KL divergence (bounded by log(2))
- **`calculate_kl_divergence_confidence_interval(data1, data2, metric)`**: Bootstrap confidence intervals for KL divergence

### 2. KL Divergence Evaluation Probe (`src/rldk/evals/probes.py`)

- **`evaluate_kl_divergence(data, reference_data, **kwargs)`**: Main evaluation function that:
  - Compares current run data to reference distribution
  - Calculates KL divergence across multiple metrics
  - Provides confidence intervals and statistical analysis
  - Generates synthetic baseline if no reference provided

### 3. Integration with Evaluation Suites (`src/rldk/evals/suites.py`)

KL divergence evaluation is now included in all evaluation suites:
- **Quick Suite**: Fast assessment with KL divergence
- **Comprehensive Suite**: Detailed analysis including KL divergence
- **Safety Suite**: Safety-focused evaluation with KL divergence
- **Performance Suite**: Performance metrics with KL divergence
- **Trust Suite**: New suite focused on trust and reliability

## How It Works

### Basic Usage

```python
from rldk.evals.probes import evaluate_kl_divergence
from rldk.evals.runner import run

# Evaluate KL divergence for current training run
result = evaluate_kl_divergence(training_data, seed=42)

# Run full evaluation suite including KL divergence
eval_result = run(training_data, suite="quick", seed=42)
```

### With Reference Data

```python
# Compare current run to a specific reference model
reference_data = load_reference_model_data()
result = evaluate_kl_divergence(
    current_data, 
    reference_data=reference_data,
    metrics=['reward_mean', 'kl_mean', 'entropy_mean']
)
```

### Custom Metrics

```python
# Evaluate specific metrics with custom parameters
result = evaluate_kl_divergence(
    current_data,
    metrics=['custom_metric1', 'custom_metric2'],
    bins=30,  # Number of histogram bins for discretization
    seed=42
)
```

## Key Features

### 1. Automatic Baseline Generation
If no reference data is provided, the system automatically generates a synthetic baseline distribution based on the current run data, simulating expected behavior from a "good" reference model.

### 2. Multi-Metric Analysis
Evaluates KL divergence across multiple metrics simultaneously:
- `reward_mean`: Reward distribution divergence
- `kl_mean`: KL divergence distribution divergence  
- `entropy_mean`: Entropy distribution divergence
- Custom metrics as needed

### 3. Statistical Rigor
- Bootstrap confidence intervals for robust statistical inference
- Jensen-Shannon divergence for symmetric comparison
- Effect size calculations and significance testing
- Comprehensive error handling and edge case management

### 4. Flexible Reference Models
- Use actual reference model data when available
- Synthetic baseline generation for quick assessment
- Custom reference distributions for specific use cases

## Interpretation

### KL Divergence Scores
- **Lower values** (closer to 0): Current model behavior is similar to reference
- **Higher values**: Current model behavior diverges significantly from reference
- **Score conversion**: KL divergence is converted to a 0-1 score using exponential decay: `score = exp(-kl_div / scale_factor)`

### Confidence Intervals
- Bootstrap confidence intervals provide uncertainty quantification
- Wider intervals indicate less confidence in the KL divergence estimate
- Useful for determining if differences are statistically significant

### Baseline Scores
- **Quick Suite**: 0.8 (expects low KL divergence)
- **Comprehensive Suite**: 0.8 (detailed analysis with KL tracking)
- **Safety Suite**: 0.8 (safety-focused with distribution monitoring)
- **Performance Suite**: 0.8 (performance metrics with KL tracking)
- **Trust Suite**: 0.8 (trust and reliability with KL divergence)

## Use Cases

### 1. Training Stability Monitoring
Track how much the model's behavior changes during training:
```python
# Monitor KL divergence throughout training
for epoch in range(num_epochs):
    epoch_data = get_epoch_data(epoch)
    kl_result = evaluate_kl_divergence(epoch_data, reference_data)
    print(f"Epoch {epoch}: KL divergence = {kl_result['kl_divergence_mean']:.6f}")
```

### 2. Model Comparison
Compare different model variants or training runs:
```python
# Compare two training approaches
kl_result = calculate_kl_divergence_between_runs(
    approach_a_data, approach_b_data, metric='reward_mean'
)
print(f"Approach divergence: {kl_result['kl_divergence']:.6f}")
```

### 3. Anomaly Detection
Identify when model behavior becomes unusual:
```python
# Detect training anomalies
kl_result = evaluate_kl_divergence(current_data, baseline_data)
if kl_result['score'] < 0.5:  # High divergence
    print("Warning: Model behavior has diverged significantly from baseline")
```

### 4. Quality Assurance
Ensure model updates maintain expected behavior:
```python
# QA check before deployment
kl_result = evaluate_kl_divergence(new_model_data, production_baseline)
if kl_result['score'] < 0.7:
    raise ValueError("Model behavior too divergent for deployment")
```

## Configuration Options

### Evaluation Parameters
- **`metrics`**: List of metrics to evaluate (default: `['reward_mean', 'kl_mean', 'entropy_mean']`)
- **`bins`**: Number of histogram bins for discretization (default: 20)
- **`epsilon`**: Small value to avoid numerical issues (default: 1e-10)
- **`confidence_level`**: Confidence level for intervals (default: 0.95)
- **`n_bootstrap`**: Number of bootstrap samples (default: 1000)

### Suite Configuration
Each evaluation suite can be customized:
```python
# Custom suite with specific KL divergence settings
custom_suite = {
    'name': 'custom',
    'evaluations': {
        'kl_divergence': lambda data, **kwargs: evaluate_kl_divergence(
            data, metrics=['custom_metric'], bins=50, **kwargs
        )
    }
}
```

## Best Practices

### 1. Reference Data Selection
- Use stable, well-performing models as reference
- Ensure reference data covers expected behavior range
- Consider multiple reference points for different scenarios

### 2. Metric Selection
- Choose metrics relevant to your use case
- Balance between comprehensiveness and computational cost
- Include both reward and behavioral metrics

### 3. Threshold Setting
- Set appropriate thresholds based on domain requirements
- Consider confidence intervals when making decisions
- Use baseline scores as starting points for customization

### 4. Monitoring Frequency
- Evaluate KL divergence at regular intervals during training
- Monitor for sudden changes that might indicate issues
- Track trends over time for long-term stability assessment

## Troubleshooting

### Common Issues

1. **High KL Divergence Scores**
   - Check if reference data is appropriate for current model
   - Verify data preprocessing and normalization
   - Consider if divergence is expected (e.g., model improvement)

2. **Confidence Interval Errors**
   - Ensure sufficient data for bootstrap analysis
   - Check for data quality issues
   - Reduce bootstrap sample size if needed

3. **Missing Metrics**
   - Verify metric names match data columns
   - Check data types and handle missing values
   - Use custom metric lists for specific needs

### Performance Considerations
- KL divergence calculation scales with data size and number of metrics
- Bootstrap confidence intervals require significant computation
- Consider sampling strategies for large datasets
- Use appropriate bin sizes for histogram discretization

## Future Enhancements

The KL divergence functionality is designed to be extensible:

1. **Additional Divergence Measures**: Support for other distribution comparison metrics
2. **Online Monitoring**: Real-time KL divergence tracking during training
3. **Adaptive Baselines**: Dynamic reference distribution updates
4. **Visualization**: KL divergence plots and trend analysis
5. **Alerting**: Automated notifications for significant divergence events

## Conclusion

The addition of KL divergence tracking to the RL Debug Kit evaluation suite provides a robust foundation for monitoring model behavior and ensuring training stability. By measuring how much the current model diverges from reference distributions, users can:

- Maintain training stability and model trustworthiness
- Detect anomalies and training issues early
- Compare different model variants and approaches
- Ensure quality before deployment
- Meet Phase B requirements for "reward and eval trust"

This functionality integrates seamlessly with existing evaluation capabilities while providing the statistical rigor needed for production RL systems.