# RLDK Configuration System

## Overview

The RLDK configuration system provides centralized management of all parameters, thresholds, and settings used throughout the codebase. This eliminates hardcoded values and makes the system highly configurable and maintainable.

## Configuration Files

### 1. Evaluation Configuration (`src/rldk/config/evaluation_config.py`)

Manages all evaluation-related parameters including thresholds, sample sizes, and analysis settings.

#### Key Parameters

**KL Divergence Thresholds:**
- `KL_DIVERGENCE_MIN`: Minimum KL divergence threshold (default: 0.01)
- `KL_DIVERGENCE_MAX`: Maximum KL divergence threshold (default: 0.5)
- `KL_DIVERGENCE_TARGET`: Target KL divergence value (default: 0.1)

**Memory Thresholds (GB):**
- `MEMORY_EFFICIENCY_THRESHOLD`: Memory efficiency threshold (default: 8.0)
- `MEMORY_STABILITY_THRESHOLD`: Memory stability threshold (default: 1.0)
- `GPU_MEMORY_EFFICIENCY_THRESHOLD`: GPU memory efficiency threshold (default: 6.0)
- `MEMORY_RANGE_THRESHOLD`: Memory range threshold (default: 2.0)
- `MEMORY_CONSISTENCY_THRESHOLD`: Memory consistency threshold (default: 16.0)

**Gradient Thresholds:**
- `GRADIENT_STABILITY_THRESHOLD`: Gradient stability threshold (default: 1.0)
- `GRADIENT_EXPLOSION_THRESHOLD`: Gradient explosion threshold (default: 10.0)
- `GRADIENT_EFFICIENCY_THRESHOLD`: Gradient efficiency threshold (default: 4.0)

**Toxicity Thresholds:**
- `HIGH_TOXICITY_THRESHOLD`: High toxicity threshold (default: 0.7)
- `CONFIDENCE_CALIBRATION_MIN`: Minimum confidence calibration (default: 0.3)
- `CONFIDENCE_CALIBRATION_MAX`: Maximum confidence calibration (default: 0.8)
- `CONFIDENCE_STABILITY_THRESHOLD`: Confidence stability threshold (default: 0.2)

**Performance Thresholds:**
- `INFERENCE_TIME_THRESHOLD`: Inference time threshold in seconds (default: 0.1)
- `LATENCY_THRESHOLD`: Latency threshold in seconds (default: 0.05)
- `STEPS_PER_SECOND_MAX`: Maximum steps per second (default: 1000.0)
- `SPEED_CONSISTENCY_THRESHOLD`: Speed consistency threshold (default: 0.01)
- `BATCH_SPEED_THRESHOLD`: Batch speed threshold (default: 1000.0)

**Sample Size Thresholds:**
- `MIN_SAMPLES_FOR_ANALYSIS`: Minimum samples for analysis (default: 10)
- `MIN_SAMPLES_FOR_CONSISTENCY`: Minimum samples for consistency (default: 5)
- `MIN_SAMPLES_FOR_DISTRIBUTION`: Minimum samples for distribution (default: 50)
- `MIN_SAMPLES_FOR_TREND`: Minimum samples for trend analysis (default: 20)

### 2. Forensics Configuration (`src/rldk/config/forensics_config.py`)

Manages parameters for forensics analysis, advantage statistics, and anomaly detection.

#### Key Parameters

**Advantage Statistics:**
- `ADVANTAGE_WINDOW_SIZE`: Window size for advantage statistics (default: 100)
- `ADVANTAGE_TREND_WINDOW`: Trend analysis window size (default: 20)
- `ADVANTAGE_BIAS_THRESHOLD`: Advantage bias threshold (default: 0.1)
- `ADVANTAGE_SCALE_THRESHOLD`: Advantage scale threshold (default: 2.0)

**Gradient Analysis:**
- `GRADIENT_NORM_WINDOW`: Gradient norm analysis window (default: 50)
- `GRADIENT_EXPLOSION_THRESHOLD`: Gradient explosion threshold (default: 10.0)
- `GRADIENT_VANISHING_THRESHOLD`: Gradient vanishing threshold (default: 1e-6)
- `GRADIENT_STABILITY_THRESHOLD`: Gradient stability threshold (default: 1.0)

**KL Divergence Tracking:**
- `KL_WINDOW_SIZE`: KL divergence window size (default: 100)
- `KL_ANOMALY_THRESHOLD`: KL anomaly threshold (default: 0.5)
- `KL_TREND_WINDOW`: KL trend analysis window (default: 20)
- `KL_STABILITY_THRESHOLD`: KL stability threshold (default: 0.1)

**Anomaly Detection:**
- `ANOMALY_SEVERITY_CRITICAL`: Critical anomaly severity threshold (default: 0.8)
- `ANOMALY_SEVERITY_WARNING`: Warning anomaly severity threshold (default: 0.5)
- `ANOMALY_SEVERITY_INFO`: Info anomaly severity threshold (default: 0.3)

### 3. Visualization Configuration (`src/rldk/config/visualization_config.py`)

Manages all plotting and visualization parameters.

#### Key Parameters

**Figure Settings:**
- `DEFAULT_FIGSIZE`: Default figure size tuple (default: (12, 8))
- `DEFAULT_DPI`: Default DPI (default: 300)
- `DEFAULT_STYLE`: Default matplotlib style (default: "seaborn-v0_8")

**Font Settings:**
- `TITLE_FONTSIZE`: Title font size (default: 16)
- `LABEL_FONTSIZE`: Label font size (default: 12)
- `TICK_FONTSIZE`: Tick font size (default: 10)
- `LEGEND_FONTSIZE`: Legend font size (default: 10)

**Color Settings:**
- `PRIMARY_COLOR`: Primary color (default: "blue")
- `SECONDARY_COLOR`: Secondary color (default: "red")
- `TERTIARY_COLOR`: Tertiary color (default: "orange")
- `GRID_ALPHA`: Grid alpha transparency (default: 0.3)

**Histogram Settings:**
- `HISTOGRAM_BINS`: Number of histogram bins (default: 30)
- `HISTOGRAM_ALPHA`: Histogram alpha transparency (default: 0.7)
- `HISTOGRAM_EDGECOLOR`: Histogram edge color (default: "black")

**Sampling Settings:**
- `MAX_POINTS_FOR_PLOT`: Maximum points for plotting (default: 1000)
- `SAMPLING_RANDOM_STATE`: Random state for sampling (default: 42)

### 4. Suite Configuration (`src/rldk/config/suite_config.py`)

Manages evaluation suite parameters including sample sizes, runtime estimates, and baseline scores.

#### Key Parameters

**Sample Sizes:**
- `QUICK_SAMPLE_SIZE`: Quick suite sample size (default: 50)
- `COMPREHENSIVE_SAMPLE_SIZE`: Comprehensive suite sample size (default: 200)
- `SAFETY_SAMPLE_SIZE`: Safety suite sample size (default: 100)
- `INTEGRITY_SAMPLE_SIZE`: Integrity suite sample size (default: 150)
- `PERFORMANCE_SAMPLE_SIZE`: Performance suite sample size (default: 150)
- `TRUST_SAMPLE_SIZE`: Trust suite sample size (default: 120)

**Runtime Estimates (minutes):**
- `QUICK_RUNTIME_MIN/MAX`: Quick suite runtime range (default: 2-5)
- `COMPREHENSIVE_RUNTIME_MIN/MAX`: Comprehensive suite runtime range (default: 10-20)
- `SAFETY_RUNTIME_MIN/MAX`: Safety suite runtime range (default: 5-10)
- `INTEGRITY_RUNTIME_MIN/MAX`: Integrity suite runtime range (default: 8-15)
- `PERFORMANCE_RUNTIME_MIN/MAX`: Performance suite runtime range (default: 8-15)
- `TRUST_RUNTIME_MIN/MAX`: Trust suite runtime range (default: 6-12)

**Baseline Scores:**
Each suite has baseline scores for all its evaluation metrics, ranging from 0.0 to 1.0.

**Suite Generation:**
- `GENERATES_PLOTS`: Whether suites generate plots (default: True)
- `ENABLE_CACHING`: Whether to enable caching (default: True)
- `CACHE_TTL_SECONDS`: Cache time-to-live in seconds (default: 3600)

## Configuration Presets

### Default Presets

**default**: Balanced settings for general use
**strict**: Conservative thresholds for production environments
**lenient**: Relaxed thresholds for development and testing
**research**: Minimal thresholds for experimentation

### Specialized Presets

**fast**: Optimized for speed with reduced sample sizes
**thorough**: Optimized for accuracy with increased sample sizes
**publication**: High-quality visualization settings for papers
**presentation**: Large, clear visualization settings for presentations
**web**: Optimized visualization settings for web display

## Usage Examples

### Basic Usage

```python
from rldk.config import get_eval_config, get_forensics_config

# Get default evaluation configuration
eval_config = get_eval_config()

# Get strict forensics configuration
forensics_config = get_forensics_config("strict")

# Use configuration values
if len(data) > eval_config.MIN_SAMPLES_FOR_ANALYSIS:
    # Process data
    pass
```

### Custom Configuration

```python
from rldk.config import create_custom_eval_config

# Create custom configuration
custom_config = create_custom_eval_config(
    MIN_SAMPLES_FOR_ANALYSIS=5,
    HIGH_TOXICITY_THRESHOLD=0.8,
    MEMORY_EFFICIENCY_THRESHOLD=6.0
)
```

### Environment Variable Override

```bash
# Set environment variables
export RLDK_MIN_SAMPLES_FOR_ANALYSIS=20
export RLDK_HIGH_TOXICITY_THRESHOLD=0.8
export RLDK_MEMORY_EFFICIENCY_THRESHOLD=6.0

# Configuration will automatically use these values
```

### Function Integration

```python
def evaluate_model(data, config=None, **kwargs):
    if config is None:
        config = get_eval_config(kwargs.get("config_name", "default"))
    
    # Use configuration values
    if len(data) > config.MIN_SAMPLES_FOR_ANALYSIS:
        # Process data
        pass
```

## Configuration Validation

The configuration system includes comprehensive validation to ensure all parameters are within valid ranges and logically consistent.

### Validation Example

```python
from rldk.config import validate_all_configs, print_validation_results

# Validate configurations
issues = validate_all_configs(
    eval_config=eval_config,
    forensics_config=forensics_config
)

# Print validation results
print_validation_results(issues)
```

### Common Validation Rules

- Thresholds must be positive where applicable
- Minimum values must be less than maximum values
- Sample sizes must be positive integers
- Confidence levels must be between 0 and 1
- Weights must sum to 1.0
- Percentiles must be between 0 and 100

## Migration Guide

### Step 1: Import Configuration

```python
# Add to imports
from ..config import get_eval_config, get_forensics_config
```

### Step 2: Update Function Signatures

```python
# Before
def evaluate_something(data, **kwargs):
    # function body

# After
def evaluate_something(data, config=None, **kwargs):
    if config is None:
        config = get_eval_config(kwargs.get("config_name", "default"))
    # function body
```

### Step 3: Replace Hardcoded Values

```python
# Before
if len(data) > 10:
    # process data

# After
if len(data) > config.MIN_SAMPLES_FOR_ANALYSIS:
    # process data
```

### Step 4: Update Function Calls

```python
# Before
result = evaluate_something(data)

# After
result = evaluate_something(data, config=config)
```

## Best Practices

1. **Always use configuration values** instead of hardcoded numbers
2. **Provide sensible defaults** for all parameters
3. **Use descriptive parameter names** that clearly indicate their purpose
4. **Group related parameters** in the same configuration file
5. **Validate configurations** before using them
6. **Document parameter ranges** and expected values
7. **Test with different presets** to ensure compatibility
8. **Use type hints** for better code clarity

## Troubleshooting

### Common Issues

**Configuration not found**: Falls back to default configuration
**Invalid parameter values**: Use validation to identify issues
**Missing parameters**: Check if parameter exists before using
**Type mismatches**: Ensure parameter types match expected values

### Debug Tips

1. Use `print_validation_results()` to identify configuration issues
2. Check configuration values with `config.to_dict()`
3. Verify parameter names match exactly
4. Test with different configuration presets
5. Check environment variable names and values

## Future Enhancements

- Dynamic configuration loading from files
- Configuration inheritance and composition
- Runtime configuration updates
- Configuration versioning and migration
- Advanced validation rules
- Configuration templates and wizards