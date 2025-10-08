# Cursor Instructions for RLDK Configuration Management

## Overview
This document provides instructions for using the centralized configuration system in RLDK. All hardcoded values should be replaced with references to the appropriate configuration files.

## Configuration Files

### 1. Evaluation Configuration (`src/rldk/config/evaluation_config.py`)
Contains all evaluation-related parameters including:
- KL Divergence thresholds
- Memory thresholds
- Gradient thresholds
- Toxicity thresholds
- Performance thresholds
- Consistency thresholds
- Robustness thresholds
- Efficiency thresholds
- Calibration thresholds

### 2. Forensics Configuration (`src/rldk/config/forensics_config.py`)
Contains all forensics and analysis parameters including:
- Advantage statistics tracking
- Gradient analysis
- KL divergence tracking
- PPO scan parameters
- Checkpoint diff analysis
- Environment audit
- Statistical analysis parameters

### 3. Visualization Configuration (`src/rldk/config/visualization_config.py`)
Contains all plotting and visualization parameters including:
- Figure settings
- Font settings
- Color settings
- Line settings
- Histogram settings
- Scatter plot settings
- Output settings

### 4. Suite Configuration (`src/rldk/config/suite_config.py`)
Contains all evaluation suite parameters including:
- Sample sizes for each suite
- Runtime estimates
- Baseline scores
- Suite generation settings
- Timeout settings
- Parallel processing settings

## Usage Guidelines

### 1. Always Use Configuration Instead of Hardcoded Values

**❌ WRONG:**
```python
if len(rewards) > 10:
    # process data
```

**✅ CORRECT:**
```python
if len(rewards) > config.MIN_SAMPLES_FOR_ANALYSIS:
    # process data
```

### 2. Import Configuration in Your Files

```python
from ..config import get_eval_config, get_forensics_config, get_visualization_config, get_suite_config
```

### 3. Initialize Configuration in Functions

```python
def your_function(data, config=None, **kwargs):
    if config is None:
        config = get_eval_config(kwargs.get("config_name", "default"))
    
    # Use config values instead of hardcoded values
    if len(data) > config.MIN_SAMPLES_FOR_ANALYSIS:
        # process data
```

### 4. Use Environment-Specific Configurations

```python
# Get strict configuration
config = get_eval_config("strict")

# Get lenient configuration  
config = get_eval_config("lenient")

# Get research configuration
config = get_eval_config("research")
```

### 5. Create Custom Configurations

```python
from ..config import create_custom_eval_config

# Create custom configuration with overridden values
custom_config = create_custom_eval_config(
    MIN_SAMPLES_FOR_ANALYSIS=5,
    HIGH_TOXICITY_THRESHOLD=0.8
)
```

## Common Patterns to Replace

### Sample Size Checks
```python
# Replace this:
if len(data) > 10:

# With this:
if len(data) > config.MIN_SAMPLES_FOR_ANALYSIS:
```

### Threshold Comparisons
```python
# Replace this:
if value > 0.7:

# With this:
if value > config.HIGH_TOXICITY_THRESHOLD:
```

### Memory Thresholds
```python
# Replace this:
if memory < 8.0:

# With this:
if memory < config.MEMORY_EFFICIENCY_THRESHOLD:
```

### Gradient Thresholds
```python
# Replace this:
if grad_norm < 10.0:

# With this:
if grad_norm < config.GRADIENT_EXPLOSION_THRESHOLD:
```

### Percentile Calculations
```python
# Replace this:
percentiles = [5, 10, 25, 50, 75, 90, 95]

# With this:
percentiles = config.PERCENTILES
```

## Configuration Validation

Always validate configurations before using them:

```python
from ..config import validate_all_configs, print_validation_results

# Validate configurations
issues = validate_all_configs(eval_config=config)
print_validation_results(issues)
```

## Environment Variables

You can override configuration values using environment variables:

```bash
export RLDK_MIN_SAMPLES_FOR_ANALYSIS=20
export RLDK_HIGH_TOXICITY_THRESHOLD=0.8
export RLDK_MEMORY_EFFICIENCY_THRESHOLD=6.0
```

## Best Practices

1. **Never use hardcoded numbers** - always reference configuration values
2. **Use descriptive config names** - make it clear what the parameter controls
3. **Group related parameters** - keep similar thresholds in the same config file
4. **Provide sensible defaults** - ensure configurations work out of the box
5. **Validate configurations** - use the validator to catch invalid values
6. **Document parameters** - add docstrings explaining what each parameter does
7. **Use type hints** - specify the expected type for each parameter
8. **Test configurations** - ensure all config combinations work correctly

## Migration Checklist

When updating existing code:

- [ ] Import the appropriate configuration functions
- [ ] Add config parameter to function signatures
- [ ] Initialize config with default if not provided
- [ ] Replace all hardcoded numbers with config references
- [ ] Update function calls to pass config parameter
- [ ] Test with different configuration presets
- [ ] Validate configuration values
- [ ] Update documentation

## Common Configuration Presets

- **default**: Balanced settings for general use
- **strict**: Conservative thresholds for production
- **lenient**: Relaxed thresholds for development
- **research**: Minimal thresholds for experimentation
- **fast**: Optimized for speed
- **thorough**: Optimized for accuracy
- **publication**: High-quality visualization settings
- **presentation**: Large, clear visualization settings
- **web**: Optimized for web display

## Troubleshooting

### Configuration Not Found
```python
# This will fall back to default config
config = get_eval_config("unknown_preset")
```

### Invalid Configuration Values
```python
# Use the validator to check for issues
from ..config import validate_all_configs
issues = validate_all_configs(eval_config=config)
if issues["evaluation"]:
    print("Configuration issues found:", issues["evaluation"])
```

### Missing Configuration Parameters
```python
# Check if parameter exists before using
if hasattr(config, 'PARAMETER_NAME'):
    value = config.PARAMETER_NAME
else:
    value = fallback_value
```

Remember: **Every number in the codebase should come from a configuration file!**