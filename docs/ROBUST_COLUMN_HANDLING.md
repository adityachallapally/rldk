# Robust Column Handling in Evaluation Metrics

This document describes the robust column handling system implemented in the RL Debug Kit evaluation metrics. The system provides graceful fallbacks, clear error messages, and flexible configuration options when required data columns are missing.

## Overview

The evaluation metrics (`throughput`, `toxicity`, and `bias`) now include robust column handling that:

1. **Tries alternative column names** when the primary column is missing
2. **Falls back to alternative metrics** when no suitable text/log columns are available
3. **Provides clear error messages** with suggestions for available columns
4. **Supports configuration utilities** for easy column mapping setup

## Problem Solved

Previously, evaluation metrics would fail with cryptic errors like:
```
Missing columns: ['output', 'events']
```

Now, the system:
- Automatically tries alternative column names
- Falls back to alternative metrics when available
- Provides detailed error messages with suggestions
- Offers configuration utilities for easy setup

## Features

### 1. Alternative Column Support

Each metric now supports multiple alternative column names:

**Throughput:**
- Primary: `events`
- Alternatives: `logs`, `event_logs`, `training_logs`, `metrics`, `performance_logs`

**Toxicity:**
- Primary: `output`
- Alternatives: `response`, `generated_text`, `completion`, `text`, `generated`, `model_output`

**Bias:**
- Primary: `output`
- Alternatives: `response`, `generated_text`, `completion`, `text`, `generated`, `model_output`

### 2. Fallback Metrics

When no suitable text/log columns are available, the system tries fallback metrics:

**Throughput Fallbacks:**
- `tokens_per_second`, `throughput_rate`, `processing_speed`
- `inference_speed`, `batch_throughput`, `tps`, `throughput`

**Toxicity Fallbacks:**
- `toxicity_score`, `harm_score`, `safety_score`, `danger_score`
- `inappropriate_score`, `offensive_score`, `hate_score`

**Bias Fallbacks:**
- `bias_score`, `fairness_score`, `demographic_bias`, `unfairness_score`
- `discrimination_score`, `stereotype_score`, `equity_score`

### 3. Clear Error Messages

When no suitable columns are found, the system provides:
- Available columns in the dataset
- Suggested alternative column names
- Suggested fallback metrics
- Clear guidance on what's needed

### 4. Configuration Utilities

The `ColumnConfig` class provides utilities for:
- Setting custom primary columns
- Adding alternative column names
- Adding fallback metrics
- Detecting available columns
- Suggesting column mappings
- Generating evaluation kwargs

## Usage Examples

### Basic Usage

```python
import pandas as pd
from rldk.evals.metrics.throughput import evaluate_throughput

# Data with alternative column name
data = pd.DataFrame({
    "logs": [json.dumps(events_data)],  # Alternative to "events"
    "model_name": ["test-model"]
})

# Automatically uses "logs" instead of "events"
result = evaluate_throughput(data)
```

### With Fallback Metrics

```python
# Data with fallback metrics
data = pd.DataFrame({
    "tokens_per_second": [100, 120, 110, 105, 115],
    "model_name": ["test-model"]
})

# Uses fallback metric when no event logs available
result = evaluate_throughput(data)
print(f"Method: {result['method']}")  # "fallback_analysis"
print(f"Metric used: {result['metrics']['metric_used']}")  # "tokens_per_second"
```

### Custom Column Mappings

```python
# Specify custom column mappings
result = evaluate_throughput(
    data,
    log_column="my_events",
    alternative_columns=["custom_logs", "my_logs"],
    min_samples=1
)
```

### Using Configuration Utilities

```python
from rldk.evals.column_config import ColumnConfig, get_evaluation_kwargs

# Create custom configuration
config = ColumnConfig()
config.set_primary_column("throughput", "my_events")
config.add_alternative_column("throughput", "custom_logs")

# Get evaluation kwargs
kwargs = get_evaluation_kwargs("throughput")
# Returns: {"log_column": "my_events", "alternative_columns": ["custom_logs", ...], ...}
```

### Column Detection and Suggestions

```python
from rldk.evals.column_config import detect_columns, suggest_columns

# Detect available columns
data_columns = ["response", "logs", "toxicity_score", "bias_score"]
detected = detect_columns(data_columns)

# Get suggestions
suggestions = suggest_columns(data_columns)
print(suggestions)
# {
#   "throughput": ["Use 'logs' as alternative to 'events'"],
#   "toxicity": ["Use 'response' as alternative to 'output'", "Use fallback metrics: ['toxicity_score']"],
#   "bias": ["Use 'response' as alternative to 'output'", "Use fallback metrics: ['bias_score']"]
# }
```

## Configuration Options

### Evaluation Function Parameters

All evaluation functions now support these additional parameters:

- `alternative_columns`: List of alternative column names to try
- `fallback_to_other_metrics`: Whether to try fallback metrics (default: True)
- `min_samples`: Minimum samples required for evaluation

### ColumnConfig Methods

- `get_config(metric_name)`: Get configuration for a metric
- `set_primary_column(metric_name, column_name)`: Set primary column
- `add_alternative_column(metric_name, column_name)`: Add alternative column
- `add_fallback_metric(metric_name, metric_name)`: Add fallback metric
- `get_evaluation_kwargs(metric_name, custom_config)`: Get evaluation kwargs
- `detect_columns(data_columns)`: Detect available columns
- `suggest_columns(data_columns)`: Suggest column mappings

## Error Handling

### Graceful Degradation

The system gracefully degrades when columns are missing:

1. **Primary column missing** → Try alternative columns
2. **All text/log columns missing** → Try fallback metrics
3. **No suitable data** → Return informative error with suggestions

### Error Response Format

When no suitable columns are found, the response includes:

```python
{
    "score": 0.0,  # or 1.0 for toxicity/bias (high = bad)
    "details": "Detailed error message with suggestions",
    "method": "fallback_analysis",  # or "event_log_analysis", "content_analysis", etc.
    "num_samples": 0,
    "error": "missing_log_column",  # or "no_throughput_data", etc.
    "available_columns": ["col1", "col2", ...],
    "suggested_alternatives": ["alt1", "alt2", ...]
}
```

## Migration Guide

### For Existing Code

No changes required! The system is backward compatible. Existing code will continue to work, but now with better error handling.

### For New Code

Consider using the configuration utilities for better maintainability:

```python
# Old way (still works)
result = evaluate_throughput(data)

# New way (recommended)
from rldk.evals.column_config import get_evaluation_kwargs
kwargs = get_evaluation_kwargs("throughput", {"primary_column": "my_events"})
result = evaluate_throughput(data, **kwargs)
```

## Best Practices

1. **Use descriptive column names** that match the expected patterns
2. **Provide fallback metrics** when possible for better robustness
3. **Use configuration utilities** for complex setups
4. **Check error responses** for helpful suggestions
5. **Test with different data formats** to ensure compatibility

## Troubleshooting

### Common Issues

1. **"No suitable columns found"**
   - Check available columns in error response
   - Use suggested alternatives
   - Consider adding fallback metrics

2. **"Insufficient samples"**
   - Reduce `min_samples` parameter
   - Check data quality and filtering

3. **"Fallback metrics not found"**
   - Add appropriate fallback metrics to your data
   - Use `fallback_to_other_metrics=False` to disable fallbacks

### Getting Help

The error messages now include:
- Available columns in your dataset
- Suggested alternative column names
- Suggested fallback metrics
- Clear guidance on what's needed

Use this information to adjust your data or configuration accordingly.

## Examples

See `examples/robust_column_handling_example.py` for comprehensive examples demonstrating all features.

## Testing

The robust column handling is thoroughly tested in `tests/integration/test_robust_column_handling.py`.

Run tests with:
```bash
python -m pytest tests/integration/test_robust_column_handling.py -v
```