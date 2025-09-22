# Evaluation Data Requirements

This document describes the data requirements for RL Debug Kit evaluation suites, including required and optional columns, data types, and handling of missing data.

## Required Columns

| Column Name | Data Type | Description | Synonyms | Example |
|-------------|-----------|-------------|----------|---------|
| `step` | numeric | Training step or global step number for temporal analysis | `global_step`, `iteration`, `epoch` | `1000` |
| `output` | text | Model output text for evaluation | `response`, `completion`, `text`, `generation` | `"This is a helpful response."` |

## Optional Columns

| Column Name | Data Type | Description | Synonyms | Example |
|-------------|-----------|-------------|----------|---------|
| `reward` | numeric | Reward signal for the output | `reward_mean`, `score`, `value` | `0.85` |
| `kl_to_ref` | numeric | KL divergence to reference model | `kl`, `kl_divergence`, `kl_mean` | `0.12` |
| `events` | object | Event logs for detailed analysis | `event_logs`, `logs`, `events_raw` | `[{"event_type": "token_generated", "timestamp": "2024-01-01T00:00:00Z", "token_count": 10}]` |
| `task` | text | Benchmark or task identifier used for regression analysis | `task_id`, `benchmark`, `dataset`, `evaluation_name` | `"gsm8k"` |
| `score` | numeric | Evaluation score for the associated task sample | `metric`, `metric_value`, `accuracy` | `0.78` |

## What Happens if a Column is Missing

### Missing Required Columns

If a required column is missing, the evaluation will fail with a clear error message indicating:

- Which column is missing
- What synonyms are accepted (if any)
- How to fix the issue

**Example Error Messages:**
```
Missing required column: output. Provide one of: output, response, completion, text
Missing required column: step. Provide one of: step, global_step, iteration, epoch
```

### Missing Optional Columns

If optional columns are missing, the evaluation will proceed with warnings:

- **Missing `events` column**: Event-based diagnostics will be skipped
- **Missing `reward` column**: Reward-based metrics will be unavailable
- **Missing `kl_to_ref` column**: KL divergence analysis will be skipped

## Data Validation

The evaluation system performs automatic data validation:

1. **Column Normalization**: Automatically maps synonyms to standard column names
2. **Type Validation**: Ensures numeric columns contain numeric data
3. **Empty Data Handling**: Warns about empty DataFrames or all-NaN columns

## Example Usage

### Basic Example

```python
import pandas as pd
from rldk.evals import run

# Create sample data with required columns
data = pd.DataFrame({
    'step': [1, 2, 3, 4, 5],
    'output': [
        "This is a helpful response.",
        "Another response here.",
        "Yet another response.",
        "More helpful text.",
        "Final response."
    ],
    'reward': [0.8, 0.9, 0.7, 0.85, 0.95]
})

# Run evaluation
result = run(data, suite="quick")
print(f"Overall Score: {result.overall_score}")
print(f"Available Metrics: {result.available_fraction:.1%}")
```

### With Event Logs

```python
import pandas as pd
from rldk.evals import run

# Create data with event logs
data = pd.DataFrame({
    'step': [1, 2, 3],
    'output': ["Response 1", "Response 2", "Response 3"],
    'events': [
        [{"event_type": "token_generated", "timestamp": "2024-01-01T00:00:00Z", "token_count": 10}],
        [{"event_type": "token_generated", "timestamp": "2024-01-01T00:00:01Z", "token_count": 15}],
        [{"event_type": "token_generated", "timestamp": "2024-01-01T00:00:02Z", "token_count": 12}]
    ]
})

# Run evaluation
result = run(data, suite="quick")
```

### Using Column Synonyms

```python
import pandas as pd
from rldk.evals import run

# Data with synonym column names (will be automatically normalized)
data = pd.DataFrame({
    'global_step': [1, 2, 3],  # Will be normalized to 'step'
    'response': ["Response 1", "Response 2", "Response 3"],  # Will be normalized to 'output'
    'reward_mean': [0.8, 0.9, 0.7]  # Will be normalized to 'reward'
})

# Run evaluation
result = run(data, suite="quick")
```

## Evaluation Results

When evaluations complete, you get an `EvalResult` object with:

- **`overall_score`**: Unweighted mean of available metrics (None if no metrics available)
- **`available_fraction`**: Fraction of metrics that produced valid values (0.0 to 1.0)
- **`warnings`**: List of warnings about missing data or other issues
- **`scores`**: Dictionary of individual metric scores
- **`metadata`**: Additional information about the evaluation run

### Handling Missing Metrics

- If no metrics can be computed, `overall_score` will be `None`
- Individual metric scores will be `None` if they cannot be computed
- Warnings will indicate which metrics failed and why

## Best Practices

1. **Always include required columns**: `step` and `output` are mandatory
2. **Use standard column names**: While synonyms work, standard names are clearer
3. **Include optional columns when available**: More data leads to more comprehensive evaluations
4. **Check warnings**: Review the `warnings` list in results for data quality issues
5. **Validate data beforehand**: Use `validate_eval_input()` to check data before running evaluations

## Troubleshooting

### Common Issues

1. **"Missing required column: output"**
   - Ensure your data has a column with model outputs
   - Use one of the accepted synonyms: `response`, `completion`, `text`, `generation`

2. **"Missing required column: step"**
   - Include a step counter or iteration number
   - Use one of the accepted synonyms: `global_step`, `iteration`, `epoch`

3. **"events column not provided, event-based diagnostics will be skipped"**
   - This is a warning, not an error
   - Evaluation will proceed but some metrics (like throughput) may not be available

4. **"No valid metrics computed"**
   - Check that your data contains the necessary columns for the evaluation suite
   - Some suites require specific columns (e.g., throughput evaluation needs event logs)

### Getting Help

If you encounter issues:

1. Check the warnings in your evaluation results
2. Verify your data contains the required columns
3. Ensure your data types are correct (numeric columns should contain numbers)
4. Review the evaluation suite documentation for specific requirements