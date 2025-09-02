# Evaluation Integrity Checks Guide

This guide explains the evaluation integrity checks that have been added to RL Debug Kit to detect prompt contamination and answer leakage in evaluation data.

## Overview

Evaluation integrity checks are essential for ensuring that evaluation results are reliable and not compromised by data contamination or information leakage. The new integrity checks detect:

1. **Prompt Contamination** - When evaluation prompts contain information that could bias the model
2. **Answer Leakage** - When expected answers or solutions are inadvertently present in prompts
3. **Data Split Integrity** - When there's contamination between train/validation/test splits
4. **Evaluation Robustness** - When evaluation metrics are unreliable due to insufficient data or systematic biases

## Quick Start

### Running Integrity Checks

```bash
# Run the dedicated integrity suite
rldk eval --suite integrity --data your_data.csv

# Run integrity checks as part of quick evaluation
rldk eval --suite quick --data your_data.csv

# Run integrity checks as part of comprehensive evaluation
rldk eval --suite comprehensive --data your_data.csv
```

### Available Suites

- **`integrity`** - Dedicated suite focused on integrity checks (8-15 minutes)
- **`quick`** - Includes basic prompt contamination and answer leakage checks (2-5 minutes)
- **`comprehensive`** - Includes all integrity checks plus other evaluations (10-20 minutes)

## Integrity Check Details

### 1. Prompt Contamination Detection

**Purpose**: Detects if evaluation prompts contain information that could bias the model or if there's contamination between data splits.

**What it checks**:
- Duplicate prompts (potential contamination)
- Prompt length anomalies (very short/long prompts)
- Test-like patterns (e.g., "Answer the following question")
- Metadata leakage (epoch, step, batch_idx values in prompts)

**Example detection**:
```python
# This would trigger contamination detection
prompts = [
    "Test prompt for epoch 1 at step 100",
    "Test prompt for epoch 1 at step 100",  # Duplicate
    "Answer the following question: What is 2 + 2?"  # Test pattern
]
```

**Score interpretation**:
- **High score (0.8-1.0)**: Clean prompts, no contamination detected
- **Medium score (0.5-0.8)**: Some contamination detected
- **Low score (0.0-0.5)**: Significant contamination detected

### 2. Answer Leakage Detection

**Purpose**: Detects if the expected answers or solutions are inadvertently present in the prompts.

**What it checks**:
- Direct answer leakage (answer appears in prompt)
- Partial answer leakage (key parts of answer in prompt)
- Numerical answer leakage (numbers from answer in prompt)
- Semantic similarity between prompts and responses

**Example detection**:
```python
# This would trigger leakage detection
prompts = [
    "What is 2 + 2? The answer is 4. Please confirm.",  # Direct leakage
    "Calculate the sum of 5 and 3"  # No leakage
]
responses = [
    "The answer is 4",  # Matches leaked answer
    "The sum is 8"      # No match
]
```

**Score interpretation**:
- **High score (0.8-1.0)**: No leakage detected
- **Medium score (0.5-0.8)**: Some leakage detected
- **Low score (0.0-0.5)**: Significant leakage detected

### 3. Data Split Integrity Detection

**Purpose**: Detects contamination between different data splits (train/validation/test).

**What it checks**:
- Proper split distribution (balanced splits)
- Duplicate content across splits
- Temporal violations (if timestamps are available)

**Example detection**:
```python
# This would trigger integrity issues
data = {
    "split": ["train"] * 190 + ["val"] * 5 + ["test"] * 5,  # Unbalanced
    "prompt": ["Same prompt"] * 200  # Cross-split duplicates
}
```

**Score interpretation**:
- **High score (0.9-1.0)**: Good split integrity
- **Medium score (0.7-0.9)**: Some integrity issues
- **Low score (0.0-0.7)**: Significant integrity issues

### 4. Evaluation Robustness Detection

**Purpose**: Checks for potential issues that could make evaluations unreliable.

**What it checks**:
- Sample size adequacy
- High variance in key metrics
- Systematic biases (correlation with metadata)
- Outliers that might skew results

**Example detection**:
```python
# This would trigger robustness issues
data = {
    "reward_mean": [0.1, 0.2, 0.1, 0.2, 0.1],  # Small sample
    "step": [1, 2, 3, 4, 5],
    "reward_mean": [0.1, 0.2, 0.3, 0.4, 0.5]  # Correlated with step
}
```

**Score interpretation**:
- **High score (0.8-1.0)**: Robust evaluation
- **Medium score (0.6-0.8)**: Some robustness issues
- **Low score (0.0-0.6)**: Significant robustness issues

## Configuration

### Thresholds

The integrity checks use configurable thresholds for different types of issues:

```python
# Default thresholds
CONTAMINATION_THRESHOLDS = {
    "duplicate_ratio": 0.1,      # >10% duplicates
    "test_pattern_ratio": 0.3,    # >30% test patterns
    "metadata_leakage_ratio": 0.05,  # >5% metadata leakage
}

LEAKAGE_THRESHOLDS = {
    "direct_leakage_ratio": 0.1,  # >10% direct leakage
    "partial_leakage_ratio": 0.2, # >20% partial leakage
    "numerical_leakage_ratio": 0.15,  # >15% numerical leakage
}
```

### Customizing Checks

You can customize the integrity checks by modifying the thresholds or adding new detection patterns:

```python
from rldk.evals.integrity import evaluate_prompt_contamination

# Custom evaluation with different thresholds
result = evaluate_prompt_contamination(
    data, 
    seed=42,
    duplicate_threshold=0.05,  # More strict
    pattern_threshold=0.2      # More strict
)
```

## Integration with Existing Workflows

### CLI Integration

The integrity checks are automatically included in the evaluation CLI:

```bash
# Check integrity of your data
rldk eval --suite integrity --data training_data.csv

# Get detailed report
rldk eval --suite integrity --data training_data.csv --output-dir integrity_report/
```

### Programmatic Usage

```python
from rldk.evals.runner import run
from rldk.evals.integrity import evaluate_prompt_contamination

# Run full evaluation with integrity checks
result = run(data, suite="integrity", seed=42)

# Run specific integrity check
contamination_result = evaluate_prompt_contamination(data, seed=42)
print(f"Contamination score: {contamination_result['score']}")
```

## Best Practices

### 1. Run Integrity Checks Early

Run integrity checks before training to catch issues early:

```bash
# Before training
rldk eval --suite integrity --data dataset.csv
```

### 2. Monitor During Training

Include integrity checks in your evaluation pipeline:

```python
# In your training loop
if step % 1000 == 0:
    eval_result = run(training_data, suite="quick", seed=42)
    if eval_result.scores["prompt_contamination"] < 0.7:
        print("Warning: Potential contamination detected")
```

### 3. Set Up Alerts

Configure alerts for integrity issues:

```python
# Alert configuration
INTEGRITY_ALERTS = {
    "prompt_contamination": 0.7,
    "answer_leakage": 0.7,
    "data_split_integrity": 0.8,
    "evaluation_robustness": 0.7,
}
```

### 4. Document Your Data Pipeline

Keep track of data processing steps to help debug integrity issues:

```python
# Data lineage tracking
data_lineage = {
    "source": "original_dataset.csv",
    "processing_steps": [
        "filtered_by_quality",
        "split_train_val_test",
        "tokenized"
    ],
    "integrity_checks": {
        "prompt_contamination": 0.85,
        "answer_leakage": 0.92,
    }
}
```

## Troubleshooting

### Common Issues

1. **High contamination scores**: Check for duplicate prompts or metadata leakage
2. **High leakage scores**: Review prompt-answer pairs for information overlap
3. **Low split integrity**: Verify train/val/test splits are properly separated
4. **Low robustness scores**: Check sample size and metric variance

### Debugging Tips

1. **Examine specific metrics**: Look at the detailed metrics in the evaluation results
2. **Check data preprocessing**: Review your data pipeline for potential issues
3. **Validate manually**: Manually inspect a sample of problematic data
4. **Compare baselines**: Compare against known good datasets

### Example Debugging Session

```python
from rldk.evals.integrity import evaluate_prompt_contamination

# Debug contamination issues
result = evaluate_prompt_contamination(data, seed=42)
print("Detailed metrics:")
for metric, value in result["metrics"]:
    print(f"  {metric}: {value}")

# Check specific problematic data
if result["score"] < 0.7:
    print("Investigating contamination...")
    # Add your debugging code here
```

## Performance Considerations

### Runtime

- **Quick suite**: 2-5 minutes (includes basic integrity checks)
- **Integrity suite**: 8-15 minutes (comprehensive integrity analysis)
- **Comprehensive suite**: 10-20 minutes (all evaluations including integrity)

### Memory Usage

- **Small datasets** (< 1K samples): Minimal memory usage
- **Medium datasets** (1K-10K samples): Moderate memory usage
- **Large datasets** (> 10K samples): Consider sampling for integrity checks

### Optimization Tips

1. **Sample large datasets**: Use `sample_size` parameter for large datasets
2. **Parallel processing**: Integrity checks can be run in parallel
3. **Caching**: Cache results for repeated evaluations

## Future Enhancements

### Planned Features

1. **Advanced NLP-based detection**: Using language models for better contamination detection
2. **Cross-dataset contamination**: Detecting contamination across different datasets
3. **Real-time monitoring**: Continuous integrity monitoring during training
4. **Automated fixes**: Suggestions for fixing integrity issues

### Contributing

To contribute to the integrity checks:

1. Add new detection patterns
2. Improve existing algorithms
3. Add new integrity metrics
4. Enhance documentation

## Conclusion

Evaluation integrity checks are crucial for ensuring reliable evaluation results. By detecting prompt contamination, answer leakage, and other integrity issues, these checks help maintain the quality and trustworthiness of your evaluation pipeline.

For more information, see the main RL Debug Kit documentation and the integrity check source code in `src/rldk/evals/integrity.py`.