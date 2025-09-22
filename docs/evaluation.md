# RL Debug Kit Evaluation Metrics

This document describes the evaluation metrics implemented in RL Debug Kit for assessing model performance, safety, and bias.

## Overview

The evaluation suite provides three core metrics designed to be lightweight, interpretable, and actionable:

1. **Throughput**: Measures model processing capacity in tokens per second
2. **Toxicity**: Detects harmful or inappropriate content in model outputs
3. **Bias**: Identifies demographic bias and stereotyping in model responses

## Core Metrics

### 1. Throughput Evaluation

**Purpose**: Measure model processing capacity and efficiency.

**Methodology**:
- Parses event logs to extract token generation and batch processing events
- Calculates tokens per second from timestamp intervals
- Computes confidence intervals using bootstrap resampling
- Normalizes scores to [0, 1] range based on expected maximum throughput

**Key Metrics**:
- `mean_tokens_per_sec`: Average processing rate
- `throughput_stability`: Consistency of processing speed
- `confidence_interval`: Statistical uncertainty bounds
- `total_tokens`: Total tokens processed

**Assumptions**:
- Event logs contain timestamp and token count information
- Processing is roughly linear over time intervals
- Maximum reasonable throughput is 1000 tokens/sec

**Limitations**:
- Requires structured event logs
- May not capture GPU/CPU utilization directly
- Assumes consistent token counting methodology

**Expected Range**: 0.0 - 1.0 (higher is better)

### 2. Toxicity Evaluation

**Purpose**: Detect harmful, inappropriate, or dangerous content in model outputs.

**Methodology**:
- Uses lightweight word-based classifier with toxic keyword lists
- Detects toxic patterns using regex matching
- Applies mitigation factors for negation and context
- Supports external classifiers (Detoxify) when available

**Key Metrics**:
- `mean_toxicity`: Average toxicity score across outputs
- `high_toxicity_ratio`: Proportion of outputs above threshold
- `pattern_score`: Score from toxic pattern detection
- `confidence_interval`: Statistical uncertainty bounds

**Toxic Categories Detected**:
- Hate speech and slurs
- Violence and threats
- Harmful content
- Discriminatory terms
- Offensive language
- Explicit content

**Assumptions**:
- Toxic content correlates with presence of specific keywords
- Negation words reduce toxicity scores
- Pattern-based detection captures structured harmful content

**Limitations**:
- May miss context-dependent toxicity
- Requires regular updates to keyword lists
- Simple classifier may have false positives/negatives

**Expected Range**: 0.0 - 1.0 (lower is better)

### 3. Bias Evaluation

**Purpose**: Identify demographic bias and stereotyping in model outputs.

**Methodology**:
- Detects demographic mentions across categories (gender, race, age, religion, nationality)
- Analyzes sentiment differences across demographic groups
- Identifies stereotype patterns using regex matching
- Calculates bias score as coefficient of variation across groups

**Key Metrics**:
- `demographic_bias_score`: Overall bias across demographic groups
- `mean_stereotype_score`: Average stereotype pattern detection
- `sentiment_variance`: Variance in sentiment across groups
- `demographic_bias_details`: Per-group bias breakdown

**Demographic Categories**:
- Gender: male, female, non-binary
- Race: white, black, asian, hispanic, middle eastern, indigenous
- Age: young, adult, elderly
- Religion: christian, muslim, jewish, hindu, buddhist, atheist
- Nationality: american, british, canadian, australian, etc.

**Assumptions**:
- Sentiment differences indicate bias
- Stereotype patterns follow predictable structures
- Demographic mentions are explicit in text

**Limitations**:
- Requires sufficient demographic mentions for analysis
- May miss implicit bias
- Sentiment analysis is simplified
- Cultural context may affect interpretation

**Expected Range**: 0.0 - 1.0 (lower is better)

## Evaluation Suites

### Quick Suite
- **Runtime**: 2-5 minutes
- **Sample Size**: 50
- **Metrics**: Throughput, Toxicity, Bias + existing probes
- **Use Case**: Rapid model assessment

### Comprehensive Suite
- **Runtime**: 10-20 minutes
- **Sample Size**: 200
- **Metrics**: All metrics + additional integrity checks
- **Use Case**: Detailed model analysis

### Safety Suite
- **Runtime**: 5-10 minutes
- **Sample Size**: 100
- **Metrics**: Toxicity, Bias + safety-focused probes
- **Use Case**: Safety-focused evaluation

## Usage

### Command Line Interface

```bash
# Run quick evaluation
python -m rldk.evals.cli evaluate data.jsonl --suite quick --output results.json

# Run comprehensive evaluation with verbose output
python -m rldk.evals.cli evaluate data.jsonl --suite comprehensive --verbose

# List available suites
python -m rldk.evals.cli list-suites

# Validate input file
python -m rldk.evals.cli validate data.jsonl
```

## Reward Model Overoptimization Detector

The reward health toolkit now tracks **reward model overoptimization** by comparing proxy reward improvements to trusted "gold" metrics such as human evaluation scores or benchmark accuracy.

### Providing Gold Metrics

- **CLI**: supply a gold metrics table with `--gold` (or embed the column directly in the run data) and specify the column via `--gold-col`. Gold metrics should align on the training step column.
- **API**: pass a pandas DataFrame or Series to `reward_health(..., gold_metrics=..., gold_metric_col="gold_metric")`. The helper auto-aligns by step and handles early/late window splits.

### How Detection Works

1. Compute early vs. late window means (default first/last 100 steps) for proxy reward and gold metrics.
2. Calculate the delta (`proxy - gold`). If the delta exceeds the configurable threshold (default `0.2`) while gold metrics stagnate or regress, the detector activates.
3. Track Pearson/Spearman correlation drift between proxy and gold metrics.
4. Pull recent KL statistics from PPO forensics; sustained high KL combined with proxy/gold divergence is treated as a red flag.

### Interpreting the Report

- **Summary Card**: shows proxy/gold delta, correlation trend, KL snapshot, and whether overoptimization was flagged.
- **JSON Summary**: `overoptimization` contains metrics, thresholds, and remediation notes.
- **Warnings**: If no gold metrics are provided, the report records a warning instead of failing.

### Tuning Thresholds

- `--overopt-window`: change the size of the early/late comparison window.
- `--overopt-delta-threshold`: tighten or relax the delta needed to warn.
- `--overopt-min-samples`: require a minimum number of overlapping proxy/gold samples before analysis runs.

Remediation tips include pausing reward optimization, revisiting KL controllers, and refreshing gold metric evaluations to realign the reward model.

### Programmatic Usage

```python
from rldk.evals.metrics import evaluate_throughput, evaluate_toxicity, evaluate_bias
import pandas as pd

# Load your data
data = pd.read_json("data.jsonl", lines=True)

# Run individual metrics
throughput_result = evaluate_throughput(data)
toxicity_result = evaluate_toxicity(data)
bias_result = evaluate_bias(data)

# Run evaluation suite
from rldk.cli import run_evaluation_suite
results = run_evaluation_suite(data, suite_name="quick")
```

## Input Data Format

### JSONL Structure
Each line should be a JSON object with the following structure:

```json
{
  "output": "Model output text here",
  "events": "[{\"event_type\": \"token_generated\", \"timestamp\": \"2024-01-01T10:00:00Z\", \"token_count\": 50}]",
  "model_name": "model-name",
  "timestamp": "2024-01-01T10:00:00Z"
}
```

### Required Columns
- `output`: Model output text (for toxicity and bias evaluation)
- `events`: JSON array of event logs (for throughput evaluation)

### Optional Columns
- `model_name`: Model identifier
- `timestamp`: Timestamp for the output
- Any additional metadata columns

## Output Format

### Individual Metric Results
```json
{
  "score": 0.75,
  "details": "Throughput: 500.0 ± 25.0 tokens/sec",
  "method": "event_log_analysis",
  "num_samples": 100,
  "metrics": {
    "mean_tokens_per_sec": 500.0,
    "std_tokens_per_sec": 25.0,
    "total_tokens": 50000,
    "throughput_stability": 0.95,
    "confidence_interval": {
      "lower": 0.70,
      "upper": 0.80,
      "level": 0.95
    }
  }
}
```

### Suite Results
```json
{
  "suite_name": "quick",
  "suite_description": "Fast evaluation suite for quick model assessment",
  "evaluations": {
    "throughput": { ... },
    "toxicity": { ... },
    "bias": { ... }
  },
  "summary": {
    "total_evaluations": 3,
    "successful_evaluations": 3,
    "failed_evaluations": 0,
    "overall_score": 0.65,
    "errors": []
  }
}
```

## Extending the Suite

### Adding New Metrics

1. Create a new metric module in `src/rldk/evals/metrics/`
2. Implement the evaluation function with standard signature:
   ```python
   def evaluate_new_metric(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
       # Implementation here
       return {
           "score": float(score),
           "details": str(details),
           "method": str(method),
           "num_samples": int(num_samples),
           "metrics": dict(additional_metrics)
       }
   ```
3. Add to `src/rldk/evals/metrics/__init__.py`
4. Update evaluation suites in `src/rldk/evals/suites.py`
5. Add tests in `tests/test_evals_new_metric.py`

### Custom Classifiers

The toxicity and bias metrics support external classifiers:

```python
# Toxicity with Detoxify
result = evaluate_toxicity(data, use_external_classifier=True)

# Bias with VADER
result = evaluate_bias(data, use_external_sentiment=True)
```

## Best Practices

### Data Preparation
- Ensure consistent timestamp formats (ISO 8601)
- Validate JSON structure before evaluation
- Include sufficient samples for statistical significance
- Clean and normalize text data

### Interpretation
- Consider confidence intervals when comparing models
- Look at detailed metrics, not just overall scores
- Account for dataset characteristics and biases
- Use multiple evaluation suites for comprehensive assessment

### Performance
- Use appropriate sample sizes for your use case
- Consider caching for expensive computations
- Monitor memory usage with large datasets
- Use external classifiers judiciously (they may be slow)

## Troubleshooting

### Common Issues

1. **Missing columns**: Ensure required columns are present in data
2. **Invalid JSON**: Check JSON formatting in events column
3. **Insufficient samples**: Increase sample size or adjust min_samples parameter
4. **Import errors**: Install optional dependencies for external classifiers

### Error Handling

The evaluation functions return error information when failures occur:

```json
{
  "score": 0.0,
  "details": "Error description",
  "error": "error_type",
  "num_samples": 0
}
```

### Debugging

Enable verbose logging for detailed debugging:

```bash
python -m rldk.evals.cli evaluate data.jsonl --verbose
```

## Contributing

When contributing new metrics or improvements:

1. Follow the established code structure
2. Add comprehensive tests
3. Update documentation
4. Include example usage
5. Consider performance implications
6. Validate with real-world data

## References

- Detoxify: https://github.com/unitaryai/detoxify
- VADER Sentiment: https://github.com/cjhutto/vaderSentiment
- Bootstrap Confidence Intervals: Efron, B. (1979). Bootstrap methods: Another look at the jackknife