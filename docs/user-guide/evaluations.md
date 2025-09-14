# Evaluation Suites

RLDK provides comprehensive evaluation suites to assess model performance, safety, and quality across multiple dimensions.

## Overview

The evaluation system includes:
- **Predefined suites** for quick, comprehensive, and safety-focused evaluation
- **Individual metrics** for bias, toxicity, throughput, and quality assessment
- **Configurable thresholds** and parameters
- **Parallel execution** for faster evaluation
- **Multiple output formats** (JSON, YAML, HTML reports)

## Available Evaluation Suites

### Quick Suite
Fast evaluation with essential metrics for rapid feedback:

```bash
rldk evals evaluate data.jsonl --suite quick
```

**Included metrics:**
- Throughput (tokens/second)
- Basic quality metrics
- Essential safety checks

**Typical runtime:** 1-5 minutes

### Comprehensive Suite
Complete evaluation with all available metrics:

```bash
rldk evals evaluate data.jsonl --suite comprehensive
```

**Included metrics:**
- All throughput metrics
- Comprehensive bias analysis
- Full toxicity evaluation
- Quality and coherence metrics
- Statistical analysis

**Typical runtime:** 10-30 minutes

### Safety Suite
Safety-focused evaluation for bias and toxicity:

```bash
rldk evals evaluate data.jsonl --suite safety
```

**Included metrics:**
- Demographic bias detection
- Toxicity classification
- Harmful content identification
- Fairness metrics

**Typical runtime:** 5-15 minutes

## Command Line Usage

### List Available Suites

```bash
# Show all available evaluation suites
rldk evals list-suites
```

### Run Evaluation

```bash
# Basic evaluation
rldk evals evaluate data.jsonl --suite quick

# Custom output location
rldk evals evaluate data.jsonl --suite comprehensive --output results.json

# Specific metrics only
rldk evals evaluate data.jsonl --metrics throughput,bias,toxicity

# Parallel execution
rldk evals evaluate data.jsonl --suite comprehensive --parallel

# Custom configuration
rldk evals evaluate data.jsonl --config custom_eval.yaml
```

## Python API

### Basic Usage

```python
from rldk.evals import get_eval_suite, run_evaluation

# Get predefined suite
suite = get_eval_suite("comprehensive")

# Run evaluation
result = run_evaluation(
    data="training_data.jsonl",
    suite=suite,
    output_path="eval_results.json",
    parallel=True
)

print(f"Overall score: {result.overall_score}")
print(f"Individual metrics: {result.metrics}")
```

### Custom Evaluation Configuration

```python
from rldk.evals import EvaluationConfig, run_evaluation

# Custom configuration
config = EvaluationConfig(
    metrics=["throughput", "bias", "toxicity"],
    thresholds={
        "throughput_min": 100.0,
        "bias_max": 0.1,
        "toxicity_max": 0.05
    },
    parallel=True,
    timeout=600
)

# Run with custom config
result = run_evaluation(
    data="data.jsonl",
    config=config,
    output_path="custom_results.json"
)
```

### Individual Metrics

```python
from rldk.evals.metrics import evaluate_throughput, evaluate_bias, evaluate_toxicity

# Throughput evaluation
throughput_result = evaluate_throughput(
    model=model,
    tokenizer=tokenizer,
    test_data=test_prompts,
    batch_size=32
)

# Bias evaluation
bias_result = evaluate_bias(
    model=model,
    tokenizer=tokenizer,
    test_data=bias_test_set,
    demographic_groups=["gender", "race", "age"]
)

# Toxicity evaluation
toxicity_result = evaluate_toxicity(
    model=model,
    tokenizer=tokenizer,
    test_data=safety_prompts,
    threshold=0.5
)
```

## Individual Metrics

### Throughput Metrics

Measure model processing speed and efficiency:

```python
from rldk.evals.metrics import ThroughputMetric

metric = ThroughputMetric(
    batch_sizes=[1, 4, 8, 16],
    sequence_lengths=[128, 256, 512],
    warmup_steps=10,
    measurement_steps=100
)

result = metric.evaluate(model, tokenizer, test_data)
print(f"Tokens/second: {result.tokens_per_second}")
print(f"Latency: {result.latency_ms}ms")
```

### Bias Detection

Detect demographic bias in model outputs:

```python
from rldk.evals.metrics import BiasMetric

metric = BiasMetric(
    demographic_groups=["gender", "race", "religion"],
    bias_types=["sentiment", "toxicity", "stereotypes"],
    threshold=0.1
)

result = metric.evaluate(model, tokenizer, bias_test_data)
print(f"Bias score: {result.bias_score}")
print(f"Affected groups: {result.biased_groups}")
```

### Toxicity Classification

Identify harmful or toxic content:

```python
from rldk.evals.metrics import ToxicityMetric

metric = ToxicityMetric(
    toxicity_threshold=0.5,
    categories=["hate", "harassment", "violence", "self-harm"],
    use_perspective_api=True
)

result = metric.evaluate(model, tokenizer, safety_prompts)
print(f"Toxicity rate: {result.toxicity_rate}")
print(f"Flagged outputs: {len(result.toxic_outputs)}")
```

## Data Requirements

### Input Data Format

Evaluation data should be in JSONL format with required fields:

```json
{"prompt": "What is the capital of France?", "expected": "Paris", "metadata": {"category": "geography"}}
{"prompt": "Explain quantum computing", "expected": null, "metadata": {"category": "science"}}
```

**Required fields:**
- `prompt`: Input text for the model
- `expected`: Expected output (optional for some metrics)
- `metadata`: Additional context (optional)

### Data Preparation

```python
import pandas as pd

# Prepare evaluation data
data = [
    {
        "prompt": "What is machine learning?",
        "expected": "Machine learning is a subset of AI...",
        "metadata": {"category": "technology", "difficulty": "easy"}
    },
    # ... more examples
]

# Save as JSONL
df = pd.DataFrame(data)
df.to_json("eval_data.jsonl", orient="records", lines=True)
```

## Configuration Files

### Custom Evaluation Configuration

```yaml
# custom_eval.yaml
name: "custom_evaluation"
description: "Custom evaluation for my model"

metrics:
  - name: "throughput"
    config:
      batch_sizes: [1, 4, 8]
      sequence_lengths: [128, 256]
      warmup_steps: 5
      measurement_steps: 50
  
  - name: "bias"
    config:
      demographic_groups: ["gender", "race"]
      threshold: 0.1
      
  - name: "toxicity"
    config:
      threshold: 0.3
      use_perspective_api: false

thresholds:
  throughput_min: 50.0
  bias_max: 0.15
  toxicity_max: 0.1

execution:
  parallel: true
  timeout: 300
  max_workers: 4

output:
  format: "json"
  include_details: true
  save_intermediate: false
```

### Global Configuration

```yaml
# ~/.rldk/eval_config.yaml
default_suite: "quick"
default_parallel: true
default_timeout: 600

thresholds:
  throughput:
    min_tokens_per_second: 100
    max_latency_ms: 1000
  bias:
    max_bias_score: 0.1
  toxicity:
    max_toxicity_rate: 0.05

perspective_api:
  enabled: false
  api_key: "${PERSPECTIVE_API_KEY}"
  rate_limit: 10
```

## Output Reports

### Evaluation Result Structure

```json
{
  "summary": {
    "overall_score": 0.85,
    "passed": true,
    "total_metrics": 5,
    "passed_metrics": 4,
    "failed_metrics": 1
  },
  "metrics": {
    "throughput": {
      "score": 0.9,
      "passed": true,
      "tokens_per_second": 150.5,
      "latency_ms": 45.2,
      "threshold": 100.0
    },
    "bias": {
      "score": 0.8,
      "passed": true,
      "bias_score": 0.08,
      "affected_groups": [],
      "threshold": 0.1
    },
    "toxicity": {
      "score": 0.6,
      "passed": false,
      "toxicity_rate": 0.12,
      "flagged_outputs": 12,
      "threshold": 0.1
    }
  },
  "details": {
    "data_size": 1000,
    "evaluation_time": 120.5,
    "model_info": {
      "name": "gpt2",
      "parameters": "124M"
    }
  },
  "recommendations": [
    "Consider additional safety filtering to reduce toxicity rate",
    "Throughput performance is excellent",
    "Bias metrics are within acceptable range"
  ]
}
```

## Integration with Training

### TRL Integration

```python
from trl import PPOTrainer
from rldk.integrations.trl import RLDKCallback

# Create evaluation callback
callback = RLDKCallback(
    enable_evaluation=True,
    eval_config={
        "suite": "safety",
        "eval_frequency": 100,  # Every 100 steps
        "eval_data": "safety_prompts.jsonl"
    }
)

trainer = PPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    callbacks=[callback]
)

# Train with periodic evaluation
trainer.train()

# Get evaluation history
eval_history = callback.get_evaluation_history()
```

### Custom Training Integration

```python
from rldk.evals import get_eval_suite, run_evaluation

# Get evaluation suite
eval_suite = get_eval_suite("comprehensive")

# Training loop with evaluation
for epoch in range(num_epochs):
    # ... training code ...
    
    # Periodic evaluation
    if epoch % eval_frequency == 0:
        result = run_evaluation(
            data=eval_data,
            suite=eval_suite,
            model=model,
            tokenizer=tokenizer
        )
        
        print(f"Epoch {epoch} - Evaluation score: {result.overall_score}")
        
        # Early stopping based on evaluation
        if result.overall_score < min_score_threshold:
            print("Evaluation score too low, stopping training")
            break
```

## Best Practices

1. **Start with Quick Suite**: Use quick evaluation during development
2. **Regular Safety Checks**: Run safety suite frequently for production models
3. **Comprehensive Before Deployment**: Run full evaluation before model deployment
4. **Track Over Time**: Monitor evaluation metrics across training runs
5. **Custom Thresholds**: Set appropriate thresholds for your use case
6. **Parallel Execution**: Use parallel evaluation for faster results
7. **Data Quality**: Ensure evaluation data is representative and high-quality

## Troubleshooting

### Common Issues

1. **Slow Evaluation**: Use parallel execution and smaller data samples
2. **Memory Issues**: Reduce batch sizes or use gradient checkpointing
3. **API Rate Limits**: Configure rate limiting for external APIs
4. **Missing Dependencies**: Install optional dependencies as needed

### Performance Tips

- Use `--parallel` for faster evaluation
- Start with smaller data samples for testing
- Cache evaluation results when possible
- Use appropriate batch sizes for your hardware

For more examples and advanced usage, see the [Examples](../examples/basic-ppo-cartpole.md) section.
