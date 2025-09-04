# Evaluation Results Card

**Suite:** quick
**Sample Size:** 5
**Seed:** 42
**Timestamp:** 2025-09-04 07:05:20

## 📊 Overall Scores

| Metric | Score | Confidence Interval | Effect Size |
|--------|-------|-------------------|-------------|
| alignment | 0.908 | [0.645, 1.000] | 0.695 |
| helpfulness | 0.725 | [0.462, 0.988] | 0.417 |
| harmlessness | 0.773 | [0.510, 1.000] | -0.092 |
| hallucination | 0.092 | [0.000, 0.355] | -0.695 |
| reward_alignment | 0.500 | [0.237, 0.763] | -0.667 |
| prompt_contamination | 0.500 | [0.237, 0.763] | -1.000 |
| answer_leakage | 0.500 | [0.237, 0.763] | -1.000 |
| throughput | 0.000 | [0.000, 0.263] | -2.000 |
| toxicity | 1.000 | [0.737, 1.000] | 2.667 |
| bias | 1.000 | [0.737, 1.000] | 2.333 |

## 🔍 Detailed Results

### alignment

- **details:** Alignment evaluation based on 0 metrics
- **method:** correlation_and_consistency
- **metrics:** []
- **sample_size:** 5

### helpfulness

- **details:** Helpfulness evaluation based on 0 metrics
- **method:** quality_metrics_and_rewards
- **metrics:** []
- **sample_size:** 5

### harmlessness

- **details:** Harmlessness evaluation based on 0 metrics
- **method:** safety_metrics_and_stability
- **metrics:** []
- **sample_size:** 5

### hallucination

- **details:** Hallucination evaluation based on 0 metrics
- **method:** accuracy_metrics_and_consistency
- **metrics:** []
- **sample_size:** 5
- **note:** Lower scores indicate better performance (less hallucination)

### reward_alignment

- **details:** Reward alignment evaluation based on 0 metrics
- **method:** correlation_and_stability
- **metrics:** []
- **sample_size:** 5

### kl_divergence

❌ **Error:** cannot import name 'calculate_kl_divergence_between_runs' from 'rldk.evals.metrics' (/workspace/src/rldk/evals/metrics/__init__.py)

### prompt_contamination

- **details:** No prompt data available for contamination analysis
- **method:** no_prompt_data
- **metrics:** []
- **sample_size:** 5

### answer_leakage

- **details:** No response/prompt data available for leakage analysis
- **method:** no_response_data
- **metrics:** []
- **sample_size:** 5

### throughput

- **details:** No event logs found in column 'events'
- **method:** event_log_analysis
- **num_samples:** 0
- **error:** missing_log_column

### toxicity

- **details:** No output data found in column 'output'
- **method:** content_analysis
- **num_samples:** 0
- **error:** missing_output_column

### bias

- **details:** No output data found in column 'output'
- **method:** demographic_analysis
- **num_samples:** 0
- **error:** missing_output_column

## 📁 Files Generated

- **Evaluation Card:** `eval_card.md`
- **Detailed Results:** `eval_results.jsonl`
- **Summary:** `eval_summary.json`
- **Plots:** `tradeoff_plots.png`
