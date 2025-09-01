# Evaluation Results Card

**Suite:** quick
**Sample Size:** 50
**Seed:** 42
**Timestamp:** 2025-09-01 04:32:37

## 📊 Overall Scores

| Metric | Score | Confidence Interval | Effect Size |
|--------|-------|-------------------|-------------|
| alignment | 0.500 | [0.417, 0.583] | -0.667 |
| helpfulness | 0.500 | [0.417, 0.583] | -0.333 |
| harmlessness | 0.500 | [0.417, 0.583] | -1.000 |
| hallucination | 0.500 | [0.417, 0.583] | 0.667 |
| reward_alignment | 1.000 | [0.917, 1.000] | 1.000 |
| kl_divergence | 1.000 | [0.917, 1.000] | 0.667 |

## 🔍 Detailed Results

### alignment

- **details:** Alignment evaluation based on 0 metrics
- **method:** correlation_and_consistency
- **metrics:** []
- **sample_size:** 50

### helpfulness

- **details:** Helpfulness evaluation based on 0 metrics
- **method:** quality_metrics_and_rewards
- **metrics:** []
- **sample_size:** 50

### harmlessness

- **details:** Harmlessness evaluation based on 0 metrics
- **method:** safety_metrics_and_stability
- **metrics:** []
- **sample_size:** 50

### hallucination

- **details:** Hallucination evaluation based on 0 metrics
- **method:** accuracy_metrics_and_consistency
- **metrics:** []
- **sample_size:** 50
- **note:** Lower scores indicate better performance (less hallucination)

### reward_alignment

- **details:** Reward alignment evaluation based on 1 metrics
- **method:** correlation_and_stability
- **metrics:** [('reward_stability', np.float64(1.0))]
- **sample_size:** 50

### kl_divergence

- **kl_divergence_mean:** 0.0
- **details:** KL divergence evaluation across 3 metrics
- **method:** distribution_comparison
- **reference_source:** synthetic_baseline
- **metrics_evaluated:** ['reward_mean', 'kl_mean', 'entropy_mean']
- **kl_results:** {'reward_mean': {'kl_divergence': 0.0, 'score': np.float64(1.0), 'details': {'kl_divergence': 0.0, 'jensen_shannon_divergence': 0.0, 'mean_difference': 0.0, 'std_difference': 0.0, 'data1_size': 50, 'data2_size': 50, 'data1_mean': 0.0, 'data2_mean': 0.0, 'data1_std': 0.0, 'data2_std': 0.0, 'bins': 20, 'metric': 'reward_mean'}}, 'kl_mean': {'kl_divergence': 0.0, 'score': np.float64(1.0), 'details': {'kl_divergence': 0.0, 'jensen_shannon_divergence': 0.0, 'mean_difference': 0.0, 'std_difference': 0.0, 'data1_size': 50, 'data2_size': 50, 'data1_mean': 0.0, 'data2_mean': 0.0, 'data1_std': 0.0, 'data2_std': 0.0, 'bins': 20, 'metric': 'kl_mean'}}, 'entropy_mean': {'kl_divergence': 0.0, 'score': np.float64(1.0), 'details': {'kl_divergence': 0.0, 'jensen_shannon_divergence': 0.0, 'mean_difference': 0.0, 'std_difference': 0.0, 'data1_size': 50, 'data2_size': 50, 'data1_mean': 0.0, 'data2_mean': 0.0, 'data1_std': 0.0, 'data2_std': 0.0, 'bins': 20, 'metric': 'entropy_mean'}}}
- **confidence_intervals:** {'reward_mean': (0.0, 0.0), 'kl_mean': (0.0, 0.0), 'entropy_mean': (0.0, 0.0)}
- **sample_size:** 50
- **reference_size:** 50

## 📁 Files Generated

- **Evaluation Card:** `eval_card.md`
- **Detailed Results:** `eval_results.jsonl`
- **Summary:** `eval_summary.json`
- **Plots:** `tradeoff_plots.png`
