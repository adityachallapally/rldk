# RLDK Blog Post Visualizations

This directory contains visualizations for the RLDK technical blog post.

## Generated Files

The visualization script creates the following files:

1. **data_summary.md** - Comprehensive data summary with all metrics and anomalies
2. **data_visualization.md** - Simple ASCII charts and visualizations

## Data Sources

All visualizations are based on real RLDK monitoring data from:

- `/workspace/comprehensive_ppo_forensics_demo/comprehensive_analysis.json`
- `/workspace/comprehensive_ppo_monitor_demo/comprehensive_demo_run_comprehensive_metrics.json`
- `/workspace/enhanced_ppo_scan_demo/enhanced_scan_results.json`

## Key Statistics

- **Overall Health Score**: 0.597
- **Training Stability Score**: 0.875
- **Convergence Quality Score**: 0.956
- **Total Steps**: 140
- **Anomalies Detected**: 5

## Anomalies Detected

1. **Controller Responsiveness Anomaly** (Warning)
   - Value: 0.100 (threshold: 0.3)
   - Low controller responsiveness

2. **Controller Overshoot Anomaly** (Warning)
   - Value: 0.517 (threshold: 0.3)
   - High controller overshoot

3. **Coefficient Adaptation Anomaly** (Warning)
   - Value: 0.000 (threshold: 0.2)
   - Poor coefficient adaptation

4. **Advantage Bias Anomaly** (Critical)
   - Value: 0.237 (threshold: 0.1)
   - High advantage bias

5. **Advantage Normalization Anomaly** (Warning)
   - Value: 0.490 (threshold: 0.5)
   - Poor advantage normalization

## Reproducing Visualizations

To regenerate the visualizations:

```bash
python3 /workspace/blog_assets/create_visualizations_simple.py
```

This will create updated data summaries and visualizations based on the current data files.