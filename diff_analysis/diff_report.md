# Divergence Analysis Report

## 🚨 Divergence Detected

**First divergence at step:** 29

**Tripped signals:** kl_mean

## 📊 Analysis

The runs have diverged significantly. Here are the most likely causes:

1. **Learning rate changes** - Sudden spikes in learning rate can cause instability
2. **Reward scaling issues** - Inconsistent reward normalization between runs
3. **Random seed differences** - Different initialization or sampling

## 📝 Additional Notes

- Divergence detected using 3-consecutive rule
- Rolling window size: 50
- Signals monitored: kl_mean

## 📈 Events CSV

Detailed divergence events saved to: `diff_analysis/diff_events.csv`
