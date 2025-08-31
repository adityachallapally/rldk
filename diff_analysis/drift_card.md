# Drift Detection Card

## 🚨 Drift Detected

**First divergence at step:** 45

**Tripped signals:** kl_mean, reward_mean

## 📊 Analysis

The runs have diverged significantly. Here are the most likely causes:

- Most affected signal: kl_mean (1 violations)

## 📈 Divergence Details

| Step | Signal | Z-Score | Run A Value | Run B Value | Consecutive Count |
|------|--------|---------|-------------|-------------|-------------------|
| 103 | kl_mean | -2.033 | 0.197400 | 0.594000 | 3 |
| 45 | reward_mean | 2.097 | 0.990500 | 0.905700 | 3 |
## 📁 Report Location

Full report saved to: `diff_analysis/drift_card.md`
Detailed events saved to: `diff_analysis/diff_events.csv`

## 🔍 Analysis Parameters

- **Total divergence events:** 2
- **First event step:** 45
- **Last event step:** 103
