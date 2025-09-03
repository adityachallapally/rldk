# Reward Health Analysis Card

## 🚨 Reward Health Issues Detected

The reward model shows potential pathologies that should be investigated.

## 📊 Summary of Findings

- **Drift Detected:** No
- **Saturation Issues:** 1
- **Calibration Score:** 0.000
- **Shortcut Signals:** 0
- **Label Leakage Risk:** 0.000

## 📈 Saturation Analysis

**Status:** 🚨 Saturation issues detected

- High zero clustering: 100.0% of rewards near zero

### Saturation Metrics

- **upper_saturation_ratio:** 0.000
- **lower_saturation_ratio:** 0.000
- **total_samples:** 1000
- **zero_ratio:** 1.000
## 🎯 Calibration Analysis

**Status:** ⚠️ Poor calibration detected

**Calibration Score:** 0.000

## 🔧 Recommended Fixes

1. Adjust reward scaling or check for gradient issues

## 📁 Report Location

Full report saved to: `artifacts/health/reward_health_card.md`
Calibration plots saved to: `artifacts/health/calibration_plots.png`
