# RLDK Data Summary

## Forensic Analysis Results
- Overall Health Score: 0.597
- Training Stability Score: 0.875
- Convergence Quality Score: 0.956
- Total Steps: 140

### Anomalies Detected
1. **Controller Responsiveness Anomaly** (warning)
   - Message: Low controller responsiveness: 0.100
   - Value: 0.100 (threshold: 0.3)

2. **Controller Overshoot Anomaly** (warning)
   - Message: High controller overshoot: 0.517
   - Value: 0.517 (threshold: 0.3)

3. **Coef Adaptation Anomaly** (warning)
   - Message: Poor coefficient adaptation: 0.000
   - Value: 0.000 (threshold: 0.2)

4. **Advantage Bias Anomaly** (critical)
   - Message: High advantage bias: 0.2371
   - Value: 0.237 (threshold: 0.1)

5. **Advantage Normalization Anomaly** (warning)
   - Message: Poor advantage normalization: 0.490
   - Value: 0.490 (threshold: 0.5)

### Tracker Details
#### KL Schedule Tracker
- Current KL: 0.107
- KL Target: 0.1
- KL Health Score: 0.916
- Schedule Health Score: 0.230
- Time in Target Range: 88.0%
- Target Range Violations: 12

#### Gradient Norms Tracker
- Policy Gradient Norm: 0.691
- Value Gradient Norm: 0.476
- Total Gradient Norm: 0.840
- Policy/Value Ratio: 1.452
- Gradient Health Score: 0.772
- Training Stability: 0.869

#### Advantage Statistics Tracker
- Advantage Mean: 0.249
- Advantage Std: 1.023
- Advantage Bias: 0.237
- Advantage Health Score: 0.470
- Quality Score: 0.956

## Training Metrics Summary
- Total Steps Recorded: 20
- KL Divergence Range: 0.100 - 0.120
- KL Coefficient Range: 0.968 - 1.100

### KL Progression (First 10 Steps)
Step 0: KL=0.100, Coef=1.100
Step 1: KL=0.102, Coef=1.100
Step 2: KL=0.104, Coef=1.098
Step 3: KL=0.106, Coef=1.096
Step 4: KL=0.108, Coef=1.092
Step 5: KL=0.110, Coef=1.088
Step 6: KL=0.111, Coef=1.083
Step 7: KL=0.113, Coef=1.076
Step 8: KL=0.114, Coef=1.070
Step 9: KL=0.116, Coef=1.062

### KL Progression (Last 10 Steps)
Step 10: KL=0.117, Coef=1.054
Step 11: KL=0.118, Coef=1.045
Step 12: KL=0.119, Coef=1.036
Step 13: KL=0.119, Coef=1.027
Step 14: KL=0.120, Coef=1.017
Step 15: KL=0.120, Coef=1.007
Step 16: KL=0.120, Coef=0.997
Step 17: KL=0.120, Coef=0.987
Step 18: KL=0.119, Coef=0.977
Step 19: KL=0.119, Coef=0.968

## Enhanced Scan Results
- Overall Health Score: 0.762
- Training Stability Score: 0.550
- Convergence Quality Score: 1.000
- Total Steps: 50

### Rules Fired
- **kl_controller_stuck**: KL controller stuck: 14 consecutive updates with KL outside [0.01, 0.15] and coef change < 5.0%
  - Step Range: 20-34
- **grad_ratio_high**: Gradient ratio too high: 15 consecutive updates with policy/value ratio > 10.0
  - Step Range: 35-49
