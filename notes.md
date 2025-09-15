# RLDK Monitor Discovery Notes

## Monitor Classes Found

### TRL Integration Monitors (src/rldk/integrations/trl/monitors.py)
1. **PPOMonitor** (line 86)
   - Base class: TrainerCallback
   - Purpose: Specialized monitor for PPO training with advanced analytics
   - Key thresholds: kl_threshold, reward_threshold, gradient_threshold, clip_frac_threshold

2. **CheckpointMonitor** (line 427)
   - Base class: TrainerCallback
   - Purpose: Monitor checkpointing behavior

3. **ComprehensivePPOMonitor** (line 594)
   - Base class: TrainerCallback
   - Purpose: Comprehensive PPO monitor with advanced forensics capabilities
   - Key features: KL schedule tracking, gradient norms analysis, advantage statistics

### TRL Integration Callbacks (src/rldk/integrations/trl/callbacks.py)
4. **RLDKCallback** (line 133)
   - Base class: TrainerCallback
   - Purpose: Base RLDK callback for TRL integration

5. **RLDKMonitor** (line 746)
   - Base class: RLDKCallback
   - Purpose: RLDK monitoring callback

### OpenRLHF Integration (src/rldk/integrations/openrlhf/callbacks.py)
6. **OpenRLHFCallback** (line 149)
   - Purpose: Base callback for OpenRLHF

7. **OpenRLHFMonitor** (line 709)
   - Base class: OpenRLHFCallback

8. **DistributedTrainingMonitor** (line 722)
   - Base class: OpenRLHFCallback

9. **MultiGPUMonitor** (line 860)
   - Base class: OpenRLHFCallback

## Files with Monitor/Callback References
- examples/trl_integration/advanced_monitoring.py
- examples/trl_integration/custom_callbacks.py
- comprehensive_ppo_monitor_demo/ (contains demo results)

## Key Import Paths
- `from rldk.integrations.trl.monitors import PPOMonitor`
- `from rldk.integrations.trl.monitors import ComprehensivePPOMonitor`
- `from rldk.integrations.trl.callbacks import RLDKCallback`