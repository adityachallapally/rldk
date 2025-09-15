# RLDK Monitor Discovery Notes

## Monitors Found

### TRL Integration Monitors (src/rldk/integrations/trl/monitors.py)

1. **PPOMonitor** (line 86)
   - Inherits from TrainerCallback
   - Monitors PPO-specific metrics: KL divergence, rewards, policy entropy, clip fraction
   - Has alert system with thresholds for KL, reward variance, gradient norm, clip fraction
   - Prints alerts to stdout with 🚨 emoji
   - Saves analysis to CSV and JSON files

2. **CheckpointMonitor** (line 427)
   - Inherits from TrainerCallback
   - Monitors model parameters, gradients, and health indicators
   - Analyzes checkpoints when saved
   - Calculates health scores and parameter drift

3. **ComprehensivePPOMonitor** (line 594)
   - Inherits from TrainerCallback
   - Most advanced monitor with comprehensive forensics
   - Uses ComprehensivePPOForensics for advanced analytics
   - Monitors KL schedule, gradient norms, advantage statistics
   - Has anomaly detection and health scoring

### TRL Integration Callbacks (src/rldk/integrations/trl/callbacks.py)

1. **RLDKCallback** (line 133)
   - Main callback class for TRL integration
   - Inherits from TrainerCallback

2. **RLDKMonitor** (line 746)
   - Inherits from RLDKCallback
   - Enhanced monitoring capabilities

## Key Features for Live Monitoring

- All monitors implement `on_log()` method which is called during training
- PPOMonitor and ComprehensivePPOMonitor have real-time alert systems
- Alerts are printed to stdout immediately when thresholds are exceeded
- Monitors track metrics like KL divergence, rewards, gradients in real-time
- ComprehensivePPOMonitor has anomaly detection that triggers during training

## File Paths
- Main monitors: `src/rldk/integrations/trl/monitors.py`
- Main callbacks: `src/rldk/integrations/trl/callbacks.py`
- Examples: `examples/trl_integration/`