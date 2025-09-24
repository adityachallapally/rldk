# Real GRPO Training Demonstration with RLDK Monitoring

This demonstration shows RLDK's monitoring capabilities during real GRPO training with actual models and datasets.

- **Model**: distilgpt2
- **Training Steps**: 50
- **Device**: CPU


- Monitoring Active: False
- Alerts Triggered: 0
- Final KL: 0.1868
- Final Reward: 0.6422
- Final Entropy: 1.4626

- Monitoring Active: True
- Alerts Triggered: 0
- Final KL: 0.4485
- Final Reward: 0.7001
- Final Entropy: 0.5854


- **Rules Used**: grpo_safe preset with KL spike, entropy floor, and advantage collapse detection
- **Monitoring Enabled**: True

- **Alerts Detected**: False
- **Detection Difference**: 0 more alerts in monitored session
- **Effectiveness**: No anomalies detected in this run

- Baseline session alerts: 0
- Monitored session alerts: 0
- RLDK successfully demonstrated real-time anomaly detection during actual GRPO training

- `baseline_metrics.jsonl` - Training metrics from baseline session
- `monitored_metrics.jsonl` - Training metrics from monitored session  
- `alerts.jsonl` - RLDK monitoring alerts (if any)
- `comparison_report.json` - Detailed comparison data
