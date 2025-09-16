#!/usr/bin/env python3
"""
Demonstration of RLDK monitoring with gating actions and robust outputs.

This example shows how to use the new monitoring features:
- Stop actions (PID signaling)
- Sentinel file actions
- Shell command actions
- HTTP request actions
- Rolling windows
- Alerts output (JSONL and human-readable)
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any

from rldk.emit import EventWriter
from rldk.monitor import MonitorEngine, load_rules, read_events_once


def create_sample_rules() -> str:
    """Create sample rules file demonstrating all action types."""
    rules_content = """
rules:
  - id: stop_on_high_kl
    where: name == "kl"
    condition: value > 0.35
    window:
      size: 5
      kind: consecutive
    cooldown_steps: 5
    actions:
      - warn:
          msg: "KL {value:.3f} exceeded at step {step} - stopping training"
      - stop:
          kill_timeout_sec: 3

  - id: rolling_reward_drop
    where: name == "reward"
    condition: mean(value) < -0.5
    window:
      size: 10
      kind: rolling
    grace_steps: 20
    actions:
      - warn:
          msg: "Rolling average reward {mean(value):.3f} below threshold at step {step}"
      - sentinel:
          path: "artifacts/reward_drop_sentinel.json"

  - id: shell_notification
    where: name == "grad_norm"
    condition: value > 10.0
    window:
      size: 3
      kind: consecutive
    actions:
      - shell:
          command: "echo 'High gradient norm detected: {value:.2f} at step {step}' | tee -a artifacts/grad_norm_alerts.log"
          timeout_sec: 10

  - id: http_webhook
    where: name == "loss"
    condition: value > 2.0
    window:
      size: 1
      kind: consecutive
    actions:
      - http:
          url: "https://httpbin.org/post"
          method: "POST"
          headers:
            "Content-Type": "application/json"
          payload: '{"alert": "High loss detected", "value": {value}, "step": {step}, "rule": "{rule_id}"}'
          timeout_sec: 30
          retries: 2
"""
    return rules_content


def create_sample_data(output_path: str) -> None:
    """Create sample training data that will trigger various rules."""
    print(f"Creating sample training data at {output_path}")
    
    with EventWriter(output_path) as writer:
        # Simulate training with various metrics
        for step in range(1, 101):
            # Simulate KL divergence that exceeds threshold
            kl_value = 0.2 + (step - 50) * 0.01 if step > 50 else 0.2
            writer.log(step=step, name="kl", value=kl_value, run_id="demo-run-123")
            
            # Simulate reward that drops in rolling window
            reward_value = 0.5 - (step - 30) * 0.02 if step > 30 else 0.5
            writer.log(step=step, name="reward", value=reward_value, run_id="demo-run-123")
            
            # Simulate gradient norm spikes
            grad_norm = 1.0 + (step % 20) * 0.5
            if step % 15 == 0:  # Occasional spikes
                grad_norm += 8.0
            writer.log(step=step, name="grad_norm", value=grad_norm, run_id="demo-run-123")
            
            # Simulate loss spikes
            loss_value = 1.0 + (step % 25) * 0.1
            if step % 20 == 0:  # Occasional spikes
                loss_value += 1.5
            writer.log(step=step, name="loss", value=loss_value, run_id="demo-run-123")
            
            # Add some metadata
            if step % 10 == 0:
                writer.log(
                    step=step, 
                    name="learning_rate", 
                    value=0.001 * (0.95 ** (step // 10)),
                    run_id="demo-run-123",
                    tags={"phase": "training", "model": "gpt2"},
                    meta={"checkpoint": f"step_{step}"}
                )


def run_monitoring_demo() -> None:
    """Run the complete monitoring demonstration."""
    print("RLDK Monitoring Actions Demo")
    print("=" * 40)
    
    # Create artifacts directory
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Create rules file
    rules_path = artifacts_dir / "demo_rules.yaml"
    rules_content = create_sample_rules()
    rules_path.write_text(rules_content)
    print(f"Created rules file: {rules_path}")
    
    # Create sample data
    data_path = artifacts_dir / "demo_data.jsonl"
    create_sample_data(str(data_path))
    print(f"Created sample data: {data_path}")
    
    # Load and display rules
    print("\nLoaded rules:")
    rules = load_rules(rules_path)
    for rule in rules:
        print(f"  - {rule.id}: {rule.condition.expression}")
        print(f"    Window: {rule.window_kind} size {rule.window_size}")
        print(f"    Actions: {len(rule.warn_actions)} warn, {len(rule.stop_actions)} stop, "
              f"{len(rule.sentinel_actions)} sentinel, {len(rule.shell_actions)} shell, "
              f"{len(rule.http_actions)} http")
    
    # Run monitoring
    print(f"\nRunning monitoring on {data_path}...")
    print("This will process the data and execute actions when rules fire.")
    print("Check artifacts/alerts.jsonl and artifacts/alerts.txt for results.\n")
    
    # Create monitor engine
    engine = MonitorEngine(rules)
    
    # Process events
    events = read_events_once(data_path)
    total_alerts = 0
    
    for event in events:
        alerts = engine.process_event(event)
        if alerts:
            total_alerts += len(alerts)
            for alert in alerts:
                print(f"[{alert.rule_id}] {alert.action}: {alert.message}")
    
    print(f"\nTotal alerts generated: {total_alerts}")
    
    # Generate reports
    report = engine.generate_report()
    
    # Save alerts to files
    alerts_jsonl_path = artifacts_dir / "alerts.jsonl"
    alerts_txt_path = artifacts_dir / "alerts.txt"
    
    # Write JSONL alerts
    with alerts_jsonl_path.open("w") as f:
        for alert in engine._alerts:
            f.write(json.dumps(alert.to_dict()) + "\n")
    print(f"Alerts saved to: {alerts_jsonl_path}")
    
    # Write human-readable alerts
    with alerts_txt_path.open("w") as f:
        f.write("RLDK Monitoring Alerts Summary\n")
        f.write("=" * 40 + "\n\n")
        
        for alert in engine._alerts:
            f.write(f"Rule: {alert.rule_id}\n")
            f.write(f"Action: {alert.action}\n")
            f.write(f"Step: {alert.event.step}\n")
            f.write(f"Time: {alert.event.time}\n")
            f.write(f"Metric: {alert.event.name} = {alert.event.value:.4f}\n")
            if alert.message:
                f.write(f"Message: {alert.message}\n")
            f.write("-" * 40 + "\n")
    print(f"Human-readable alerts saved to: {alerts_txt_path}")
    
    # Save detailed report
    report_path = artifacts_dir / "monitoring_report.json"
    with report_path.open("w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"Detailed report saved to: {report_path}")
    
    print("\nDemo completed! Check the artifacts/ directory for all outputs.")


def demonstrate_cli_usage() -> None:
    """Show how to use the CLI with the new features."""
    print("\nCLI Usage Examples:")
    print("=" * 40)
    
    print("1. Basic monitoring with streaming:")
    print("   rldk monitor --stream artifacts/demo_data.jsonl --rules artifacts/demo_rules.yaml")
    
    print("\n2. Monitoring with PID control and custom alerts directory:")
    print("   rldk monitor --stream artifacts/demo_data.jsonl --rules artifacts/demo_rules.yaml \\")
    print("                --pid 12345 --alerts custom_alerts/")
    
    print("\n3. Batch analysis with custom timeouts:")
    print("   rldk monitor --once artifacts/demo_data.jsonl --rules artifacts/demo_rules.yaml \\")
    print("                --http-timeout-sec 60 --retries 5 --report analysis.json")
    
    print("\n4. Monitoring with global cooldown and grace periods:")
    print("   rldk monitor --stream - --rules artifacts/demo_rules.yaml \\")
    print("                --cooldown-steps 10 --grace-steps 5")
    
    print("\n5. Using field mapping for different data formats:")
    print("   rldk monitor --stream data.jsonl --rules rules.yaml \\")
    print("                --field-map '{\"s\":\"step\",\"metric\":\"name\",\"v\":\"value\"}'")


if __name__ == "__main__":
    try:
        run_monitoring_demo()
        demonstrate_cli_usage()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        sys.exit(1)
    except Exception as exc:
        print(f"Demo failed: {exc}")
        sys.exit(1)