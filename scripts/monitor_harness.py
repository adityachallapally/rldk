#!/usr/bin/env python3
"""Unit harness to test real-time alerts from RLDK monitors."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

# Import the monitor class from RLDK
from rldk.integrations.trl.monitors import PPOMonitor as Monitor

def simulate_trl_callback_contract():
    """Simulate the TRL callback contract with fake logs."""
    print("🧪 Starting RLDK Monitor Harness Test")
    print("=" * 50)
    
    # Create artifacts directory
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Initialize the monitor
    monitor = Monitor(
        output_dir=str(artifacts_dir),
        kl_threshold=0.1,  # Low threshold to trigger alerts
        reward_threshold=0.05,
        gradient_threshold=1.0,
        clip_frac_threshold=0.2,
        run_id="harness_test"
    )
    
    print(f"✅ Monitor initialized with thresholds: KL={monitor.kl_threshold}")
    
    # Create fake logs dict and state
    alerts_log = []
    first_alert_step = None
    first_alert_wall_clock_sec = None
    
    # Simulate 200 iterations with KL spike around iteration 120
    for i in range(200):
        # Start with stable values
        if i < 120:
            kl = 0.05  # Below threshold
            reward_mean = 0.3
            reward_std = 0.1
            gradient_norm = 0.5
            clip_frac = 0.1
        else:
            # Increase KL sharply over a few iterations
            kl = 0.05 + (i - 120) * 0.02  # Gradually increase KL
            reward_mean = 0.3
            reward_std = 0.1
            gradient_norm = 0.5
            clip_frac = 0.1
        
        # Create fake logs dict
        logs = {
            "global_step": i,
            "ppo/policy/kl_mean": kl,
            "ppo/rewards/mean": reward_mean,
            "ppo/rewards/std": reward_std,
            "ppo/policy/entropy": 2.0 - i * 0.01,
            "ppo/policy/clipfrac": clip_frac,
            "ppo/val/value_loss": 0.3 - i * 0.001,
            "grad_norm": gradient_norm,
            "learning_rate": 1e-5,
        }
        
        # Create fake state
        state = SimpleNamespace(global_step=i, epoch=i/10.0)
        
        # Capture stdout to detect alerts
        import sys
        from io import StringIO
        
        # Redirect stdout to capture alerts
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            # Call monitor.on_log
            monitor.on_log(args=None, state=state, control=None, logs=logs)
            
            # Get captured output
            output = captured_output.getvalue()
            
            # Check if any alert was printed
            if "🚨 PPO Alert:" in output:
                alert_time = time.time()
                if first_alert_step is None:
                    first_alert_step = i
                    first_alert_wall_clock_sec = alert_time
                
                # Record the alert
                alert_record = {
                    "step": i,
                    "wall_clock_sec": alert_time,
                    "message": output.strip(),
                    "kl_value": kl
                }
                alerts_log.append(alert_record)
                
                # Print the alert to console
                print(f"🚨 ALERT at step {i}: {output.strip()}")
        
        finally:
            # Restore stdout
            sys.stdout = old_stdout
        
        # Print progress every 20 iterations
        if i % 20 == 0:
            print(f"   Iteration {i}: KL={kl:.4f}")
    
    # Save alerts to JSONL file
    alerts_file = artifacts_dir / "harness_alerts.jsonl"
    with open(alerts_file, "w") as f:
        for alert in alerts_log:
            json.dump(alert, f)
            f.write("\n")
    
    # Create summary
    summary = {
        "first_alert_step": first_alert_step,
        "first_alert_wall_clock_sec": first_alert_wall_clock_sec,
        "total_iterations": 200,
        "total_alerts": len(alerts_log),
        "kl_threshold": monitor.kl_threshold
    }
    
    summary_file = artifacts_dir / "harness_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print results
    print(f"\n📊 Harness Test Results:")
    print(f"   Total iterations: {summary['total_iterations']}")
    print(f"   Total alerts: {summary['total_alerts']}")
    if first_alert_step is not None:
        print(f"   First alert at step: {first_alert_step}")
        print(f"   First alert wall clock: {first_alert_wall_clock_sec}")
        print(f"   KL threshold: {monitor.kl_threshold}")
    else:
        print("   No alerts detected")
    
    print(f"   Alerts saved to: {alerts_file}")
    print(f"   Summary saved to: {summary_file}")
    
    return summary

if __name__ == "__main__":
    try:
        summary = simulate_trl_callback_contract()
        print("\n✅ Harness test completed successfully")
        exit(0)
    except Exception as e:
        print(f"\n❌ Harness test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)