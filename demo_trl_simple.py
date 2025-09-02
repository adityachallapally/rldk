#!/usr/bin/env python3
"""
RLDK TRL Integration Simple Demo

This script demonstrates the RLDK TRL integration working components,
focusing on the monitoring and callback functionality that we've verified works.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transformers import TrainingArguments, TrainerState, TrainerControl

# Import RLDK components
from rldk.integrations.trl import RLDKCallback, PPOMonitor, CheckpointMonitor, RLDKDashboard


def run_simple_demo():
    """Run a simple demo of RLDK TRL integration."""
    print("🎯 RLDK TRL Integration Simple Demo")
    print("=" * 50)
    
    # Create output directory
    output_dir = "./demo_simple_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Initializing RLDK monitoring components...")
    
    # Initialize RLDK components
    rldk_callback = RLDKCallback(
        output_dir=output_dir,
        log_interval=2,
        run_id="simple_demo",
        alert_thresholds={
            'kl_divergence': 0.1,
            'clip_fraction': 0.2,
            'gradient_norm': 1.0,
            'reward_std': 0.3,
        }
    )
    
    ppo_monitor = PPOMonitor(
        output_dir=output_dir,
        kl_threshold=0.1,
        reward_threshold=0.3,
        gradient_threshold=1.0,
        run_id="simple_demo"
    )
    
    checkpoint_monitor = CheckpointMonitor(
        output_dir=output_dir,
        enable_parameter_analysis=True,
        run_id="simple_demo"
    )
    
    print("✅ RLDK components initialized")
    
    # Test dashboard functionality
    print("\n📊 Testing Dashboard Functionality...")
    dashboard = RLDKDashboard(
        output_dir=output_dir,
        port=8507,
        run_id="simple_demo"
    )
    
    app_file = dashboard.output_dir / "simple_demo_app.py"
    dashboard._create_dashboard_app(app_file)
    
    if app_file.exists():
        print("✅ Dashboard app created successfully")
    else:
        print("❌ Dashboard app creation failed")
    
    # Simulate training with realistic metrics
    print("\n🎯 Starting training simulation...")
    print("=" * 50)
    
    args = TrainingArguments(output_dir=output_dir)
    state = TrainerState()
    control = TrainerControl()
    
    # Start training
    rldk_callback.on_train_begin(args, state, control)
    ppo_monitor.on_train_begin(args, state, control)
    checkpoint_monitor.on_train_begin(args, state, control)
    
    # Simulate 8 training steps with realistic progression
    for step in range(8):
        state.global_step = step + 1
        state.epoch = step / 4.0
        
        # Simulate realistic PPO training metrics
        logs = {
            'ppo/rewards/mean': 0.2 + step * 0.08,      # Improving rewards
            'ppo/rewards/std': 0.15 + step * 0.02,      # Slightly increasing variance
            'ppo/policy/kl_mean': 0.12 - step * 0.01,   # Decreasing KL divergence
            'ppo/policy/entropy': 2.2 - step * 0.15,    # Decreasing entropy
            'ppo/policy/clipfrac': 0.18 - step * 0.02,  # Decreasing clip fraction
            'ppo/val/value_loss': 0.5 - step * 0.05,    # Decreasing value loss
            'learning_rate': 1e-5 * (0.95 ** (step // 2)),  # Learning rate decay
            'grad_norm': 0.9 - step * 0.08,             # Decreasing gradient norm
        }
        
        # Call callbacks
        rldk_callback.on_step_end(args, state, control)
        rldk_callback.on_log(args, state, control, logs=logs)
        
        ppo_monitor.on_step_end(args, state, control)
        ppo_monitor.on_log(args, state, control, logs=logs)
        
        checkpoint_monitor.on_step_end(args, state, control)
        checkpoint_monitor.on_log(args, state, control, logs=logs)
        
        # Simulate checkpoint saves
        if step % 3 == 0 and step > 0:
            import torch.nn as nn
            dummy_model = nn.Linear(10, 1)
            rldk_callback.on_save(args, state, control, model=dummy_model)
            checkpoint_monitor.on_save(args, state, control, model=dummy_model)
            print(f"    💾 Checkpoint saved at step {step + 1}")
        
        # Show progress
        print(f"Step {step + 1:2d}: Reward={logs['ppo/rewards/mean']:.3f}, "
              f"KL={logs['ppo/policy/kl_mean']:.3f}, "
              f"Entropy={logs['ppo/policy/entropy']:.2f}, "
              f"ClipFrac={logs['ppo/policy/clipfrac']:.3f}")
        
        time.sleep(0.1)  # Small delay for demo effect
    
    # End training
    rldk_callback.on_train_end(args, state, control)
    ppo_monitor.on_train_end(args, state, control)
    checkpoint_monitor.on_train_end(args, state, control)
    
    print("\n🎉 Training simulation completed!")
    print("=" * 50)
    
    # Show results
    print("📊 Generated Files:")
    expected_files = [
        f"{output_dir}/simple_demo_metrics.csv",
        f"{output_dir}/simple_demo_ppo_metrics.csv", 
        f"{output_dir}/simple_demo_checkpoint_summary.csv",
        f"{output_dir}/simple_demo_final_report.json",
        f"{output_dir}/simple_demo_app.py",
    ]
    
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
    
    # Show sample metrics
    if os.path.exists(f"{output_dir}/simple_demo_metrics.csv"):
        print(f"\n📈 Sample Metrics (first 3 lines):")
        with open(f"{output_dir}/simple_demo_metrics.csv", 'r') as f:
            for i, line in enumerate(f):
                if i < 3:
                    print(f"   {line.strip()}")
                else:
                    break
    
    # Show alerts
    if len(rldk_callback.alerts) > 0:
        print(f"\n⚠️  RLDK Alerts Generated: {len(rldk_callback.alerts)}")
        for alert in rldk_callback.alerts[:3]:  # Show first 3 alerts
            print(f"   - {alert['message']}")
    else:
        print(f"\n✅ No RLDK alerts generated (training was stable)")
    
    if len(ppo_monitor.ppo_alerts) > 0:
        print(f"\n🚨 PPO Alerts Generated: {len(ppo_monitor.ppo_alerts)}")
        for alert in ppo_monitor.ppo_alerts[:3]:  # Show first 3 alerts
            print(f"   - {alert['message']}")
    
    # Show final reports
    if os.path.exists(f"{output_dir}/simple_demo_final_report.json"):
        print(f"\n📋 Final Training Report:")
        import json
        with open(f"{output_dir}/simple_demo_final_report.json", 'r') as f:
            report = json.load(f)
            print(f"   - Total Steps: {report.get('total_steps', 'N/A')}")
            print(f"   - Final Reward: {report.get('final_reward', 'N/A')}")
            print(f"   - Training Stability: {report.get('training_stability', 'N/A')}")
            print(f"   - Total Alerts: {report.get('total_alerts', 'N/A')}")
    
    print(f"\n🎯 Demo completed! Check the '{output_dir}' directory for detailed results.")
    print("\n💡 Key Features Demonstrated:")
    print("   ✅ Real-time metrics collection")
    print("   ✅ Proactive alerting system")
    print("   ✅ PPO-specific monitoring")
    print("   ✅ Checkpoint analysis")
    print("   ✅ Dashboard generation")
    print("   ✅ Comprehensive reporting")
    
    print("\n🔧 To use RLDK in your own TRL training:")
    print("   from rldk.integrations.trl import RLDKCallback, PPOMonitor")
    print("   monitor = RLDKCallback(output_dir='./logs')")
    print("   ppo_monitor = PPOMonitor(output_dir='./logs')")
    print("   # Add to your PPOTrainer callbacks list")


if __name__ == "__main__":
    try:
        run_simple_demo()
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()