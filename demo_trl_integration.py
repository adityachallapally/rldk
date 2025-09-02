#!/usr/bin/env python3
"""
RLDK TRL Integration Demo

This script demonstrates the RLDK TRL integration in action,
showing real-time monitoring, alerting, and comprehensive metrics collection.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerState, TrainerControl
from datasets import Dataset

# Import RLDK components
from rldk.integrations.trl import RLDKCallback, PPOMonitor, CheckpointMonitor, RLDKDashboard

try:
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    TRL_AVAILABLE = True
except ImportError:
    print("❌ TRL not available. Install with: pip install trl")
    TRL_AVAILABLE = False
    sys.exit(1)


def create_demo_dataset():
    """Create a demo dataset for PPO training."""
    prompts = [
        "The weather today is",
        "Python programming is",
        "Machine learning helps",
        "Artificial intelligence can",
        "Data science involves",
    ] * 4  # 20 samples total
    
    responses = [
        "sunny and warm.",
        "versatile and powerful.",
        "solve complex problems.",
        "transform industries.",
        "analyzing large datasets.",
    ] * 4
    
    return Dataset.from_dict({
        "prompt": prompts,
        "response": responses,
    })


def run_demo():
    """Run the RLDK TRL integration demo."""
    print("🎯 RLDK TRL Integration Demo")
    print("=" * 50)
    
    # Create output directory
    output_dir = "./demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Initializing RLDK monitoring components...")
    
    # Initialize RLDK components
    rldk_callback = RLDKCallback(
        output_dir=output_dir,
        log_interval=2,
        run_id="demo_run",
        alert_thresholds={
            'kl_divergence': 0.1,
            'clip_fraction': 0.2,
            'gradient_norm': 1.0,
            'reward_std': 0.3,  # More lenient for demo
        }
    )
    
    ppo_monitor = PPOMonitor(
        output_dir=output_dir,
        kl_threshold=0.1,
        reward_threshold=0.3,  # More lenient for demo
        gradient_threshold=1.0,
        run_id="demo_run"
    )
    
    checkpoint_monitor = CheckpointMonitor(
        output_dir=output_dir,
        enable_parameter_analysis=True,
        run_id="demo_run"
    )
    
    print("✅ RLDK components initialized")
    
    # Load a small model for demo
    print("📦 Loading GPT-2 model...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    print("✅ Model loaded")
    
    # Create demo dataset
    dataset = create_demo_dataset()
    print(f"📊 Dataset created with {len(dataset)} samples")
    
    # PPO configuration
    ppo_config = PPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_ppo_epochs=1,
        max_grad_norm=0.5,
        num_train_epochs=1,
        do_train=True,
        save_steps=1000,
        eval_steps=1000,
        bf16=False,
        fp16=False,
    )
    
    # Note: PPOTrainer requires additional parameters (reward_model, value_model, etc.)
    # For this demo, we'll focus on testing the RLDK callbacks directly
    # In real usage, you would create the full PPOTrainer with all required parameters
    
    print("✅ RLDK callbacks ready for PPOTrainer integration")
    print("   Note: PPOTrainer requires reward_model, value_model, and other parameters")
    print("   This demo focuses on testing the RLDK monitoring components")
    
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
        f"{output_dir}/demo_run_metrics.csv",
        f"{output_dir}/demo_run_ppo_metrics.csv", 
        f"{output_dir}/demo_run_checkpoint_summary.csv",
        f"{output_dir}/demo_run_final_report.json",
    ]
    
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
    
    # Show sample metrics
    if os.path.exists(f"{output_dir}/demo_run_metrics.csv"):
        print(f"\n📈 Sample Metrics (first 3 lines):")
        with open(f"{output_dir}/demo_run_metrics.csv", 'r') as f:
            for i, line in enumerate(f):
                if i < 3:
                    print(f"   {line.strip()}")
                else:
                    break
    
    # Show alerts
    if len(rldk_callback.alerts) > 0:
        print(f"\n⚠️  Alerts Generated: {len(rldk_callback.alerts)}")
        for alert in rldk_callback.alerts[:3]:  # Show first 3 alerts
            print(f"   - {alert['message']}")
    else:
        print(f"\n✅ No alerts generated (training was stable)")
    
    if len(ppo_monitor.ppo_alerts) > 0:
        print(f"\n🚨 PPO Alerts Generated: {len(ppo_monitor.ppo_alerts)}")
        for alert in ppo_monitor.ppo_alerts[:3]:  # Show first 3 alerts
            print(f"   - {alert['message']}")
    
    print(f"\n🎯 Demo completed! Check the '{output_dir}' directory for detailed results.")
    print("\n💡 To use RLDK in your own training:")
    print("   from rldk.integrations.trl import RLDKCallback, PPOMonitor")
    print("   monitor = RLDKCallback(output_dir='./logs')")
    print("   trainer = PPOTrainer(..., callbacks=[monitor])")


if __name__ == "__main__":
    if not TRL_AVAILABLE:
        print("❌ TRL not available. Please install with: pip install trl")
        sys.exit(1)
    
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()