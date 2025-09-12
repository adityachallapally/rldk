#!/usr/bin/env python3
"""
GRPO Toy Text Example

This script demonstrates GRPO (Group Relative Policy Optimization) training
on synthetic text data with RLDK tracking and anomaly detection.

It shows:
1. Synthetic GRPO training data generation
2. RLDK tracking setup
3. Training loop simulation with GRPO-specific metrics
4. Anomaly detection and analysis
5. GRPO-specific forensics analysis
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import sys

# RLDK imports
from rldk.tracking import ExperimentTracker, TrackingConfig
from rldk.forensics import scan_logs
from rldk.ingest import ingest_runs
from rldk.utils.seed import set_global_seed, DEFAULT_SEED

def main():
    print("🚀 GRPO Toy Text Training Example")
    print("=" * 50)
    
    # Set global seed for reproducibility
    set_global_seed(42)
    print(f"Global seed set to: 42")
    
    # 1. Generate GRPO demo data
    print("\n📊 Generating synthetic GRPO training data...")
    result = subprocess.run([
        "python", "-m", "rldk", "demo", 
        "--out", "./grpo_demo", 
        "--seed", "42", 
        "--steps", "150", 
        "--variants", "1"
    ], capture_output=True, text=True, cwd="/workspace")
    
    if result.returncode == 0:
        print("✅ GRPO demo data generated successfully!")
    else:
        print("❌ Demo generation failed:")
        print(result.stderr)
        return
    
    # 2. Set up RLDK tracking
    print("\n🔧 Setting up RLDK tracking...")
    config = TrackingConfig(
        experiment_name="grpo_toy_text",
        enable_dataset_tracking=True,
        enable_model_tracking=True,
        enable_environment_tracking=True,
        enable_seed_tracking=True,
        enable_git_tracking=True,
        save_to_wandb=False,
        tags=["grpo", "text", "toy", "demo"],
        notes="GRPO toy text training demonstration"
    )
    
    tracker = ExperimentTracker(config)
    tracking_data = tracker.start_experiment()
    print(f"✅ Experiment started: {tracking_data['experiment_id']}")
    
    # 3. Simulate GRPO training loop
    print("\n🚀 Starting simulated GRPO training...")
    
    # GRPO-specific parameters
    model_name = "gpt2-small"
    dataset_name = "toy_text_prompts"
    group_size = 4
    learning_rate = 1e-5
    batch_size = 8
    n_epochs = 1
    clip_range = 0.2
    vf_coef = 0.1
    ent_coef = 0.01
    
    # Add metadata
    tracker.add_metadata("model_name", model_name)
    tracker.add_metadata("dataset_name", dataset_name)
    tracker.add_metadata("group_size", group_size)
    tracker.add_metadata("learning_rate", learning_rate)
    tracker.add_metadata("batch_size", batch_size)
    tracker.add_metadata("n_epochs", n_epochs)
    tracker.add_metadata("clip_range", clip_range)
    tracker.add_metadata("vf_coef", vf_coef)
    tracker.add_metadata("ent_coef", ent_coef)
    
    print(f"📊 GRPO parameters:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Group size: {group_size}")
    print(f"  Learning rate: {learning_rate}")
    
    # Simulate GRPO training with specific anomalies
    log_file = Path("./grpo_training.jsonl")
    
    with open(log_file, 'w') as f:
        for step in range(1, 151):  # 150 training steps
            # Simulate GRPO-specific metrics
            # GRPO typically shows different patterns than PPO
            
            # Base metrics with GRPO characteristics
            reward_mean = 0.6 + 0.0008 * step + 0.05 * np.sin(step / 40)
            reward_std = 0.08 + 0.0001 * step
            kl = 0.04 + 0.0003 * step
            entropy = 1.8 - 0.0008 * step
            loss = 0.8 - 0.0015 * step
            
            # Add realistic noise
            reward_mean += np.random.normal(0, 0.03)
            reward_std += np.random.normal(0, 0.01)
            kl += np.random.normal(0, 0.008)
            entropy += np.random.normal(0, 0.04)
            loss += np.random.normal(0, 0.04)
            
            # GRPO-specific metrics
            policy_grad_norm = 0.9 + np.random.normal(0, 0.08)
            value_grad_norm = 0.7 + np.random.normal(0, 0.08)
            advantage_mean = np.random.normal(0, 0.08)
            advantage_std = 0.9 + np.random.normal(0, 0.08)
            
            # Pass rate is GRPO-specific
            pass_rate = 0.4 + 0.0008 * step + np.random.normal(0, 0.02)
            
            # Add GRPO-specific anomalies for forensics to detect
            if step > 80 and step % 40 == 0:
                # Pass rate spike anomaly
                pass_rate *= 1.5
                print(f"🚨 Pass rate spike at step {step}")
            
            if step > 120 and step % 25 == 0:
                # KL spike anomaly
                kl *= 1.8
                print(f"🚨 KL spike at step {step}")
            
            if step > 160 and step % 35 == 0:
                # Reward drop anomaly
                reward_mean *= 0.7
                print(f"🚨 Reward drop at step {step}")
            
            # Create log entry
            log_entry = {
                "step": step,
                "reward_mean": round(reward_mean, 6),
                "reward_std": round(reward_std, 6),
                "kl": round(kl, 6),
                "entropy": round(entropy, 6),
                "loss": round(loss, 6),
                "policy_grad_norm": round(policy_grad_norm, 6),
                "value_grad_norm": round(value_grad_norm, 6),
                "advantage_mean": round(advantage_mean, 6),
                "advantage_std": round(advantage_std, 6),
                "pass_rate": round(pass_rate, 6),
                "model_name": model_name,
                "group_size": group_size,
                "learning_rate": learning_rate
            }
            
            # Write to JSONL
            f.write(json.dumps(log_entry) + "\n")
            
            # Log to tracker (every 15 steps)
            if step % 15 == 0:
                tracker.log_metric("reward", reward_mean)
                tracker.log_metric("kl_divergence", kl)
                tracker.log_metric("entropy", entropy)
                tracker.log_metric("loss", loss)
                tracker.log_metric("pass_rate", pass_rate)
    
    print(f"✅ GRPO training completed! Logs saved to {log_file}")
    
    # 4. Finish experiment
    summary = tracker.finish_experiment()
    print("✅ Experiment tracking completed!")
    
    # 5. Analyze training data
    print("\n📊 Loading and analyzing GRPO training data...")
    df = ingest_runs(log_file, adapter_hint="demo_jsonl")
    print(f"✅ Loaded {len(df)} training steps")
    
    # Display key metrics
    print(f"\n📋 Key GRPO metrics:")
    print(f"  Final reward: {df['reward_mean'].iloc[-1]:.4f}")
    print(f"  Final pass rate: {df['pass_rate'].iloc[-1]:.4f}")
    print(f"  Final KL: {df['kl_mean'].iloc[-1]:.4f}")
    print(f"  Final entropy: {df['entropy_mean'].iloc[-1]:.4f}")
    
    # 6. Run forensics analysis
    print("\n🔍 Running RLDK forensics analysis...")
    scan_result = scan_logs(log_file.parent)
    
    if scan_result and 'rules_fired' in scan_result:
        rules_fired = scan_result['rules_fired']
        print(f"\n📋 Anomalies detected: {len(rules_fired)}")
        
        if rules_fired:
            print("\n🚨 GRPO-specific anomalies found:")
            for rule in rules_fired:
                print(f"  - {rule.get('rule', 'Unknown')}: {rule.get('description', 'No description')}")
        else:
            print("\n✅ No anomalies detected - GRPO training looks healthy!")
    else:
        print("\n✅ No anomalies detected - GRPO training looks healthy!")
    
    # 7. Create visualizations
    print("\n📊 Creating GRPO training visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('GRPO Toy Text Training Progress', fontsize=16)
    
    # Reward over time
    axes[0, 0].plot(df['step'], df['reward_mean'], 'b-', linewidth=2)
    axes[0, 0].set_title('Reward')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Pass rate over time (GRPO-specific)
    axes[0, 1].plot(df['step'], df['pass_rate'], 'g-', linewidth=2)
    axes[0, 1].set_title('Pass Rate (GRPO-specific)')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Pass Rate')
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL Divergence over time
    axes[1, 0].plot(df['step'], df['kl_mean'], 'r-', linewidth=2)
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Entropy over time
    axes[1, 1].plot(df['step'], df['entropy_mean'], 'm-', linewidth=2)
    axes[1, 1].set_title('Entropy')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./grpo_training_progress.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 8. Training summary
    print("\n📊 GRPO Training Summary:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Total steps: {len(df)}")
    print(f"  Initial reward: {df['reward_mean'].iloc[0]:.4f}")
    print(f"  Final reward: {df['reward_mean'].iloc[-1]:.4f}")
    print(f"  Initial pass rate: {df['pass_rate'].iloc[0]:.4f}")
    print(f"  Final pass rate: {df['pass_rate'].iloc[-1]:.4f}")
    print(f"  Reward improvement: {df['reward_mean'].iloc[-1] - df['reward_mean'].iloc[0]:.4f}")
    print(f"  Pass rate improvement: {df['pass_rate'].iloc[-1] - df['pass_rate'].iloc[0]:.4f}")
    
    # Check for GRPO-specific success criteria
    final_pass_rate = df['pass_rate'].iloc[-1]
    if final_pass_rate > 0.8:
        print("\n🎉 GRPO training successful! High pass rate achieved!")
    elif final_pass_rate > 0.6:
        print("\n✅ GRPO training shows good progress!")
    else:
        print("\n⚠️  GRPO training may need more steps or hyperparameter tuning.")
    
    # 9. Cleanup
    print("\n🧹 Cleaning up...")
    files_created = [
        "./grpo_demo",
        "./grpo_training.jsonl",
        "./grpo_training_progress.png",
        "./rldk_reports"
    ]
    
    print("📁 Files created during this demo:")
    for file_path in files_created:
        path = Path(file_path)
        if path.exists():
            if path.is_file():
                print(f"  📄 {file_path}")
            elif path.is_dir():
                files = list(path.rglob("*"))
                print(f"  📁 {file_path} ({len(files)} files)")
    
    print("\n✅ GRPO Toy Text example completed!")
    print("\n💡 Next steps:")
    print("  - Try the research reproducibility workflow")
    print("  - Explore hyperparameter tuning")
    print("  - Run multi-run analysis")

if __name__ == "__main__":
    main()