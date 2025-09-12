#!/usr/bin/env python3
"""
Research Reproducibility Workflow Example

This script demonstrates the complete RLDK research workflow:
1. Track experiment with comprehensive metadata
2. Scan logs for anomalies
3. Find first divergence between runs
4. Replay training with exact seeds

This shows how RLDK enables reproducible research in RL.
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import sys
import tempfile

# RLDK imports
from rldk.tracking import ExperimentTracker, TrackingConfig
from rldk.forensics import scan_logs
from rldk.ingest import ingest_runs
from rldk.diff import first_divergence
from rldk.determinism import check
from rldk.replay import replay
from rldk.utils.seed import set_global_seed, DEFAULT_SEED

def main():
    print("🔬 Research Reproducibility Workflow Example")
    print("=" * 60)
    
    # Set global seed for reproducibility
    set_global_seed(42)
    print(f"Global seed set to: 42")
    
    # 1. Generate multiple demo runs for comparison
    print("\n📊 Generating multiple training runs for comparison...")
    
    runs_data = []
    for run_id in range(3):
        print(f"  Generating run {run_id + 1}/3...")
        
        # Generate demo data with different seeds
        seed = 42 + run_id * 100
        result = subprocess.run([
            "python", "-m", "rldk", "demo", 
            "--out", f"./research_run_{run_id:02d}", 
            "--seed", str(seed), 
            "--steps", "100", 
            "--variants", "1"
        ], capture_output=True, text=True, cwd="/workspace")
        
        if result.returncode == 0:
            print(f"    ✅ Run {run_id + 1} generated successfully")
            runs_data.append(f"./research_run_{run_id:02d}")
        else:
            print(f"    ❌ Run {run_id + 1} generation failed")
            return
    
    # 2. Set up comprehensive experiment tracking
    print("\n🔧 Setting up comprehensive experiment tracking...")
    
    config = TrackingConfig(
        experiment_name="research_repro_workflow",
        enable_dataset_tracking=True,
        enable_model_tracking=True,
        enable_environment_tracking=True,
        enable_seed_tracking=True,
        enable_git_tracking=True,
        save_to_wandb=False,
        tags=["research", "reproducibility", "workflow", "demo"],
        notes="Complete research reproducibility workflow demonstration"
    )
    
    tracker = ExperimentTracker(config)
    tracking_data = tracker.start_experiment()
    print(f"✅ Experiment started: {tracking_data['experiment_id']}")
    
    # Add research-specific metadata
    research_metadata = {
        "research_question": "How does seed variation affect PPO training stability?",
        "hypothesis": "Different seeds should produce similar but not identical training curves",
        "methodology": "Compare 3 runs with different seeds, analyze divergence patterns",
        "expected_outcome": "Runs should diverge gradually, not catastrophically",
        "evaluation_metrics": ["reward_mean", "kl", "entropy", "loss"],
        "success_criteria": "All runs converge to similar final performance"
    }
    
    for key, value in research_metadata.items():
        tracker.add_metadata(key, value)
    
    print("📋 Research metadata added to tracking")
    
    # 3. Simulate training runs with controlled variations
    print("\n🚀 Simulating training runs with controlled variations...")
    
    training_logs = []
    for run_id, run_path in enumerate(runs_data):
        log_file = Path(f"./research_training_{run_id:02d}.jsonl")
        
        # Simulate training with slight variations
        base_seed = 42 + run_id * 100
        np.random.seed(base_seed)
        
        with open(log_file, 'w') as f:
            for step in range(1, 101):
                # Base metrics
                reward_mean = 0.5 + 0.001 * step + 0.1 * np.sin(step / 50)
                reward_std = 0.1 + 0.0001 * step
                kl = 0.05 + 0.0005 * step
                entropy = 2.0 - 0.001 * step
                loss = 1.0 - 0.002 * step
                
                # Add run-specific variations
                variation_factor = 1.0 + (run_id - 1) * 0.05  # Slight variation between runs
                reward_mean *= variation_factor
                
                # Add realistic noise
                reward_mean += np.random.normal(0, 0.05)
                reward_std += np.random.normal(0, 0.01)
                kl += np.random.normal(0, 0.01)
                entropy += np.random.normal(0, 0.05)
                loss += np.random.normal(0, 0.05)
                
                # Simulate gradient norms
                policy_grad_norm = 1.0 + np.random.normal(0, 0.1)
                value_grad_norm = 0.8 + np.random.normal(0, 0.1)
                advantage_mean = np.random.normal(0, 0.1)
                advantage_std = 1.0 + np.random.normal(0, 0.1)
                
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
                    "run_id": f"research_run_{run_id:02d}",
                    "seed": base_seed
                }
                
                f.write(json.dumps(log_entry) + "\n")
        
        training_logs.append(log_file)
        print(f"  ✅ Training log {run_id + 1} created: {log_file}")
    
    # 4. Scan logs for anomalies
    print("\n🔍 Step 1: Scanning logs for anomalies...")
    
    scan_results = []
    for i, log_file in enumerate(training_logs):
        print(f"  Scanning run {i + 1}...")
        scan_result = scan_logs(log_file.parent)
        scan_results.append(scan_result)
        
        if scan_result and 'rules_fired' in scan_result:
            rules_fired = scan_result['rules_fired']
            print(f"    📋 Anomalies detected: {len(rules_fired)}")
            if rules_fired:
                for rule in rules_fired[:2]:  # Show first 2 anomalies
                    print(f"      - {rule.get('rule', 'Unknown')}")
        else:
            print(f"    ✅ No anomalies detected")
    
    # 5. Find first divergence between runs
    print("\n🔍 Step 2: Finding first divergence between runs...")
    
    # Load data for comparison
    df_a = ingest_runs(training_logs[0], adapter_hint="demo_jsonl")
    df_b = ingest_runs(training_logs[1], adapter_hint="demo_jsonl")
    
    print(f"  Comparing run 1 ({len(df_a)} steps) vs run 2 ({len(df_b)} steps)")
    
    # Find divergence
    signals = ["reward_mean", "kl", "entropy", "loss"]
    divergence_report = first_divergence(
        df_a=df_a,
        df_b=df_b,
        signals=signals,
        k_consecutive=3,
        window=20,
        tolerance=2.0
    )
    
    print(f"  📊 Divergence analysis:")
    print(f"    Diverged: {divergence_report.diverged}")
    if divergence_report.diverged:
        print(f"    First divergence at step: {divergence_report.first_step}")
        print(f"    Tripped signals: {[s['signal'] for s in divergence_report.tripped_signals]}")
        print(f"    Notes: {divergence_report.notes}")
    else:
        print(f"    ✅ No significant divergence detected")
    
    # 6. Check determinism
    print("\n🔍 Step 3: Checking determinism...")
    
    # Create a simple deterministic script for testing
    det_script = Path("./deterministic_test.py")
    with open(det_script, 'w') as f:
        f.write("""
import json
import random
random.seed(42)
for i in range(10):
    print(json.dumps({"step": i, "loss": 1.0 - i*0.1}))
""")
    
    # Check determinism
    det_report = check(
        cmd=f"python {det_script}",
        compare=["loss"],
        replicas=3
    )
    
    print(f"  📊 Determinism check:")
    print(f"    Passed: {det_report.passed}")
    if not det_report.passed:
        print(f"    Issues: {len(det_report.mismatches)} mismatches")
        print(f"    Culprit: {det_report.culprit}")
    else:
        print(f"    ✅ Training is deterministic")
    
    # 7. Replay training with exact seed
    print("\n🔍 Step 4: Replaying training with exact seed...")
    
    # Create a simple replay script
    replay_script = Path("./replay_test.py")
    with open(replay_script, 'w') as f:
        f.write("""
import json
import random
import sys
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
random.seed(seed)
for i in range(10):
    print(json.dumps({"step": i, "loss": 1.0 - i*0.1}))
""")
    
    # Replay with original seed
    original_seed = 42
    replay_report = replay(
        run_path="./research_run_00",  # Use first run as reference
        training_command=f"python {replay_script} {{seed}}",
        metrics_to_compare=["loss"],
        tolerance=0.01,
        max_steps=10
    )
    
    print(f"  📊 Replay analysis:")
    print(f"    Passed: {replay_report.passed}")
    print(f"    Original seed: {replay_report.original_seed}")
    print(f"    Replay seed: {replay_report.replay_seed}")
    if not replay_report.passed:
        print(f"    Mismatches: {len(replay_report.mismatches)}")
    else:
        print(f"    ✅ Replay successful")
    
    # 8. Create comprehensive analysis visualization
    print("\n📊 Creating comprehensive analysis visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Research Reproducibility Analysis', fontsize=16)
    
    # Load all runs for comparison
    all_runs = []
    for log_file in training_logs:
        df = ingest_runs(log_file, adapter_hint="demo_jsonl")
        all_runs.append(df)
    
    # Plot 1: Reward comparison across runs
    for i, df in enumerate(all_runs):
        axes[0, 0].plot(df['step'], df['reward_mean'], 
                       label=f'Run {i+1}', linewidth=2, alpha=0.8)
    axes[0, 0].set_title('Reward Comparison Across Runs')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: KL divergence comparison
    for i, df in enumerate(all_runs):
        axes[0, 1].plot(df['step'], df['kl_mean'], 
                       label=f'Run {i+1}', linewidth=2, alpha=0.8)
    axes[0, 1].set_title('KL Divergence Comparison')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('KL Divergence')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Entropy comparison
    for i, df in enumerate(all_runs):
        axes[1, 0].plot(df['step'], df['entropy_mean'], 
                       label=f'Run {i+1}', linewidth=2, alpha=0.8)
    axes[1, 0].set_title('Entropy Comparison')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Loss comparison
    for i, df in enumerate(all_runs):
        axes[1, 1].plot(df['step'], df['loss'], 
                       label=f'Run {i+1}', linewidth=2, alpha=0.8)
    axes[1, 1].set_title('Loss Comparison')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./research_repro_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 9. Finish experiment tracking
    summary = tracker.finish_experiment()
    print("✅ Experiment tracking completed!")
    
    # 10. Research summary
    print("\n📊 Research Reproducibility Summary:")
    print(f"  Research question: {research_metadata['research_question']}")
    print(f"  Number of runs: {len(all_runs)}")
    print(f"  Divergence detected: {divergence_report.diverged}")
    print(f"  Determinism check: {'✅ Passed' if det_report.passed else '❌ Failed'}")
    print(f"  Replay test: {'✅ Passed' if replay_report.passed else '❌ Failed'}")
    
    # Calculate final performance statistics
    final_rewards = [df['reward_mean'].iloc[-1] for df in all_runs]
    reward_std = np.std(final_rewards)
    reward_mean = np.mean(final_rewards)
    
    print(f"  Final reward mean: {reward_mean:.4f}")
    print(f"  Final reward std: {reward_std:.4f}")
    print(f"  Coefficient of variation: {reward_std/reward_mean:.4f}")
    
    # Research conclusions
    print(f"\n🔬 Research Conclusions:")
    if reward_std/reward_mean < 0.1:
        print("  ✅ Runs show good reproducibility (low variance)")
    elif reward_std/reward_mean < 0.2:
        print("  ⚠️  Runs show moderate reproducibility (moderate variance)")
    else:
        print("  ❌ Runs show poor reproducibility (high variance)")
    
    if divergence_report.diverged:
        print(f"  📊 Runs diverge at step {divergence_report.first_step}")
        print(f"  🔍 This suggests the training is sensitive to seed variations")
    else:
        print(f"  ✅ Runs remain consistent throughout training")
    
    # 11. Cleanup
    print("\n🧹 Cleaning up...")
    files_created = [
        "./research_run_00", "./research_run_01", "./research_run_02",
        "./research_training_00.jsonl", "./research_training_01.jsonl", "./research_training_02.jsonl",
        "./deterministic_test.py", "./replay_test.py",
        "./research_repro_analysis.png", "./rldk_reports"
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
    
    print("\n✅ Research reproducibility workflow completed!")
    print("\n💡 Key takeaways:")
    print("  - RLDK enables comprehensive experiment tracking")
    print("  - Anomaly detection helps identify training issues")
    print("  - Divergence analysis reveals when runs start to differ")
    print("  - Determinism checking ensures reproducible results")
    print("  - Replay functionality enables exact reproduction")

if __name__ == "__main__":
    main()