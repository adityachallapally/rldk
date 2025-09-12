#!/usr/bin/env python3
"""
Hyperparameter Tuning Example

This script demonstrates hyperparameter tuning with RLDK:
1. Generate multiple runs with different hyperparameters
2. Track experiments with hyperparameter metadata
3. Compare runs to find optimal settings
4. Analyze hyperparameter sensitivity
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import sys
from itertools import product

# RLDK imports
from rldk.tracking import ExperimentTracker, TrackingConfig
from rldk.forensics import scan_logs
from rldk.ingest import ingest_runs
from rldk.diff import first_divergence
from rldk.utils.seed import set_global_seed, DEFAULT_SEED

def main():
    print("🎛️ Hyperparameter Tuning Example")
    print("=" * 50)
    
    # Set global seed for reproducibility
    set_global_seed(42)
    print(f"Global seed set to: 42")
    
    # Define hyperparameter grid
    hyperparams = {
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'batch_size': [32, 64, 128],
        'clip_range': [0.1, 0.2, 0.3]
    }
    
    print(f"\n📊 Hyperparameter grid:")
    for param, values in hyperparams.items():
        print(f"  {param}: {values}")
    
    # Generate all combinations
    param_combinations = list(product(*hyperparams.values()))
    param_names = list(hyperparams.keys())
    
    print(f"\n🔢 Total combinations: {len(param_combinations)}")
    
    # 1. Generate runs for each hyperparameter combination
    print("\n🚀 Generating runs for each hyperparameter combination...")
    
    run_results = []
    for i, combination in enumerate(param_combinations):
        print(f"  Generating run {i+1}/{len(param_combinations)}...")
        
        # Create parameter dict
        params = dict(zip(param_names, combination))
        
        # Generate demo data with different seeds
        seed = 42 + i * 10
        run_dir = f"./hp_tuning_run_{i:02d}"
        
        result = subprocess.run([
            "python", "-m", "rldk", "demo", 
            "--out", run_dir, 
            "--seed", str(seed), 
            "--steps", "80", 
            "--variants", "1"
        ], capture_output=True, text=True, cwd="/workspace")
        
        if result.returncode == 0:
            print(f"    ✅ Run {i+1} generated successfully")
            run_results.append({
                'run_id': i,
                'run_dir': run_dir,
                'params': params,
                'seed': seed
            })
        else:
            print(f"    ❌ Run {i+1} generation failed")
            continue
    
    # 2. Set up experiment tracking for hyperparameter tuning
    print("\n🔧 Setting up hyperparameter tuning experiment tracking...")
    
    config = TrackingConfig(
        experiment_name="hyperparameter_tuning",
        enable_dataset_tracking=True,
        enable_model_tracking=True,
        enable_environment_tracking=True,
        enable_seed_tracking=True,
        enable_git_tracking=True,
        save_to_wandb=False,
        tags=["hyperparameter", "tuning", "optimization", "demo"],
        notes="Hyperparameter tuning demonstration with grid search"
    )
    
    tracker = ExperimentTracker(config)
    tracking_data = tracker.start_experiment()
    print(f"✅ Experiment started: {tracking_data['experiment_id']}")
    
    # Add hyperparameter tuning metadata
    tuning_metadata = {
        "tuning_method": "grid_search",
        "objective": "maximize_final_reward",
        "total_combinations": len(param_combinations),
        "hyperparameters": hyperparams,
        "evaluation_metric": "reward_mean",
        "optimization_goal": "find_best_hyperparameters"
    }
    
    for key, value in tuning_metadata.items():
        tracker.add_metadata(key, value)
    
    # 3. Simulate training with hyperparameter effects
    print("\n🚀 Simulating training with hyperparameter effects...")
    
    training_logs = []
    for run_data in run_results:
        log_file = Path(f"./hp_training_{run_data['run_id']:02d}.jsonl")
        params = run_data['params']
        
        # Simulate training with hyperparameter effects
        np.random.seed(run_data['seed'])
        
        with open(log_file, 'w') as f:
            for step in range(1, 81):
                # Base metrics
                reward_mean = 0.5 + 0.001 * step + 0.1 * np.sin(step / 50)
                reward_std = 0.1 + 0.0001 * step
                kl = 0.05 + 0.0005 * step
                entropy = 2.0 - 0.001 * step
                loss = 1.0 - 0.002 * step
                
                # Apply hyperparameter effects
                lr_effect = params['learning_rate'] / 3e-4  # Normalize to default
                batch_effect = params['batch_size'] / 64   # Normalize to default
                clip_effect = params['clip_range'] / 0.2   # Normalize to default
                
                # Learning rate affects convergence speed
                reward_mean *= (0.8 + 0.4 * lr_effect)
                
                # Batch size affects stability
                reward_std *= (0.5 + 0.5 * batch_effect)
                
                # Clip range affects KL divergence
                kl *= (0.5 + 0.5 * clip_effect)
                
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
                    "learning_rate": params['learning_rate'],
                    "batch_size": params['batch_size'],
                    "clip_range": params['clip_range'],
                    "run_id": run_data['run_id']
                }
                
                f.write(json.dumps(log_entry) + "\n")
        
        training_logs.append(log_file)
        print(f"  ✅ Training log {run_data['run_id']+1} created: {log_file}")
    
    # 4. Analyze hyperparameter performance
    print("\n📊 Analyzing hyperparameter performance...")
    
    # Load all training data
    all_runs_data = []
    for log_file in training_logs:
        df = ingest_runs(log_file, adapter_hint="demo_jsonl")
        all_runs_data.append(df)
    
    # Calculate performance metrics
    performance_results = []
    for i, (df, run_data) in enumerate(zip(all_runs_data, run_results)):
        final_reward = df['reward_mean'].iloc[-1]
        avg_reward = df['reward_mean'].mean()
        reward_std = df['reward_std'].mean()
        final_kl = df['kl_mean'].iloc[-1]
        convergence_step = None
        
        # Find convergence step (when reward stops improving significantly)
        for step in range(20, len(df)):
            if df['reward_mean'].iloc[step] > df['reward_mean'].iloc[step-10] + 0.01:
                convergence_step = step
                break
        
        performance_results.append({
            'run_id': run_data['run_id'],
            'params': run_data['params'],
            'final_reward': final_reward,
            'avg_reward': avg_reward,
            'reward_std': reward_std,
            'final_kl': final_kl,
            'convergence_step': convergence_step or len(df),
            'stability': 1.0 / (reward_std + 1e-6)  # Higher is more stable
        })
    
    # Sort by performance
    performance_results.sort(key=lambda x: x['final_reward'], reverse=True)
    
    print(f"\n🏆 Top 5 performing hyperparameter combinations:")
    for i, result in enumerate(performance_results[:5]):
        params = result['params']
        print(f"  {i+1}. Final reward: {result['final_reward']:.4f}")
        print(f"     Learning rate: {params['learning_rate']}")
        print(f"     Batch size: {params['batch_size']}")
        print(f"     Clip range: {params['clip_range']}")
        print(f"     Stability: {result['stability']:.2f}")
        print()
    
    # 5. Hyperparameter sensitivity analysis
    print("\n📊 Hyperparameter sensitivity analysis...")
    
    # Analyze each hyperparameter's effect
    for param_name in param_names:
        print(f"\n  {param_name} sensitivity:")
        
        # Group by hyperparameter value
        param_values = {}
        for result in performance_results:
            value = result['params'][param_name]
            if value not in param_values:
                param_values[value] = []
            param_values[value].append(result['final_reward'])
        
        # Calculate statistics
        for value, rewards in param_values.items():
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            print(f"    {value}: {mean_reward:.4f} ± {std_reward:.4f} (n={len(rewards)})")
    
    # 6. Create hyperparameter tuning visualizations
    print("\n📊 Creating hyperparameter tuning visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Hyperparameter Tuning Analysis', fontsize=16)
    
    # Plot 1: Performance vs Learning Rate
    lr_performance = {}
    for result in performance_results:
        lr = result['params']['learning_rate']
        if lr not in lr_performance:
            lr_performance[lr] = []
        lr_performance[lr].append(result['final_reward'])
    
    lr_values = sorted(lr_performance.keys())
    lr_means = [np.mean(lr_performance[lr]) for lr in lr_values]
    lr_stds = [np.std(lr_performance[lr]) for lr in lr_values]
    
    axes[0, 0].errorbar(lr_values, lr_means, yerr=lr_stds, marker='o', capsize=5)
    axes[0, 0].set_title('Performance vs Learning Rate')
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Final Reward')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Performance vs Batch Size
    bs_performance = {}
    for result in performance_results:
        bs = result['params']['batch_size']
        if bs not in bs_performance:
            bs_performance[bs] = []
        bs_performance[bs].append(result['final_reward'])
    
    bs_values = sorted(bs_performance.keys())
    bs_means = [np.mean(bs_performance[bs]) for bs in bs_values]
    bs_stds = [np.std(bs_performance[bs]) for bs in bs_values]
    
    axes[0, 1].errorbar(bs_values, bs_means, yerr=bs_stds, marker='o', capsize=5)
    axes[0, 1].set_title('Performance vs Batch Size')
    axes[0, 1].set_xlabel('Batch Size')
    axes[0, 1].set_ylabel('Final Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Performance vs Clip Range
    cr_performance = {}
    for result in performance_results:
        cr = result['params']['clip_range']
        if cr not in cr_performance:
            cr_performance[cr] = []
        cr_performance[cr].append(result['final_reward'])
    
    cr_values = sorted(cr_performance.keys())
    cr_means = [np.mean(cr_performance[cr]) for cr in cr_values]
    cr_stds = [np.std(cr_performance[cr]) for cr in cr_values]
    
    axes[1, 0].errorbar(cr_values, cr_means, yerr=cr_stds, marker='o', capsize=5)
    axes[1, 0].set_title('Performance vs Clip Range')
    axes[1, 0].set_xlabel('Clip Range')
    axes[1, 0].set_ylabel('Final Reward')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Best vs Worst runs comparison
    best_run = performance_results[0]
    worst_run = performance_results[-1]
    
    best_df = all_runs_data[best_run['run_id']]
    worst_df = all_runs_data[worst_run['run_id']]
    
    axes[1, 1].plot(best_df['step'], best_df['reward_mean'], 
                   label=f'Best (LR={best_run["params"]["learning_rate"]})', linewidth=2)
    axes[1, 1].plot(worst_df['step'], worst_df['reward_mean'], 
                   label=f'Worst (LR={worst_run["params"]["learning_rate"]})', linewidth=2)
    axes[1, 1].set_title('Best vs Worst Runs')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./hyperparameter_tuning_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 7. Finish experiment tracking
    summary = tracker.finish_experiment()
    print("✅ Experiment tracking completed!")
    
    # 8. Hyperparameter tuning summary
    print("\n📊 Hyperparameter Tuning Summary:")
    print(f"  Total combinations tested: {len(performance_results)}")
    print(f"  Best final reward: {performance_results[0]['final_reward']:.4f}")
    print(f"  Worst final reward: {performance_results[-1]['final_reward']:.4f}")
    print(f"  Performance range: {performance_results[0]['final_reward'] - performance_results[-1]['final_reward']:.4f}")
    
    best_params = performance_results[0]['params']
    print(f"\n🏆 Optimal hyperparameters:")
    print(f"  Learning rate: {best_params['learning_rate']}")
    print(f"  Batch size: {best_params['batch_size']}")
    print(f"  Clip range: {best_params['clip_range']}")
    
    # 9. Cleanup
    print("\n🧹 Cleaning up...")
    files_created = []
    for run_data in run_results:
        files_created.extend([
            run_data['run_dir'],
            f"./hp_training_{run_data['run_id']:02d}.jsonl"
        ])
    files_created.extend([
        "./hyperparameter_tuning_analysis.png",
        "./rldk_reports"
    ])
    
    print("📁 Files created during this demo:")
    for file_path in files_created:
        path = Path(file_path)
        if path.exists():
            if path.is_file():
                print(f"  📄 {file_path}")
            elif path.is_dir():
                files = list(path.rglob("*"))
                print(f"  📁 {file_path} ({len(files)} files)")
    
    print("\n✅ Hyperparameter tuning example completed!")
    print("\n💡 Key insights:")
    print("  - Grid search helps identify optimal hyperparameters")
    print("  - Performance varies significantly with hyperparameter choices")
    print("  - Stability and final performance are both important metrics")
    print("  - RLDK enables systematic hyperparameter comparison")

if __name__ == "__main__":
    main()