#!/usr/bin/env python3
"""
Multi-Run Analysis Example

This script demonstrates comprehensive multi-run analysis with RLDK:
1. Generate multiple runs with different configurations
2. Compare runs to identify regressions and improvements
3. Analyze run-to-run variability and stability
4. Detect anomalies across multiple runs
5. Generate comprehensive reports
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# RLDK imports
from rldk.tracking import ExperimentTracker, TrackingConfig
from rldk.forensics import scan_logs
from rldk.ingest import ingest_runs
from rldk.diff import first_divergence
from rldk.utils.seed import set_global_seed, DEFAULT_SEED

def main():
    print("📊 Multi-Run Analysis Example")
    print("=" * 50)
    
    # Set global seed for reproducibility
    set_global_seed(42)
    print(f"Global seed set to: 42")
    
    # 1. Generate multiple runs with different configurations
    print("\n🚀 Generating multiple runs with different configurations...")
    
    run_configs = [
        {"name": "baseline", "seed": 42, "steps": 100, "variant": "stable"},
        {"name": "high_lr", "seed": 43, "steps": 100, "variant": "unstable"},
        {"name": "low_lr", "seed": 44, "steps": 100, "variant": "slow"},
        {"name": "noisy", "seed": 45, "steps": 100, "variant": "noisy"},
        {"name": "converged", "seed": 46, "steps": 100, "variant": "fast"},
        {"name": "diverged", "seed": 47, "steps": 100, "variant": "unstable"},
    ]
    
    run_results = []
    for i, config in enumerate(run_configs):
        print(f"  Generating run {i+1}/{len(run_configs)}: {config['name']}...")
        
        run_dir = f"./multi_run_{config['name']}"
        
        result = subprocess.run([
            "python", "-m", "rldk", "demo", 
            "--out", run_dir, 
            "--seed", str(config['seed']), 
            "--steps", str(config['steps']), 
            "--variants", "1"
        ], capture_output=True, text=True, cwd="/workspace")
        
        if result.returncode == 0:
            print(f"    ✅ Run '{config['name']}' generated successfully")
            run_results.append({
                'name': config['name'],
                'run_dir': run_dir,
                'seed': config['seed'],
                'variant': config['variant']
            })
        else:
            print(f"    ❌ Run '{config['name']}' generation failed")
            continue
    
    # 2. Set up multi-run experiment tracking
    print("\n🔧 Setting up multi-run experiment tracking...")
    
    config = TrackingConfig(
        experiment_name="multi_run_analysis",
        enable_dataset_tracking=True,
        enable_model_tracking=True,
        enable_environment_tracking=True,
        enable_seed_tracking=True,
        enable_git_tracking=True,
        save_to_wandb=False,
        tags=["multi-run", "analysis", "comparison", "regression", "demo"],
        notes="Multi-run analysis demonstration with different configurations"
    )
    
    tracker = ExperimentTracker(config)
    tracking_data = tracker.start_experiment()
    print(f"✅ Multi-run experiment started: {tracking_data['experiment_id']}")
    
    # Add multi-run metadata
    multi_run_metadata = {
        "analysis_type": "multi_run_comparison",
        "total_runs": len(run_results),
        "run_configurations": run_configs,
        "comparison_metrics": ["reward_mean", "kl_mean", "entropy", "loss"],
        "analysis_goals": ["regression_detection", "stability_analysis", "anomaly_detection"]
    }
    
    for key, value in multi_run_metadata.items():
        tracker.add_metadata(key, value)
    
    # 3. Load and analyze all runs
    print("\n📊 Loading and analyzing all runs...")
    
    all_runs_data = {}
    run_summaries = {}
    
    for run_data in run_results:
        print(f"  Loading run: {run_data['name']}...")
        
        # Find the JSONL file in the run directory
        run_path = Path(run_data['run_dir'])
        jsonl_files = list(run_path.glob("*.jsonl"))
        
        if jsonl_files:
            jsonl_file = jsonl_files[0]  # Take the first JSONL file
            df = ingest_runs(jsonl_file, adapter_hint="demo_jsonl")
            all_runs_data[run_data['name']] = df
            
            # Calculate run summary
            summary = {
                'name': run_data['name'],
                'variant': run_data['variant'],
                'seed': run_data['seed'],
                'total_steps': len(df),
                'final_reward': df['reward_mean'].iloc[-1],
                'avg_reward': df['reward_mean'].mean(),
                'reward_std': df['reward_std'].mean(),
                'final_kl': df['kl_mean'].iloc[-1],
                'avg_kl': df['kl_mean'].mean(),
                'final_entropy': df['entropy'].iloc[-1],
                'avg_entropy': df['entropy'].mean(),
                'final_loss': df['loss'].iloc[-1],
                'avg_loss': df['loss'].mean(),
                'convergence_step': None,
                'stability_score': 0.0
            }
            
            # Find convergence step
            for step in range(10, len(df)):
                if df['reward_mean'].iloc[step] > df['reward_mean'].iloc[step-5] + 0.01:
                    summary['convergence_step'] = step
                    break
            
            # Calculate stability score (lower variance = higher stability)
            reward_variance = df['reward_mean'].var()
            summary['stability_score'] = 1.0 / (reward_variance + 1e-6)
            
            run_summaries[run_data['name']] = summary
            print(f"    ✅ Loaded {len(df)} steps")
        else:
            print(f"    ❌ No JSONL files found in {run_data['run_dir']}")
    
    # 4. Run-to-run comparison analysis
    print("\n📊 Run-to-run comparison analysis...")
    
    # Create comparison matrix
    comparison_results = []
    run_names = list(all_runs_data.keys())
    
    for i, run1_name in enumerate(run_names):
        for j, run2_name in enumerate(run_names):
            if i < j:  # Only compare each pair once
                print(f"  Comparing {run1_name} vs {run2_name}...")
                
                df1 = all_runs_data[run1_name]
                df2 = all_runs_data[run2_name]
                
                # Find first divergence
                divergence_result = first_divergence(df1, df2, signals=['reward_mean', 'kl_mean'])
                
                comparison_results.append({
                    'run1': run1_name,
                    'run2': run2_name,
                    'divergence_step': divergence_result.get('divergence_step'),
                    'divergence_signal': divergence_result.get('divergence_signal'),
                    'final_reward_diff': df1['reward_mean'].iloc[-1] - df2['reward_mean'].iloc[-1],
                    'final_kl_diff': df1['kl_mean'].iloc[-1] - df2['kl_mean'].iloc[-1],
                    'avg_reward_diff': df1['reward_mean'].mean() - df2['reward_mean'].mean(),
                    'stability_diff': run_summaries[run1_name]['stability_score'] - run_summaries[run2_name]['stability_score']
                })
    
    # 5. Regression detection
    print("\n🔍 Regression detection...")
    
    # Sort runs by performance
    sorted_runs = sorted(run_summaries.values(), key=lambda x: x['final_reward'], reverse=True)
    
    print(f"\n🏆 Performance ranking:")
    for i, run in enumerate(sorted_runs):
        print(f"  {i+1}. {run['name']}: {run['final_reward']:.4f} (stability: {run['stability_score']:.2f})")
    
    # Detect regressions (runs that perform significantly worse)
    baseline_performance = sorted_runs[0]['final_reward']
    regression_threshold = 0.1  # 10% performance drop
    
    regressions = []
    for run in sorted_runs[1:]:
        performance_drop = (baseline_performance - run['final_reward']) / baseline_performance
        if performance_drop > regression_threshold:
            regressions.append({
                'run_name': run['name'],
                'performance_drop': performance_drop,
                'final_reward': run['final_reward'],
                'stability_score': run['stability_score']
            })
    
    if regressions:
        print(f"\n⚠️  Detected {len(regressions)} regressions:")
        for reg in regressions:
            print(f"  - {reg['run_name']}: {reg['performance_drop']:.1%} performance drop")
    else:
        print(f"\n✅ No significant regressions detected")
    
    # 6. Stability analysis
    print("\n📊 Stability analysis...")
    
    # Calculate stability metrics
    stability_metrics = []
    for run_name, df in all_runs_data.items():
        reward_variance = df['reward_mean'].var()
        reward_trend = np.polyfit(df['step'], df['reward_mean'], 1)[0]  # Slope
        reward_volatility = df['reward_mean'].rolling(window=5).std().mean()
        
        stability_metrics.append({
            'run_name': run_name,
            'reward_variance': reward_variance,
            'reward_trend': reward_trend,
            'reward_volatility': reward_volatility,
            'stability_score': run_summaries[run_name]['stability_score']
        })
    
    # Sort by stability
    stability_metrics.sort(key=lambda x: x['stability_score'], reverse=True)
    
    print(f"\n📈 Stability ranking:")
    for i, metric in enumerate(stability_metrics):
        print(f"  {i+1}. {metric['run_name']}: stability={metric['stability_score']:.2f}, "
              f"variance={metric['reward_variance']:.4f}, trend={metric['reward_trend']:.6f}")
    
    # 7. Anomaly detection across runs
    print("\n🔍 Anomaly detection across runs...")
    
    # Detect anomalous runs
    anomalous_runs = []
    
    for run_name, df in all_runs_data.items():
        anomalies = []
        
        # Check for reward anomalies
        reward_mean = df['reward_mean'].mean()
        reward_std = df['reward_mean'].std()
        reward_outliers = df[abs(df['reward_mean'] - reward_mean) > 2 * reward_std]
        
        if len(reward_outliers) > 0:
            anomalies.append(f"reward_outliers: {len(reward_outliers)} steps")
        
        # Check for KL anomalies
        kl_mean = df['kl_mean'].mean()
        kl_std = df['kl_mean'].std()
        kl_outliers = df[abs(df['kl_mean'] - kl_mean) > 2 * kl_std]
        
        if len(kl_outliers) > 0:
            anomalies.append(f"kl_outliers: {len(kl_outliers)} steps")
        
        # Check for entropy collapse
        entropy_min = df['entropy'].min()
        if entropy_min < 0.1:
            anomalies.append(f"entropy_collapse: min={entropy_min:.3f}")
        
        # Check for loss explosion
        loss_max = df['loss'].max()
        if loss_max > 10.0:
            anomalies.append(f"loss_explosion: max={loss_max:.3f}")
        
        if anomalies:
            anomalous_runs.append({
                'run_name': run_name,
                'anomalies': anomalies
            })
    
    if anomalous_runs:
        print(f"\n⚠️  Detected {len(anomalous_runs)} anomalous runs:")
        for run in anomalous_runs:
            print(f"  - {run['run_name']}: {', '.join(run['anomalies'])}")
    else:
        print(f"\n✅ No significant anomalies detected")
    
    # 8. Create comprehensive visualizations
    print("\n📊 Creating comprehensive visualizations...")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Multi-Run Analysis Dashboard', fontsize=16)
    
    # Plot 1: All runs reward curves
    for run_name, df in all_runs_data.items():
        axes[0, 0].plot(df['step'], df['reward_mean'], label=run_name, alpha=0.7)
    axes[0, 0].set_title('Reward Curves - All Runs')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Performance comparison
    run_names = list(run_summaries.keys())
    final_rewards = [run_summaries[name]['final_reward'] for name in run_names]
    stability_scores = [run_summaries[name]['stability_score'] for name in run_names]
    
    scatter = axes[0, 1].scatter(final_rewards, stability_scores, s=100, alpha=0.7)
    for i, name in enumerate(run_names):
        axes[0, 1].annotate(name, (final_rewards[i], stability_scores[i]), 
                           xytext=(5, 5), textcoords='offset points')
    axes[0, 1].set_title('Performance vs Stability')
    axes[0, 1].set_xlabel('Final Reward')
    axes[0, 1].set_ylabel('Stability Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: KL divergence comparison
    for run_name, df in all_runs_data.items():
        axes[1, 0].plot(df['step'], df['kl_mean'], label=run_name, alpha=0.7)
    axes[1, 0].set_title('KL Divergence - All Runs')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Entropy comparison
    for run_name, df in all_runs_data.items():
        axes[1, 1].plot(df['step'], df['entropy'], label=run_name, alpha=0.7)
    axes[1, 1].set_title('Entropy - All Runs')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Run comparison heatmap
    comparison_matrix = np.zeros((len(run_names), len(run_names)))
    for comp in comparison_results:
        i = run_names.index(comp['run1'])
        j = run_names.index(comp['run2'])
        comparison_matrix[i, j] = comp['final_reward_diff']
        comparison_matrix[j, i] = -comp['final_reward_diff']
    
    im = axes[2, 0].imshow(comparison_matrix, cmap='RdBu_r', aspect='auto')
    axes[2, 0].set_title('Run Comparison Matrix (Reward Difference)')
    axes[2, 0].set_xlabel('Run 2')
    axes[2, 0].set_ylabel('Run 1')
    axes[2, 0].set_xticks(range(len(run_names)))
    axes[2, 0].set_yticks(range(len(run_names)))
    axes[2, 0].set_xticklabels(run_names, rotation=45)
    axes[2, 0].set_yticklabels(run_names)
    plt.colorbar(im, ax=axes[2, 0])
    
    # Plot 6: Stability vs Performance scatter
    axes[2, 1].scatter(final_rewards, stability_scores, s=100, alpha=0.7)
    for i, name in enumerate(run_names):
        axes[2, 1].annotate(name, (final_rewards[i], stability_scores[i]), 
                           xytext=(5, 5), textcoords='offset points')
    axes[2, 1].set_title('Stability vs Performance')
    axes[2, 1].set_xlabel('Final Reward')
    axes[2, 1].set_ylabel('Stability Score')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./multi_run_analysis_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 9. Generate comprehensive report
    print("\n📊 Generating comprehensive multi-run report...")
    
    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'total_runs': len(run_results),
        'analysis_summary': {
            'best_performing_run': sorted_runs[0]['name'],
            'most_stable_run': stability_metrics[0]['run_name'],
            'total_regressions': len(regressions),
            'total_anomalies': len(anomalous_runs),
            'performance_range': sorted_runs[0]['final_reward'] - sorted_runs[-1]['final_reward']
        },
        'run_summaries': run_summaries,
        'comparison_results': comparison_results,
        'regressions': regressions,
        'anomalous_runs': anomalous_runs,
        'stability_metrics': stability_metrics
    }
    
    # Save report
    with open('./multi_run_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✅ Multi-run analysis report saved: multi_run_analysis_report.json")
    
    # 10. Finish experiment tracking
    summary = tracker.finish_experiment()
    print("✅ Multi-run experiment tracking completed!")
    
    # 11. Multi-run analysis summary
    print("\n📊 Multi-Run Analysis Summary:")
    print(f"  Total runs analyzed: {len(run_results)}")
    print(f"  Best performing run: {sorted_runs[0]['name']} ({sorted_runs[0]['final_reward']:.4f})")
    print(f"  Most stable run: {stability_metrics[0]['run_name']} ({stability_metrics[0]['stability_score']:.2f})")
    print(f"  Regressions detected: {len(regressions)}")
    print(f"  Anomalous runs: {len(anomalous_runs)}")
    print(f"  Performance range: {report['analysis_summary']['performance_range']:.4f}")
    
    # 12. Cleanup
    print("\n🧹 Cleaning up...")
    files_created = []
    for run_data in run_results:
        files_created.append(run_data['run_dir'])
    files_created.extend([
        "./multi_run_analysis_dashboard.png",
        "./multi_run_analysis_report.json",
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
    
    print("\n✅ Multi-run analysis example completed!")
    print("\n💡 Key insights:")
    print("  - Multi-run analysis helps identify regressions and improvements")
    print("  - Stability and performance are both important metrics")
    print("  - Anomaly detection can catch problematic runs early")
    print("  - RLDK enables comprehensive run comparison and analysis")

if __name__ == "__main__":
    main()