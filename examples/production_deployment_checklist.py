#!/usr/bin/env python3
"""
Production Deployment Checklist Example

This script demonstrates pre-deployment validation using RLDK:
1. Generate production-like demo data
2. Run comprehensive validation checks
3. Verify reproducibility and determinism
4. Check for common deployment issues
5. Generate deployment readiness report
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import sys
import hashlib
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# RLDK imports
from rldk.tracking import ExperimentTracker, TrackingConfig
from rldk.forensics import scan_logs
from rldk.ingest import ingest_runs
from rldk.diff import first_divergence
from rldk.determinism import check as det_check
from rldk.utils.seed import set_global_seed, DEFAULT_SEED

def main():
    print("🚀 Production Deployment Checklist Example")
    print("=" * 50)
    
    # Set global seed for reproducibility
    set_global_seed(42)
    print(f"Global seed set to: 42")
    
    # 1. Generate production-like demo data
    print("\n🏭 Generating production-like demo data...")
    
    production_configs = [
        {"name": "production_baseline", "seed": 42, "steps": 200, "variant": "stable"},
        {"name": "production_candidate", "seed": 43, "steps": 200, "variant": "stable"},
        {"name": "production_edge_case", "seed": 44, "steps": 200, "variant": "unstable"},
    ]
    
    production_runs = []
    for i, config in enumerate(production_configs):
        print(f"  Generating production run {i+1}/{len(production_configs)}: {config['name']}...")
        
        run_dir = f"./production_{config['name']}"
        
        result = subprocess.run([
            "python", "-m", "rldk", "demo", 
            "--out", run_dir, 
            "--seed", str(config['seed']), 
            "--steps", str(config['steps']), 
            "--variants", "1"
        ], capture_output=True, text=True, cwd="/workspace")
        
        if result.returncode == 0:
            print(f"    ✅ Production run '{config['name']}' generated successfully")
            production_runs.append({
                'name': config['name'],
                'run_dir': run_dir,
                'seed': config['seed'],
                'variant': config['variant']
            })
        else:
            print(f"    ❌ Production run '{config['name']}' generation failed")
            continue
    
    # 2. Set up production deployment tracking
    print("\n🔧 Setting up production deployment tracking...")
    
    config = TrackingConfig(
        experiment_name="production_deployment",
        enable_dataset_tracking=True,
        enable_model_tracking=True,
        enable_environment_tracking=True,
        enable_seed_tracking=True,
        enable_git_tracking=True,
        save_to_wandb=False,
        tags=["production", "deployment", "validation", "checklist", "demo"],
        notes="Production deployment validation checklist demonstration"
    )
    
    tracker = ExperimentTracker(config)
    tracking_data = tracker.start_experiment()
    print(f"✅ Production deployment experiment started: {tracking_data['experiment_id']}")
    
    # Add production deployment metadata
    deployment_metadata = {
        "deployment_type": "production",
        "validation_phase": "pre_deployment",
        "total_runs": len(production_runs),
        "deployment_checks": [
            "reproducibility",
            "determinism",
            "performance_regression",
            "anomaly_detection",
            "stability_validation",
            "edge_case_handling"
        ],
        "deployment_environment": "production",
        "rollback_threshold": 0.05  # 5% performance drop
    }
    
    for key, value in deployment_metadata.items():
        tracker.add_metadata(key, value)
    
    # 3. Load production runs data
    print("\n📊 Loading production runs data...")
    
    production_data = {}
    for run_data in production_runs:
        print(f"  Loading production run: {run_data['name']}...")
        
        # Find the JSONL file in the run directory
        run_path = Path(run_data['run_dir'])
        jsonl_files = list(run_path.glob("*.jsonl"))
        
        if jsonl_files:
            jsonl_file = jsonl_files[0]  # Take the first JSONL file
            df = ingest_runs(jsonl_file, adapter_hint="demo_jsonl")
            production_data[run_data['name']] = df
            print(f"    ✅ Loaded {len(df)} steps")
        else:
            print(f"    ❌ No JSONL files found in {run_data['run_dir']}")
    
    # 4. Reproducibility validation
    print("\n🔄 Reproducibility validation...")
    
    reproducibility_results = []
    
    # Test reproducibility by running the same seed multiple times
    for run_data in production_runs:
        print(f"  Testing reproducibility for {run_data['name']}...")
        
        # Generate two runs with the same seed
        run1_dir = f"./repro_test_{run_data['name']}_1"
        run2_dir = f"./repro_test_{run_data['name']}_2"
        
        # Generate first run
        result1 = subprocess.run([
            "python", "-m", "rldk", "demo", 
            "--out", run1_dir, 
            "--seed", str(run_data['seed']), 
            "--steps", "50", 
            "--variants", "1"
        ], capture_output=True, text=True, cwd="/workspace")
        
        # Generate second run
        result2 = subprocess.run([
            "python", "-m", "rldk", "demo", 
            "--out", run2_dir, 
            "--seed", str(run_data['seed']), 
            "--steps", "50", 
            "--variants", "1"
        ], capture_output=True, text=True, cwd="/workspace")
        
        if result1.returncode == 0 and result2.returncode == 0:
            # Load both runs
            run1_path = Path(run1_dir)
            run2_path = Path(run2_dir)
            
            run1_jsonl = list(run1_path.glob("*.jsonl"))[0]
            run2_jsonl = list(run2_path.glob("*.jsonl"))[0]
            
            df1 = ingest_runs(run1_jsonl, adapter_hint="demo_jsonl")
            df2 = ingest_runs(run2_jsonl, adapter_hint="demo_jsonl")
            
            # Check if runs are identical
            is_reproducible = df1.equals(df2)
            
            reproducibility_results.append({
                'run_name': run_data['name'],
                'is_reproducible': is_reproducible,
                'run1_final_reward': df1['reward_mean'].iloc[-1],
                'run2_final_reward': df2['reward_mean'].iloc[-1],
                'reward_difference': abs(df1['reward_mean'].iloc[-1] - df2['reward_mean'].iloc[-1])
            })
            
            print(f"    {'✅' if is_reproducible else '❌'} Reproducibility: {is_reproducible}")
        else:
            print(f"    ❌ Failed to generate reproducibility test runs")
    
    # 5. Determinism validation
    print("\n🎯 Determinism validation...")
    
    determinism_results = []
    
    for run_data in production_runs:
        print(f"  Testing determinism for {run_data['name']}...")
        
        # Create a simple deterministic script
        det_script = f"./determinism_test_{run_data['name']}.py"
        with open(det_script, 'w') as f:
            f.write(f"""
import json
import random
random.seed({run_data['seed']})

for i in range(50):
    print(json.dumps({{
        "step": i,
        "loss": 1.0 - i*0.01,
        "reward": 0.5 + i*0.001
    }}))
""")
        
        # Test determinism using RLDK
        try:
            det_result = det_check(
                cmd=f"python {det_script}",
                compare=['loss', 'reward'],
                replicas=3
            )
            
            determinism_results.append({
                'run_name': run_data['name'],
                'is_deterministic': det_result.get('is_deterministic', False),
                'determinism_score': det_result.get('determinism_score', 0.0),
                'max_difference': det_result.get('max_difference', float('inf'))
            })
            
            print(f"    {'✅' if det_result.get('is_deterministic', False) else '❌'} Determinism: {det_result.get('is_deterministic', False)}")
        except Exception as e:
            print(f"    ❌ Determinism test failed: {e}")
            determinism_results.append({
                'run_name': run_data['name'],
                'is_deterministic': False,
                'determinism_score': 0.0,
                'max_difference': float('inf')
            })
    
    # 6. Performance regression validation
    print("\n📈 Performance regression validation...")
    
    # Compare production runs
    baseline_run = production_data.get('production_baseline')
    candidate_run = production_data.get('production_candidate')
    
    regression_results = {}
    
    if baseline_run is not None and candidate_run is not None:
        print("  Comparing baseline vs candidate performance...")
        
        # Calculate performance metrics
        baseline_final_reward = baseline_run['reward_mean'].iloc[-1]
        candidate_final_reward = candidate_run['reward_mean'].iloc[-1]
        
        baseline_avg_reward = baseline_run['reward_mean'].mean()
        candidate_avg_reward = candidate_run['reward_mean'].mean()
        
        baseline_stability = 1.0 / (baseline_run['reward_mean'].var() + 1e-6)
        candidate_stability = 1.0 / (candidate_run['reward_mean'].var() + 1e-6)
        
        # Calculate regression metrics
        final_reward_regression = (baseline_final_reward - candidate_final_reward) / baseline_final_reward
        avg_reward_regression = (baseline_avg_reward - candidate_avg_reward) / baseline_avg_reward
        stability_regression = (baseline_stability - candidate_stability) / baseline_stability
        
        regression_results = {
            'baseline_final_reward': baseline_final_reward,
            'candidate_final_reward': candidate_final_reward,
            'final_reward_regression': final_reward_regression,
            'avg_reward_regression': avg_reward_regression,
            'stability_regression': stability_regression,
            'has_regression': abs(final_reward_regression) > 0.05,  # 5% threshold
            'regression_severity': 'high' if abs(final_reward_regression) > 0.1 else 'medium' if abs(final_reward_regression) > 0.05 else 'low'
        }
        
        print(f"    Final reward regression: {final_reward_regression:.2%}")
        print(f"    Average reward regression: {avg_reward_regression:.2%}")
        print(f"    Stability regression: {stability_regression:.2%}")
        print(f"    {'❌' if regression_results['has_regression'] else '✅'} Regression detected: {regression_results['has_regression']}")
    
    # 7. Anomaly detection validation
    print("\n🔍 Anomaly detection validation...")
    
    anomaly_results = {}
    
    for run_name, df in production_data.items():
        print(f"  Detecting anomalies in {run_name}...")
        
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
        
        # Check for gradient explosion
        policy_grad_max = df['policy_grad_norm'].max()
        if policy_grad_max > 100.0:
            anomalies.append(f"policy_grad_explosion: max={policy_grad_max:.3f}")
        
        anomaly_results[run_name] = {
            'anomalies': anomalies,
            'anomaly_count': len(anomalies),
            'has_anomalies': len(anomalies) > 0,
            'anomaly_severity': 'high' if len(anomalies) > 3 else 'medium' if len(anomalies) > 1 else 'low'
        }
        
        print(f"    {'❌' if len(anomalies) > 0 else '✅'} Anomalies: {len(anomalies)}")
        if anomalies:
            for anomaly in anomalies:
                print(f"      - {anomaly}")
    
    # 8. Stability validation
    print("\n📊 Stability validation...")
    
    stability_results = {}
    
    for run_name, df in production_data.items():
        print(f"  Validating stability for {run_name}...")
        
        # Calculate stability metrics
        reward_variance = df['reward_mean'].var()
        reward_trend = np.polyfit(df['step'], df['reward_mean'], 1)[0]  # Slope
        reward_volatility = df['reward_mean'].rolling(window=10).std().mean()
        
        # Check for stability issues
        stability_issues = []
        
        if reward_variance > 0.1:
            stability_issues.append(f"high_variance: {reward_variance:.4f}")
        
        if abs(reward_trend) > 0.01:
            stability_issues.append(f"strong_trend: {reward_trend:.6f}")
        
        if reward_volatility > 0.05:
            stability_issues.append(f"high_volatility: {reward_volatility:.4f}")
        
        stability_results[run_name] = {
            'reward_variance': reward_variance,
            'reward_trend': reward_trend,
            'reward_volatility': reward_volatility,
            'stability_issues': stability_issues,
            'stability_score': 1.0 / (reward_variance + 1e-6),
            'is_stable': len(stability_issues) == 0,
            'stability_grade': 'A' if len(stability_issues) == 0 else 'B' if len(stability_issues) <= 1 else 'C'
        }
        
        print(f"    Stability grade: {stability_results[run_name]['stability_grade']}")
        print(f"    {'✅' if stability_results[run_name]['is_stable'] else '❌'} Stable: {stability_results[run_name]['is_stable']}")
    
    # 9. Edge case handling validation
    print("\n🧪 Edge case handling validation...")
    
    edge_case_run = production_data.get('production_edge_case')
    edge_case_results = {}
    
    if edge_case_run is not None:
        print("  Validating edge case handling...")
        
        # Check for edge case behaviors
        edge_case_issues = []
        
        # Check for extreme values
        if edge_case_run['reward_mean'].min() < -10:
            edge_case_issues.append("extreme_negative_reward")
        
        if edge_case_run['reward_mean'].max() > 100:
            edge_case_issues.append("extreme_positive_reward")
        
        # Check for NaN values
        if edge_case_run['reward_mean'].isna().any():
            edge_case_issues.append("nan_values")
        
        # Check for infinite values
        if np.isinf(edge_case_run['reward_mean']).any():
            edge_case_issues.append("infinite_values")
        
        # Check for convergence issues
        final_reward = edge_case_run['reward_mean'].iloc[-1]
        if final_reward < 0:
            edge_case_issues.append("negative_convergence")
        
        edge_case_results = {
            'edge_case_issues': edge_case_issues,
            'handles_edge_cases': len(edge_case_issues) == 0,
            'edge_case_grade': 'A' if len(edge_case_issues) == 0 else 'B' if len(edge_case_issues) <= 1 else 'C'
        }
        
        print(f"    Edge case grade: {edge_case_results['edge_case_grade']}")
        print(f"    {'✅' if edge_case_results['handles_edge_cases'] else '❌'} Handles edge cases: {edge_case_results['handles_edge_cases']}")
    
    # 10. Generate deployment readiness report
    print("\n📊 Generating deployment readiness report...")
    
    # Calculate overall deployment readiness
    reproducibility_pass = all(r['is_reproducible'] for r in reproducibility_results)
    determinism_pass = all(r['is_deterministic'] for r in determinism_results)
    regression_pass = not regression_results.get('has_regression', False)
    anomaly_pass = all(not r['has_anomalies'] for r in anomaly_results.values())
    stability_pass = all(r['is_stable'] for r in stability_results.values())
    edge_case_pass = edge_case_results.get('handles_edge_cases', True)
    
    deployment_readiness = {
        'reproducibility': reproducibility_pass,
        'determinism': determinism_pass,
        'performance_regression': regression_pass,
        'anomaly_detection': anomaly_pass,
        'stability': stability_pass,
        'edge_case_handling': edge_case_pass
    }
    
    overall_readiness = all(deployment_readiness.values())
    
    # Generate comprehensive report
    deployment_report = {
        'deployment_timestamp': datetime.now().isoformat(),
        'overall_readiness': overall_readiness,
        'deployment_checks': deployment_readiness,
        'reproducibility_results': reproducibility_results,
        'determinism_results': determinism_results,
        'regression_results': regression_results,
        'anomaly_results': anomaly_results,
        'stability_results': stability_results,
        'edge_case_results': edge_case_results,
        'deployment_recommendation': 'APPROVED' if overall_readiness else 'REJECTED',
        'deployment_risks': [],
        'deployment_notes': []
    }
    
    # Add deployment risks
    if not reproducibility_pass:
        deployment_report['deployment_risks'].append("Reproducibility issues detected")
    if not determinism_pass:
        deployment_report['deployment_risks'].append("Determinism issues detected")
    if not regression_pass:
        deployment_report['deployment_risks'].append("Performance regression detected")
    if not anomaly_pass:
        deployment_report['deployment_risks'].append("Anomalies detected")
    if not stability_pass:
        deployment_report['deployment_risks'].append("Stability issues detected")
    if not edge_case_pass:
        deployment_report['deployment_risks'].append("Edge case handling issues detected")
    
    # Save deployment report
    with open('./production_deployment_report.json', 'w') as f:
        json.dump(deployment_report, f, indent=2)
    
    print("✅ Production deployment report saved: production_deployment_report.json")
    
    # 11. Create deployment validation dashboard
    print("\n📊 Creating deployment validation dashboard...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Production Deployment Validation Dashboard', fontsize=16)
    
    # Plot 1: Reproducibility results
    repro_names = [r['run_name'] for r in reproducibility_results]
    repro_scores = [1 if r['is_reproducible'] else 0 for r in reproducibility_results]
    
    axes[0, 0].bar(repro_names, repro_scores, color=['green' if s == 1 else 'red' for s in repro_scores])
    axes[0, 0].set_title('Reproducibility Validation')
    axes[0, 0].set_ylabel('Pass (1) / Fail (0)')
    axes[0, 0].set_ylim(0, 1.2)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Determinism results
    det_names = [r['run_name'] for r in determinism_results]
    det_scores = [1 if r['is_deterministic'] else 0 for r in determinism_results]
    
    axes[0, 1].bar(det_names, det_scores, color=['green' if s == 1 else 'red' for s in det_scores])
    axes[0, 1].set_title('Determinism Validation')
    axes[0, 1].set_ylabel('Pass (1) / Fail (0)')
    axes[0, 1].set_ylim(0, 1.2)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Performance comparison
    if baseline_run is not None and candidate_run is not None:
        axes[0, 2].plot(baseline_run['step'], baseline_run['reward_mean'], label='Baseline', linewidth=2)
        axes[0, 2].plot(candidate_run['step'], candidate_run['reward_mean'], label='Candidate', linewidth=2)
        axes[0, 2].set_title('Performance Comparison')
        axes[0, 2].set_xlabel('Training Step')
        axes[0, 2].set_ylabel('Reward')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Anomaly detection results
    anomaly_names = list(anomaly_results.keys())
    anomaly_counts = [r['anomaly_count'] for r in anomaly_results.values()]
    
    axes[1, 0].bar(anomaly_names, anomaly_counts, color=['green' if c == 0 else 'orange' if c <= 2 else 'red' for c in anomaly_counts])
    axes[1, 0].set_title('Anomaly Detection Results')
    axes[1, 0].set_ylabel('Anomaly Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 5: Stability results
    stability_names = list(stability_results.keys())
    stability_scores = [r['stability_score'] for r in stability_results.values()]
    
    axes[1, 1].bar(stability_names, stability_scores, color=['green' if s > 10 else 'orange' if s > 5 else 'red' for s in stability_scores])
    axes[1, 1].set_title('Stability Validation')
    axes[1, 1].set_ylabel('Stability Score')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Plot 6: Overall deployment readiness
    check_names = list(deployment_readiness.keys())
    check_scores = [1 if v else 0 for v in deployment_readiness.values()]
    
    axes[1, 2].bar(check_names, check_scores, color=['green' if s == 1 else 'red' for s in check_scores])
    axes[1, 2].set_title('Overall Deployment Readiness')
    axes[1, 2].set_ylabel('Pass (1) / Fail (0)')
    axes[1, 2].set_ylim(0, 1.2)
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('./production_deployment_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 12. Finish experiment tracking
    summary = tracker.finish_experiment()
    print("✅ Production deployment experiment tracking completed!")
    
    # 13. Deployment readiness summary
    print("\n🚀 Production Deployment Readiness Summary:")
    print(f"  Overall readiness: {'✅ APPROVED' if overall_readiness else '❌ REJECTED'}")
    print(f"  Reproducibility: {'✅ PASS' if reproducibility_pass else '❌ FAIL'}")
    print(f"  Determinism: {'✅ PASS' if determinism_pass else '❌ FAIL'}")
    print(f"  Performance regression: {'✅ PASS' if regression_pass else '❌ FAIL'}")
    print(f"  Anomaly detection: {'✅ PASS' if anomaly_pass else '❌ FAIL'}")
    print(f"  Stability: {'✅ PASS' if stability_pass else '❌ FAIL'}")
    print(f"  Edge case handling: {'✅ PASS' if edge_case_pass else '❌ FAIL'}")
    
    if deployment_report['deployment_risks']:
        print(f"\n⚠️  Deployment risks:")
        for risk in deployment_report['deployment_risks']:
            print(f"  - {risk}")
    
    print(f"\n📋 Deployment recommendation: {deployment_report['deployment_recommendation']}")
    
    # 14. Cleanup
    print("\n🧹 Cleaning up...")
    files_created = []
    for run_data in production_runs:
        files_created.append(run_data['run_dir'])
    files_created.extend([
        "./production_deployment_report.json",
        "./production_deployment_dashboard.png",
        "./rldk_reports"
    ])
    
    # Clean up temporary files
    temp_files = list(Path('.').glob('repro_test_*'))
    temp_files.extend(list(Path('.').glob('determinism_test_*.py')))
    
    for temp_file in temp_files:
        if temp_file.is_dir():
            import shutil
            shutil.rmtree(temp_file)
        else:
            temp_file.unlink()
    
    print("📁 Files created during this demo:")
    for file_path in files_created:
        path = Path(file_path)
        if path.exists():
            if path.is_file():
                print(f"  📄 {file_path}")
            elif path.is_dir():
                files = list(path.rglob("*"))
                print(f"  📁 {file_path} ({len(files)} files)")
    
    print("\n✅ Production deployment checklist example completed!")
    print("\n💡 Key insights:")
    print("  - Pre-deployment validation is crucial for production systems")
    print("  - Multiple validation checks ensure deployment readiness")
    print("  - RLDK provides comprehensive deployment validation tools")
    print("  - Automated validation reduces deployment risks")

if __name__ == "__main__":
    main()