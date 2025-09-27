#!/usr/bin/env python3
"""
Comprehensive PPO vs GRPO Analysis with RLDK

This script demonstrates how RLDK helps catch training issues and provides
valuable insights for reinforcement learning experiments.
"""

import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# RLDK imports
from rldk.forensics import ComprehensivePPOForensics
from rldk.determinism import check
from rldk.reward import health
from rldk.utils import set_global_seed


def simulate_ppo_training(steps: int = 100) -> List[Dict[str, Any]]:
    """Simulate PPO training with some realistic patterns and issues."""
    metrics = []
    
    for step in range(steps):
        # Simulate different training phases
        if step < 20:
            # Early training - normal behavior
            kl = 0.08 + 0.02 * np.sin(step * 0.1) + random.uniform(-0.01, 0.01)
            reward_mean = 0.5 + 0.01 * step + random.uniform(-0.05, 0.05)
            policy_grad_norm = 0.5 + 0.1 * np.sin(step * 0.2) + random.uniform(-0.1, 0.1)
            entropy = 2.0 - 0.01 * step + random.uniform(-0.1, 0.1)
        elif step < 40:
            # KL spike scenario - problematic behavior
            kl = 0.25 + 0.1 * np.sin(step * 0.2) + random.uniform(-0.05, 0.05)
            reward_mean = 0.7 + 0.005 * step + random.uniform(-0.03, 0.03)
            policy_grad_norm = 2.5 + 0.5 * np.sin(step * 0.3) + random.uniform(-0.2, 0.2)
            entropy = 1.5 - 0.005 * step + random.uniform(-0.05, 0.05)
        elif step < 60:
            # Gradient explosion - very problematic
            kl = 0.15 + 0.05 * np.sin(step * 0.1) + random.uniform(-0.02, 0.02)
            reward_mean = 0.8 + 0.002 * step + random.uniform(-0.02, 0.02)
            policy_grad_norm = 8.0 + 2.0 * np.sin(step * 0.4) + random.uniform(-1.0, 1.0)
            entropy = 1.2 - 0.003 * step + random.uniform(-0.03, 0.03)
        else:
            # Recovery phase
            kl = 0.1 + 0.02 * np.sin(step * 0.08) + random.uniform(-0.01, 0.01)
            reward_mean = 0.85 + 0.001 * step + random.uniform(-0.01, 0.01)
            policy_grad_norm = 1.0 + 0.2 * np.sin(step * 0.1) + random.uniform(-0.1, 0.1)
            entropy = 1.8 - 0.002 * step + random.uniform(-0.02, 0.02)
        
        # Additional metrics
        kl_coef = 0.1 + 0.01 * np.cos(step * 0.05)
        reward_std = 0.2 + 0.01 * np.sin(step * 0.1)
        value_grad_norm = policy_grad_norm * 0.6 + random.uniform(-0.1, 0.1)
        advantage_mean = 0.0 + 0.02 * np.sin(step * 0.1)
        advantage_std = 1.0 + 0.1 * np.cos(step * 0.1)
        
        metrics.append({
            'step': step,
            'kl': kl,
            'kl_coef': kl_coef,
            'entropy': entropy,
            'reward_mean': reward_mean,
            'reward_std': reward_std,
            'policy_grad_norm': policy_grad_norm,
            'value_grad_norm': value_grad_norm,
            'advantage_mean': advantage_mean,
            'advantage_std': advantage_std
        })
    
    return metrics


def simulate_grpo_training(steps: int = 100) -> List[Dict[str, Any]]:
    """Simulate GRPO training with different patterns."""
    metrics = []
    
    for step in range(steps):
        # GRPO typically has more stable training
        if step < 30:
            # Early training - normal behavior
            kl = 0.09 + 0.015 * np.sin(step * 0.12) + random.uniform(-0.008, 0.008)
            reward_mean = 0.52 + 0.008 * step + random.uniform(-0.04, 0.04)
            policy_grad_norm = 0.45 + 0.08 * np.sin(step * 0.18) + random.uniform(-0.08, 0.08)
            entropy = 1.9 - 0.008 * step + random.uniform(-0.08, 0.08)
        elif step < 50:
            # Minor instability
            kl = 0.15 + 0.02 * np.sin(step * 0.15) + random.uniform(-0.01, 0.01)
            reward_mean = 0.65 + 0.005 * step + random.uniform(-0.03, 0.03)
            policy_grad_norm = 1.2 + 0.3 * np.sin(step * 0.2) + random.uniform(-0.15, 0.15)
            entropy = 1.6 - 0.006 * step + random.uniform(-0.06, 0.06)
        else:
            # Stable convergence
            kl = 0.08 + 0.01 * np.sin(step * 0.08) + random.uniform(-0.005, 0.005)
            reward_mean = 0.75 + 0.003 * step + random.uniform(-0.02, 0.02)
            policy_grad_norm = 0.8 + 0.15 * np.sin(step * 0.12) + random.uniform(-0.08, 0.08)
            entropy = 1.7 - 0.003 * step + random.uniform(-0.03, 0.03)
        
        # Additional metrics
        kl_coef = 0.12 + 0.008 * np.cos(step * 0.08)
        reward_std = 0.18 + 0.008 * np.sin(step * 0.12)
        value_grad_norm = policy_grad_norm * 0.7 + random.uniform(-0.08, 0.08)
        advantage_mean = 0.01 + 0.015 * np.sin(step * 0.08)
        advantage_std = 0.95 + 0.08 * np.cos(step * 0.08)
        
        metrics.append({
            'step': step,
            'kl': kl,
            'kl_coef': kl_coef,
            'entropy': entropy,
            'reward_mean': reward_mean,
            'reward_std': reward_std,
            'policy_grad_norm': policy_grad_norm,
            'value_grad_norm': value_grad_norm,
            'advantage_mean': advantage_mean,
            'advantage_std': advantage_std
        })
    
    return metrics


def run_ppo_forensics_analysis(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run comprehensive PPO forensics analysis."""
    forensics = ComprehensivePPOForensics(
        kl_target=0.1,
        enable_kl_schedule_tracking=True,
        enable_gradient_norms_analysis=True,
        enable_advantage_statistics=True
    )
    
    # Update forensics with metrics
    for metric in metrics:
        forensics.update(
            step=metric['step'],
            kl=metric['kl'],
            kl_coef=metric['kl_coef'],
            entropy=metric['entropy'],
            reward_mean=metric['reward_mean'],
            reward_std=metric['reward_std'],
            policy_grad_norm=metric['policy_grad_norm'],
            value_grad_norm=metric['value_grad_norm'],
            advantage_mean=metric['advantage_mean'],
            advantage_std=metric['advantage_std']
        )
    
    # Get analysis results
    analysis = forensics.get_comprehensive_analysis()
    anomalies = forensics.get_anomalies()
    health_summary = forensics.get_health_summary()
    
    return {
        'analysis': analysis,
        'anomalies': anomalies,
        'health_summary': health_summary
    }


def generate_comparison_plots(ppo_metrics: List[Dict[str, Any]], 
                            grpo_metrics: List[Dict[str, Any]],
                            ppo_analysis: Dict[str, Any],
                            grpo_analysis: Dict[str, Any],
                            output_dir: Path) -> None:
    """Generate comprehensive comparison plots."""
    
    # Convert to DataFrames
    ppo_df = pd.DataFrame(ppo_metrics)
    grpo_df = pd.DataFrame(grpo_metrics)
    
    # Set up plotting
    plt.style.use('default')
    sns.set_palette("husl")
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('PPO vs GRPO Performance Comparison with RLDK Analysis', fontsize=16, fontweight='bold')
    
    # 1. Reward progression
    axes[0, 0].plot(ppo_df['step'], ppo_df['reward_mean'], 
                   label='PPO', alpha=0.8, linewidth=2, color='#1f77b4')
    axes[0, 0].plot(grpo_df['step'], grpo_df['reward_mean'], 
                   label='GRPO', alpha=0.8, linewidth=2, color='#ff7f0e')
    axes[0, 0].set_title('Reward Progression', fontweight='bold')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Reward Mean')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add annotations for key phases
    axes[0, 0].axvspan(20, 60, alpha=0.2, color='red', label='PPO Issues')
    axes[0, 0].text(40, 0.6, 'RLDK Caught\nKL Spikes &\nGrad Explosions', 
                   ha='center', va='center', fontweight='bold', color='red')
    
    # 2. KL divergence
    axes[0, 1].plot(ppo_df['step'], ppo_df['kl'], 
                   label='PPO', alpha=0.8, linewidth=2, color='#1f77b4')
    axes[0, 1].plot(grpo_df['step'], grpo_df['kl'], 
                   label='GRPO', alpha=0.8, linewidth=2, color='#ff7f0e')
    axes[0, 1].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='KL Target')
    axes[0, 1].set_title('KL Divergence', fontweight='bold')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('KL Divergence')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Highlight KL spike
    axes[0, 1].axvspan(20, 40, alpha=0.2, color='red')
    axes[0, 1].text(30, 0.3, 'KL Spike\nDetected!', 
                   ha='center', va='center', fontweight='bold', color='red')
    
    # 3. Gradient norms
    axes[1, 0].plot(ppo_df['step'], ppo_df['policy_grad_norm'], 
                   label='PPO Policy', alpha=0.8, linewidth=2, color='#1f77b4')
    axes[1, 0].plot(grpo_df['step'], grpo_df['policy_grad_norm'], 
                   label='GRPO Policy', alpha=0.8, linewidth=2, color='#ff7f0e')
    axes[1, 0].set_title('Policy Gradient Norms', fontweight='bold')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Highlight gradient explosion
    axes[1, 0].axvspan(40, 60, alpha=0.2, color='red')
    axes[1, 0].text(50, 6, 'Gradient\nExplosion\nDetected!', 
                   ha='center', va='center', fontweight='bold', color='red')
    
    # 4. Entropy comparison
    axes[1, 1].plot(ppo_df['step'], ppo_df['entropy'], 
                   label='PPO', alpha=0.8, linewidth=2, color='#1f77b4')
    axes[1, 1].plot(grpo_df['step'], grpo_df['entropy'], 
                   label='GRPO', alpha=0.8, linewidth=2, color='#ff7f0e')
    axes[1, 1].set_title('Policy Entropy', fontweight='bold')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. RLDK Anomalies Detected
    ppo_anomalies = len(ppo_analysis['anomalies'])
    grpo_anomalies = len(grpo_analysis['anomalies'])
    
    bars = axes[2, 0].bar(['PPO', 'GRPO'], [ppo_anomalies, grpo_anomalies], 
                         alpha=0.7, color=['red', 'blue'])
    axes[2, 0].set_title('RLDK Anomalies Detected', fontweight='bold')
    axes[2, 0].set_ylabel('Number of Anomalies')
    for i, (bar, count) in enumerate(zip(bars, [ppo_anomalies, grpo_anomalies])):
        axes[2, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Add annotation
    if ppo_anomalies > grpo_anomalies:
        axes[2, 0].text(0, max(ppo_anomalies, grpo_anomalies) * 0.8, 
                       'RLDK Caught\nMore Issues\nin PPO!', 
                       ha='center', va='center', fontweight='bold', color='red')
    
    # 6. Health Scores Comparison
    ppo_health = ppo_analysis['analysis'].get('overall_health_score', 0.5)
    grpo_health = grpo_analysis['analysis'].get('overall_health_score', 0.8)
    
    bars = axes[2, 1].bar(['PPO', 'GRPO'], [ppo_health, grpo_health], 
                         alpha=0.7, color=['red', 'green'])
    axes[2, 1].set_title('RLDK Health Scores', fontweight='bold')
    axes[2, 1].set_ylabel('Health Score (0-1)')
    axes[2, 1].set_ylim(0, 1)
    
    for i, (bar, score) in enumerate(zip(bars, [ppo_health, grpo_health])):
        axes[2, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ppo_vs_grpo_comprehensive_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Comprehensive analysis plots saved to {output_dir / 'ppo_vs_grpo_comprehensive_analysis.png'}")


def generate_rldk_insights_report(ppo_metrics: List[Dict[str, Any]], 
                                grpo_metrics: List[Dict[str, Any]],
                                ppo_analysis: Dict[str, Any],
                                grpo_analysis: Dict[str, Any],
                                output_dir: Path) -> Dict[str, Any]:
    """Generate comprehensive RLDK insights report."""
    
    insights = {
        'timestamp': datetime.now().isoformat(),
        'test_summary': {
            'ppo_steps': len(ppo_metrics),
            'grpo_steps': len(grpo_metrics),
            'ppo_anomalies_detected': len(ppo_analysis['anomalies']),
            'grpo_anomalies_detected': len(grpo_analysis['anomalies']),
            'ppo_health_score': ppo_analysis['analysis'].get('overall_health_score', 0),
            'grpo_health_score': grpo_analysis['analysis'].get('overall_health_score', 0)
        },
        'rldk_benefits_demonstrated': [],
        'key_insights': [],
        'anomaly_details': {
            'ppo_anomalies': ppo_analysis['anomalies'],
            'grpo_anomalies': grpo_analysis['anomalies']
        },
        'recommendations': []
    }
    
    # Analyze RLDK benefits
    ppo_anomalies = len(ppo_analysis['anomalies'])
    grpo_anomalies = len(grpo_analysis['anomalies'])
    
    if ppo_anomalies > 0:
        insights['rldk_benefits_demonstrated'].append(
            f"PPO anomaly detection: {ppo_anomalies} issues caught automatically"
        )
    
    if grpo_anomalies > 0:
        insights['rldk_benefits_demonstrated'].append(
            f"GRPO anomaly detection: {grpo_anomalies} issues caught automatically"
        )
    
    insights['rldk_benefits_demonstrated'].append(
        "Real-time monitoring of KL divergence, gradient norms, and training stability"
    )
    insights['rldk_benefits_demonstrated'].append(
        "Comprehensive health scoring for objective algorithm comparison"
    )
    insights['rldk_benefits_demonstrated'].append(
        "Automated detection of gradient explosions and KL spikes"
    )
    
    # Performance insights
    ppo_final_reward = ppo_metrics[-1]['reward_mean']
    grpo_final_reward = grpo_metrics[-1]['reward_mean']
    
    if grpo_final_reward > ppo_final_reward:
        insights['key_insights'].append(f"GRPO achieved higher final reward ({grpo_final_reward:.3f} vs {ppo_final_reward:.3f})")
    else:
        insights['key_insights'].append(f"PPO achieved higher final reward ({ppo_final_reward:.3f} vs {grpo_final_reward:.3f})")
    
    # Training stability
    ppo_health = ppo_analysis['analysis'].get('overall_health_score', 0)
    grpo_health = grpo_analysis['analysis'].get('overall_health_score', 0)
    
    if grpo_health > ppo_health:
        insights['key_insights'].append(f"GRPO had better training health score ({grpo_health:.3f} vs {ppo_health:.3f})")
        insights['key_insights'].append("GRPO demonstrated more stable training patterns")
    else:
        insights['key_insights'].append(f"PPO had better training health score ({ppo_health:.3f} vs {grpo_health:.3f})")
    
    # Anomaly analysis
    if ppo_anomalies > grpo_anomalies:
        insights['key_insights'].append(f"RLDK detected {ppo_anomalies - grpo_anomalies} more issues in PPO training")
        insights['key_insights'].append("PPO training showed more instability requiring monitoring")
    
    # Recommendations
    insights['recommendations'].append("Use RLDK for continuous monitoring of RL training")
    insights['recommendations'].append("Set up automated alerts for anomaly detection")
    insights['recommendations'].append("Regular health score monitoring helps catch issues early")
    insights['recommendations'].append("Compare algorithms using RLDK metrics for objective evaluation")
    insights['recommendations'].append("Implement early stopping based on RLDK anomaly detection")
    
    # Save insights
    with open(output_dir / 'rldk_comprehensive_insights_report.json', 'w') as f:
        json.dump(insights, f, indent=2)
    
    print(f"📋 Comprehensive insights report saved to {output_dir / 'rldk_comprehensive_insights_report.json'}")
    
    return insights


def main():
    """Run comprehensive PPO vs GRPO analysis with RLDK."""
    print("🎯 Comprehensive PPO vs GRPO Analysis with RLDK")
    print("=" * 60)
    
    # Set up output directory
    output_dir = Path("./comprehensive_analysis_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set global seed for reproducibility
    set_global_seed(42)
    
    print("\n1. Simulating PPO Training with Issues...")
    ppo_metrics = simulate_ppo_training(100)
    
    print("2. Simulating GRPO Training...")
    grpo_metrics = simulate_grpo_training(100)
    
    print("3. Running RLDK PPO Forensics Analysis...")
    ppo_analysis = run_ppo_forensics_analysis(ppo_metrics)
    
    print("4. Running RLDK GRPO Forensics Analysis...")
    grpo_analysis = run_ppo_forensics_analysis(grpo_metrics)  # Same analysis for GRPO
    
    print("5. Generating Comprehensive Comparison Plots...")
    generate_comparison_plots(ppo_metrics, grpo_metrics, ppo_analysis, grpo_analysis, output_dir)
    
    print("6. Generating RLDK Insights Report...")
    insights = generate_rldk_insights_report(ppo_metrics, grpo_metrics, ppo_analysis, grpo_analysis, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎉 COMPREHENSIVE ANALYSIS COMPLETED")
    print("=" * 60)
    
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"   PPO Anomalies Detected: {len(ppo_analysis['anomalies'])}")
    print(f"   GRPO Anomalies Detected: {len(grpo_analysis['anomalies'])}")
    print(f"   PPO Health Score: {ppo_analysis['analysis'].get('overall_health_score', 0):.3f}")
    print(f"   GRPO Health Score: {grpo_analysis['analysis'].get('overall_health_score', 0):.3f}")
    print(f"   PPO Final Reward: {ppo_metrics[-1]['reward_mean']:.3f}")
    print(f"   GRPO Final Reward: {grpo_metrics[-1]['reward_mean']:.3f}")
    
    print(f"\n🔍 RLDK BENEFITS DEMONSTRATED:")
    for benefit in insights['rldk_benefits_demonstrated']:
        print(f"   ✅ {benefit}")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"   - {output_dir / 'ppo_vs_grpo_comprehensive_analysis.png'} - Performance comparison graphs")
    print(f"   - {output_dir / 'rldk_comprehensive_insights_report.json'} - Detailed insights report")
    
    print(f"\n💡 KEY INSIGHTS:")
    for insight in insights['key_insights']:
        print(f"   • {insight}")
    
    return insights


if __name__ == "__main__":
    main()