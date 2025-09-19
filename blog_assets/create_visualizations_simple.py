import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path

def create_visualizations():
    """Create all required visualizations for the RLDK blog post."""
    
    run_df = pd.read_json('artifacts/run.jsonl', lines=True)
    alerts_df = pd.read_json('artifacts/alerts.jsonl', lines=True)
    
    with open('comprehensive_ppo_forensics_demo/comprehensive_analysis.json', 'r') as f:
        analysis_data = json.load(f)
    
    kl_data = run_df[run_df['name'] == 'kl'].sort_values('step')
    reward_data = run_df[run_df['name'] == 'reward'].sort_values('step')
    grad_norm_data = run_df[run_df['name'] == 'grad_norm'].sort_values('step')
    
    stop_alerts = alerts_df[alerts_df.get('action') == 'stop']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('RLDK Real-Time RL Monitoring Dashboard', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    if not kl_data.empty:
        ax1.plot(kl_data['step'], kl_data['value'], 'b-', linewidth=2, label='KL Divergence')
        ax1.axhline(y=0.4, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold (0.4)')
        ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Critical Threshold (0.8)')
        
        critical_points = kl_data[kl_data['value'] >= 0.8]
        if not critical_points.empty:
            ax1.scatter(critical_points['step'], critical_points['value'], 
                       color='red', s=100, zorder=5, label='Critical Spikes')
    
    ax1.set_title('KL Divergence Spike Detection')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('KL Divergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    health_scores = [
        analysis_data['overall_health_score'],
        analysis_data['training_stability_score'],
        analysis_data['convergence_quality_score']
    ]
    health_labels = ['Overall\n(0.603)', 'Stability\n(0.855)', 'Convergence\n(0.959)']
    colors = ['red' if score < 0.7 else 'orange' if score < 0.8 else 'green' for score in health_scores]
    
    bars = ax2.bar(health_labels, health_scores, color=colors, alpha=0.7)
    ax2.set_title('Training Health Scores')
    ax2.set_ylabel('Health Score')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars, health_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax3 = axes[1, 0]
    if not reward_data.empty:
        reward_values = reward_data['value'].values.astype(float)
        if np.std(reward_values) > 0:
            normalized_rewards = (reward_values - np.mean(reward_values)) / np.std(reward_values)
        else:
            normalized_rewards = reward_values
        ax3.plot(reward_data['step'], normalized_rewards, 'g-', linewidth=2, label='Reward (normalized)')
    
    if not grad_norm_data.empty:
        ax3.plot(grad_norm_data['step'], grad_norm_data['value'], 'purple', linewidth=2, label='Gradient Norm')
    
    ax3.set_title('Training Metrics Timeline')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Metric Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    if not alerts_df.empty:
        severity_colors = {'info': 'blue', 'warning': 'orange', 'critical': 'red'}
        for severity in ['info', 'warning', 'critical']:
            severity_alerts = alerts_df[alerts_df['severity'] == severity]
            if not severity_alerts.empty:
                ax4.scatter(severity_alerts['step'], [severity] * len(severity_alerts),
                           c=severity_colors[severity], s=100, alpha=0.7, label=f'{severity.title()} Alerts')
        
        if not stop_alerts.empty:
            termination_step = stop_alerts.iloc[0]['step']
            ax4.axvline(x=termination_step, color='red', linestyle=':', linewidth=3, 
                       label=f'Training Terminated (Step {termination_step})')
    
    ax4.set_title('Alert Timeline')
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Alert Severity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/rldk_monitoring_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    create_individual_plots(kl_data, alerts_df, analysis_data, reward_data, grad_norm_data)
    
    print("✅ All visualizations created successfully!")
    print("Generated files:")
    print("  - images/rldk_monitoring_dashboard.png")
    print("  - images/kl_spike_detection.png")
    print("  - images/health_dashboard.png")
    print("  - images/training_metrics.png")
    print("  - images/alerts_timeline.png")

def create_individual_plots(kl_data, alerts_df, analysis_data, reward_data, grad_norm_data):
    """Create individual plots for embedding in blog post."""
    
    plt.figure(figsize=(10, 6))
    if not kl_data.empty:
        plt.plot(kl_data['step'], kl_data['value'], 'b-', linewidth=3, label='KL Divergence', marker='o', markersize=8)
        plt.axhline(y=0.4, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='Warning Threshold (0.4)')
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Critical Threshold (0.8)')
        
        critical_points = kl_data[kl_data['value'] >= 0.8]
        if not critical_points.empty:
            plt.scatter(critical_points['step'], critical_points['value'], 
                       color='red', s=150, zorder=5, label='Critical Spikes', edgecolors='darkred', linewidth=2)
    
    plt.title('KL Divergence Spike Detection', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('KL Divergence', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/kl_spike_detection.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    health_scores = [
        analysis_data['overall_health_score'],
        analysis_data['training_stability_score'],
        analysis_data['convergence_quality_score']
    ]
    health_labels = ['Overall Health\n(0.603)', 'Training Stability\n(0.855)', 'Convergence Quality\n(0.959)']
    colors = ['#ff4444' if score < 0.7 else '#ff8800' if score < 0.8 else '#44aa44' for score in health_scores]
    
    bars = plt.bar(health_labels, health_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    plt.title('Training Health Assessment', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Health Score', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars, health_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('images/health_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    if not reward_data.empty:
        ax1.plot(reward_data['step'], reward_data['value'], 'g-', linewidth=3, marker='s', markersize=6, label='Reward')
        ax1.set_title('Reward Progression', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Reward Value', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    if not grad_norm_data.empty:
        ax2.plot(grad_norm_data['step'], grad_norm_data['value'], 'purple', linewidth=3, marker='^', markersize=6, label='Gradient Norm')
        ax2.set_title('Gradient Norm Evolution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Training Step', fontsize=12)
        ax2.set_ylabel('Gradient Norm', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Training Metrics Timeline', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    if not alerts_df.empty:
        severity_colors = {'info': '#4488cc', 'warning': '#ff8800', 'critical': '#cc4444'}
        severity_positions = {'info': 1, 'warning': 2, 'critical': 3}
        
        for severity in ['info', 'warning', 'critical']:
            severity_alerts = alerts_df[alerts_df['severity'] == severity]
            if not severity_alerts.empty:
                plt.scatter(severity_alerts['step'], [severity_positions[severity]] * len(severity_alerts),
                           c=severity_colors[severity], s=150, alpha=0.8, 
                           label=f'{severity.title()} Alerts ({len(severity_alerts)})', 
                           edgecolors='black', linewidth=1)
        
        stop_alerts = alerts_df[alerts_df.get('action') == 'stop']
        if not stop_alerts.empty:
            termination_step = stop_alerts.iloc[0]['step']
            plt.axvline(x=termination_step, color='red', linestyle=':', linewidth=4, alpha=0.8,
                       label=f'Training Terminated (Step {termination_step})')
    
    plt.title('Real-Time Alert Timeline', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Alert Severity Level', fontsize=12)
    plt.yticks([1, 2, 3], ['Info', 'Warning', 'Critical'])
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/alerts_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    Path('images').mkdir(exist_ok=True)
    create_visualizations()
