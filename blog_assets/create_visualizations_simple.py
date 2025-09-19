#!/usr/bin/env python3
"""
RLDK Blog Post Visualization Generator

Creates visualizations for the RLDK technical blog post using real training data.
This script generates 4 key charts that demonstrate RLDK's monitoring capabilities.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_comprehensive_analysis():
    """Load the comprehensive analysis data"""
    with open('../comprehensive_ppo_forensics_demo/comprehensive_analysis.json', 'r') as f:
        return json.load(f)

def load_training_metrics():
    """Load the training metrics data"""
    with open('../comprehensive_ppo_monitor_demo/comprehensive_demo_run_comprehensive_metrics.json', 'r') as f:
        return json.load(f)

def create_kl_health_dashboard():
    """Create KL divergence and health score dashboard"""
    print("Creating KL Health Dashboard...")
    
    # Load data
    training_data = load_training_metrics()
    analysis_data = load_comprehensive_analysis()
    
    # Extract KL and health scores from training data
    steps = [entry['step'] for entry in training_data]
    kl_values = [entry['kl'] for entry in training_data]
    overall_health = [entry['overall_health_score'] for entry in training_data]
    stability_health = [entry['training_stability_score'] for entry in training_data]
    convergence_health = [entry['convergence_quality_score'] for entry in training_data]
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top plot: KL Divergence
    ax1.plot(steps, kl_values, 'g-', linewidth=2, label='KL Divergence', alpha=0.8)
    ax1.axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='Target KL (0.1)')
    ax1.axhline(y=0.4, color='orange', linestyle=':', alpha=0.7, label='Warning Threshold (0.4)')
    ax1.axhline(y=0.8, color='red', linestyle=':', alpha=0.7, label='Critical Threshold (0.8)')
    ax1.set_ylabel('KL Divergence', fontsize=12)
    ax1.set_title('KL Divergence Monitoring', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.15)
    
    # Bottom plot: Health Scores
    ax2.plot(steps, overall_health, 'r-', linewidth=3, label='Overall Health', alpha=0.8)
    ax2.plot(steps, stability_health, 'b-', linewidth=2, label='Training Stability', alpha=0.7)
    ax2.plot(steps, convergence_health, 'g-', linewidth=2, label='Convergence Quality', alpha=0.7)
    ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Critical Threshold (0.7)')
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Health Score', fontsize=12)
    ax2.set_title('Multi-Dimensional Health Monitoring', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # Add final health score annotation
    final_health = analysis_data['overall_health_score']
    ax2.annotate(f'Final Health: {final_health:.3f}', 
                xy=(steps[-1], final_health), 
                xytext=(steps[-1]-5, final_health+0.1),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('images/kl_health_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ KL Health Dashboard saved")

def create_advantage_bias_analysis():
    """Create advantage bias analysis chart"""
    print("Creating Advantage Bias Analysis...")
    
    # Load data
    training_data = load_training_metrics()
    analysis_data = load_comprehensive_analysis()
    
    # Extract advantage data
    steps = [entry['step'] for entry in training_data]
    advantage_means = [entry['advantage_advantage_mean'] for entry in training_data]
    advantage_bias_risk = [entry['advantage_advantage_bias_risk'] for entry in training_data]
    advantage_bias = [entry['advantage_advantage_bias'] for entry in training_data]
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top plot: Advantage Mean (shows bias)
    ax1.plot(steps, advantage_means, 'purple', linewidth=2, label='Advantage Mean', alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Zero Bias (Ideal)')
    ax1.axhline(y=0.1, color='orange', linestyle=':', alpha=0.7, label='Warning Threshold')
    ax1.axhline(y=-0.1, color='orange', linestyle=':', alpha=0.7)
    ax1.set_ylabel('Advantage Mean', fontsize=12)
    ax1.set_title('Advantage Bias Detection', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Bias Risk and Actual Bias
    ax2.plot(steps, advantage_bias_risk, 'r-', linewidth=2, label='Bias Risk Score', alpha=0.8)
    ax2.plot(steps, advantage_bias, 'orange', linewidth=2, label='Actual Bias', alpha=0.8)
    ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Critical Threshold (0.1)')
    ax2.axhline(y=0.3, color='red', linestyle=':', alpha=0.7, label='Risk Threshold (0.3)')
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Bias Score', fontsize=12)
    ax2.set_title('Advantage Bias Risk Assessment', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)
    
    # Add final bias annotation
    final_bias = analysis_data['trackers']['advantage_statistics']['advantage_bias']
    ax2.annotate(f'Final Bias: {final_bias:.3f}', 
                xy=(steps[-1], final_bias), 
                xytext=(steps[-1]-5, final_bias+0.1),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('images/advantage_bias_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Advantage Bias Analysis saved")

def create_training_stability():
    """Create training stability metrics chart"""
    print("Creating Training Stability Overview...")
    
    # Load data
    training_data = load_training_metrics()
    
    # Extract gradient and stability data
    steps = [entry['step'] for entry in training_data]
    policy_grad_norm = [entry['gradient_policy_grad_norm'] for entry in training_data]
    value_grad_norm = [entry['gradient_value_grad_norm'] for entry in training_data]
    total_grad_norm = [entry['gradient_total_grad_norm'] for entry in training_data]
    grad_ratio = [entry['gradient_policy_value_ratio'] for entry in training_data]
    
    # Create the plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Gradient norms
    ax1.plot(steps, policy_grad_norm, 'b-', linewidth=2, label='Policy Grad Norm', alpha=0.8)
    ax1.plot(steps, value_grad_norm, 'g-', linewidth=2, label='Value Grad Norm', alpha=0.8)
    ax1.plot(steps, total_grad_norm, 'r-', linewidth=2, label='Total Grad Norm', alpha=0.8)
    ax1.set_ylabel('Gradient Norm', fontsize=12)
    ax1.set_title('Gradient Norms Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Policy/Value ratio
    ax2.plot(steps, grad_ratio, 'purple', linewidth=2, label='Policy/Value Ratio', alpha=0.8)
    ax2.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, label='Balanced (1.0)')
    ax2.axhline(y=5.0, color='orange', linestyle=':', alpha=0.7, label='Warning (5.0)')
    ax2.axhline(y=10.0, color='red', linestyle=':', alpha=0.7, label='Critical (10.0)')
    ax2.set_ylabel('Policy/Value Ratio', fontsize=12)
    ax2.set_title('Gradient Balance', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training stability score
    stability_scores = [entry['training_stability_score'] for entry in training_data]
    ax3.plot(steps, stability_scores, 'orange', linewidth=3, label='Training Stability', alpha=0.8)
    ax3.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
    ax3.set_xlabel('Training Step', fontsize=12)
    ax3.set_ylabel('Stability Score', fontsize=12)
    ax3.set_title('Training Stability Score', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)
    
    # Gradient health score
    grad_health = [entry['gradient_gradient_health_score'] for entry in training_data]
    ax4.plot(steps, grad_health, 'green', linewidth=3, label='Gradient Health', alpha=0.8)
    ax4.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
    ax4.set_xlabel('Training Step', fontsize=12)
    ax4.set_ylabel('Health Score', fontsize=12)
    ax4.set_title('Gradient Health Score', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('images/training_stability.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Training Stability Overview saved")

def create_anomaly_timeline():
    """Create anomaly detection timeline"""
    print("Creating Anomaly Detection Timeline...")
    
    # Load analysis data
    analysis_data = load_comprehensive_analysis()
    
    # Extract anomalies
    anomalies = analysis_data['anomalies']
    
    # Create timeline data
    anomaly_types = []
    severities = []
    messages = []
    values = []
    thresholds = []
    
    for anomaly in anomalies:
        anomaly_types.append(anomaly['type'].replace('_anomaly', '').replace('_', ' ').title())
        severities.append(anomaly['severity'])
        messages.append(anomaly['message'])
        values.append(anomaly['value'])
        thresholds.append(anomaly['threshold'])
    
    # Create color mapping for severities
    severity_colors = {'warning': 'orange', 'critical': 'red'}
    colors = [severity_colors.get(sev, 'blue') for sev in severities]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(anomaly_types))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.7)
    
    # Add threshold lines
    for i, threshold in enumerate(thresholds):
        ax.axvline(x=threshold, ymin=i/len(anomaly_types), ymax=(i+1)/len(anomaly_types), 
                  color='red', linestyle=':', alpha=0.7)
        # Add threshold labels
        ax.text(threshold + 0.01, i, f'Threshold: {threshold:.3f}', 
               verticalalignment='center', fontsize=9, color='red', alpha=0.8)
    
    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(anomaly_types, fontsize=11)
    ax.set_xlabel('Anomaly Value', fontsize=12)
    ax.set_title('RLDK Anomaly Detection Results', fontsize=16, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, value, severity) in enumerate(zip(bars, values, severities)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{value:.3f} ({severity})', 
               ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='orange', alpha=0.7, label='Warning'),
                      Patch(facecolor='red', alpha=0.7, label='Critical')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Add summary text
    total_anomalies = len(anomalies)
    critical_count = sum(1 for a in anomalies if a['severity'] == 'critical')
    warning_count = sum(1 for a in anomalies if a['severity'] == 'warning')
    
    summary_text = f'Total Anomalies: {total_anomalies}\nCritical: {critical_count}\nWarning: {warning_count}'
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('images/anomaly_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Anomaly Detection Timeline saved")

def main():
    """Main function to generate all visualizations"""
    print("🚀 Generating RLDK Blog Post Visualizations")
    print("=" * 50)
    
    # Create images directory if it doesn't exist
    Path('images').mkdir(exist_ok=True)
    
    try:
        # Generate all visualizations
        create_kl_health_dashboard()
        create_advantage_bias_analysis()
        create_training_stability()
        create_anomaly_timeline()
        
        print("\n" + "=" * 50)
        print("✅ All visualizations generated successfully!")
        print("\nGenerated files:")
        print("  📊 images/kl_health_dashboard.png")
        print("  📊 images/advantage_bias_analysis.png")
        print("  📊 images/training_stability.png")
        print("  📊 images/anomaly_timeline.png")
        
    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        raise

if __name__ == "__main__":
    main()
