#!/usr/bin/env python3
"""
Create visualizations for RLDK blog post from real demo data.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import os

np.random.seed(42)

plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_run_data():
    """Load training run data from run.jsonl"""
    data = []
    with open('artifacts/run.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return pd.DataFrame(data)

def load_alerts_data():
    """Load alerts data from alerts.jsonl"""
    alerts = []
    with open('artifacts/alerts.jsonl', 'r') as f:
        for line in f:
            alerts.append(json.loads(line.strip()))
    return pd.DataFrame(alerts)

def load_forensics_data():
    """Load comprehensive forensics analysis"""
    with open('comprehensive_ppo_forensics_demo/comprehensive_analysis.json', 'r') as f:
        return json.load(f)

def create_kl_spike_detection_plot():
    """Create the main KL spike detection visualization"""
    run_df = load_run_data()
    alerts_df = load_alerts_data()
    
    kl_data = run_df[run_df['metric'] == 'kl'].copy()
    kl_data = kl_data.sort_values('step')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(kl_data['step'], kl_data['value'], 'b-', linewidth=2, label='KL Divergence', alpha=0.8)
    
    ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold (0.4)')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Critical Threshold (0.8)')
    
    for _, alert in alerts_df.iterrows():
        if alert['action'] == 'warn':
            ax.scatter(alert['step'], alert['kl'], color='orange', s=100, marker='!', 
                      label='Warning Alert' if _ == 0 else "", zorder=5)
        elif alert['action'] == 'stop':
            ax.scatter(alert['step'], alert['kl'], color='red', s=150, marker='X', 
                      label='Stop Alert' if alert['step'] == 44 else "", zorder=5)
    
    ax.annotate('Training Stopped\nKL=0.937', xy=(44, 0.937), xytext=(50, 0.85),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, ha='left', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel('KL Divergence', fontsize=14)
    ax.set_title('RLDK Real-Time KL Spike Detection\nAutomatic Training Termination at Step 44', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('images/kl_spike_detection.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_health_scores_dashboard():
    """Create health scores dashboard from forensics data"""
    forensics = load_forensics_data()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    health_metrics = {
        'Overall Health': forensics['overall_health_score'],
        'Training Stability': forensics['training_stability_score'],
        'Convergence Quality': forensics['convergence_quality_score']
    }
    
    colors = ['#2E8B57' if v > 0.8 else '#FF6B35' if v < 0.6 else '#FFB347' for v in health_metrics.values()]
    bars1 = ax1.bar(health_metrics.keys(), health_metrics.values(), color=colors, alpha=0.8)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Health Score', fontsize=12)
    ax1.set_title('Overall Training Health Metrics', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    for bar, value in zip(bars1, health_metrics.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    kl_data = forensics['trackers']['kl_schedule']
    kl_metrics = {
        'KL Health': kl_data['kl_health_score'],
        'Schedule Health': kl_data['schedule_health_score'],
        'Time in Target': kl_data['time_in_target_range']
    }
    
    colors2 = ['#2E8B57' if v > 0.8 else '#FF6B35' if v < 0.6 else '#FFB347' for v in kl_metrics.values()]
    bars2 = ax2.bar(kl_metrics.keys(), kl_metrics.values(), color=colors2, alpha=0.8)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('KL Schedule Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    for bar, value in zip(bars2, kl_metrics.values()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    grad_data = forensics['trackers']['gradient_norms']
    grad_metrics = {
        'Gradient Health': grad_data['gradient_health_score'],
        'Training Stability': grad_data['training_stability'],
        'Gradient Balance': grad_data['gradient_balance']
    }
    
    colors3 = ['#2E8B57' if v > 0.8 else '#FF6B35' if v < 0.6 else '#FFB347' for v in grad_metrics.values()]
    bars3 = ax3.bar(grad_metrics.keys(), grad_metrics.values(), color=colors3, alpha=0.8)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Gradient Norms Analysis', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    for bar, value in zip(bars3, grad_metrics.values()):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    anomaly_counts = {}
    for anomaly in forensics['anomalies']:
        tracker = anomaly['tracker']
        if tracker not in anomaly_counts:
            anomaly_counts[tracker] = 0
        anomaly_counts[tracker] += 1
    
    if anomaly_counts:
        bars4 = ax4.bar(anomaly_counts.keys(), anomaly_counts.values(), 
                       color=['#FF6B35', '#FFB347', '#87CEEB'], alpha=0.8)
        ax4.set_ylabel('Number of Anomalies', fontsize=12)
        ax4.set_title('Anomalies Detected by Tracker', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        for bar, value in zip(bars4, anomaly_counts.values()):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('images/health_scores_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_training_metrics_plot():
    """Create comprehensive training metrics visualization"""
    run_df = load_run_data()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    kl_data = run_df[run_df['metric'] == 'kl'].sort_values('step')
    ax1.plot(kl_data['step'], kl_data['value'], 'b-', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('KL Divergence')
    ax1.set_title('KL Divergence Progression', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    reward_data = run_df[run_df['metric'] == 'reward'].sort_values('step')
    ax2.plot(reward_data['step'], reward_data['value'], 'g-', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Reward')
    ax2.set_title('Reward Progression', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    grad_data = run_df[run_df['metric'] == 'gradient_norm'].sort_values('step')
    ax3.plot(grad_data['step'], grad_data['value'], 'r-', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Gradient Norm')
    ax3.set_title('Gradient Norm Progression', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    ax4_twin = ax4.twinx()
    
    kl_norm = kl_data['value'] / kl_data['value'].max()
    reward_norm = (reward_data['value'] - reward_data['value'].min()) / (reward_data['value'].max() - reward_data['value'].min())
    grad_norm = grad_data['value'] / grad_data['value'].max()
    
    ax4.plot(kl_data['step'], kl_norm, 'b-', linewidth=2, alpha=0.8, label='KL (normalized)')
    ax4.plot(reward_data['step'], reward_norm, 'g-', linewidth=2, alpha=0.8, label='Reward (normalized)')
    ax4.plot(grad_data['step'], grad_norm, 'r-', linewidth=2, alpha=0.8, label='Gradient (normalized)')
    
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Normalized Values')
    ax4.set_title('Combined Training Metrics (Normalized)', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_anomaly_timeline():
    """Create anomaly detection timeline"""
    forensics = load_forensics_data()
    alerts_df = load_alerts_data()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    anomaly_types = {}
    for i, anomaly in enumerate(forensics['anomalies']):
        atype = anomaly['type'].replace('_anomaly', '').replace('_', ' ').title()
        severity = anomaly['severity']
        
        if atype not in anomaly_types:
            anomaly_types[atype] = []
        
        color = 'red' if severity == 'critical' else 'orange'
        marker = 'X' if severity == 'critical' else '!'
        
        ax.scatter(i*10, len(anomaly_types), color=color, s=150, marker=marker, alpha=0.8)
        
        ax.text(i*10, len(anomaly_types) + 0.1, f"{atype}\n{anomaly['value']:.3f}", 
               ha='center', va='bottom', fontsize=9, rotation=45)
    
    for _, alert in alerts_df.iterrows():
        y_pos = len(anomaly_types) + 1
        color = 'red' if alert['action'] == 'stop' else 'orange'
        marker = 'X' if alert['action'] == 'stop' else '!'
        
        ax.scatter(alert['step'], y_pos, color=color, s=150, marker=marker, alpha=0.8)
        ax.text(alert['step'], y_pos + 0.1, f"KL Alert\n{alert['kl']:.3f}", 
               ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Step / Analysis Point', fontsize=14)
    ax.set_ylabel('Anomaly Type', fontsize=14)
    ax.set_title('RLDK Anomaly Detection Timeline\nReal-time Alerts + Forensic Analysis', fontsize=16, fontweight='bold')
    
    ax.scatter([], [], color='red', s=150, marker='X', label='Critical/Stop')
    ax.scatter([], [], color='orange', s=150, marker='!', label='Warning')
    ax.legend(loc='upper right')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/anomaly_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all visualizations"""
    os.makedirs('images', exist_ok=True)
    
    print("Creating KL spike detection plot...")
    create_kl_spike_detection_plot()
    
    print("Creating health scores dashboard...")
    create_health_scores_dashboard()
    
    print("Creating training metrics plot...")
    create_training_metrics_plot()
    
    print("Creating anomaly timeline...")
    create_anomaly_timeline()
    
    print("All visualizations created successfully!")
    print("Generated files:")
    for img in ['kl_spike_detection.png', 'health_scores_dashboard.png', 
                'training_metrics.png', 'anomaly_timeline.png']:
        print(f"  - images/{img}")

if __name__ == "__main__":
    main()
