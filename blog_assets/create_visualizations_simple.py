#!/usr/bin/env python3
"""
RLDK Visualization Script - Simplified Version
Creates visualizations from actual RLDK monitoring data using only built-in libraries
"""

import json
import math
from pathlib import Path

def load_data():
    """Load the actual data files"""
    data = {}
    
    # Load comprehensive analysis (main forensic data)
    try:
        with open('/workspace/comprehensive_ppo_forensics_demo/comprehensive_analysis.json', 'r') as f:
            data['forensics'] = json.load(f)
    except FileNotFoundError:
        print("Warning: comprehensive_analysis.json not found")
        data['forensics'] = None
    
    # Load monitoring metrics (step-by-step data)
    try:
        with open('/workspace/comprehensive_ppo_monitor_demo/comprehensive_demo_run_comprehensive_metrics.json', 'r') as f:
            data['metrics'] = json.load(f)
    except FileNotFoundError:
        print("Warning: comprehensive_metrics.json not found")
        data['metrics'] = None
    
    # Load enhanced scan results
    try:
        with open('/workspace/enhanced_ppo_scan_demo/enhanced_scan_results.json', 'r') as f:
            data['scan'] = json.load(f)
    except FileNotFoundError:
        print("Warning: enhanced_scan_results.json not found")
        data['scan'] = None
    
    return data

def create_data_summary(data):
    """Create a text summary of the data for the blog post"""
    summary = []
    summary.append("# RLDK Data Summary")
    summary.append("")
    
    if data['forensics']:
        summary.append("## Forensic Analysis Results")
        summary.append(f"- Overall Health Score: {data['forensics']['overall_health_score']:.3f}")
        summary.append(f"- Training Stability Score: {data['forensics']['training_stability_score']:.3f}")
        summary.append(f"- Convergence Quality Score: {data['forensics']['convergence_quality_score']:.3f}")
        summary.append(f"- Total Steps: {data['forensics']['total_steps']}")
        summary.append("")
        
        summary.append("### Anomalies Detected")
        for i, anomaly in enumerate(data['forensics']['anomalies'], 1):
            summary.append(f"{i}. **{anomaly['type'].replace('_', ' ').title()}** ({anomaly['severity']})")
            summary.append(f"   - Message: {anomaly['message']}")
            summary.append(f"   - Value: {anomaly['value']:.3f} (threshold: {anomaly['threshold']})")
            summary.append("")
        
        if 'trackers' in data['forensics']:
            summary.append("### Tracker Details")
            
            # KL Schedule
            if 'kl_schedule' in data['forensics']['trackers']:
                kl_tracker = data['forensics']['trackers']['kl_schedule']
                summary.append("#### KL Schedule Tracker")
                summary.append(f"- Current KL: {kl_tracker['current_kl']:.3f}")
                summary.append(f"- KL Target: {kl_tracker['kl_target']}")
                summary.append(f"- KL Health Score: {kl_tracker['kl_health_score']:.3f}")
                summary.append(f"- Schedule Health Score: {kl_tracker['schedule_health_score']:.3f}")
                summary.append(f"- Time in Target Range: {kl_tracker['time_in_target_range']:.1%}")
                summary.append(f"- Target Range Violations: {kl_tracker['target_range_violations']}")
                summary.append("")
            
            # Gradient Norms
            if 'gradient_norms' in data['forensics']['trackers']:
                grad_tracker = data['forensics']['trackers']['gradient_norms']
                summary.append("#### Gradient Norms Tracker")
                summary.append(f"- Policy Gradient Norm: {grad_tracker['current_policy_grad_norm']:.3f}")
                summary.append(f"- Value Gradient Norm: {grad_tracker['current_value_grad_norm']:.3f}")
                summary.append(f"- Total Gradient Norm: {grad_tracker['current_total_grad_norm']:.3f}")
                summary.append(f"- Policy/Value Ratio: {grad_tracker['current_policy_value_ratio']:.3f}")
                summary.append(f"- Gradient Health Score: {grad_tracker['gradient_health_score']:.3f}")
                summary.append(f"- Training Stability: {grad_tracker['training_stability']:.3f}")
                summary.append("")
            
            # Advantage Statistics
            if 'advantage_statistics' in data['forensics']['trackers']:
                adv_tracker = data['forensics']['trackers']['advantage_statistics']
                summary.append("#### Advantage Statistics Tracker")
                summary.append(f"- Advantage Mean: {adv_tracker['current_advantage_mean']:.3f}")
                summary.append(f"- Advantage Std: {adv_tracker['current_advantage_std']:.3f}")
                summary.append(f"- Advantage Bias: {adv_tracker['advantage_bias']:.3f}")
                summary.append(f"- Advantage Health Score: {adv_tracker['advantage_health_score']:.3f}")
                summary.append(f"- Quality Score: {adv_tracker['advantage_quality_score']:.3f}")
                summary.append("")
    
    if data['metrics']:
        summary.append("## Training Metrics Summary")
        summary.append(f"- Total Steps Recorded: {len(data['metrics'])}")
        
        # Find min/max values
        kl_values = [step['kl'] for step in data['metrics']]
        kl_coef_values = [step['kl_coef'] for step in data['metrics']]
        
        summary.append(f"- KL Divergence Range: {min(kl_values):.3f} - {max(kl_values):.3f}")
        summary.append(f"- KL Coefficient Range: {min(kl_coef_values):.3f} - {max(kl_coef_values):.3f}")
        summary.append("")
        
        # Show progression
        summary.append("### KL Progression (First 10 Steps)")
        for i, step in enumerate(data['metrics'][:10]):
            summary.append(f"Step {step['step']}: KL={step['kl']:.3f}, Coef={step['kl_coef']:.3f}")
        summary.append("")
        
        summary.append("### KL Progression (Last 10 Steps)")
        for step in data['metrics'][-10:]:
            summary.append(f"Step {step['step']}: KL={step['kl']:.3f}, Coef={step['kl_coef']:.3f}")
        summary.append("")
    
    if data['scan']:
        summary.append("## Enhanced Scan Results")
        summary.append(f"- Overall Health Score: {data['scan']['comprehensive_analysis']['overall_health_score']:.3f}")
        summary.append(f"- Training Stability Score: {data['scan']['comprehensive_analysis']['training_stability_score']:.3f}")
        summary.append(f"- Convergence Quality Score: {data['scan']['comprehensive_analysis']['convergence_quality_score']:.3f}")
        summary.append(f"- Total Steps: {data['scan']['comprehensive_analysis']['total_steps']}")
        summary.append("")
        
        summary.append("### Rules Fired")
        for rule in data['scan']['rules_fired']:
            summary.append(f"- **{rule['rule']}**: {rule['description']}")
            summary.append(f"  - Step Range: {rule['step_range'][0]}-{rule['step_range'][1]}")
        summary.append("")
    
    return "\n".join(summary)

def create_simple_charts(data):
    """Create simple ASCII charts for visualization"""
    charts = []
    charts.append("# RLDK Data Visualization")
    charts.append("")
    
    if data['forensics']:
        charts.append("## Health Scores")
        charts.append("")
        
        # Overall health scores
        overall = data['forensics']['overall_health_score']
        stability = data['forensics']['training_stability_score']
        convergence = data['forensics']['convergence_quality_score']
        
        charts.append("Overall Health Score:")
        charts.append("█" * int(overall * 20) + "░" * int((1 - overall) * 20) + f" {overall:.3f}")
        charts.append("")
        
        charts.append("Training Stability Score:")
        charts.append("█" * int(stability * 20) + "░" * int((1 - stability) * 20) + f" {stability:.3f}")
        charts.append("")
        
        charts.append("Convergence Quality Score:")
        charts.append("█" * int(convergence * 20) + "░" * int((1 - convergence) * 20) + f" {convergence:.3f}")
        charts.append("")
        
        # KL progression if we have metrics
        if data['metrics']:
            charts.append("## KL Divergence Progression")
            charts.append("")
            
            # Sample every 5th step for readability
            sample_steps = data['metrics'][::5]
            
            charts.append("Step  | KL Divergence | KL Coefficient")
            charts.append("------|---------------|---------------")
            
            for step in sample_steps[:10]:  # Show first 10 sampled steps
                kl_bar = "█" * int(step['kl'] * 50) + "░" * int((0.2 - step['kl']) * 50)
                charts.append(f"{step['step']:4d}  | {kl_bar[:13]:13s} | {step['kl_coef']:.3f}")
            
            charts.append("")
            charts.append("(█ = KL value, ░ = remaining to 0.2 threshold)")
            charts.append("")
    
    return "\n".join(charts)

def main():
    """Main function to create data summaries and visualizations"""
    print("Creating RLDK data summaries...")
    
    # Create blog_assets directory
    Path('/workspace/blog_assets').mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_data()
    
    # Create data summary
    summary = create_data_summary(data)
    with open('/workspace/blog_assets/data_summary.md', 'w') as f:
        f.write(summary)
    
    # Create simple charts
    charts = create_simple_charts(data)
    with open('/workspace/blog_assets/data_visualization.md', 'w') as f:
        f.write(charts)
    
    print("Data summaries created successfully!")
    print("- /workspace/blog_assets/data_summary.md")
    print("- /workspace/blog_assets/data_visualization.md")
    
    # Print key statistics
    if data['forensics']:
        print("\nKey Statistics:")
        print(f"Overall Health Score: {data['forensics']['overall_health_score']:.3f}")
        print(f"Training Stability Score: {data['forensics']['training_stability_score']:.3f}")
        print(f"Convergence Quality Score: {data['forensics']['convergence_quality_score']:.3f}")
        print(f"Total Steps: {data['forensics']['total_steps']}")
        print(f"Anomalies Detected: {len(data['forensics']['anomalies'])}")

if __name__ == "__main__":
    main()