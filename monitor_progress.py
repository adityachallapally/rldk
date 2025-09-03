#!/usr/bin/env python3
"""
Monitor training progress and provide status updates.
"""

import os
import time
import json
from datetime import datetime

def check_file_exists(filepath):
    """Check if a file exists and return its size."""
    if os.path.exists(filepath):
        return os.path.getsize(filepath)
    return 0

def monitor_progress():
    """Monitor the training pipeline progress."""
    print("RLHF Training Pipeline Progress Monitor")
    print("=" * 50)
    
    # Check data preparation
    data_files = [
        "./rldk_demos/rm_pairs_train.jsonl",
        "./rldk_demos/rm_pairs_val.jsonl", 
        "./rldk_demos/ppo_prompts.jsonl",
        "./rldk_demos/probes.jsonl"
    ]
    
    print("1. Data Preparation:")
    for file in data_files:
        size = check_file_exists(file)
        status = "✓" if size > 0 else "✗"
        print(f"   {status} {os.path.basename(file)} ({size} bytes)")
    
    # Check reward model training
    print("\n2. Reward Model Training:")
    rm_files = [
        "./rldk_demos/rm_a/pytorch_model.bin",
        "./rldk_demos/rm_a/config.json",
        "./rldk_demos/rm_a/tokenizer.json",
        "./rldk_demos/rm_a/metadata.json"
    ]
    
    for file in rm_files:
        size = check_file_exists(file)
        status = "✓" if size > 0 else "✗"
        print(f"   {status} {os.path.basename(file)} ({size} bytes)")
    
    # Check PPO training runs
    ppo_runs = ["a", "b", "c", "d"]
    print("\n3. PPO Training Runs:")
    for run in ppo_runs:
        run_dir = f"./rldk_demos/ppo_{run}"
        metrics_file = os.path.join(run_dir, "metrics.jsonl")
        metadata_file = os.path.join(run_dir, "metadata.json")
        
        metrics_size = check_file_exists(metrics_file)
        metadata_size = check_file_exists(metadata_file)
        
        if metrics_size > 0 and metadata_size > 0:
            status = "✓"
            # Try to get step count
            try:
                with open(metrics_file, 'r') as f:
                    lines = f.readlines()
                    step_count = len(lines)
                    status += f" ({step_count} steps)"
            except:
                pass
        else:
            status = "✗"
        
        print(f"   {status} PPO Run {run.upper()}")
    
    # Check RLDK reports
    print("\n4. RLDK Validation Reports:")
    report_files = [
        "./rldk_demos/reports/determinism_a_b.json",
        "./rldk_demos/reports/reward_drift_a_c.json", 
        "./rldk_demos/reports/reward_health_d.json",
        "./rldk_demos/reports/calibration_rm_a.json"
    ]
    
    for file in report_files:
        size = check_file_exists(file)
        status = "✓" if size > 0 else "✗"
        print(f"   {status} {os.path.basename(file)} ({size} bytes)")
    
    # Check final report
    print("\n5. Final Report:")
    report_size = check_file_exists("./rldk_demos/report.md")
    status = "✓" if report_size > 0 else "✗"
    print(f"   {status} report.md ({report_size} bytes)")
    
    print(f"\nLast updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    monitor_progress()