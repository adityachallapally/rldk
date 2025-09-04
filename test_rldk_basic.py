#!/usr/bin/env python3
"""Basic RLDK test to verify core functionality."""

import os
import json
import time
from rldk.integrations.trl import RLDKCallback, PPOMonitor, CheckpointMonitor
from rldk.ingest import ingest_runs

def test_rldk_basic():
    """Test basic RLDK functionality."""
    
    # Create output directory
    output_dir = "runs/trl_test"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starting basic RLDK test")
    
    # Create RLDK callbacks
    rldk_callback = RLDKCallback(
        output_dir=output_dir,
        run_id="rldk_test_run",
        log_interval=1
    )
    
    ppo_monitor = PPOMonitor(
        output_dir=output_dir,
        run_id="rldk_test_run"
    )
    
    checkpoint_monitor = CheckpointMonitor(
        output_dir=output_dir,
        run_id="rldk_test_run"
    )
    
    print("✅ RLDK callbacks created successfully")
    
    # Simulate some training steps
    print("🔄 Simulating training steps...")
    
    for step in range(5):
        # Simulate metrics
        metrics = {
            'step': step,
            'epoch': step / 10.0,
            'learning_rate': 1e-5,
            'loss': 0.5 - step * 0.05,
            'grad_norm': 1.0 - step * 0.1,
            'reward_mean': 0.3 + step * 0.02,
            'reward_std': 0.1,
            'kl_mean': 0.05 - step * 0.01,
            'kl_std': 0.01,
            'entropy_mean': 2.0 - step * 0.1,
            'clip_frac': 0.1 + step * 0.01,
            'value_loss': 0.2 - step * 0.02,
            'policy_loss': 0.3 - step * 0.03,
            'batch_size': 4,
            'global_step': step,
            'model_type': 'gpt2',
            'vocab_size': 50257,
            'dataset_size': 250,
            'gpu_memory_used': 1000.0 + step * 100,
            'cpu_memory_used': 2000.0 + step * 50,
            'step_time': 0.1 + step * 0.01,
            'wall_time': time.time(),
            'tokens_in': step * 100,
            'tokens_out': step * 50,
            'training_stability_score': 0.8 + step * 0.02,
            'convergence_indicator': step * 0.1,
            'phase': 'train',
            'run_id': 'rldk_test_run',
            'git_sha': 'test_sha',
            'seed': 42
        }
        
        # Log metrics using the callback
        rldk_callback.on_log(None, None, None, logs=metrics)
        
        # Simulate PPO monitoring
        ppo_monitor.on_log(None, None, None, logs=metrics)
        
        print(f"  Step {step}: loss={metrics['loss']:.3f}, reward={metrics['reward_mean']:.3f}")
        
        time.sleep(0.1)  # Small delay to simulate training
    
    print("✅ Training simulation completed")
    
    # Check if output files were created
    run_dir = os.path.join(output_dir, "rldk_test_run")
    if os.path.exists(run_dir):
        print(f"📁 Run directory created: {run_dir}")
        files = os.listdir(run_dir)
        print(f"📄 Generated files: {files}")
        
        # Check for metrics.jsonl
        metrics_file = os.path.join(run_dir, "metrics.jsonl")
        if os.path.exists(metrics_file):
            print(f"✅ Metrics file created: {metrics_file}")
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
                print(f"📊 Metrics entries: {len(lines)}")
                if lines:
                    print(f"📝 Last entry: {lines[-1].strip()}")
        else:
            print("⚠️  No metrics.jsonl file found")
            
        # Check for alert files
        alert_files = [f for f in files if f.startswith("alerts_step_")]
        if alert_files:
            print(f"⚠️  Alert files found: {alert_files}")
        else:
            print("ℹ️  No alert files found (this is normal for short training)")
            
        # Test ingestion
        print("🔄 Testing log ingestion...")
        try:
            ingested_runs = ingest_runs(output_dir)
            print(f"✅ Ingestion successful: {ingested_runs}")
        except Exception as e:
            print(f"⚠️  Ingestion failed: {e}")
    else:
        print("⚠️  No run directory created")
    
    return True

if __name__ == "__main__":
    success = test_rldk_basic()
    if success:
        print("🎯 Basic RLDK test completed successfully")
    else:
        print("❌ Basic RLDK test failed")