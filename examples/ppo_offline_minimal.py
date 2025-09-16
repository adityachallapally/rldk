#!/usr/bin/env python3
"""Minimal offline PPO simulation with RLDK monitoring."""

import json
import os
import time
from pathlib import Path

import torch
import numpy as np
from datasets import Dataset

# Set offline mode
os.environ["HF_HUB_OFFLINE"] = "1"

# Import RLDK components
from rldk.forensics.comprehensive_ppo_forensics import ComprehensivePPOForensics


def create_synthetic_dataset():
    """Create a tiny synthetic dataset of short prompts."""
    prompts = [
        "Write a one word positive review:",
        "Say something nice:",
        "Give a positive comment:",
        "Express something good:",
        "Share something awesome:",
        "Tell me something great:",
        "Write something wonderful:",
        "Say something amazing:",
        "Give a good response:",
        "Express something positive:",
    ] * 7  # 70 total prompts
    
    return Dataset.from_dict({"prompt": prompts})


def compute_reward(text: str) -> float:
    """Simple reward function: +1 if contains positive words, else 0."""
    positive_words = ["good", "great", "awesome", "wonderful", "amazing", "excellent", "fantastic", "perfect"]
    return 1.0 if any(word in text.lower() for word in positive_words) else 0.0


def simulate_ppo_step(step: int, batch_size: int = 2):
    """Simulate a PPO training step with realistic metrics."""
    # Simulate some training dynamics
    base_kl = 0.1 + 0.02 * np.sin(step * 0.1)  # Oscillating KL
    base_reward = 0.3 + 0.1 * step / 100  # Slowly increasing reward
    base_advantage = 0.05 * np.cos(step * 0.15)  # Oscillating advantage
    
    # Add some noise
    kl_noise = np.random.normal(0, 0.01)
    reward_noise = np.random.normal(0, 0.05)
    advantage_noise = np.random.normal(0, 0.02)
    
    # Generate batch metrics
    kl_values = [max(0, base_kl + kl_noise + np.random.normal(0, 0.005)) for _ in range(batch_size)]
    reward_values = [max(0, base_reward + reward_noise + np.random.normal(0, 0.02)) for _ in range(batch_size)]
    advantage_values = [base_advantage + advantage_noise + np.random.normal(0, 0.01) for _ in range(batch_size)]
    
    # Compute statistics
    stats = {
        'ppo/policy/kl_mean': np.mean(kl_values),
        'ppo/policy/kl_std': np.std(kl_values),
        'ppo/policy/entropy': 2.0 - 0.01 * step,
        'ppo/policy/clipfrac': 0.1 + 0.02 * np.sin(step * 0.2),
        'ppo/policy/grad_norm': 0.5 + 0.1 * np.sin(step * 0.1),
        'ppo/rewards/mean': np.mean(reward_values),
        'ppo/rewards/std': np.std(reward_values),
        'ppo/rewards/min': np.min(reward_values),
        'ppo/rewards/max': np.max(reward_values),
        'ppo/val/value_loss': 0.3 - 0.01 * step,
        'ppo/val/grad_norm': 0.3 + 0.05 * np.cos(step * 0.1),
        'ppo/advantages/mean': np.mean(advantage_values),
        'ppo/advantages/std': np.std(advantage_values),
        'ppo/advantages/min': np.min(advantage_values),
        'ppo/advantages/max': np.max(advantage_values),
        'learning_rate': 1e-5,
        'grad_norm': 0.5 + 0.1 * np.sin(step * 0.1),
    }
    
    return stats


def main():
    """Main PPO simulation loop."""
    print("🚀 Starting offline PPO simulation with RLDK monitoring")
    
    # Create output directory
    output_dir = Path("artifacts/phase2_offline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    dataset = create_synthetic_dataset()
    print(f"✅ Dataset created: {len(dataset)} samples")
    
    # Initialize RLDK monitoring
    print("🔍 Setting up RLDK monitoring...")
    
    # Create comprehensive PPO forensics
    forensics = ComprehensivePPOForensics(
        kl_target=0.1,
        kl_target_tolerance=0.05,
        enable_kl_schedule_tracking=True,
        enable_gradient_norms_analysis=True,
        enable_advantage_statistics=True,
    )
    
    print("✅ RLDK monitoring initialized")
    
    # Training loop
    print("🎯 Starting training simulation...")
    
    start_time = time.time()
    step_times = []
    all_metrics = []
    
    for step in range(100):
        step_start = time.time()
        
        # Simulate PPO step
        stats = simulate_ppo_step(step, batch_size=2)
        
        # Extract metrics
        kl_mean = stats['ppo/policy/kl_mean']
        reward_mean = stats['ppo/rewards/mean']
        advantage_mean = stats['ppo/advantages/mean']
        grad_norm = stats['ppo/policy/grad_norm']
        
        # Update comprehensive forensics
        forensics.update(
            step=step,
            kl=kl_mean,
            kl_coef=1.0,
            entropy=stats['ppo/policy/entropy'],
            reward_mean=reward_mean,
            reward_std=stats['ppo/rewards/std'],
            policy_grad_norm=grad_norm,
            value_grad_norm=stats['ppo/val/grad_norm'],
            advantage_mean=advantage_mean,
            advantage_std=stats['ppo/advantages/std'],
            advantage_min=stats['ppo/advantages/min'],
            advantage_max=stats['ppo/advantages/max'],
            advantage_samples=[advantage_mean + np.random.normal(0, 0.01) for _ in range(10)],
        )
        
        # Log metrics
        metrics = {
            "step": step,
            "kl_mean": kl_mean,
            "reward_mean": reward_mean,
            "advantage_mean": advantage_mean,
            "grad_norm": grad_norm,
            "timestamp": time.time(),
        }
        all_metrics.append(metrics)
        
        # Log to JSONL
        with open(output_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")
        
        if step % 10 == 0:
            print(f"Step {step}: KL={kl_mean:.4f}, Reward={reward_mean:.4f}, "
                  f"Advantage={advantage_mean:.4f}, GradNorm={grad_norm:.4f}")
        
        step_time = time.time() - step_start
        step_times.append(step_time)
    
    # Final analysis
    total_time = time.time() - start_time
    
    # Get comprehensive analysis
    analysis = forensics.get_comprehensive_analysis()
    health_summary = forensics.get_health_summary()
    
    # Create summary
    summary = {
        "total_steps": 100,
        "total_time_seconds": total_time,
        "steps_per_second": 100 / total_time,
        "average_step_time": sum(step_times) / len(step_times),
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
        "final_kl_mean": all_metrics[-1]["kl_mean"] if all_metrics else 0.0,
        "final_reward_mean": all_metrics[-1]["reward_mean"] if all_metrics else 0.0,
        "final_advantage_mean": all_metrics[-1]["advantage_mean"] if all_metrics else 0.0,
        "comprehensive_analysis": {
            "total_steps": analysis.get("total_steps", 0),
            "overall_health_score": float(analysis.get("overall_health_score", 0.0)),
            "training_stability_score": float(analysis.get("training_stability_score", 0.0)),
            "convergence_quality_score": float(analysis.get("convergence_quality_score", 0.0)),
            "anomaly_count": len(analysis.get("anomalies", [])),
        },
        "health_summary": {
            "overall_health": float(health_summary.get("overall_health", 0.0)),
            "training_stability": float(health_summary.get("training_stability", 0.0)),
            "convergence_quality": float(health_summary.get("convergence_quality", 0.0)),
        },
    }
    
    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save comprehensive analysis (skip for now due to JSON serialization issues)
    # forensics.save_analysis(str(output_dir / "comprehensive_analysis.json"))
    
    print(f"\n🎉 Training simulation completed!")
    print(f"📊 Total time: {total_time:.2f}s")
    print(f"📊 Steps per second: {100/total_time:.2f}")
    print(f"📊 Final KL: {summary['final_kl_mean']:.4f}")
    print(f"📊 Final Reward: {summary['final_reward_mean']:.4f}")
    print(f"📊 Overall Health Score: {analysis['overall_health_score']:.3f}")
    
    # Check for anomalies
    anomalies = forensics.get_anomalies()
    if anomalies:
        print(f"🚨 Detected {len(anomalies)} anomalies:")
        for anomaly in anomalies[-3:]:  # Show last 3
            print(f"   - {anomaly['type']}: {anomaly['message']}")
    else:
        print("✅ No anomalies detected")
    
    print(f"💾 All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()