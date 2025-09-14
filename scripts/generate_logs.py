#!/usr/bin/env python3
"""
Generate training log files for RLDK demo.
Creates realistic PPO training logs with clean and doctored data.
"""

import json
import os
from pathlib import Path

import numpy as np


def generate_clean_logs(output_path, num_steps=1000):
    """Generate clean PPO training logs with steady KL around 0.05."""
    print(f"Generating clean logs with {num_steps} steps...")

    # Set random seed for reproducibility
    np.random.seed(42)

    logs = []
    base_kl = 0.05
    base_entropy = 0.8

    for step in range(1, num_steps + 1):
        # Add natural variations to KL
        kl_variation = np.random.normal(0, 0.01)
        kl = max(0.01, base_kl + kl_variation)

        # KL coefficient adapts to keep KL near target
        kl_coef = max(0.1, min(2.0, 0.5 / kl))

        # Entropy decreases slowly over training
        entropy = base_entropy * (1 - 0.0001 * step)

        # Advantage statistics
        advantage_mean = np.random.normal(0.1, 0.05)
        advantage_std = np.random.uniform(0.8, 1.2)

        # Gradient norms
        grad_norm_policy = np.random.uniform(0.5, 2.0)
        grad_norm_value = np.random.uniform(0.3, 1.5)

        log_entry = {
            "step": step,
            "kl": round(kl, 6),
            "kl_coef": round(kl_coef, 6),
            "entropy": round(entropy, 6),
            "advantage_mean": round(advantage_mean, 6),
            "advantage_std": round(advantage_std, 6),
            "grad_norm_policy": round(grad_norm_policy, 6),
            "grad_norm_value": round(grad_norm_value, 6)
        }

        logs.append(log_entry)

        if step % 100 == 0:
            print(f"  Generated {step}/{num_steps} clean log entries")

    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for log in logs:
            f.write(json.dumps(log) + '\n')

    print(f"Clean logs written to {output_path}")

def generate_doctored_logs(output_path, num_steps=1000):
    """Generate doctored PPO training logs with KL spike at step 800."""
    print(f"Generating doctored logs with {num_steps} steps...")

    # Set random seed for reproducibility
    np.random.seed(42)

    logs = []
    base_kl = 0.05
    base_entropy = 0.8

    for step in range(1, num_steps + 1):
        # Add natural variations to KL
        kl_variation = np.random.normal(0, 0.01)
        kl = max(0.01, base_kl + kl_variation)

        # Introduce KL spike starting at step 800
        if step >= 800:
            spike_intensity = min(0.15, 0.05 + (step - 800) * 0.001)
            kl = max(0.01, kl + spike_intensity)

        # KL coefficient adapts but gets stuck during spike
        if step < 800:
            kl_coef = max(0.1, min(2.0, 0.5 / kl))
        else:
            # Controller gets stuck during spike
            kl_coef = max(0.1, min(2.0, 0.3 / kl))

        # Entropy decreases slowly over training
        entropy = base_entropy * (1 - 0.0001 * step)

        # Advantage statistics
        advantage_mean = np.random.normal(0.1, 0.05)
        advantage_std = np.random.uniform(0.8, 1.2)

        # Gradient norms
        grad_norm_policy = np.random.uniform(0.5, 2.0)
        grad_norm_value = np.random.uniform(0.3, 1.5)

        log_entry = {
            "step": step,
            "kl": round(kl, 6),
            "kl_coef": round(kl_coef, 6),
            "entropy": round(entropy, 6),
            "advantage_mean": round(advantage_mean, 6),
            "advantage_std": round(advantage_std, 6),
            "grad_norm_policy": round(grad_norm_policy, 6),
            "grad_norm_value": round(grad_norm_value, 6)
        }

        logs.append(log_entry)

        if step % 100 == 0:
            print(f"  Generated {step}/{num_steps} doctored log entries")

    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for log in logs:
            f.write(json.dumps(log) + '\n')

    print(f"Doctored logs written to {output_path}")

def main():
    """Main function to generate all required log files."""
    print("RLDK Demo Log Generator")
    print("=" * 50)

    # Define output paths
    clean_logs_path = "test_artifacts/logs_clean/training.jsonl"
    doctored_logs_path = "test_artifacts/logs_doctored_kl_spike/training.jsonl"

    try:
        # Generate clean logs
        generate_clean_logs(clean_logs_path)

        # Generate doctored logs
        generate_doctored_logs(doctored_logs_path)

        print("\n" + "=" * 50)
        print("Log generation completed successfully!")
        print(f"Clean logs: {clean_logs_path}")
        print(f"Doctored logs: {doctored_logs_path}")
        print("\nKey features:")
        print("- Clean logs: Steady KL around 0.05 with natural variations")
        print("- Doctored logs: KL spike to ~0.15 starting at step 800")
        print("- Both logs use deterministic random seeds for reproducibility")

    except Exception as e:
        print(f"Error generating logs: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
