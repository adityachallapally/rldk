#!/usr/bin/env python3
"""Generate golden fixtures for testing."""

import json
import random
from pathlib import Path


def create_clean_ppo_log():
    """Create a clean PPO training log."""
    metrics = []

    for step in range(200):
        # Simulate realistic PPO training metrics
        reward_mean = 0.5 + 0.01 * step + random.uniform(-0.05, 0.05)
        reward_std = 0.1 + 0.001 * step + random.uniform(0.01, 0.02)
        kl_mean = 0.1 + 0.001 * step + random.uniform(-0.01, 0.01)
        entropy_mean = 0.8 - 0.002 * step + random.uniform(-0.02, 0.02)
        clip_frac = max(0.0, 0.1 - 0.001 * step + random.uniform(-0.01, 0.01))
        grad_norm = 1.0 + random.uniform(-0.1, 0.1)
        lr = 0.001 * (0.95 ** (step // 50))
        loss = 0.5 - 0.002 * step + random.uniform(-0.05, 0.05)

        metric = {
            "step": step,
            "phase": "train",
            "reward_mean": round(reward_mean, 4),
            "reward_std": round(reward_std, 4),
            "kl_mean": round(kl_mean, 4),
            "entropy_mean": round(entropy_mean, 4),
            "clip_frac": round(clip_frac, 4),
            "grad_norm": round(grad_norm, 4),
            "lr": round(lr, 6),
            "loss": round(loss, 4),
            "tokens_in": 512,
            "tokens_out": 128,
            "wall_time": 100 + random.uniform(-10, 10),
            "seed": 42,
            "run_id": "clean_ppo_run",
            "git_sha": "abc123def456",
        }
        metrics.append(metric)

    return metrics


def create_kl_spike_log():
    """Create a PPO log with an injected KL spike."""
    metrics = create_clean_ppo_log()

    # Inject KL spike around step 100
    spike_start, spike_end = 95, 105
    for i in range(spike_start, spike_end):
        metrics[i]["kl_mean"] *= 3.0  # 3x spike
        metrics[i]["run_id"] = "kl_spike_run"

    return metrics


def create_nondeterminism_toggle():
    """Create a log with nondeterminism toggle."""
    metrics = create_clean_ppo_log()

    # Add some nondeterministic variation
    for i in range(len(metrics)):
        if i % 10 == 0:  # Every 10th step
            metrics[i]["reward_mean"] += random.uniform(-0.1, 0.1)
            metrics[i]["kl_mean"] += random.uniform(-0.02, 0.02)
        metrics[i]["run_id"] = "nondet_run"

    return metrics


def main():
    """Generate all fixtures."""
    fixtures_dir = Path("runs_fixtures")
    fixtures_dir.mkdir(exist_ok=True)

    # Generate clean PPO log
    clean_metrics = create_clean_ppo_log()
    with open(fixtures_dir / "clean_ppo.jsonl", "w") as f:
        for metric in clean_metrics:
            json.dump(metric, f)
            f.write("\n")

    # Generate KL spike log
    spike_metrics = create_kl_spike_log()
    with open(fixtures_dir / "kl_spike.jsonl", "w") as f:
        for metric in spike_metrics:
            json.dump(metric, f)
            f.write("\n")

    # Generate nondeterminism log
    nondet_metrics = create_nondeterminism_toggle()
    with open(fixtures_dir / "nondeterminism.jsonl", "w") as f:
        for metric in nondet_metrics:
            json.dump(metric, f)
            f.write("\n")

    print(f"Generated fixtures in {fixtures_dir}:")
    print("  - clean_ppo.jsonl (baseline)")
    print("  - kl_spike.jsonl (with injected KL spike)")
    print("  - nondeterminism.jsonl (with nondeterministic variation)")

    # Create a simple test script that demonstrates the KL spike
    test_script = """#!/usr/bin/env python3
\"\"\"Test script to demonstrate KL spike detection.\"\"\"

import pandas as pd
from rldk.ingest import ingest_runs
from rldk.diff import first_divergence

# Load the fixtures
df_clean = ingest_runs('runs_fixtures/clean_ppo.jsonl')
df_spike = ingest_runs('runs_fixtures/kl_spike.jsonl')

# Find divergence
report = first_divergence(df_clean, df_spike, ['kl_mean'], k_consecutive=3, window=20)

print(f"Divergence detected: {report.diverged}")
if report.diverged:
    print(f"First divergence at step: {report.first_step}")
    print(f"Tripped signals: {report.tripped_signals}")
"""

    with open(fixtures_dir / "test_spike_detection.py", "w") as f:
        f.write(test_script)

    print("  - test_spike_detection.py (demo script)")


if __name__ == "__main__":
    main()
