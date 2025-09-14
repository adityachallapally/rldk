#!/usr/bin/env python3
"""
Demonstration of the Seeded Replay Utility

This script shows how to use the replay functionality to verify
training run reproducibility. It creates mock training data and
demonstrates the replay workflow.
"""

import json
import os
import tempfile
from pathlib import Path


# Mock implementation for demonstration purposes
# In a real environment, you would import from rldk.replay
class MockReplayReport:
    """Mock replay report for demonstration."""

    def __init__(
        self,
        passed,
        original_seed,
        replay_seed,
        metrics_compared,
        tolerance,
        mismatches,
        comparison_stats,
        replay_command,
        replay_duration,
    ):
        self.passed = passed
        self.original_seed = original_seed
        self.replay_seed = replay_seed
        self.metrics_compared = metrics_compared
        self.tolerance = tolerance
        self.mismatches = mismatches
        self.comparison_stats = comparison_stats
        self.replay_command = replay_command
        self.replay_duration = replay_duration


def create_mock_training_run(seed=42, num_steps=10):
    """Create mock training run data."""
    run_data = []

    for step in range(num_steps):
        # Generate deterministic metrics based on seed and step
        reward_mean = 0.5 + 0.1 * step + 0.01 * seed
        reward_std = 0.1 + 0.01 * step
        kl_mean = 0.1 + 0.01 * step
        entropy_mean = 0.8 - 0.01 * step

        metrics = {
            "step": step,
            "phase": "train",
            "reward_mean": round(reward_mean, 4),
            "reward_std": round(reward_std, 4),
            "kl_mean": round(kl_mean, 4),
            "entropy_mean": round(entropy_mean, 4),
            "clip_frac": 0.1,
            "grad_norm": 1.0,
            "lr": 0.001,
            "loss": 0.5,
            "tokens_in": 512,
            "tokens_out": 128,
            "wall_time": 100.0 + step,
            "seed": seed,
            "run_id": f"mock_run_{seed}",
            "git_sha": "demo123",
        }
        run_data.append(metrics)

    return run_data


def save_training_run(run_data, filepath):
    """Save training run data to JSONL file."""
    with open(filepath, "w") as f:
        for metrics in run_data:
            f.write(json.dumps(metrics) + "\n")
    print(f"Saved training run data to: {filepath}")


def demonstrate_replay_workflow():
    """Demonstrate the complete replay workflow."""
    print("🧪 Seeded Replay Utility Demonstration")
    print("=" * 50)

    # Step 1: Create original training run
    print("\n1️⃣ Creating original training run...")
    original_seed = 42
    original_run = create_mock_training_run(seed=original_seed, num_steps=10)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        original_file = f.name

    save_training_run(original_run, original_file)

    # Step 2: Simulate replay execution
    print("\n2️⃣ Simulating replay execution...")

    # In a real scenario, this would run the actual training command
    # For demo purposes, we'll create "replay" data with slight variations
    replay_run = create_mock_training_run(seed=original_seed, num_steps=10)

    # Introduce small variations to simulate real-world conditions
    import random

    random.seed(123)  # Fixed seed for demo reproducibility

    for i, metrics in enumerate(replay_run):
        # Add small random variations (within tolerance)
        variation = random.uniform(-0.001, 0.001)
        metrics["reward_mean"] = round(metrics["reward_mean"] + variation, 4)

    # Save replay data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        replay_file = f.name

    save_training_run(replay_run, replay_file)

    # Step 3: Compare metrics
    print("\n3️⃣ Comparing metrics...")

    # Load both datasets
    with open(original_file) as f:
        original_data = [json.loads(line) for line in f]

    with open(replay_file) as f:
        replay_data = [json.loads(line) for line in f]

    # Compare metrics
    metrics_to_compare = ["reward_mean", "kl_mean", "entropy_mean"]
    tolerance = 0.01
    mismatches = []
    comparison_stats = {}

    for metric in metrics_to_compare:
        metric_mismatches = []
        metric_stats = {
            "max_diff": 0.0,
            "max_diff_step": None,
            "mean_diff": 0.0,
            "tolerance_violations": 0,
        }

        diffs = []

        for step in range(len(original_data)):
            orig_val = original_data[step][metric]
            replay_val = replay_data[step][metric]

            # Calculate relative difference
            if abs(orig_val) > 1e-8:
                rel_diff = abs(replay_val - orig_val) / abs(orig_val)
            else:
                rel_diff = abs(replay_val - orig_val)

            diffs.append(rel_diff)

            # Check tolerance
            if rel_diff > tolerance:
                metric_stats["tolerance_violations"] += 1
                metric_mismatches.append(
                    {
                        "step": step,
                        "original": orig_val,
                        "replay": replay_val,
                        "relative_diff": rel_diff,
                        "tolerance": tolerance,
                    }
                )

        if diffs:
            metric_stats["max_diff"] = max(diffs)
            metric_stats["max_diff_step"] = diffs.index(max(diffs))
            metric_stats["mean_diff"] = sum(diffs) / len(diffs)

        comparison_stats[metric] = metric_stats

        if metric_mismatches:
            mismatches.extend(metric_mismatches)
            print(f"  {metric}: {len(metric_mismatches)} tolerance violations")
        else:
            print(f"  {metric}: ✅ within tolerance")

    # Step 4: Generate replay report
    print("\n4️⃣ Generating replay report...")

    passed = len(mismatches) == 0
    replay_duration = 15.5  # Mock duration

    report = MockReplayReport(
        passed=passed,
        original_seed=original_seed,
        replay_seed=original_seed,
        metrics_compared=metrics_to_compare,
        tolerance=tolerance,
        mismatches=mismatches,
        comparison_stats=comparison_stats,
        replay_command=f"python train.py --seed {original_seed}",
        replay_duration=replay_duration,
    )

    # Step 5: Display results
    print("\n5️⃣ Replay Results")
    print("-" * 30)

    if report.passed:
        print("✅ Seeded replay PASSED - metrics match within tolerance")
    else:
        print(
            f"🚨 Seeded replay FAILED - {len(report.mismatches)} tolerance violations"
        )

    print(f"Original seed: {report.original_seed}")
    print(f"Replay seed: {report.replay_seed}")
    print(f"Tolerance: {report.tolerance}")
    print(f"Replay duration: {report.replay_duration:.2f} seconds")
    print(f"Command used: {report.replay_command}")

    # Show detailed comparison stats
    print("\n📊 Metric Comparison Statistics:")
    for metric, stats in comparison_stats.items():
        violations = stats["tolerance_violations"]
        max_diff = stats["max_diff"]
        mean_diff = stats["mean_diff"]

        status = "✅" if violations == 0 else "🚨"
        print(f"  {status} {metric}:")
        print(f"    Max difference: {max_diff:.6f}")
        print(f"    Mean difference: {mean_diff:.6f}")
        print(f"    Tolerance violations: {violations}")

    # Step 6: Save results
    print("\n6️⃣ Saving replay results...")

    output_dir = Path("replay_demo_results")
    output_dir.mkdir(exist_ok=True)

    # Save replay metrics
    replay_output = output_dir / "replay_metrics.jsonl"
    save_training_run(replay_data, replay_output)

    # Save comparison report
    comparison_output = output_dir / "replay_comparison.json"
    comparison_data = {
        "passed": report.passed,
        "original_seed": report.original_seed,
        "replay_seed": report.replay_seed,
        "metrics_compared": report.metrics_compared,
        "tolerance": report.tolerance,
        "tolerance_violations": len(report.mismatches),
        "comparison_stats": report.comparison_stats,
        "replay_command": report.replay_command,
        "replay_duration": report.replay_duration,
    }

    with open(comparison_output, "w") as f:
        json.dump(comparison_data, f, indent=2)

    # Save mismatches if any
    if report.mismatches:
        mismatches_output = output_dir / "replay_mismatches.json"
        with open(mismatches_output, "w") as f:
            json.dump(report.mismatches, f, indent=2)

    print(f"Results saved to: {output_dir}/")

    # Cleanup temporary files
    try:
        os.unlink(original_file)
        os.unlink(replay_file)
    except OSError as e:
        # Log cleanup errors but don't fail the demo
        print(f"Warning: Could not clean up temporary files: {e}")
        pass

    print("\n🎉 Demonstration completed!")
    print("\nTo use the real replay utility:")
    print("1. Install RL Debug Kit: pip install rldk")
    print(
        "2. Use CLI: rldk replay --run <run_file> --command <training_cmd> --metrics <metrics>"
    )
    print("3. Or use Python API: from rldk.replay import replay")


if __name__ == "__main__":
    demonstrate_replay_workflow()
