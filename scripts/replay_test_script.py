#!/usr/bin/env python3
"""Test script for the seeded replay functionality."""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rldk.replay import replay


def create_mock_training_script():
    """Create a mock training script that outputs metrics."""
    script_content = """#!/usr/bin/env python3
import json
import random
import numpy as np
import os
import sys

# Set seed from command line argument
seed = 42
if len(sys.argv) > 2 and sys.argv[1] == '--seed':
    seed = int(sys.argv[2])

# Set deterministic seeds
random.seed(seed)
np.random.seed(seed)

# Get output path from environment
output_path = os.environ.get('RLDK_METRICS_PATH', 'metrics.jsonl')

# Simulate training steps
for step in range(10):
    # Generate deterministic metrics based on seed and step
    reward_mean = 0.5 + 0.1 * step + 0.01 * seed
    reward_std = 0.1 + 0.01 * step
    kl_mean = 0.1 + 0.01 * step
    entropy_mean = 0.8 - 0.01 * step

    metrics = {
        "step": step,
        "phase": "train",
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "kl_mean": kl_mean,
        "entropy_mean": entropy_mean,
        "clip_frac": 0.1,
        "grad_norm": 1.0,
        "lr": 0.001,
        "loss": 0.5,
        "tokens_in": 512,
        "tokens_out": 128,
        "wall_time": 100.0 + step,
        "seed": seed,
        "run_id": f"mock_run_{seed}",
        "git_sha": "test123"
    }

    # Write to output file
    with open(output_path, 'a') as f:
        f.write(json.dumps(metrics) + '\\n')

    print(f"Step {step}: reward={reward_mean:.4f}, kl={kl_mean:.4f}")

print(f"Training completed with seed {seed}")
"""

    script_path = Path("mock_training.py")
    with open(script_path, "w") as f:
        f.write(script_content)

    # Make executable
    os.chmod(script_path, 0o755)
    return script_path


def create_mock_run_data():
    """Create mock training run data for testing."""
    run_data = []
    seed = 42

    for step in range(10):
        reward_mean = 0.5 + 0.1 * step + 0.01 * seed
        reward_std = 0.1 + 0.01 * step
        kl_mean = 0.1 + 0.01 * step
        entropy_mean = 0.8 - 0.01 * step

        metrics = {
            "step": step,
            "phase": "train",
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "kl_mean": kl_mean,
            "entropy_mean": entropy_mean,
            "clip_frac": 0.1,
            "grad_norm": 1.0,
            "lr": 0.001,
            "loss": 0.5,
            "tokens_in": 512,
            "tokens_out": 128,
            "wall_time": 100.0 + step,
            "seed": seed,
            "run_id": f"mock_run_{seed}",
            "git_sha": "test123",
        }
        run_data.append(metrics)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for metrics in run_data:
            f.write(json.dumps(metrics) + "\n")
        run_file = f.name

    return run_file


def main():
    """Test the replay functionality."""
    print("🧪 Testing Seeded Replay Functionality")
    print("=" * 50)

    # Create mock training script
    print("Creating mock training script...")
    script_path = create_mock_training_script()

    # Create mock run data
    print("Creating mock run data...")
    run_file = create_mock_run_data()

    try:
        # Test replay
        print("\nRunning replay test...")
        replay_report = replay(
            run_path=run_file,
            training_command=f"python {script_path}",
            metrics_to_compare=["reward_mean", "reward_std", "kl_mean", "entropy_mean"],
            tolerance=0.001,  # Very strict tolerance for deterministic test
            max_steps=5,  # Limit to 5 steps for faster testing
            output_dir="test_replay_results",
        )

        # Display results
        print("\n📊 Replay Test Results")
        print(f"Passed: {replay_report.passed}")
        print(f"Original seed: {replay_report.original_seed}")
        print(f"Replay seed: {replay_report.replay_seed}")
        print(f"Duration: {replay_report.replay_duration:.2f} seconds")
        print(f"Tolerance violations: {len(replay_report.mismatches)}")

        if replay_report.mismatches:
            print("\n🚨 Tolerance violations found:")
            for mismatch in replay_report.mismatches[:3]:  # Show first 3
                print(
                    f"  Step {mismatch['step']}: {mismatch['relative_diff']:.6f} > {mismatch['tolerance']}"
                )
        else:
            print("\n✅ All metrics within tolerance - replay successful!")

        print("\nResults saved to: test_replay_results/")

    except Exception as e:
        print(f"❌ Replay test failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        print("\nCleaning up...")
        try:
            os.unlink(script_path)
            os.unlink(run_file)
        except OSError as e:
            # Log cleanup errors but don't fail the test
            print(f"Warning: Could not clean up test files: {e}")
            pass


if __name__ == "__main__":
    main()
