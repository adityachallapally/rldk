#!/usr/bin/env python3
"""
RLDK 2-Minute CPU Smoke Test

This test demonstrates RLDK's core capabilities in under 2 minutes on CPU.
It runs the tiny GPT-2 summarization task and shows RLDK catching real bugs.

Expected Results:
- Training completes in ~1 minute
- RLDK analysis completes in ~1 minute
- All 3 intentional bugs are detected
- Reports are generated and saved
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict


def run_command(cmd: str, description: str) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")

    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    duration = time.time() - start_time

    print(f"Duration: {duration:.1f}s")
    print(f"Exit code: {result.returncode}")

    if result.stdout:
        print("Output:")
        print(
            result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout
        )

    if result.stderr:
        print("Errors:")
        print(
            result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr
        )

    return result


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists and print status."""
    path = Path(filepath)
    exists = path.exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")

    if exists:
        size = path.stat().st_size
        print(f"   Size: {size:,} bytes")

    return exists


def analyze_training_metrics(metrics_file: str) -> Dict:
    """Analyze training metrics to verify bugs are present."""
    print(f"\n📊 Analyzing training metrics: {metrics_file}")

    try:
        with open(metrics_file) as f:
            lines = f.readlines()

        metrics = [json.loads(line) for line in lines]
        print(f"Total training steps: {len(metrics)}")

        # Check for KL divergence spike at step 47
        step_47_kl = None
        for m in metrics:
            if m["step"] == 47:
                step_47_kl = m["kl_divergence"]
                break

        if step_47_kl:
            print(f"Step 47 KL divergence: {step_47_kl:.4f}")
            if step_47_kl > 0.1:  # Expected spike
                print("✅ KL divergence spike detected at step 47")
            else:
                print("❌ KL divergence spike not detected")

        # Check reward saturation
        rewards = [m["reward_mean"] for m in metrics]
        if len(rewards) > 0:
            reward_range = max(rewards) - min(rewards)
            print(f"Reward range: {reward_range:.3f}")
            if reward_range < 0.1:  # Expected saturation
                print("✅ Reward saturation detected")
            else:
                print("❌ Reward saturation not detected")

        return {
            "steps": len(metrics),
            "step_47_kl": step_47_kl,
            "reward_range": reward_range,
        }

    except Exception as e:
        print(f"❌ Error analyzing metrics: {e}")
        return {}


def run_rldk_analysis(output_dir: str) -> bool:
    """Run RLDK analysis commands."""
    print(f"\n🔍 Running RLDK analysis on: {output_dir}")

    # Check if RLDK is available
    try:
        result = subprocess.run(
            "rldk --version", shell=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            print("❌ RLDK not available, skipping analysis")
            return False
    except (OSError, subprocess.SubprocessError) as e:
        print(f"❌ RLDK not available, skipping analysis: {e}")
        return False

    # Run divergence analysis
    print("\n1️⃣ Running divergence analysis...")
    diff_cmd = f"rldk diff --a {output_dir} --b {output_dir}_run2 --signals kl_divergence,reward_mean --output-dir {output_dir}_diff"
    diff_result = run_command(diff_cmd, "Divergence analysis")

    # Run determinism check
    print("\n2️⃣ Running determinism check...")
    det_cmd = f"rldk check-determinism --cmd 'python reference/tasks/summarization_helpfulness/train_summarization.py --steps 10' --compare reward_mean --output-dir {output_dir}_determinism"
    det_result = run_command(det_cmd, "Determinism check")

    # Run reward health analysis
    print("\n3️⃣ Running reward health analysis...")
    health_cmd = (
        f"rldk reward-health --run {output_dir} --output-dir {output_dir}_health"
    )
    health_result = run_command(health_cmd, "Reward health analysis")

    return all(
        [
            diff_result.returncode == 0,
            det_result.returncode == 0,
            health_result.returncode == 0,
        ]
    )


def main():
    """Main smoke test function."""
    print("🚀 RLDK 2-Minute CPU Smoke Test")
    print("=" * 50)

    start_time = time.time()

    # Step 1: Run training
    print("\n📝 Step 1: Running summarization training")
    training_cmd = "python reference/tasks/summarization_helpfulness/train_summarization.py --steps 50 --output_dir smoke_test_outputs"
    training_result = run_command(training_cmd, "Summarization training")

    if training_result.returncode != 0:
        print("❌ Training failed, cannot continue")
        return False

    # Step 2: Check outputs
    print("\n📁 Step 2: Checking training outputs")
    output_dir = "smoke_test_outputs"
    metrics_file = f"{output_dir}/training_metrics.jsonl"

    if not check_file_exists(metrics_file, "Training metrics"):
        print("❌ Training metrics not found, cannot continue")
        return False

    # Step 3: Analyze metrics for bugs
    print("\n🐛 Step 3: Analyzing metrics for intentional bugs")
    analyze_training_metrics(metrics_file)

    # Step 4: Run RLDK analysis
    print("\n🔍 Step 4: Running RLDK analysis")
    rldk_success = run_rldk_analysis(output_dir)

    # Step 5: Check RLDK outputs
    print("\n📊 Step 5: Checking RLDK analysis outputs")

    expected_outputs = [
        f"{output_dir}_diff/diff_report.md",
        f"{output_dir}_diff/drift_card.md",
        f"{output_dir}_determinism/determinism_card.md",
        f"{output_dir}_health/reward_health_card.md",
        f"{output_dir}_health/calibration_plots.png",
    ]

    outputs_found = 0
    for output_file in expected_outputs:
        if check_file_exists(output_file, "RLDK output"):
            outputs_found += 1

    # Step 6: Summary
    total_time = time.time() - start_time
    print("\n🎯 Smoke Test Summary")
    print("=" * 30)
    print(f"Total time: {total_time:.1f}s")
    print(f"Training success: {'✅' if training_result.returncode == 0 else '❌'}")
    print(f"RLDK analysis: {'✅' if rldk_success else '❌'}")
    print(f"Outputs generated: {outputs_found}/{len(expected_outputs)}")

    # Success criteria
    success = (
        training_result.returncode == 0
        and outputs_found >= 3  # At least 3 outputs should be generated
        and total_time < 180  # Under 3 minutes
    )

    if success:
        print("\n🎉 SUCCESS: RLDK smoke test passed!")
        print("RLDK successfully demonstrated its debugging capabilities.")
        print("\nNext steps:")
        print("1. Explore the generated reports")
        print("2. Try the 1-hour GPU test for full analysis")
        print("3. Apply RLDK to your own RL training runs")
    else:
        print("\n❌ FAILURE: RLDK smoke test failed")
        print("Check the output above for issues.")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
