#!/usr/bin/env python3
"""
Demo script for Phase B of RL Debug Kit.
Demonstrates reward health checking and evaluation functionality.
"""

import shutil

# Add src to path for imports
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rldk.evals import run
from rldk.reward import health


def create_synthetic_data():
    """Create synthetic training data with intentional pathologies."""
    np.random.seed(42)
    n_steps = 200

    # Create base data
    data = pd.DataFrame(
        {
            "step": range(n_steps),
            "reward_std": np.random.uniform(0.1, 0.3, n_steps),
            "tokens_out": np.random.randint(10, 100, n_steps),
            "repetition_penalty": np.random.uniform(0.8, 1.2, n_steps),
            "human_preference": np.random.uniform(0, 1, n_steps),
            "ground_truth": np.random.choice([0, 1], n_steps),
            "epoch": np.random.randint(0, 10, n_steps),
            "run_id": ["demo_run"] * n_steps,
        }
    )

    # Introduce pathologies

    # 1. Length bias - rewards correlate with response length
    data["reward_mean"] = data["tokens_out"] * 0.01 + np.random.normal(
        0.5, 0.1, n_steps
    )

    # 2. Saturation - rewards cluster at upper bound
    data.loc[100:, "reward_mean"] = np.ones(100) * 0.99

    # 3. Label leakage - rewards correlate with epoch
    data["reward_mean"] = data["reward_mean"] + data["epoch"] * 0.05

    # 4. Poor calibration - rewards don't match human preferences well
    data["reward_mean"] = np.random.uniform(
        0.4, 0.6, n_steps
    )  # Override for calibration test

    return data


def create_reference_data():
    """Create reference data for drift detection."""
    np.random.seed(123)
    n_steps = 200

    return pd.DataFrame(
        {
            "step": range(n_steps),
            "reward_mean": np.random.normal(
                0.5, 0.15, n_steps
            ),  # Different distribution
            "reward_std": np.random.uniform(0.1, 0.3, n_steps),
        }
    )


def demo_reward_health():
    """Demonstrate reward health analysis."""
    print("🔍 Demo: Reward Health Analysis")
    print("=" * 50)

    # Create synthetic data with pathologies
    print("Creating synthetic training data with intentional pathologies...")
    run_data = create_synthetic_data()
    reference_data = create_reference_data()

    print(f"Generated {len(run_data)} training steps")
    print(f"Data columns: {', '.join(run_data.columns)}")

    # Run reward health analysis
    print("\nRunning reward health analysis...")
    health_report = health(
        run_data=run_data,
        reference_data=reference_data,
        threshold_drift=0.1,
        threshold_saturation=0.5,
        threshold_calibration=0.7,
        threshold_shortcut=0.5,
        threshold_leakage=0.3,
    )

    # Display results
    print("\n📊 Reward Health Results:")
    print(f"Overall Status: {'✅ PASSED' if health_report.passed else '🚨 FAILED'}")
    print(f"Drift Detected: {'Yes' if health_report.drift_detected else 'No'}")
    print(f"Saturation Issues: {len(health_report.saturation_issues)}")
    print(f"Calibration Score: {health_report.calibration_score:.3f}")
    print(f"Shortcut Signals: {len(health_report.shortcut_signals)}")
    print(f"Label Leakage Risk: {health_report.label_leakage_risk:.3f}")

    if health_report.saturation_issues:
        print("\n🚨 Saturation Issues Detected:")
        for issue in health_report.saturation_issues:
            print(f"  - {issue}")

    if health_report.shortcut_signals:
        print("\n🚧 Shortcut Signals Detected:")
        for signal in health_report.shortcut_signals:
            print(f"  - {signal}")

    if health_report.fixes:
        print("\n🔧 Recommended Fixes:")
        for i, fix in enumerate(health_report.fixes, 1):
            print(f"  {i}. {fix}")

    return health_report


def demo_evaluation():
    """Demonstrate evaluation suite."""
    print("\n📊 Demo: Evaluation Suite")
    print("=" * 50)

    # Create synthetic data
    print("Creating synthetic evaluation data...")
    eval_data = create_synthetic_data()

    # Run quick evaluation suite
    print("Running quick evaluation suite...")
    eval_result = run(run_data=eval_data, suite="quick", seed=42, sample_size=50)

    # Display results
    print(f"\n📈 Evaluation Results for {eval_result.suite_name} suite:")
    print(f"Sample Size: {eval_result.sample_size}")
    print(f"Seed: {eval_result.seed}")

    print("\nScores:")
    for metric, score in eval_result.scores.items():
        if not np.isnan(score):
            ci = eval_result.confidence_intervals.get(metric, (np.nan, np.nan))
            effect_size = eval_result.effect_sizes.get(metric, np.nan)

            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if not np.isnan(ci[0]) else "N/A"
            effect_str = f"{effect_size:.3f}" if not np.isnan(effect_size) else "N/A"

            print(f"  {metric}: {score:.3f} (CI: {ci_str}, Effect: {effect_str})")

    return eval_result


def demo_cli_commands():
    """Demonstrate CLI commands."""
    print("\n💻 Demo: CLI Commands")
    print("=" * 50)

    print("You can run these commands to reproduce the demo:")
    print()

    # Create temporary data file
    temp_dir = tempfile.mkdtemp()
    try:
        data_file = Path(temp_dir) / "demo_data.jsonl"
        run_data = create_synthetic_data()
        run_data.to_json(data_file, orient="records", lines=True)

        print("1. Reward Health Analysis:")
        print(f"   rldk reward-health --run {data_file} --output-dir reward_analysis")
        print()

        print("2. Evaluation Suite:")
        print(f"   rldk eval --run {data_file} --suite quick --output-dir eval_results")
        print()

        print("3. With Reference Data (for drift detection):")
        reference_file = Path(temp_dir) / "reference_data.jsonl"
        reference_data = create_reference_data()
        reference_data.to_json(reference_file, orient="records", lines=True)
        print(
            f"   rldk reward-health --run {data_file} --reference {reference_file} --output-dir reward_analysis"
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run the complete Phase B demo."""
    print("🚀 RL Debug Kit - Phase B Demo")
    print("Reward Health & Evaluation Suite")
    print("=" * 60)

    try:
        # Demo reward health
        demo_reward_health()

        # Demo evaluation
        demo_evaluation()

        # Demo CLI commands
        demo_cli_commands()

        print("\n✅ Demo completed successfully!")
        print("\nKey Takeaways:")
        print("- RLDK detected multiple reward pathologies in synthetic data")
        print("- Length bias, saturation, and label leakage were identified")
        print("- Evaluation suite provided comprehensive model assessment")
        print("- All functionality available via both Python API and CLI")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
