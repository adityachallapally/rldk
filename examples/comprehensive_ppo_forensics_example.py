"""Comprehensive PPO forensics example with advanced tracking and analysis."""

import json
import os
from pathlib import Path

import numpy as np

# Import RLDK components
from rldk.forensics.comprehensive_ppo_forensics import ComprehensivePPOForensics
from rldk.integrations.trl.monitors import ComprehensivePPOMonitor

try:
    from datasets import Dataset
    from transformers import AutoTokenizer, TrainingArguments
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
    TRL_AVAILABLE = True
except ImportError:
    print("TRL not available. Install with: pip install trl")
    TRL_AVAILABLE = False


def create_sample_dataset():
    """Create a sample dataset for PPO training."""
    prompts = [
        "The capital of France is",
        "Python is a programming language that",
        "Machine learning is",
        "The weather today is",
        "Artificial intelligence can",
    ] * 20  # Repeat to have enough data

    responses = [
        "Paris, the beautiful city of lights.",
        "is widely used for data science and AI.",
        "a subset of artificial intelligence.",
        "sunny and warm.",
        "help solve complex problems.",
    ] * 20

    return Dataset.from_dict({
        "prompt": prompts,
        "response": responses,
    })


def demonstrate_comprehensive_ppo_forensics():
    """Demonstrate comprehensive PPO forensics capabilities."""
    print("🔍 Comprehensive PPO Forensics Demonstration")
    print("=" * 60)

    # Initialize comprehensive forensics
    forensics = ComprehensivePPOForensics(
        kl_target=0.1,
        kl_target_tolerance=0.05,
        enable_kl_schedule_tracking=True,
        enable_gradient_norms_analysis=True,
        enable_advantage_statistics=True,
    )

    print("✅ Comprehensive PPO Forensics initialized")

    # Simulate PPO training with various scenarios
    scenarios = [
        {
            "name": "Healthy Training",
            "kl_range": (0.08, 0.12),
            "gradient_range": (0.3, 0.7),
            "advantage_bias": 0.0,
            "steps": 50
        },
        {
            "name": "KL Spike Scenario",
            "kl_range": (0.05, 0.3),
            "gradient_range": (0.3, 0.7),
            "advantage_bias": 0.0,
            "steps": 30
        },
        {
            "name": "Gradient Explosion",
            "kl_range": (0.08, 0.12),
            "gradient_range": (0.3, 15.0),
            "advantage_bias": 0.0,
            "steps": 30
        },
        {
            "name": "Advantage Bias",
            "kl_range": (0.08, 0.12),
            "gradient_range": (0.3, 0.7),
            "advantage_bias": 0.2,
            "steps": 30
        }
    ]

    step_counter = 0

    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print("-" * 40)

        for i in range(scenario['steps']):
            # Generate synthetic data based on scenario
            kl = np.random.uniform(*scenario['kl_range'])
            kl_coef = 1.0 + 0.1 * np.sin(step_counter * 0.1)
            entropy = 2.0 - 0.01 * step_counter
            reward_mean = 0.5 + 0.01 * step_counter
            reward_std = 0.2

            policy_grad_norm = np.random.uniform(*scenario['gradient_range'])
            value_grad_norm = np.random.uniform(0.2, 0.5)

            advantage_mean = scenario['advantage_bias'] + 0.05 * np.sin(step_counter * 0.1)
            advantage_std = 1.0 + 0.1 * np.cos(step_counter * 0.1)
            advantage_min = advantage_mean - advantage_std
            advantage_max = advantage_mean + advantage_std

            # Generate advantage samples for distribution analysis
            advantage_samples = np.random.normal(advantage_mean, advantage_std, 100).tolist()

            # Update forensics
            metrics = forensics.update(
                step=step_counter,
                kl=kl,
                kl_coef=kl_coef,
                entropy=entropy,
                reward_mean=reward_mean,
                reward_std=reward_std,
                policy_grad_norm=policy_grad_norm,
                value_grad_norm=value_grad_norm,
                advantage_mean=advantage_mean,
                advantage_std=advantage_std,
                advantage_min=advantage_min,
                advantage_max=advantage_max,
                advantage_samples=advantage_samples
            )

            step_counter += 1

            # Log progress every 10 steps
            if step_counter % 10 == 0:
                print(f"   Step {step_counter}: Health={metrics.overall_health_score:.3f}, "
                      f"Stability={metrics.training_stability_score:.3f}")

        # Get anomalies for this scenario
        anomalies = forensics.get_anomalies()
        if anomalies:
            print(f"   🚨 Detected {len(anomalies)} anomalies:")
            for anomaly in anomalies[-3:]:  # Show last 3 anomalies
                print(f"      - {anomaly['type']}: {anomaly['message']}")
        else:
            print("   ✅ No anomalies detected")

    # Generate comprehensive analysis
    print("\n📋 Comprehensive Analysis")
    print("-" * 40)

    analysis = forensics.get_comprehensive_analysis()
    health_summary = forensics.get_health_summary()

    print(f"Total Steps: {analysis['total_steps']}")
    print(f"Overall Health Score: {analysis['overall_health_score']:.3f}")
    print(f"Training Stability: {analysis['training_stability_score']:.3f}")
    print(f"Convergence Quality: {analysis['convergence_quality_score']:.3f}")
    print(f"Total Anomalies: {len(analysis['anomalies'])}")

    # Show tracker-specific summaries
    for tracker_name, tracker_summary in analysis['trackers'].items():
        print(f"\n{tracker_name.replace('_', ' ').title()} Summary:")
        if tracker_name == 'kl_schedule':
            print(f"  KL Health: {tracker_summary.get('kl_health_score', 0):.3f}")
            print(f"  Schedule Health: {tracker_summary.get('schedule_health_score', 0):.3f}")
            print(f"  Time in Target Range: {tracker_summary.get('time_in_target_range', 0):.2%}")
        elif tracker_name == 'gradient_norms':
            print(f"  Gradient Health: {tracker_summary.get('gradient_health_score', 0):.3f}")
            print(f"  Training Stability: {tracker_summary.get('training_stability', 0):.3f}")
            print(f"  Policy/Value Ratio: {tracker_summary.get('current_policy_value_ratio', 0):.3f}")
        elif tracker_name == 'advantage_statistics':
            print(f"  Advantage Health: {tracker_summary.get('advantage_health_score', 0):.3f}")
            print(f"  Advantage Quality: {tracker_summary.get('advantage_quality_score', 0):.3f}")
            print(f"  Advantage Bias: {tracker_summary.get('advantage_bias', 0):.4f}")

    # Save analysis
    output_dir = Path("./comprehensive_ppo_forensics_demo")
    output_dir.mkdir(exist_ok=True)

    forensics.save_analysis(str(output_dir / "comprehensive_analysis.json"))

    with open(output_dir / "health_summary.json", "w") as f:
        json.dump(health_summary, f, indent=2)

    print(f"\n💾 Analysis saved to: {output_dir}")

    return forensics, analysis, health_summary


def demonstrate_comprehensive_ppo_monitor():
    """Demonstrate comprehensive PPO monitor integration."""
    if not TRL_AVAILABLE:
        print("⚠️  TRL not available - skipping monitor demonstration")
        return

    print("\n🔍 Comprehensive PPO Monitor Demonstration")
    print("=" * 60)

    # Create output directory
    output_dir = "./comprehensive_ppo_monitor_demo"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize comprehensive PPO monitor
    monitor = ComprehensivePPOMonitor(
        output_dir=output_dir,
        kl_target=0.1,
        kl_target_tolerance=0.05,
        enable_kl_schedule_tracking=True,
        enable_gradient_norms_analysis=True,
        enable_advantage_statistics=True,
        log_interval=5,
        run_id="comprehensive_demo_run"
    )

    print("✅ Comprehensive PPO Monitor initialized")

    # Simulate training logs
    from transformers import TrainerControl, TrainerState, TrainingArguments

    args = TrainingArguments(output_dir=output_dir)
    state = TrainerState()
    control = TrainerControl()

    # Simulate training steps
    for step in range(20):
        state.global_step = step
        state.epoch = step / 10.0

        # Simulate logs with various metrics
        logs = {
            'ppo/rewards/mean': 0.5 + 0.1 * step,
            'ppo/rewards/std': 0.2,
            'ppo/policy/kl_mean': 0.1 + 0.02 * np.sin(step * 0.1),
            'ppo/policy/kl_coef': 1.0 + 0.1 * np.cos(step * 0.1),
            'ppo/policy/entropy': 2.0 - 0.05 * step,
            'ppo/policy/clipfrac': 0.1,
            'ppo/val/value_loss': 0.3 - 0.01 * step,
            'ppo/policy/grad_norm': 0.5 + 0.1 * np.sin(step * 0.1),
            'ppo/val/grad_norm': 0.3 + 0.05 * np.cos(step * 0.1),
            'ppo/advantages/mean': 0.0 + 0.05 * np.sin(step * 0.1),
            'ppo/advantages/std': 1.0 + 0.1 * np.cos(step * 0.1),
            'learning_rate': 1e-5,
            'grad_norm': 0.5,
        }

        # Call monitor callbacks
        monitor.on_step_end(args, state, control)
        monitor.on_log(args, state, control, logs)

        if step % 5 == 0:
            print(f"   Step {step} completed")

    # Simulate training end
    monitor.on_train_end(args, state, control)

    # Get current health summary
    health_summary = monitor.get_current_health_summary()
    print(f"\n📋 Final Health Summary: {health_summary}")

    # Get current anomalies
    anomalies = monitor.get_anomalies()
    if anomalies:
        print(f"🚨 Current Anomalies: {len(anomalies)}")
        for anomaly in anomalies[:3]:  # Show first 3
            print(f"   - {anomaly['type']}: {anomaly['message']}")
    else:
        print("✅ No current anomalies")

    print(f"💾 Monitor data saved to: {output_dir}")


def demonstrate_ppo_scan_integration():
    """Demonstrate integration with existing PPO scan functionality."""
    print("\n🔍 PPO Scan Integration Demonstration")
    print("=" * 60)

    # Create sample PPO events
    events = []
    for step in range(50):
        # Create some interesting patterns
        if step < 20:
            # Normal training
            kl = 0.1
            policy_grad_norm = 0.5
            advantage_mean = 0.0
        elif step < 35:
            # KL spike
            kl = 0.3
            policy_grad_norm = 0.5
            advantage_mean = 0.0
        else:
            # Gradient explosion
            kl = 0.1
            policy_grad_norm = 8.0
            advantage_mean = 0.0

        event = {
            "step": step,
            "kl": kl,
            "kl_coef": 1.0,
            "entropy": 2.0,
            "advantage_mean": advantage_mean,
            "advantage_std": 1.0,
            "grad_norm_policy": policy_grad_norm,
            "grad_norm_value": 0.3,
            "reward_mean": 0.5,
            "reward_std": 0.2,
        }
        events.append(event)

    # Run comprehensive scan
    forensics = ComprehensivePPOForensics()
    result = forensics.scan_ppo_events_comprehensive(iter(events))

    print("✅ Comprehensive PPO scan completed")
    print(f"Original rules fired: {len(result['rules_fired'])}")
    print(f"Enhanced version: {result['enhanced_version']}")

    # Show comprehensive analysis summary
    comp_analysis = result['comprehensive_analysis']
    print(f"Total steps analyzed: {comp_analysis['total_steps']}")
    print(f"Overall health score: {comp_analysis['overall_health_score']:.3f}")
    print(f"Total anomalies detected: {len(comp_analysis['anomalies'])}")

    # Show anomaly breakdown by tracker
    anomaly_by_tracker = {}
    for anomaly in comp_analysis['anomalies']:
        tracker = anomaly.get('tracker', 'unknown')
        anomaly_by_tracker[tracker] = anomaly_by_tracker.get(tracker, 0) + 1

    print("\nAnomaly breakdown by tracker:")
    for tracker, count in anomaly_by_tracker.items():
        print(f"  {tracker}: {count} anomalies")

    # Save enhanced scan results
    output_dir = Path("./enhanced_ppo_scan_demo")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "enhanced_scan_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"💾 Enhanced scan results saved to: {output_dir}")


if __name__ == "__main__":
    print("🎯 Comprehensive PPO Forensics Demo Suite")
    print("=" * 80)

    # Run demonstrations
    try:
        # 1. Basic comprehensive forensics
        forensics, analysis, health_summary = demonstrate_comprehensive_ppo_forensics()

        # 2. Comprehensive PPO monitor (if TRL available)
        demonstrate_comprehensive_ppo_monitor()

        # 3. PPO scan integration
        demonstrate_ppo_scan_integration()

        print("\n🎉 All demonstrations completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ KL Schedule Tracking with adaptive coefficient monitoring")
        print("✅ Gradient Norms Analysis with exploding/vanishing detection")
        print("✅ Advantage Statistics with bias and distribution analysis")
        print("✅ Comprehensive anomaly detection across all trackers")
        print("✅ Health scoring and training stability assessment")
        print("✅ Integration with existing PPO scan functionality")
        print("✅ TRL callback integration for real-time monitoring")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
