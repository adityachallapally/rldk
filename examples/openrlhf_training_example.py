#!/usr/bin/env python3
"""
Example OpenRLHF training script with RLDK monitoring integration.

This example demonstrates how to integrate RLDK monitoring into an OpenRLHF training workflow,
including real-time monitoring, distributed training support, and analytics.
"""

import argparse
import os

# Add src to path for imports
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import openrlhf
    import torch
    from openrlhf.models import ActorCritic
    from openrlhf.trainer import PPOTrainer
    OPENRLHF_AVAILABLE = True
except ImportError:
    OPENRLHF_AVAILABLE = False
    print("⚠️  OpenRLHF not available. Install with: pip install openrlhf")

from rldk.integrations.openrlhf import (
    DistributedTrainingMonitor,
    MultiGPUMonitor,
    OpenRLHFAnalytics,
    OpenRLHFDashboard,
    OpenRLHFMetrics,
    OpenRLHFMonitor,
)


def create_mock_trainer():
    """Create a mock trainer for demonstration purposes."""
    class MockTrainer:
        def __init__(self):
            self.step_count = 0
            self.loss = 1.0
            self.reward_mean = 0.5
            self.kl_mean = 0.2
            self.learning_rate = 1e-4

        def step(self):
            """Simulate a training step."""
            self.step_count += 1

            # Simulate training progress
            self.loss = max(0.1, 1.0 - (self.step_count * 0.01))
            self.reward_mean = min(3.0, 0.5 + (self.step_count * 0.02))
            self.kl_mean = max(0.05, 0.2 - (self.step_count * 0.005))

            # Add some noise
            import random
            self.loss += random.uniform(-0.05, 0.05)
            self.reward_mean += random.uniform(-0.1, 0.1)
            self.kl_mean += random.uniform(-0.02, 0.02)

            # Simulate step time
            time.sleep(0.1)

    return MockTrainer()


def train_with_basic_monitoring(
    output_dir: str = "./rldk_logs",
    num_steps: int = 100,
    log_interval: int = 10
):
    """Example of basic training with RLDK monitoring."""
    print("🚀 Starting basic training with RLDK monitoring...")

    # Initialize trainer (mock for demonstration)
    trainer = create_mock_trainer()

    # Initialize RLDK monitor
    monitor = OpenRLHFMonitor(
        output_dir=output_dir,
        log_interval=log_interval,
        run_id=f"basic_training_{int(time.time())}",
        enable_resource_monitoring=True,
        enable_distributed_monitoring=False,  # Disable for single-node demo
    )

    print(f"📊 Monitoring training in: {output_dir}")
    print(f"📝 Logging every {log_interval} steps")

    try:
        # Training loop
        for step in range(num_steps):
            # Training step
            trainer.step()

            # Create metrics
            metrics = OpenRLHFMetrics(
                step=step,
                loss=trainer.loss,
                reward_mean=trainer.reward_mean,
                kl_mean=trainer.kl_mean,
                learning_rate=trainer.learning_rate,
                step_time=0.1,
                gpu_memory_used=8.0 + (step * 0.01),  # Simulate increasing memory usage
                run_id=monitor.run_id
            )

            # Add metrics to monitor
            monitor.metrics_history.append(metrics)
            monitor.current_metrics = metrics

            # Simulate step end
            monitor.on_step_end(trainer, step)

            # Print progress
            if step % log_interval == 0:
                print(f"Step {step:3d}: Loss={trainer.loss:.4f}, Reward={trainer.reward_mean:.3f}, KL={trainer.kl_mean:.3f}")

        # Save final metrics
        monitor._save_metrics()
        print("✅ Training completed successfully!")

        # Show summary
        df = monitor.get_metrics_dataframe()
        print(f"📈 Collected {len(df)} metric points")
        print(f"📁 Logs saved to: {output_dir}")

    except KeyboardInterrupt:
        print("⏹️  Training interrupted by user")
        monitor._save_metrics()

    return monitor


def train_with_dashboard(
    output_dir: str = "./rldk_logs",
    num_steps: int = 200,
    dashboard_port: int = 5000
):
    """Example of training with real-time dashboard."""
    print("🚀 Starting training with real-time dashboard...")

    # Initialize trainer
    trainer = create_mock_trainer()

    # Initialize monitor
    monitor = OpenRLHFMonitor(
        output_dir=output_dir,
        log_interval=5,
        run_id=f"dashboard_training_{int(time.time())}",
        enable_resource_monitoring=True,
    )

    # Initialize dashboard
    dashboard = OpenRLHFDashboard(
        output_dir=output_dir,
        port=dashboard_port,
        host="localhost",
        enable_auto_refresh=True,
        refresh_interval=1.0
    )

    print(f"📊 Dashboard will be available at: http://localhost:{dashboard_port}")
    print("🌐 Open the dashboard in your browser to see real-time metrics")

    # Start dashboard in background thread
    import threading
    dashboard_thread = threading.Thread(
        target=dashboard.start_dashboard,
        daemon=True
    )
    dashboard_thread.start()

    # Give dashboard time to start
    time.sleep(2)

    try:
        # Training loop
        for step in range(num_steps):
            # Training step
            trainer.step()

            # Create metrics
            metrics = OpenRLHFMetrics(
                step=step,
                loss=trainer.loss,
                reward_mean=trainer.reward_mean,
                kl_mean=trainer.kl_mean,
                learning_rate=trainer.learning_rate,
                step_time=0.1,
                gpu_memory_used=8.0 + (step * 0.01),
                run_id=monitor.run_id
            )

            # Add to monitor and dashboard
            monitor.metrics_history.append(metrics)
            monitor.current_metrics = metrics
            dashboard.add_metrics(metrics)

            # Simulate step end
            monitor.on_step_end(trainer, step)

            # Print progress
            if step % 20 == 0:
                print(f"Step {step:3d}: Loss={trainer.loss:.4f}, Reward={trainer.reward_mean:.3f}")

        print("✅ Training completed!")
        print(f"📊 Check the dashboard at: http://localhost:{dashboard_port}")

    except KeyboardInterrupt:
        print("⏹️  Training interrupted by user")

    finally:
        monitor._save_metrics()
        print("📁 Logs saved to:", output_dir)


def train_with_analytics(
    output_dir: str = "./rldk_logs",
    num_steps: int = 300
):
    """Example of training with comprehensive analytics."""
    print("🚀 Starting training with comprehensive analytics...")

    # Initialize trainer
    trainer = create_mock_trainer()

    # Initialize monitor
    monitor = OpenRLHFMonitor(
        output_dir=output_dir,
        log_interval=10,
        run_id=f"analytics_training_{int(time.time())}",
        enable_resource_monitoring=True,
    )

    # Initialize analytics
    analytics = OpenRLHFAnalytics(output_dir=output_dir)

    print("📊 Running analytics on training data...")

    try:
        # Training loop
        metrics_history = []

        for step in range(num_steps):
            # Training step
            trainer.step()

            # Create metrics
            metrics = OpenRLHFMetrics(
                step=step,
                loss=trainer.loss,
                reward_mean=trainer.reward_mean,
                kl_mean=trainer.kl_mean,
                learning_rate=trainer.learning_rate,
                step_time=0.1,
                gpu_memory_used=8.0 + (step * 0.01),
                run_id=monitor.run_id
            )

            # Add to monitor
            monitor.metrics_history.append(metrics)
            monitor.current_metrics = metrics
            metrics_history.append(metrics)

            # Simulate step end
            monitor.on_step_end(trainer, step)

            # Print progress
            if step % 50 == 0:
                print(f"Step {step:3d}: Loss={trainer.loss:.4f}, Reward={trainer.reward_mean:.3f}")

        # Perform comprehensive analysis
        print("📈 Performing comprehensive analysis...")
        analysis_results = analytics.analyze_training_run(metrics_history)

        # Display results
        print("\n📊 Analysis Results:")
        print("=" * 50)

        health = analysis_results.get('training_health', {})
        print(f"Overall Health Score: {health.get('overall_health', 0):.3f}")
        print(f"Stability Score: {health.get('stability_score', 0):.3f}")
        print(f"Convergence Rate: {health.get('convergence_rate', 0):.3f}")
        print(f"Reward Trend: {health.get('reward_trend', 0):.3f}")
        print(f"KL Trend: {health.get('kl_trend', 0):.3f}")

        metrics_summary = analysis_results.get('metrics_summary', {})
        print(f"\nTotal Steps: {metrics_summary.get('total_steps', 0)}")
        print(f"Final Loss: {metrics_summary.get('latest_loss', 0):.4f}")
        print(f"Final Reward: {metrics_summary.get('latest_reward', 0):.3f}")
        print(f"Average Step Time: {metrics_summary.get('avg_step_time', 0):.3f}s")

        resource_summary = analysis_results.get('resource_summary', {})
        if resource_summary:
            print(f"\nPeak GPU Memory: {resource_summary.get('peak_memory_usage', 0):.2f} GB")
            print(f"Average CPU Usage: {resource_summary.get('avg_cpu_utilization', 0):.1f}%")

        print("✅ Analysis completed!")
        print(f"📁 Detailed analysis saved to: {output_dir}")

    except KeyboardInterrupt:
        print("⏹️  Training interrupted by user")

    finally:
        monitor._save_metrics()

    return analysis_results


def train_with_distributed_monitoring(
    output_dir: str = "./rldk_logs",
    num_steps: int = 150
):
    """Example of training with distributed monitoring (simulated)."""
    print("🚀 Starting training with distributed monitoring...")

    # Initialize trainer
    trainer = create_mock_trainer()

    # Initialize distributed monitor
    monitor = DistributedTrainingMonitor(
        output_dir=output_dir,
        log_interval=10,
        run_id=f"distributed_training_{int(time.time())}",
        sync_interval=5,
        network_monitoring=True
    )

    print("🌐 Simulating distributed training monitoring...")
    print("📊 Monitoring resources across multiple nodes/GPUs")

    try:
        # Training loop
        for step in range(num_steps):
            # Training step
            trainer.step()

            # Create metrics with distributed info
            metrics = OpenRLHFMetrics(
                step=step,
                loss=trainer.loss,
                reward_mean=trainer.reward_mean,
                kl_mean=trainer.kl_mean,
                learning_rate=trainer.learning_rate,
                step_time=0.1,
                gpu_memory_used=8.0 + (step * 0.01),
                world_size=4,  # Simulate 4 GPUs
                local_rank=0,
                global_rank=0,
                node_id="node_0",
                run_id=monitor.run_id
            )

            # Add to monitor
            monitor.metrics_history.append(metrics)
            monitor.current_metrics = metrics

            # Simulate step end
            monitor.on_step_end(trainer, step)

            # Print progress
            if step % 25 == 0:
                print(f"Step {step:3d}: Loss={trainer.loss:.4f}, Reward={trainer.reward_mean:.3f}")

        print("✅ Distributed training completed!")

    except KeyboardInterrupt:
        print("⏹️  Training interrupted by user")

    finally:
        monitor._save_metrics()
        print("📁 Distributed logs saved to:", output_dir)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="OpenRLHF Training with RLDK Monitoring")
    parser.add_argument(
        "--mode",
        choices=["basic", "dashboard", "analytics", "distributed"],
        default="basic",
        help="Training mode to run"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./rldk_logs",
        help="Output directory for logs"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of training steps"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Steps between detailed logging"
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=5000,
        help="Port for dashboard (dashboard mode only)"
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("🤖 OpenRLHF Training with RLDK Monitoring")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Training Steps: {args.num_steps}")
    print("=" * 50)

    # Check if OpenRLHF is available
    if not OPENRLHF_AVAILABLE:
        print("⚠️  OpenRLHF not available. Using mock trainer for demonstration.")
        print("   Install OpenRLHF with: pip install openrlhf")

    # Run selected mode
    if args.mode == "basic":
        train_with_basic_monitoring(
            output_dir=args.output_dir,
            num_steps=args.num_steps,
            log_interval=args.log_interval
        )
    elif args.mode == "dashboard":
        train_with_dashboard(
            output_dir=args.output_dir,
            num_steps=args.num_steps,
            dashboard_port=args.dashboard_port
        )
    elif args.mode == "analytics":
        train_with_analytics(
            output_dir=args.output_dir,
            num_steps=args.num_steps
        )
    elif args.mode == "distributed":
        train_with_distributed_monitoring(
            output_dir=args.output_dir,
            num_steps=args.num_steps
        )

    print("\n🎉 Example completed!")
    print("📚 Check the documentation for more advanced usage examples.")


if __name__ == "__main__":
    main()
