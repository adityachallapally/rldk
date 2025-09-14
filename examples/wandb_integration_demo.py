#!/usr/bin/env python3
"""
W&B Integration Demo

This script demonstrates the new W&B integration features in RLDK:
- W&B enabled by default
- --no-wandb flag to disable W&B
- Graceful fallback to file logging
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rldk.tracking import ExperimentTracker, TrackingConfig


def demo_wandb_default():
    """Demonstrate W&B enabled by default."""
    print("=" * 60)
    print("DEMO: W&B Enabled by Default")
    print("=" * 60)

    # Create config with default settings
    config = TrackingConfig(
        experiment_name="wandb_default_demo",
        tags=["demo", "wandb-default"]
    )

    print(f"W&B enabled: {config.save_to_wandb}")
    print(f"W&B project: {config.wandb_project}")
    print(f"Output directory: {config.output_dir}")

    # Create tracker
    tracker = ExperimentTracker(config)

    # Start experiment
    start_info = tracker.start_experiment()
    print(f"Experiment started with ID: {start_info['experiment_id']}")

    # Log some metrics
    tracker.log_metric("loss", 0.5)
    tracker.log_metric("accuracy", 0.8)
    tracker.log_metric("learning_rate", 0.001)

    # Add some metadata
    tracker.add_metadata("model_type", "transformer")
    tracker.add_metadata("dataset_size", 10000)

    # Finish experiment
    summary = tracker.finish_experiment()
    print(f"Experiment finished. Summary: {summary['experiment_name']}")
    print()


def demo_no_wandb():
    """Demonstrate --no-wandb flag functionality."""
    print("=" * 60)
    print("DEMO: --no-wandb Flag (File Logging Only)")
    print("=" * 60)

    # Create config with W&B disabled
    config = TrackingConfig(
        experiment_name="no_wandb_demo",
        save_to_wandb=False,  # This is what --no-wandb does
        tags=["demo", "file-only"]
    )

    print(f"W&B enabled: {config.save_to_wandb}")
    print(f"W&B project: {config.wandb_project}")
    print(f"Output directory: {config.output_dir}")

    # Create tracker
    tracker = ExperimentTracker(config)

    # Start experiment
    start_info = tracker.start_experiment()
    print(f"Experiment started with ID: {start_info['experiment_id']}")

    # Log some metrics
    tracker.log_metric("loss", 0.3)
    tracker.log_metric("accuracy", 0.9)
    tracker.log_metric("learning_rate", 0.0005)

    # Add some metadata
    tracker.add_metadata("model_type", "cnn")
    tracker.add_metadata("dataset_size", 5000)

    # Finish experiment
    summary = tracker.finish_experiment()
    print(f"Experiment finished. Summary: {summary['experiment_name']}")

    # Check that files were created
    json_files = list(config.output_dir.glob("*.json"))
    yaml_files = list(config.output_dir.glob("*.yaml"))
    print(f"JSON files created: {len(json_files)}")
    print(f"YAML files created: {len(yaml_files)}")
    print()


def demo_custom_wandb_project():
    """Demonstrate custom W&B project."""
    print("=" * 60)
    print("DEMO: Custom W&B Project")
    print("=" * 60)

    # Create config with custom W&B project
    config = TrackingConfig(
        experiment_name="custom_project_demo",
        wandb_project="my-custom-project",
        tags=["demo", "custom-project"]
    )

    print(f"W&B enabled: {config.save_to_wandb}")
    print(f"W&B project: {config.wandb_project}")
    print(f"Output directory: {config.output_dir}")

    # Create tracker
    tracker = ExperimentTracker(config)

    # Start experiment
    start_info = tracker.start_experiment()
    print(f"Experiment started with ID: {start_info['experiment_id']}")

    # Log some metrics
    tracker.log_metric("loss", 0.7)
    tracker.log_metric("accuracy", 0.75)

    # Finish experiment
    summary = tracker.finish_experiment()
    print(f"Experiment finished. Summary: {summary['experiment_name']}")
    print()


def main():
    """Run all demos."""
    print("RLDK W&B Integration Demo")
    print("This demonstrates the new W&B integration features:")
    print("- W&B enabled by default")
    print("- --no-wandb flag to disable W&B")
    print("- Graceful fallback to file logging")
    print("- Custom W&B project support")
    print()

    # Run demos
    demo_wandb_default()
    demo_no_wandb()
    demo_custom_wandb_project()

    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print()
    print("Key Points:")
    print("1. W&B is now enabled by default (save_to_wandb=True)")
    print("2. Default project is 'rldk-experiments'")
    print("3. Use --no-wandb flag to disable W&B and use file logging only")
    print("4. File logging still works when W&B is disabled")
    print("5. Graceful fallback if W&B is not available")
    print()
    print("CLI Usage:")
    print("  rldk track my_experiment                    # W&B enabled by default")
    print("  rldk track my_experiment --no-wandb         # File logging only")
    print("  rldk track my_experiment --wandb-project my-project  # Custom project")
    print()


if __name__ == "__main__":
    main()
