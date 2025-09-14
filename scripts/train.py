#!/usr/bin/env python3
"""
RLHF Training Script with Integrated Profiler.

This script demonstrates how to integrate the profiler system with a training loop.
It supports the --profiler on/off argument to enable/disable profiling.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from profiler.hooks import StepProfiler, profiler_registry
from profiler.profiler_context import ProfilerContext
from profiler.torch_profiler import TorchProfiler

from rlhf_core.profiler import ProfilerManager


class SimpleModel(nn.Module):
    """Simple model for demonstration purposes."""

    def __init__(self, input_size: int = 100, hidden_size: int = 50, output_size: int = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)


class TrainingConfig:
    """Training configuration."""

    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 5
        self.num_steps_per_epoch = 20
        self.input_size = 100
        self.hidden_size = 50
        self.output_size = 10
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


class TrainingMetrics:
    """Track training metrics."""

    def __init__(self):
        self.losses: List[float] = []
        self.accuracies: List[float] = []
        self.step_times: List[float] = []
        self.epoch_times: List[float] = []

    def add_step(self, loss: float, accuracy: float, step_time: float):
        """Add metrics for a training step."""
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.step_times.append(step_time)

    def add_epoch(self, epoch_time: float):
        """Add metrics for an epoch."""
        self.epoch_times.append(epoch_time)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of metrics."""
        return {
            "total_steps": len(self.losses),
            "total_epochs": len(self.epoch_times),
            "final_loss": self.losses[-1] if self.losses else 0.0,
            "final_accuracy": self.accuracies[-1] if self.accuracies else 0.0,
            "average_step_time": np.mean(self.step_times) if self.step_times else 0.0,
            "average_epoch_time": np.mean(self.epoch_times) if self.epoch_times else 0.0
        }


class RLHFTrainer:
    """RLHF Trainer with integrated profiler."""

    def __init__(self, config: TrainingConfig, profiler_enabled: bool = False, output_dir: str = "runs"):
        self.config = config
        self.profiler_enabled = profiler_enabled
        self.output_dir = Path(output_dir)

        # Create run directory
        self.run_id = f"run_{int(time.time())}"
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize profiler components - all use the same run directory for unified artifacts
        if self.profiler_enabled:
            self.profiler_manager = ProfilerManager(str(self.run_dir), enabled=True)
            # Use ProfilerManager for all profiling - it already includes TorchProfiler functionality
            self.torch_profiler = None  # Disabled to avoid conflicts
            self.profiler_context = ProfilerContext(str(self.run_dir))  # Use main run directory
            self.step_profiler = StepProfiler()
        else:
            self.profiler_manager = None
            self.torch_profiler = None
            self.profiler_context = None
            self.step_profiler = None

        # Initialize model and training components
        self.model = SimpleModel(
            self.config.input_size,
            self.config.hidden_size,
            self.config.output_size
        ).to(self.config.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Training metrics
        self.metrics = TrainingMetrics()

        # Set random seeds
        self._set_seeds()

    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)

    def _create_dummy_data(self) -> DataLoader:
        """Create dummy training data."""
        # Generate random data
        X = torch.randn(
            self.config.num_steps_per_epoch * self.config.batch_size,
            self.config.input_size
        )
        y = torch.randint(0, self.config.output_size, (X.size(0),))

        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

    def _train_step(self, batch, step: int) -> Dict[str, float]:
        """Perform a single training step."""
        step_start_time = time.time()

        # Start step profiling
        if self.step_profiler:
            self.step_profiler.start_step()

        inputs, targets = batch
        inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)

        # Forward pass
        with self.profiler_context.stage("forward") if self.profiler_context else nullcontext(), \
             self.profiler_manager.stage("forward") if self.profiler_manager else nullcontext():
            if self.step_profiler:
                self.step_profiler.forward_pass()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

        # Backward pass
        with self.profiler_context.stage("backward") if self.profiler_context else nullcontext(), \
             self.profiler_manager.stage("backward") if self.profiler_manager else nullcontext():
            if self.step_profiler:
                self.step_profiler.backward_pass()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Optimizer step profiling
        if self.step_profiler:
            self.step_profiler.optimizer_step()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == targets).float().mean().item()

        step_time = time.time() - step_start_time

        # End step profiling
        if self.step_profiler:
            self.step_profiler.end_step(
                model=self.model,
                optimizer=self.optimizer,
                loss=loss.item(),  # Convert tensor to float for JSON serialization
                batch_size=inputs.size(0),
                accuracy=accuracy
            )

        # Step profilers
        if self.profiler_manager:
            self.profiler_manager.step_profiler()

        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "step_time": step_time
        }

    def train(self):
        """Main training loop."""
        print(f"Starting training with profiler {'enabled' if self.profiler_enabled else 'disabled'}")
        print(f"Run directory: {self.run_dir}")
        print(f"Device: {self.config.device}")
        print("-" * 50)

        # Start profiling
        if self.profiler_enabled:
            print("Starting profilers...")
            try:
                self.profiler_manager.start_profiling(warmup_steps=1, active_steps=3, repeat=1)
                print("Profilers started successfully")
            except Exception as e:
                print(f"Warning: Failed to start some profilers: {e}")
                print("Continuing with training without full profiling...")

        # Create data loader
        data_loader = self._create_dummy_data()

        # Training loop
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()

            with self.profiler_context.stage(f"epoch_{epoch}") if self.profiler_context else nullcontext():
                print(f"Epoch {epoch + 1}/{self.config.num_epochs}")

                for step, batch in enumerate(data_loader):
                    step_metrics = self._train_step(batch, step)
                    self.metrics.add_step(**step_metrics)

                    if step % 5 == 0:
                        print(f"  Step {step}: Loss = {step_metrics['loss']:.4f}, "
                              f"Accuracy = {step_metrics['accuracy']:.4f}, "
                              f"Time = {step_metrics['step_time']:.4f}s")

            epoch_time = time.time() - epoch_start_time
            self.metrics.add_epoch(epoch_time)

            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")

        # Stop profiling
        if self.profiler_enabled:
            print("Stopping profilers...")
            try:
                self.profiler_manager.stop_profiling()
                self.profiler_context.save_stage_times()
                self.profiler_manager.save_stage_times()
                print("Profilers stopped successfully")
            except Exception as e:
                print(f"Warning: Error stopping profilers: {e}")
                # Try to save what we can
                try:
                    self.profiler_context.save_stage_times()
                    self.profiler_manager.save_stage_times()
                except Exception as e:
                    print(f"Warning: Could not save profiler stage times: {e}")
                    pass

        # Save training results
        self._save_training_results()

        print("Training completed!")
        return self.metrics.get_summary()

    def _save_training_results(self):
        """Save training results and configuration."""
        # Save configuration
        config_path = self.run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_epochs,
                "num_steps_per_epoch": self.config.num_steps_per_epoch,
                "input_size": self.config.input_size,
                "hidden_size": self.config.hidden_size,
                "output_size": self.config.output_size,
                "seed": self.config.seed,
                "device": self.config.device,
                "profiler_enabled": self.profiler_enabled
            }, f, indent=2)

        # Save metrics
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                "summary": self.metrics.get_summary(),
                "losses": self.metrics.losses,
                "accuracies": self.metrics.accuracies,
                "step_times": self.metrics.step_times,
                "epoch_times": self.metrics.epoch_times
            }, f, indent=2)

        # Save profiler summary if enabled
        if self.profiler_enabled:
            profiler_summary = {
                "profiler_manager": self.profiler_manager.get_profiler_summary(),
                "profiler_context": self.profiler_context.get_summary(),
                "step_profiler": self.step_profiler.get_summary()
            }

            profiler_summary_path = self.run_dir / "profiler_summary.json"
            with open(profiler_summary_path, 'w') as f:
                json.dump(profiler_summary, f, indent=2)

    def cleanup(self):
        """Cleanup profiler resources."""
        if self.profiler_manager:
            self.profiler_manager.cleanup()


class nullcontext:
    """Null context manager for when profiler is disabled."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="RLHF Training with Profiler")
    parser.add_argument("--profiler", choices=["on", "off"], default="off",
                       help="Enable or disable profiler")
    parser.add_argument("--output-dir", default="runs", help="Output directory for runs")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=20, help="Steps per epoch")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Create training configuration
    config = TrainingConfig()
    config.num_epochs = args.epochs
    config.num_steps_per_epoch = args.steps_per_epoch
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.seed = args.seed

    # Create trainer
    profiler_enabled = args.profiler == "on"
    trainer = RLHFTrainer(config, profiler_enabled, args.output_dir)

    try:
        # Run training
        summary = trainer.train()

        # Print summary
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        for key, value in summary.items():
            print(f"{key}: {value}")

        if profiler_enabled:
            print(f"\nProfiler artifacts saved to: {trainer.run_dir}")
            print("Use tools/check_profile.py to validate profiler results")
            print("Use monitor/app.py to view profiler dashboard")

    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
