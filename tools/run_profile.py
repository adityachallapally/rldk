#!/usr/bin/env python3
"""
Standalone profiler runner for RLHF training.

This script can be used to run profiling on existing training scripts
or as a standalone profiler for testing purposes.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from profiler.profiler_context import ProfilerContext
from profiler.torch_profiler import TorchProfiler

from rlhf_core.profiler import ProfilerManager


def create_dummy_model():
    """Create a dummy model for testing profiler."""
    return nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
        nn.Softmax(dim=1)
    )


def run_dummy_training(profiler_manager: ProfilerManager, steps: int = 20):
    """Run dummy training with profiling."""
    print(f"Starting dummy training for {steps} steps...")

    # Create dummy model and data
    model = create_dummy_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Dummy data
    batch_size = 32
    input_data = torch.randn(batch_size, 100)
    target = torch.randint(0, 10, (batch_size,))

    # Start profiling
    profiler_manager.start_profiling(warmup_steps=2, active_steps=5, repeat=2)

    for step in range(steps):
        with profiler_manager.stage(f"step_{step}"):
            # Forward pass
            with profiler_manager.stage("forward"):
                output = model(input_data)
                loss = criterion(output, target)

            # Backward pass
            with profiler_manager.stage("backward"):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Step profiler
            profiler_manager.step_profiler()

            if step % 5 == 0:
                print(f"Step {step}: Loss = {loss.item():.4f}")

    # Stop profiling
    profiler_manager.stop_profiling()
    profiler_manager.save_stage_times()

    print("Training completed!")
    return profiler_manager.get_profiler_summary()


def run_torch_profiler_test(output_dir: str, steps: int = 20):
    """Test the TorchProfiler directly."""
    print(f"Testing TorchProfiler for {steps} steps...")

    profiler = TorchProfiler(output_dir, warmup_steps=2, active_steps=5, repeat=2)

    # Create dummy model and data
    model = create_dummy_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    batch_size = 32
    input_data = torch.randn(batch_size, 100)
    target = torch.randint(0, 10, (batch_size,))

    # Start profiling
    profiler.start()

    for step in range(steps):
        # Forward pass
        output = model(input_data)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step profiler
        profiler.step()

        if step % 5 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    # Stop profiling
    profiler.stop()

    print("TorchProfiler test completed!")
    return profiler.get_summary()


def run_profiler_context_test(output_dir: str, steps: int = 20):
    """Test the ProfilerContext for stage timing."""
    print(f"Testing ProfilerContext for {steps} steps...")

    context = ProfilerContext(output_dir)

    # Create dummy model and data
    model = create_dummy_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    batch_size = 32
    input_data = torch.randn(batch_size, 100)
    target = torch.randint(0, 10, (batch_size,))

    for step in range(steps):
        with context.stage(f"step_{step}"):
            # Forward pass
            with context.stage("forward"):
                output = model(input_data)
                loss = criterion(output, target)

            # Backward pass
            with context.stage("backward"):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if step % 5 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    # Save stage times
    context.save_stage_times()

    print("ProfilerContext test completed!")
    return context.get_summary()


def main():
    """Main function for standalone profiler runner."""
    parser = argparse.ArgumentParser(description="Standalone profiler runner")
    parser.add_argument("--output-dir", default="runs/profiler_test", help="Output directory for profiler artifacts")
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps")
    parser.add_argument("--test-type", choices=["manager", "torch", "context", "all"], default="all", help="Type of profiler test to run")
    parser.add_argument("--enabled", action="store_true", default=True, help="Enable profiling")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running profiler tests in: {output_dir}")
    print(f"Steps: {args.steps}")
    print(f"Test type: {args.test_type}")
    print("-" * 50)

    results = {}

    if args.test_type in ["manager", "all"]:
        print("\n1. Testing ProfilerManager...")
        profiler_manager = ProfilerManager(str(output_dir / "manager"), enabled=args.enabled)
        results["manager"] = run_dummy_training(profiler_manager, args.steps)
        profiler_manager.cleanup()

    if args.test_type in ["torch", "all"]:
        print("\n2. Testing TorchProfiler...")
        results["torch"] = run_torch_profiler_test(str(output_dir / "torch"), args.steps)

    if args.test_type in ["context", "all"]:
        print("\n3. Testing ProfilerContext...")
        results["context"] = run_profiler_context_test(str(output_dir / "context"), args.steps)

    # Print summary
    print("\n" + "=" * 50)
    print("PROFILER TEST SUMMARY")
    print("=" * 50)

    for test_name, result in results.items():
        print(f"\n{test_name.upper()} TEST:")
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print(f"  Result: {result}")

    print(f"\nProfiler artifacts saved to: {output_dir}")
    print("You can now use tools/check_profile.py to validate the results.")


if __name__ == "__main__":
    main()
