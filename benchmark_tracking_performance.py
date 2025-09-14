#!/usr/bin/env python3
"""
Quick performance benchmark for RLDK tracking system.

This script provides a quick way to test the performance improvements
without running the full test suite.
"""

import time
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil

# Import RLDK components
from src.rldk.config.settings import RLDKSettings
from src.rldk.tracking.config import TrackingConfig
from src.rldk.tracking.tracker import ExperimentTracker


def benchmark_settings_initialization():
    """Benchmark settings initialization."""
    print("Benchmarking settings initialization...")
    
    # Test sync initialization
    start_time = time.time()
    settings = RLDKSettings()
    sync_time = time.time() - start_time
    
    # Test async initialization
    start_time = time.time()
    settings.initialize_async()
    async_time = time.time() - start_time
    
    print(f"  Sync initialization: {sync_time:.3f}s")
    print(f"  Async initialization: {async_time:.3f}s")
    print(f"  Speedup: {sync_time / max(async_time, 0.001):.1f}x")
    
    return {"sync_time": sync_time, "async_time": async_time}


def benchmark_dataset_tracking():
    """Benchmark dataset tracking with different sizes."""
    print("Benchmarking dataset tracking...")
    
    from src.rldk.tracking.dataset_tracker import DatasetTracker
    
    tracker = DatasetTracker()
    results = {}
    
    # Test different dataset sizes
    sizes = [
        (1000, "Small (1K samples)"),
        (10000, "Medium (10K samples)"),
        (100000, "Large (100K samples)"),
        (1000000, "Very Large (1M samples)")
    ]
    
    for size, name in sizes:
        data = np.random.randn(size, 10)
        start_time = time.time()
        tracker.track_dataset(data, f"dataset_{size}")
        elapsed = time.time() - start_time
        results[name] = elapsed
        print(f"  {name}: {elapsed:.3f}s")
    
    return results


def benchmark_model_tracking():
    """Benchmark model tracking with different sizes."""
    print("Benchmarking model tracking...")
    
    from src.rldk.tracking.model_tracker import ModelTracker
    
    tracker = ModelTracker()
    results = {}
    
    # Test different model sizes
    models = [
        (torch.nn.Linear(10, 50), "Small (10->50)"),
        (torch.nn.Sequential(
            torch.nn.Linear(100, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 100)
        ), "Medium (100->500->100)"),
        (torch.nn.Sequential(
            torch.nn.Linear(1000, 2000),
            torch.nn.ReLU(),
            torch.nn.Linear(2000, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 100)
        ), "Large (1000->2000->1000->100)")
    ]
    
    for model, name in models:
        start_time = time.time()
        tracker.track_model(model, f"model_{name}")
        elapsed = time.time() - start_time
        results[name] = elapsed
        print(f"  {name}: {elapsed:.3f}s")
    
    return results


def benchmark_environment_tracking():
    """Benchmark environment tracking with caching."""
    print("Benchmarking environment tracking...")
    
    from src.rldk.tracking.environment_tracker import EnvironmentTracker
    
    tracker = EnvironmentTracker()
    
    # First capture
    start_time = time.time()
    tracker.capture_environment()
    first_time = time.time() - start_time
    
    # Cached capture
    start_time = time.time()
    tracker.capture_environment()
    cached_time = time.time() - start_time
    
    print(f"  First capture: {first_time:.3f}s")
    print(f"  Cached capture: {cached_time:.3f}s")
    print(f"  Cache speedup: {first_time / max(cached_time, 0.001):.1f}x")
    
    return {"first_time": first_time, "cached_time": cached_time}


def benchmark_experiment_tracker():
    """Benchmark experiment tracker initialization and start."""
    print("Benchmarking experiment tracker...")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        config = TrackingConfig(
            experiment_name="benchmark_test",
            output_dir=temp_dir / "experiments"
        )
        
        # Test sync initialization
        tracker_sync = ExperimentTracker(config)
        start_time = time.time()
        tracker_sync.initialize_sync()
        sync_init_time = time.time() - start_time
        
        # Test async initialization
        import asyncio
        tracker_async = ExperimentTracker(config)
        start_time = time.time()
        asyncio.run(tracker_async.initialize_async())
        async_init_time = time.time() - start_time
        
        # Test experiment start
        tracker = ExperimentTracker(config)
        start_time = time.time()
        tracking_data = tracker.start_experiment()
        start_time_total = time.time() - start_time
        
        print(f"  Sync initialization: {sync_init_time:.3f}s")
        print(f"  Async initialization: {async_init_time:.3f}s")
        print(f"  Experiment start: {start_time_total:.3f}s")
        print(f"  Initialization speedup: {sync_init_time / max(async_init_time, 0.001):.1f}x")
        
        return {
            "sync_init_time": sync_init_time,
            "async_init_time": async_init_time,
            "start_time": start_time_total
        }
        
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    """Run all benchmarks."""
    print("RLDK Tracking Performance Benchmark")
    print("=" * 40)
    
    all_results = {}
    
    try:
        # Run benchmarks
        all_results["settings"] = benchmark_settings_initialization()
        print()
        
        all_results["dataset_tracking"] = benchmark_dataset_tracking()
        print()
        
        all_results["model_tracking"] = benchmark_model_tracking()
        print()
        
        all_results["environment_tracking"] = benchmark_environment_tracking()
        print()
        
        all_results["experiment_tracker"] = benchmark_experiment_tracker()
        print()
        
        # Summary
        print("=" * 40)
        print("BENCHMARK SUMMARY")
        print("=" * 40)
        
        # Check if performance targets are met
        targets_met = True
        
        # Settings initialization should be fast
        if all_results["settings"]["sync_time"] > 1.0:
            print("⚠️  Settings sync initialization is slow")
            targets_met = False
        else:
            print("✅ Settings initialization is fast")
        
        # Dataset tracking should handle large datasets
        if all_results["dataset_tracking"]["Very Large (1M samples)"] > 30.0:
            print("⚠️  Large dataset tracking is slow")
            targets_met = False
        else:
            print("✅ Large dataset tracking is fast")
        
        # Model tracking should be fast
        if all_results["model_tracking"]["Large (1000->2000->1000->100)"] > 10.0:
            print("⚠️  Large model tracking is slow")
            targets_met = False
        else:
            print("✅ Model tracking is fast")
        
        # Environment tracking should benefit from caching
        env_ratio = all_results["environment_tracking"]["first_time"] / max(all_results["environment_tracking"]["cached_time"], 0.001)
        if env_ratio < 2.0:
            print("⚠️  Environment caching not effective")
            targets_met = False
        else:
            print("✅ Environment caching is effective")
        
        # Experiment tracker should initialize quickly
        if all_results["experiment_tracker"]["start_time"] > 5.0:
            print("⚠️  Experiment start is slow")
            targets_met = False
        else:
            print("✅ Experiment start is fast")
        
        print()
        if targets_met:
            print("🎉 All performance targets met!")
        else:
            print("❌ Some performance targets not met")
        
        # Save results
        import json
        with open("benchmark_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to benchmark_results.json")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())