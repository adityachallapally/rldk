#!/usr/bin/env python3
"""
Performance tests for RLDK experiment tracking system.

This script tests the performance improvements made to the tracking system,
including async initialization, caching, timeouts, and intelligent sampling.
"""

import asyncio
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import shutil

# Import RLDK components
from src.rldk.config.settings import RLDKSettings
from src.rldk.tracking.config import TrackingConfig
from src.rldk.tracking.tracker import ExperimentTracker
from src.rldk.tracking.dataset_tracker import DatasetTracker
from src.rldk.tracking.model_tracker import ModelTracker
from src.rldk.tracking.environment_tracker import EnvironmentTracker
from src.rldk.tracking.git_tracker import GitTracker


class PerformanceTester:
    """Test performance of RLDK tracking components."""
    
    def __init__(self):
        self.results = {}
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_settings_initialization(self) -> Dict[str, Any]:
        """Test settings initialization performance."""
        print("Testing settings initialization...")
        
        start_time = time.time()
        settings = RLDKSettings()
        init_time = time.time() - start_time
        
        # Test async initialization
        start_time = time.time()
        settings.initialize_async()
        async_init_time = time.time() - start_time
        
        # Test caching
        start_time = time.time()
        cache_dir = settings.get_cache_dir()
        output_dir = settings.get_output_dir()
        runs_dir = settings.get_runs_dir()
        cache_time = time.time() - start_time
        
        return {
            "sync_init_time": init_time,
            "async_init_time": async_init_time,
            "cache_access_time": cache_time,
            "performance_config": settings.get_performance_config()
        }
    
    def test_dataset_tracking_performance(self) -> Dict[str, Any]:
        """Test dataset tracking performance with various dataset sizes."""
        print("Testing dataset tracking performance...")
        
        tracker = DatasetTracker()
        results = {}
        
        # Test small dataset
        small_data = np.random.randn(1000, 10)
        start_time = time.time()
        tracker.track_dataset(small_data, "small_dataset")
        small_time = time.time() - start_time
        
        # Test medium dataset
        medium_data = np.random.randn(10000, 50)
        start_time = time.time()
        tracker.track_dataset(medium_data, "medium_dataset")
        medium_time = time.time() - start_time
        
        # Test large dataset
        large_data = np.random.randn(100000, 100)
        start_time = time.time()
        tracker.track_dataset(large_data, "large_dataset")
        large_time = time.time() - start_time
        
        # Test very large dataset
        very_large_data = np.random.randn(1000000, 50)
        start_time = time.time()
        tracker.track_dataset(very_large_data, "very_large_dataset")
        very_large_time = time.time() - start_time
        
        # Test caching
        start_time = time.time()
        tracker.track_dataset(small_data, "small_dataset")  # Should use cache
        cache_time = time.time() - start_time
        
        return {
            "small_dataset_time": small_time,
            "medium_dataset_time": medium_time,
            "large_dataset_time": large_time,
            "very_large_dataset_time": very_large_time,
            "cache_time": cache_time,
            "tracking_summary": tracker.get_tracking_summary()
        }
    
    def test_model_tracking_performance(self) -> Dict[str, Any]:
        """Test model tracking performance with various model sizes."""
        print("Testing model tracking performance...")
        
        tracker = ModelTracker()
        results = {}
        
        # Test small model
        small_model = torch.nn.Sequential(
            torch.nn.Linear(10, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 1)
        )
        start_time = time.time()
        tracker.track_model(small_model, "small_model")
        small_time = time.time() - start_time
        
        # Test medium model
        medium_model = torch.nn.Sequential(
            torch.nn.Linear(100, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 10)
        )
        start_time = time.time()
        tracker.track_model(medium_model, "medium_model")
        medium_time = time.time() - start_time
        
        # Test large model
        large_model = torch.nn.Sequential(
            torch.nn.Linear(1000, 2000),
            torch.nn.ReLU(),
            torch.nn.Linear(2000, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 100)
        )
        start_time = time.time()
        tracker.track_model(large_model, "large_model")
        large_time = time.time() - start_time
        
        # Test caching
        start_time = time.time()
        tracker.track_model(small_model, "small_model")  # Should use cache
        cache_time = time.time() - start_time
        
        return {
            "small_model_time": small_time,
            "medium_model_time": medium_time,
            "large_model_time": large_time,
            "cache_time": cache_time,
            "tracking_summary": tracker.get_tracking_summary()
        }
    
    def test_environment_tracking_performance(self) -> Dict[str, Any]:
        """Test environment tracking performance."""
        print("Testing environment tracking performance...")
        
        tracker = EnvironmentTracker()
        
        # Test first capture
        start_time = time.time()
        env_info = tracker.capture_environment()
        first_capture_time = time.time() - start_time
        
        # Test cached capture
        start_time = time.time()
        env_info_cached = tracker.capture_environment()
        cached_capture_time = time.time() - start_time
        
        return {
            "first_capture_time": first_capture_time,
            "cached_capture_time": cached_capture_time,
            "cache_ratio": first_capture_time / max(cached_capture_time, 0.001),
            "tracking_summary": tracker.get_tracking_summary()
        }
    
    def test_git_tracking_performance(self) -> Dict[str, Any]:
        """Test git tracking performance."""
        print("Testing git tracking performance...")
        
        tracker = GitTracker()
        
        # Test first capture
        start_time = time.time()
        git_info = tracker.capture_git_state()
        first_capture_time = time.time() - start_time
        
        # Test cached capture
        start_time = time.time()
        git_info_cached = tracker.capture_git_state()
        cached_capture_time = time.time() - start_time
        
        return {
            "first_capture_time": first_capture_time,
            "cached_capture_time": cached_capture_time,
            "cache_ratio": first_capture_time / max(cached_capture_time, 0.001),
            "tracking_summary": tracker.get_tracking_summary()
        }
    
    async def test_tracker_async_initialization(self) -> Dict[str, Any]:
        """Test async initialization of ExperimentTracker."""
        print("Testing async tracker initialization...")
        
        config = TrackingConfig(
            experiment_name="performance_test",
            output_dir=self.temp_dir / "experiments"
        )
        
        # Test async initialization
        tracker = ExperimentTracker(config)
        start_time = time.time()
        await tracker.initialize_async()
        async_init_time = time.time() - start_time
        
        # Test sync initialization
        tracker_sync = ExperimentTracker(config)
        start_time = time.time()
        tracker_sync.initialize_sync()
        sync_init_time = time.time() - start_time
        
        return {
            "async_init_time": async_init_time,
            "sync_init_time": sync_init_time,
            "speedup_ratio": sync_init_time / max(async_init_time, 0.001)
        }
    
    def test_tracker_start_experiment_performance(self) -> Dict[str, Any]:
        """Test experiment start performance."""
        print("Testing experiment start performance...")
        
        config = TrackingConfig(
            experiment_name="performance_test",
            output_dir=self.temp_dir / "experiments"
        )
        
        tracker = ExperimentTracker(config)
        start_time = time.time()
        tracking_data = tracker.start_experiment()
        start_time_total = time.time() - start_time
        
        return {
            "start_experiment_time": start_time_total,
            "tracking_data_keys": list(tracking_data.keys()),
            "has_environment": bool(tracking_data.get("environment")),
            "has_git": bool(tracking_data.get("git")),
            "has_seeds": bool(tracking_data.get("seeds"))
        }
    
    def test_timeout_handling(self) -> Dict[str, Any]:
        """Test timeout handling for various operations."""
        print("Testing timeout handling...")
        
        results = {}
        
        # Test dataset timeout
        tracker = DatasetTracker()
        very_large_data = np.random.randn(10000000, 100)  # Very large dataset
        
        start_time = time.time()
        try:
            tracker.track_dataset(very_large_data, "timeout_test")
            timeout_success = True
        except Exception as e:
            timeout_success = False
            timeout_error = str(e)
        timeout_time = time.time() - start_time
        
        results["dataset_timeout_success"] = timeout_success
        results["dataset_timeout_time"] = timeout_time
        if not timeout_success:
            results["dataset_timeout_error"] = timeout_error
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all performance tests."""
        print("Running RLDK tracking performance tests...")
        print("=" * 50)
        
        all_results = {}
        
        try:
            # Test settings
            all_results["settings"] = self.test_settings_initialization()
            
            # Test dataset tracking
            all_results["dataset_tracking"] = self.test_dataset_tracking_performance()
            
            # Test model tracking
            all_results["model_tracking"] = self.test_model_tracking_performance()
            
            # Test environment tracking
            all_results["environment_tracking"] = self.test_environment_tracking_performance()
            
            # Test git tracking
            all_results["git_tracking"] = self.test_git_tracking_performance()
            
            # Test async initialization
            all_results["async_initialization"] = asyncio.run(self.test_tracker_async_initialization())
            
            # Test experiment start
            all_results["experiment_start"] = self.test_tracker_start_experiment_performance()
            
            # Test timeout handling
            all_results["timeout_handling"] = self.test_timeout_handling()
            
        except Exception as e:
            print(f"Error during testing: {e}")
            all_results["error"] = str(e)
        
        finally:
            self.cleanup()
        
        return all_results
    
    def print_results(self, results: Dict[str, Any]):
        """Print performance test results."""
        print("\n" + "=" * 50)
        print("PERFORMANCE TEST RESULTS")
        print("=" * 50)
        
        # Settings performance
        if "settings" in results:
            settings = results["settings"]
            print(f"\nSettings Initialization:")
            print(f"  Sync init time: {settings['sync_init_time']:.3f}s")
            print(f"  Async init time: {settings['async_init_time']:.3f}s")
            print(f"  Cache access time: {settings['cache_access_time']:.3f}s")
        
        # Dataset tracking performance
        if "dataset_tracking" in results:
            dataset = results["dataset_tracking"]
            print(f"\nDataset Tracking:")
            print(f"  Small dataset (1K samples): {dataset['small_dataset_time']:.3f}s")
            print(f"  Medium dataset (10K samples): {dataset['medium_dataset_time']:.3f}s")
            print(f"  Large dataset (100K samples): {dataset['large_dataset_time']:.3f}s")
            print(f"  Very large dataset (1M samples): {dataset['very_large_dataset_time']:.3f}s")
            print(f"  Cache access time: {dataset['cache_time']:.3f}s")
        
        # Model tracking performance
        if "model_tracking" in results:
            model = results["model_tracking"]
            print(f"\nModel Tracking:")
            print(f"  Small model: {model['small_model_time']:.3f}s")
            print(f"  Medium model: {model['medium_model_time']:.3f}s")
            print(f"  Large model: {model['large_model_time']:.3f}s")
            print(f"  Cache access time: {model['cache_time']:.3f}s")
        
        # Environment tracking performance
        if "environment_tracking" in results:
            env = results["environment_tracking"]
            print(f"\nEnvironment Tracking:")
            print(f"  First capture: {env['first_capture_time']:.3f}s")
            print(f"  Cached capture: {env['cached_capture_time']:.3f}s")
            print(f"  Cache speedup: {env['cache_ratio']:.1f}x")
        
        # Git tracking performance
        if "git_tracking" in results:
            git = results["git_tracking"]
            print(f"\nGit Tracking:")
            print(f"  First capture: {git['first_capture_time']:.3f}s")
            print(f"  Cached capture: {git['cached_capture_time']:.3f}s")
            print(f"  Cache speedup: {git['cache_ratio']:.1f}x")
        
        # Async initialization performance
        if "async_initialization" in results:
            async_init = results["async_initialization"]
            print(f"\nAsync Initialization:")
            print(f"  Async init time: {async_init['async_init_time']:.3f}s")
            print(f"  Sync init time: {async_init['sync_init_time']:.3f}s")
            print(f"  Speedup ratio: {async_init['speedup_ratio']:.1f}x")
        
        # Experiment start performance
        if "experiment_start" in results:
            exp_start = results["experiment_start"]
            print(f"\nExperiment Start:")
            print(f"  Total start time: {exp_start['start_experiment_time']:.3f}s")
            print(f"  Environment captured: {exp_start['has_environment']}")
            print(f"  Git captured: {exp_start['has_git']}")
            print(f"  Seeds captured: {exp_start['has_seeds']}")
        
        # Timeout handling
        if "timeout_handling" in results:
            timeout = results["timeout_handling"]
            print(f"\nTimeout Handling:")
            print(f"  Dataset timeout success: {timeout['dataset_timeout_success']}")
            print(f"  Dataset timeout time: {timeout['dataset_timeout_time']:.3f}s")
        
        print("\n" + "=" * 50)
        print("Performance tests completed!")
        print("=" * 50)


def main():
    """Main function to run performance tests."""
    tester = PerformanceTester()
    
    try:
        results = tester.run_all_tests()
        tester.print_results(results)
        
        # Save results to file
        import json
        with open("tracking_performance_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to tracking_performance_results.json")
        
    except Exception as e:
        print(f"Performance testing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())