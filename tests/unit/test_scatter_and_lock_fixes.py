#!/usr/bin/env python3
"""Test to verify scatter rank handling and lock initialization fixes."""

import os
import sys
import threading
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_scatter_rank_handling():
    """Test that scatter operations handle different ranks correctly."""
    print("Testing Scatter Rank Handling")
    print("=" * 30)

    try:
        # Import directly from the network_monitor module
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'rldk', 'integrations', 'openrlhf'))
        from network_monitor import DistributedNetworkMonitor

        # Test with rank 0 (source rank)
        print("\n1. Testing source rank (rank 0):")
        monitor_source = DistributedNetworkMonitor(
            world_size=2,
            rank=0,
            enable_distributed_measurements=True
        )

        try:
            scatter_bw_source = monitor_source._measure_scatter_bandwidth()
            print(f"   Source rank scatter bandwidth: {scatter_bw_source:.2f} Mbps")
            print("   ✅ Source rank scatter operation completed")
        except Exception as e:
            print(f"   ❌ Source rank scatter failed: {e}")
            return False

        # Test with rank 1 (non-source rank)
        print("\n2. Testing non-source rank (rank 1):")
        monitor_non_source = DistributedNetworkMonitor(
            world_size=2,
            rank=1,
            enable_distributed_measurements=True
        )

        try:
            scatter_bw_non_source = monitor_non_source._measure_scatter_bandwidth()
            print(f"   Non-source rank scatter bandwidth: {scatter_bw_non_source:.2f} Mbps")
            print("   ✅ Non-source rank scatter operation completed")
        except Exception as e:
            print(f"   ❌ Non-source rank scatter failed: {e}")
            return False

        print("\n✅ Scatter rank handling verified")
        return True

    except Exception as e:
        print(f"✗ Scatter rank handling test failed: {e}")
        return False

def test_lock_initialization_race_condition():
    """Test that lock initialization is thread-safe."""
    print("\nTesting Lock Initialization Race Condition")
    print("=" * 40)

    try:
        from rldk.integrations.openrlhf.callbacks import DistributedTrainingMonitor

        # Create multiple threads that will trigger lock initialization
        results = []
        errors = []
        locks_created = []

        def init_monitor_with_lock(thread_id):
            try:
                callback = DistributedTrainingMonitor()
                # This should trigger network monitor and lock initialization
                callback._collect_network_metrics()

                # Check if lock was created
                if hasattr(callback, '_network_monitor_lock'):
                    lock_id = id(callback._network_monitor_lock)
                    locks_created.append(lock_id)
                    results.append(f"Thread {thread_id}: Success - Lock ID: {lock_id}")
                else:
                    errors.append(f"Thread {thread_id}: No lock created")

            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Create and start multiple threads
        threads = []
        for i in range(10):  # More threads to stress test
            thread = threading.Thread(target=init_monitor_with_lock, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        print(f"Thread results: {results}")
        print(f"Locks created: {locks_created}")

        if errors:
            print(f"Thread errors: {errors}")
            return False

        # Check if all locks are the same (indicating no race condition)
        unique_locks = set(locks_created)
        if len(unique_locks) == 1:
            print("✅ Lock initialization race condition fixed - all threads use same lock")
            return True
        else:
            print(f"❌ Lock initialization race condition still exists - {len(unique_locks)} different locks created")
            return False

    except Exception as e:
        print(f"✗ Lock initialization test failed: {e}")
        return False

def test_scatter_code_analysis():
    """Analyze the scatter code to verify the fix."""
    print("\nTesting Scatter Code Analysis")
    print("=" * 30)

    try:
        # Read the scatter method to verify the fix
        scatter_file = os.path.join(os.path.dirname(__file__), 'src', 'rldk', 'integrations', 'openrlhf', 'network_monitor.py')

        with open(scatter_file) as f:
            content = f.read()

        # Look for the scatter method
        if 'def _measure_scatter_bandwidth' in content:
            # Check for rank-based branching
            if 'if self.rank == 0:' in content and 'else:' in content:
                print("✅ Scatter method has rank-based branching")

                # Check for proper scatter calls
                if 'dist.scatter(test_tensor, scattered_tensors, src=0)' in content and 'dist.scatter(test_tensor, src=0)' in content:
                    print("✅ Scatter method has proper rank-specific calls")
                    return True
                else:
                    print("❌ Scatter method missing proper rank-specific calls")
                    return False
            else:
                print("❌ Scatter method missing rank-based branching")
                return False
        else:
            print("❌ Scatter method not found")
            return False

    except Exception as e:
        print(f"✗ Scatter code analysis failed: {e}")
        return False

def test_lock_code_analysis():
    """Analyze the lock initialization code to verify the fix."""
    print("\nTesting Lock Code Analysis")
    print("=" * 30)

    try:
        # Read the callbacks file to verify the lock fix
        callbacks_file = os.path.join(os.path.dirname(__file__), 'src', 'rldk', 'integrations', 'openrlhf', 'callbacks.py')

        with open(callbacks_file) as f:
            content = f.read()

        # Look for the lock initialization pattern
        if '_class_lock' in content and 'self.__class__._class_lock' in content:
            print("✅ Lock initialization uses class-level lock")
            return True
        else:
            print("❌ Lock initialization missing class-level lock")
            return False

    except Exception as e:
        print(f"✗ Lock code analysis failed: {e}")
        return False

if __name__ == "__main__":
    print("Scatter and Lock Fixes Test Suite")
    print("=" * 35)

    tests = [
        ("Scatter Rank Handling", test_scatter_rank_handling),
        ("Lock Initialization Race Condition", test_lock_initialization_race_condition),
        ("Scatter Code Analysis", test_scatter_code_analysis),
        ("Lock Code Analysis", test_lock_code_analysis),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")

        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")

    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    print(f"{'='*50}")

    if passed == total:
        print("🎉 All scatter and lock fixes verified successfully!")
        sys.exit(0)
    else:
        print("❌ Some fixes need attention.")
        sys.exit(1)
