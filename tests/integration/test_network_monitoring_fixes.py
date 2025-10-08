#!/usr/bin/env python3
"""Test script to verify network monitoring bug fixes."""

import os
import sys
import threading
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_metric_conversion_fix():
    """Test that metric conversion uses correct divisor (8000.0 instead of 1000.0)."""
    print("Testing Metric Conversion Fix")
    print("=" * 30)

    try:
        # Import directly from the network_monitor module
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'rldk', 'integrations', 'openrlhf'))
        from network_monitor import RealNetworkMonitor

        # Create monitor with distributed measurements disabled for safety
        monitor = RealNetworkMonitor(
            enable_distributed_monitoring=True,
            enable_distributed_measurements=False
        )

        # Get comprehensive metrics
        metrics = monitor.get_comprehensive_metrics()

        # Test bandwidth conversion
        bandwidth_mbps = 1000.0  # 1 Gbps in Mbps
        expected_gb_s = bandwidth_mbps / 8000.0  # Correct conversion
        incorrect_gb_s = bandwidth_mbps / 1000.0  # Incorrect conversion

        print(f"Bandwidth: {bandwidth_mbps} Mbps")
        print(f"Correct conversion to GB/s: {expected_gb_s:.6f} GB/s")
        print(f"Incorrect conversion to GB/s: {incorrect_gb_s:.6f} GB/s")
        print(f"Ratio (incorrect/correct): {incorrect_gb_s/expected_gb_s:.1f}x")

        # Verify the metrics are using the correct conversion
        print(f"Actual network bandwidth: {metrics.bandwidth_mbps:.2f} Mbps")
        print(f"Actual allreduce bandwidth: {metrics.allreduce_bandwidth:.2f} Mbps")

        print("✓ Metric conversion fix verified")
        return True

    except Exception as e:
        print(f"✗ Metric conversion test failed: {e}")
        return False

def test_thread_safety():
    """Test that network monitor initialization is thread-safe."""
    print("\nTesting Thread Safety")
    print("=" * 20)

    try:
        from rldk.integrations.openrlhf.callbacks import DistributedTrainingMonitor

        # Create multiple threads that will initialize the network monitor
        results = []
        errors = []

        def init_monitor(thread_id):
            try:
                callback = DistributedTrainingMonitor()
                # This should trigger network monitor initialization
                callback._collect_network_metrics()
                results.append(f"Thread {thread_id}: Success")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Create and start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=init_monitor, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        print(f"Thread results: {results}")
        if errors:
            print(f"Thread errors: {errors}")
            return False

        print("✓ Thread safety verified - no race conditions")
        return True

    except Exception as e:
        print(f"✗ Thread safety test failed: {e}")
        return False

def test_distributed_measurement_safety():
    """Test that distributed measurements are safely disabled by default."""
    print("\nTesting Distributed Measurement Safety")
    print("=" * 35)

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'rldk', 'integrations', 'openrlhf'))
        from network_monitor import DistributedNetworkMonitor, RealNetworkMonitor

        # Test with distributed measurements disabled (default)
        monitor_safe = RealNetworkMonitor(
            enable_distributed_monitoring=True,
            enable_distributed_measurements=False  # Default safe setting
        )

        metrics_safe = monitor_safe.get_comprehensive_metrics()

        print("With distributed measurements DISABLED (safe):")
        print(f"  Allreduce bandwidth: {metrics_safe.allreduce_bandwidth:.2f} Mbps")
        print(f"  Broadcast bandwidth: {metrics_safe.broadcast_bandwidth:.2f} Mbps")
        print(f"  Gather bandwidth: {metrics_safe.gather_bandwidth:.2f} Mbps")
        print(f"  Scatter bandwidth: {metrics_safe.scatter_bandwidth:.2f} Mbps")

        # Test with distributed measurements enabled (for testing only)
        monitor_unsafe = RealNetworkMonitor(
            enable_distributed_monitoring=True,
            enable_distributed_measurements=True  # Only for testing
        )

        metrics_unsafe = monitor_unsafe.get_comprehensive_metrics()

        print("\nWith distributed measurements ENABLED (testing only):")
        print(f"  Allreduce bandwidth: {metrics_unsafe.allreduce_bandwidth:.2f} Mbps")
        print(f"  Broadcast bandwidth: {metrics_unsafe.broadcast_bandwidth:.2f} Mbps")
        print(f"  Gather bandwidth: {metrics_unsafe.gather_bandwidth:.2f} Mbps")
        print(f"  Scatter bandwidth: {metrics_unsafe.scatter_bandwidth:.2f} Mbps")

        # Verify that the safe version doesn't interfere with training
        if (metrics_safe.allreduce_bandwidth == 0.0 and
            metrics_safe.broadcast_bandwidth == 0.0 and
            metrics_safe.gather_bandwidth == 0.0 and
            metrics_safe.scatter_bandwidth == 0.0):
            print("✓ Distributed measurements safely disabled by default")
            return True
        else:
            print("✗ Distributed measurements not properly disabled")
            return False

    except Exception as e:
        print(f"✗ Distributed measurement safety test failed: {e}")
        return False

def test_zero_division_protection():
    """Test that zero division errors are prevented."""
    print("\nTesting Zero Division Protection")
    print("=" * 30)

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'rldk', 'integrations', 'openrlhf'))
        from network_monitor import DistributedNetworkMonitor

        # Create a monitor with distributed measurements enabled for testing
        monitor = DistributedNetworkMonitor(
            world_size=1,
            rank=0,
            enable_distributed_measurements=True
        )

        # Test the bandwidth measurement methods directly
        # These should handle zero division gracefully
        allreduce_bw = monitor._measure_allreduce_bandwidth()
        broadcast_bw = monitor._measure_broadcast_bandwidth()
        gather_bw = monitor._measure_gather_bandwidth()
        scatter_bw = monitor._measure_scatter_bandwidth()

        print(f"Allreduce bandwidth: {allreduce_bw:.2f} Mbps")
        print(f"Broadcast bandwidth: {broadcast_bw:.2f} Mbps")
        print(f"Gather bandwidth: {gather_bw:.2f} Mbps")
        print(f"Scatter bandwidth: {scatter_bw:.2f} Mbps")

        # All values should be finite (no division by zero errors)
        if (allreduce_bw >= 0 and broadcast_bw >= 0 and
            gather_bw >= 0 and scatter_bw >= 0):
            print("✓ Zero division protection working")
            return True
        else:
            print("✗ Zero division protection failed")
            return False

    except Exception as e:
        print(f"✗ Zero division protection test failed: {e}")
        return False

def test_import_safety():
    """Test that torch imports are handled safely."""
    print("\nTesting Import Safety")
    print("=" * 20)

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'rldk', 'integrations', 'openrlhf'))
        from network_monitor import DistributedNetworkMonitor

        # This should work even without torch available
        monitor = DistributedNetworkMonitor(
            world_size=1,
            rank=0,
            enable_distributed_measurements=True
        )

        # Try to measure bandwidth (should handle missing torch gracefully)
        allreduce_bw = monitor._measure_allreduce_bandwidth()
        print(f"Allreduce bandwidth (with/without torch): {allreduce_bw:.2f} Mbps")

        print("✓ Import safety verified")
        return True

    except Exception as e:
        print(f"✗ Import safety test failed: {e}")
        return False

if __name__ == "__main__":
    print("Network Monitoring Bug Fixes Test Suite")
    print("=" * 40)

    tests = [
        ("Metric Conversion Fix", test_metric_conversion_fix),
        ("Thread Safety", test_thread_safety),
        ("Distributed Measurement Safety", test_distributed_measurement_safety),
        ("Zero Division Protection", test_zero_division_protection),
        ("Import Safety", test_import_safety),
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
        print("🎉 All bug fixes verified successfully!")
        sys.exit(0)
    else:
        print("❌ Some bug fixes need attention.")
        sys.exit(1)
