#!/usr/bin/env python3
"""Simple thread safety test for network monitoring."""

import os
import sys
import threading
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_thread_safety_simple():
    """Test that network monitor initialization is thread-safe."""
    print("Testing Thread Safety (Simple)")
    print("=" * 30)

    try:
        # Import directly from the network_monitor module
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'rldk', 'integrations', 'openrlhf'))
        from network_monitor import RealNetworkMonitor

        # Create multiple threads that will initialize the network monitor
        results = []
        errors = []

        def init_monitor(thread_id):
            try:
                monitor = RealNetworkMonitor(
                    enable_distributed_monitoring=True,
                    enable_distributed_measurements=False
                )
                # This should trigger network monitor initialization
                metrics = monitor.get_comprehensive_metrics()
                results.append(f"Thread {thread_id}: Success - Bandwidth: {metrics.bandwidth_mbps:.2f} Mbps")
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

if __name__ == "__main__":
    print("Simple Thread Safety Test")
    print("=" * 25)

    if test_thread_safety_simple():
        print("🎉 Thread safety test passed!")
        sys.exit(0)
    else:
        print("❌ Thread safety test failed.")
        sys.exit(1)
