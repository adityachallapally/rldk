#!/usr/bin/env python3
"""Simple test for lock initialization fix."""

import os
import sys
import threading
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_lock_initialization_simple():
    """Test that lock initialization is thread-safe."""
    print("Testing Lock Initialization (Simple)")
    print("=" * 35)

    try:
        # Import directly from the network_monitor module
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'rldk', 'integrations', 'openrlhf'))
        from network_monitor import RealNetworkMonitor

        # Create multiple threads that will initialize the network monitor
        results = []
        errors = []
        monitor_ids = []

        def init_monitor(thread_id):
            try:
                monitor = RealNetworkMonitor(
                    enable_distributed_monitoring=True,
                    enable_distributed_measurements=False
                )
                # This should trigger network monitor initialization
                monitor.get_comprehensive_metrics()

                # Check if monitor was created
                monitor_id = id(monitor)
                monitor_ids.append(monitor_id)
                results.append(f"Thread {thread_id}: Success - Monitor ID: {monitor_id}")

            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Create and start multiple threads
        threads = []
        for i in range(10):  # More threads to stress test
            thread = threading.Thread(target=init_monitor, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        print(f"Thread results: {results}")
        print(f"Monitor IDs: {monitor_ids}")

        if errors:
            print(f"Thread errors: {errors}")
            return False

        # All threads should complete successfully
        if len(results) == 10:
            print("✅ Lock initialization race condition fixed - all threads completed successfully")
            return True
        else:
            print(f"❌ Lock initialization race condition still exists - only {len(results)} threads completed")
            return False

    except Exception as e:
        print(f"✗ Lock initialization test failed: {e}")
        return False

if __name__ == "__main__":
    print("Simple Lock Initialization Test")
    print("=" * 30)

    if test_lock_initialization_simple():
        print("🎉 Lock initialization fix verified!")
        sys.exit(0)
    else:
        print("❌ Lock initialization fix needs attention.")
        sys.exit(1)
