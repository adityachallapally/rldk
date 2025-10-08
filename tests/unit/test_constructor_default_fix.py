#!/usr/bin/env python3
"""Test to verify constructor default fix for safety."""

import os
import sys
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_constructor_default_safety():
    """Test that constructor defaults to safe settings."""
    print("Testing Constructor Default Safety")
    print("=" * 35)

    try:
        # Import directly from the network_monitor module
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'rldk', 'integrations', 'openrlhf'))
        from network_monitor import DistributedNetworkMonitor, RealNetworkMonitor

        # Test 1: DistributedNetworkMonitor default
        print("\n1. Testing DistributedNetworkMonitor default:")
        monitor1 = DistributedNetworkMonitor()  # Should default to False
        print(f"   enable_distributed_measurements: {monitor1.enable_distributed_measurements}")

        if monitor1.enable_distributed_measurements:
            print("   ❌ FAILED: Should default to False for safety")
            return False
        else:
            print("   ✅ PASSED: Correctly defaults to False")

        # Test 2: RealNetworkMonitor default
        print("\n2. Testing RealNetworkMonitor default:")
        monitor2 = RealNetworkMonitor()  # Should default to False
        print(f"   enable_distributed_measurements: {monitor2.enable_distributed_measurements}")

        if monitor2.enable_distributed_measurements:
            print("   ❌ FAILED: Should default to False for safety")
            return False
        else:
            print("   ✅ PASSED: Correctly defaults to False")

        # Test 3: Explicit True setting
        print("\n3. Testing explicit True setting:")
        monitor3 = DistributedNetworkMonitor(enable_distributed_measurements=True)
        print(f"   enable_distributed_measurements: {monitor3.enable_distributed_measurements}")

        if monitor3.enable_distributed_measurements:
            print("   ✅ PASSED: Correctly accepts explicit True")
        else:
            print("   ❌ FAILED: Should accept explicit True")
            return False

        # Test 4: Explicit False setting
        print("\n4. Testing explicit False setting:")
        monitor4 = DistributedNetworkMonitor(enable_distributed_measurements=False)
        print(f"   enable_distributed_measurements: {monitor4.enable_distributed_measurements}")

        if not monitor4.enable_distributed_measurements:
            print("   ✅ PASSED: Correctly accepts explicit False")
        else:
            print("   ❌ FAILED: Should accept explicit False")
            return False

        print("\n✅ All constructor default tests passed!")
        return True

    except Exception as e:
        print(f"✗ Constructor default test failed: {e}")
        return False

def test_safety_behavior():
    """Test that the safe default behavior works correctly."""
    print("\nTesting Safety Behavior")
    print("=" * 25)

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'rldk', 'integrations', 'openrlhf'))
        from network_monitor import DistributedNetworkMonitor

        # Test with default (safe) settings
        monitor_safe = DistributedNetworkMonitor()  # Defaults to False
        metrics_safe = monitor_safe.measure_distributed_metrics()

        print("With default (safe) settings:")
        print(f"  enable_distributed_measurements: {monitor_safe.enable_distributed_measurements}")
        print(f"  Allreduce bandwidth: {metrics_safe.allreduce_bandwidth:.2f} Mbps")
        print(f"  Broadcast bandwidth: {metrics_safe.broadcast_bandwidth:.2f} Mbps")
        print(f"  Gather bandwidth: {metrics_safe.gather_bandwidth:.2f} Mbps")
        print(f"  Scatter bandwidth: {metrics_safe.scatter_bandwidth:.2f} Mbps")

        # Verify that distributed measurements are disabled
        if (metrics_safe.allreduce_bandwidth == 0.0 and
            metrics_safe.broadcast_bandwidth == 0.0 and
            metrics_safe.gather_bandwidth == 0.0 and
            metrics_safe.scatter_bandwidth == 0.0):
            print("✅ Safety behavior verified - no active distributed measurements")
            return True
        else:
            print("❌ Safety behavior failed - distributed measurements active")
            return False

    except Exception as e:
        print(f"✗ Safety behavior test failed: {e}")
        return False

if __name__ == "__main__":
    print("Constructor Default Safety Test")
    print("=" * 30)

    test1_passed = test_constructor_default_safety()
    test2_passed = test_safety_behavior()

    if test1_passed and test2_passed:
        print("\n🎉 Constructor default safety verified!")
        print("✅ Distributed measurements correctly default to False for safety")
        sys.exit(0)
    else:
        print("\n❌ Constructor default safety issues found.")
        sys.exit(1)
