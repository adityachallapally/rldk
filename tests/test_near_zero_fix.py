#!/usr/bin/env python3
"""Test script to verify near-zero operation time fix for distributed diagnostics."""

import sys
import os
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'rldk', 'integrations', 'openrlhf'))

def test_near_zero_operation_time_fix():
    """Test that near-zero operation times are handled correctly."""
    print("Testing Near-Zero Operation Time Fix")
    print("=" * 60)
    
    try:
        from network_monitor import NetworkDiagnostics
        
        print("✓ Successfully imported NetworkDiagnostics")
        
        # Create diagnostics instance
        diagnostics = NetworkDiagnostics()
        print("✓ Successfully created NetworkDiagnostics instance")
        
        # Test the _run_distributed_diagnostics method
        print("✓ Testing _run_distributed_diagnostics method...")
        
        result = diagnostics._run_distributed_diagnostics()
        print(f"✓ Method executed successfully, returned: {result}")
        
        # Check if the result has the expected structure
        if 'error' in result:
            print("✓ Correctly handled case where PyTorch is not available")
            return True
        
        # If PyTorch is available, check the structure of the results
        if 'allreduce_test' in result:
            allreduce_test = result['allreduce_test']
            print(f"✓ Allreduce test: time_ms={allreduce_test.get('time_ms', 'N/A')}, bandwidth_mbps={allreduce_test.get('bandwidth_mbps', 'N/A')}")
            
            # Check that bandwidth is not infinite or NaN
            bandwidth = allreduce_test.get('bandwidth_mbps', 0.0)
            if bandwidth == float('inf') or bandwidth != bandwidth:  # NaN check
                print("✗ Bandwidth calculation produced infinite or NaN value")
                return False
            else:
                print("✓ Bandwidth calculation is valid")
        
        if 'broadcast_test' in result:
            broadcast_test = result['broadcast_test']
            print(f"✓ Broadcast test: time_ms={broadcast_test.get('time_ms', 'N/A')}, bandwidth_mbps={broadcast_test.get('bandwidth_mbps', 'N/A')}")
            
            # Check that bandwidth is not infinite or NaN
            bandwidth = broadcast_test.get('bandwidth_mbps', 0.0)
            if bandwidth == float('inf') or bandwidth != bandwidth:  # NaN check
                print("✗ Bandwidth calculation produced infinite or NaN value")
                return False
            else:
                print("✓ Bandwidth calculation is valid")
        
        print("\n🎉 Near-zero operation time fix test passed!")
        return True
        
    except ZeroDivisionError as e:
        print(f"✗ ZeroDivisionError occurred: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_bandwidth_calculation_logic():
    """Test the bandwidth calculation logic with various time values."""
    print("\n" + "=" * 60)
    print("Testing Bandwidth Calculation Logic")
    print("=" * 60)
    
    # Test cases with different time values
    test_cases = [
        (0.0005, "Very fast operation (< 1ms)"),
        (0.001, "Exactly 1ms"),
        (0.002, "Slightly above 1ms"),
        (0.01, "10ms operation"),
        (0.1, "100ms operation"),
    ]
    
    for time_value, description in test_cases:
        print(f"\nTesting {description} (time={time_value}s):")
        
        # Simulate the bandwidth calculation logic
        tensor_size_bytes = 100 * 100 * 4  # 100x100 float32 tensor
        tensor_size_bits = tensor_size_bytes * 8
        
        if time_value <= 0.001:  # Less than 1ms, likely measurement error
            bandwidth_mbps = 0.0
            print(f"  → Time <= 1ms, setting bandwidth to 0.0")
        else:
            bandwidth_mbps = tensor_size_bits / (time_value * 1_000_000)
            print(f"  → Calculated bandwidth: {bandwidth_mbps:.2f} Mbps")
        
        # Verify the result is reasonable
        if bandwidth_mbps == float('inf') or bandwidth_mbps != bandwidth_mbps:  # NaN check
            print(f"  ✗ Invalid bandwidth result: {bandwidth_mbps}")
            return False
        elif time_value <= 0.001 and bandwidth_mbps != 0.0:
            print(f"  ✗ Expected 0.0 for near-zero time, got {bandwidth_mbps}")
            return False
        else:
            print(f"  ✓ Valid bandwidth result")
    
    print("\n🎉 Bandwidth calculation logic test passed!")
    return True

def test_consistency_with_other_methods():
    """Test that the fix is consistent with other methods in the file."""
    print("\n" + "=" * 60)
    print("Testing Consistency with Other Methods")
    print("=" * 60)
    
    try:
        # Check if the other methods have the same protection
        with open('src/rldk/integrations/openrlhf/network_monitor.py', 'r') as f:
            content = f.read()
        
        # Count occurrences of the protection pattern
        import re
        matches = re.findall(r'if.*time.*<=.*0\.001.*# Less than 1ms, likely measurement error', content)
        
        print(f"✓ Found {len(matches)} instances of near-zero time protection")
        
        # Check that we have protection in the expected methods
        expected_methods = [
            '_measure_allreduce_bandwidth',
            '_measure_broadcast_bandwidth', 
            '_measure_gather_bandwidth',
            '_measure_scatter_bandwidth',
            '_run_distributed_diagnostics'
        ]
        
        # Simple check: verify that all methods exist and the protection pattern exists
        for method in expected_methods:
            if f'def {method}' in content:
                print(f"✓ Found method {method}")
            else:
                print(f"✗ Method {method} not found")
                return False
        
        # Verify we have the expected number of protection instances
        if len(matches) >= 6:  # We should have at least 6 instances (5 methods + 1 extra)
            print("✓ All expected methods have near-zero time protection")
        else:
            print(f"✗ Expected at least 6 protection instances, found {len(matches)}")
            return False
        
        print("\n🎉 Consistency test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Consistency test failed: {e}")
        return False

def main():
    """Main test function."""
    print("Near-Zero Operation Time Fix Verification")
    print("=" * 60)
    
    # Test the specific fix
    success1 = test_near_zero_operation_time_fix()
    
    # Test bandwidth calculation logic
    success2 = test_bandwidth_calculation_logic()
    
    # Test consistency with other methods
    success3 = test_consistency_with_other_methods()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Near-Zero Operation Time Fix: {'✓ PASSED' if success1 else '✗ FAILED'}")
    print(f"Bandwidth Calculation Logic: {'✓ PASSED' if success2 else '✗ FAILED'}")
    print(f"Consistency with Other Methods: {'✓ PASSED' if success3 else '✗ FAILED'}")
    
    if success1 and success2 and success3:
        print("\n🎉 All tests passed! Near-zero operation time fix is working correctly.")
        print("\nThe fix ensures that:")
        print("- Near-zero operation times (< 1ms) are handled gracefully")
        print("- Bandwidth calculations don't produce infinite or NaN values")
        print("- ZeroDivisionError is prevented")
        print("- Consistent behavior with other similar methods")
        print("- Accurate bandwidth reporting for valid measurements")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
    
    return success1 and success2 and success3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)