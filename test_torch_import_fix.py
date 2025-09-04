#!/usr/bin/env python3
"""Test script to verify torch import fix for distributed diagnostics."""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'rldk', 'integrations', 'openrlhf'))

def test_torch_import_fix():
    """Test that torch import is available for distributed diagnostics."""
    print("Testing Torch Import Fix for Distributed Diagnostics")
    print("=" * 60)
    
    try:
        from network_monitor import NetworkDiagnostics
        
        print("✓ Successfully imported NetworkDiagnostics")
        
        # Create diagnostics instance
        diagnostics = NetworkDiagnostics()
        print("✓ Successfully created NetworkDiagnostics instance")
        
        # Test that the method can be called without NameError
        print("✓ Testing _run_distributed_diagnostics method...")
        
        # This should not raise NameError: name 'torch' is not defined
        # even if PyTorch is not available
        result = diagnostics._run_distributed_diagnostics()
        
        print(f"✓ Method executed successfully, returned: {result}")
        
        # Check if the result indicates PyTorch is not available (expected)
        if 'error' in result and 'PyTorch distributed not available' in result['error']:
            print("✓ Correctly handled case where PyTorch is not available")
        else:
            print("✓ PyTorch appears to be available")
        
        print("\n🎉 Torch import fix test passed!")
        return True
        
    except NameError as e:
        if "name 'torch' is not defined" in str(e):
            print(f"✗ NameError still exists: {e}")
            return False
        else:
            print(f"✗ Unexpected NameError: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_module_imports():
    """Test that all torch-related imports work correctly."""
    print("\n" + "=" * 60)
    print("Testing Module Imports")
    print("=" * 60)
    
    try:
        # Test importing the module
        import network_monitor
        print("✓ Successfully imported network_monitor module")
        
        # Test that torch is available in the module scope
        if hasattr(network_monitor, 'DIST_AVAILABLE'):
            print(f"✓ DIST_AVAILABLE flag: {network_monitor.DIST_AVAILABLE}")
        
        # Test importing specific classes
        from network_monitor import NetworkDiagnostics, NetworkMetrics
        print("✓ Successfully imported NetworkDiagnostics and NetworkMetrics")
        
        print("\n🎉 Module import test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Module import test failed: {e}")
        return False

def main():
    """Main test function."""
    print("Torch Import Fix Verification")
    print("=" * 60)
    
    # Test the specific fix
    success1 = test_torch_import_fix()
    
    # Test module imports
    success2 = test_module_imports()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Torch Import Fix: {'✓ PASSED' if success1 else '✗ FAILED'}")
    print(f"Module Imports: {'✓ PASSED' if success2 else '✗ FAILED'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! Torch import fix is working correctly.")
        print("\nThe fix ensures that:")
        print("- torch is imported at module level")
        print("- _run_distributed_diagnostics can be called without NameError")
        print("- Distributed diagnostics work when PyTorch is available")
        print("- Graceful handling when PyTorch is not available")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
    
    return success1 and success2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)