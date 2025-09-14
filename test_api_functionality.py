#!/usr/bin/env python3
"""Test basic API functionality after import optimizations."""

import sys
import time

def test_basic_api():
    """Test basic API functionality."""
    print("Testing basic API functionality...")
    
    try:
        import rldk
        print("✓ RLDK imported successfully")
        
        version = rldk.__version__
        print(f"✓ Version: {version}")
        
        print("✓ Functions accessible:", hasattr(rldk, 'ingest_runs'))
        print("✓ Classes accessible:", hasattr(rldk, 'ExperimentTracker'))
        
        settings = rldk.settings
        print(f"✓ Settings accessible: {type(settings)}")
        
        return True
        
    except Exception as e:
        print(f"✗ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("RLDK API Functionality Test")
    print("=" * 40)
    
    success = test_basic_api()
    
    if success:
        print("\n🟢 SUCCESS: Basic API functionality working")
        return 0
    else:
        print("\n🔴 FAILED: Basic API functionality broken")
        return 1

if __name__ == "__main__":
    sys.exit(main())
