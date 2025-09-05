#!/usr/bin/env python3
"""Test script to verify public API is accessible."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, '/workspace/src')

def test_public_api_imports():
    """Test that the main public API can be imported."""
    print("Testing public API imports...")
    
    try:
        # Test that we can import the main module
        import rldk
        print("✓ rldk module imported")
        
        # Test version
        assert hasattr(rldk, '__version__'), "Missing __version__"
        print(f"✓ Version: {rldk.__version__}")
        
        # Test main public API
        assert hasattr(rldk, 'ExperimentTracker'), "Missing ExperimentTracker"
        print("✓ ExperimentTracker available")
        
        assert hasattr(rldk, 'TrackingConfig'), "Missing TrackingConfig"
        print("✓ TrackingConfig available")
        
        # Test core functions
        core_functions = [
            'ingest_runs',
            'first_divergence', 
            'check',
            'bisect_commits',
            'health',
            'RewardHealthReport',
            'run',
            'EvalResult'
        ]
        
        for func in core_functions:
            assert hasattr(rldk, func), f"Missing {func}"
            print(f"✓ {func} available")
        
        print("✓ All public API imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_module_structure():
    """Test that individual modules can be imported."""
    print("Testing module structure...")
    
    modules = ['tracking', 'forensics', 'ingest', 'diff', 'determinism', 'reward', 'evals']
    
    for module_name in modules:
        try:
            module = __import__(f'rldk.{module_name}', fromlist=[''])
            print(f"✓ {module_name} module imported")
        except ImportError as e:
            print(f"❌ Failed to import {module_name}: {e}")
            return False
    
    print("✓ All modules can be imported")
    return True

def test_old_imports_removed():
    """Test that old import paths no longer work."""
    print("Testing old imports are removed...")
    
    old_imports = [
        'rldk.artifacts',
        'rldk.adapters', 
        'rldk.replay',
        'rldk.bisect',
        'rldk.cards'
    ]
    
    for old_import in old_imports:
        try:
            __import__(old_import)
            print(f"❌ Old import {old_import} still works")
            return False
        except ImportError:
            print(f"✓ Old import {old_import} correctly removed")
    
    print("✓ All old imports properly removed")
    return True

def main():
    """Run all public API tests."""
    print("=== RLDK Public API Test ===")
    print()
    
    try:
        if not test_public_api_imports():
            return 1
        print()
        
        if not test_module_structure():
            return 1
        print()
        
        if not test_old_imports_removed():
            return 1
        print()
        
        print("✅ All public API tests passed!")
        return 0
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())