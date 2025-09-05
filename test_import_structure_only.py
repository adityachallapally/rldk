#!/usr/bin/env python3
"""Test import structure without requiring dependencies."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(src_path))

def test_adapters_import_structure():
    """Test that adapters import structure is correct."""
    print("Testing adapters import structure...")
    
    # Check that ingest.py has correct imports
    ingest_path = src_path / 'rldk' / 'ingest' / 'ingest.py'
    with open(ingest_path) as f:
        content = f.read()
    
    # Should import from local modules, not ..adapters
    if 'from .trl import TRLAdapter' in content and 'from .openrlhf import OpenRLHFAdapter' in content:
        print("✓ Adapters imported from local modules")
    else:
        print("❌ Adapters not imported from local modules")
        return False
    
    # Should not import from ..adapters
    if 'from ..adapters import' not in content:
        print("✓ No old adapters import path")
    else:
        print("❌ Still has old adapters import path")
        return False
    
    return True

def test_package_structure():
    """Test package structure with relative paths."""
    print("Testing package structure...")
    
    # Check that core modules exist
    core_modules = ['tracking', 'forensics', 'ingest', 'diff', 'determinism', 'reward', 'evals']
    
    for module in core_modules:
        module_path = src_path / 'rldk' / module
        if module_path.exists() and (module_path / '__init__.py').exists():
            print(f"✓ {module} module exists")
        else:
            print(f"❌ {module} module missing")
            return False
    
    return True

def test_card_functions_exist():
    """Test that card function files exist."""
    print("Testing card function files...")
    
    card_files = [
        ('determinism', 'determinism.py'),
        ('diff', 'drift.py'),
        ('reward', 'reward.py')
    ]
    
    for module, filename in card_files:
        file_path = src_path / 'rldk' / module / filename
        if file_path.exists():
            print(f"✓ {module}/{filename} exists")
        else:
            print(f"❌ {module}/{filename} missing")
            return False
    
    return True

def test_module_exports():
    """Test that modules export the correct functions."""
    print("Testing module exports...")
    
    # Check diff module exports
    diff_init_path = src_path / 'rldk' / 'diff' / '__init__.py'
    with open(diff_init_path) as f:
        content = f.read()
    
    if 'generate_drift_card' in content:
        print("✓ diff module exports generate_drift_card")
    else:
        print("❌ diff module missing generate_drift_card export")
        return False
    
    # Check reward module exports
    reward_init_path = src_path / 'rldk' / 'reward' / '__init__.py'
    with open(reward_init_path) as f:
        content = f.read()
    
    if 'generate_reward_card' in content:
        print("✓ reward module exports generate_reward_card")
    else:
        print("❌ reward module missing generate_reward_card export")
        return False
    
    # Check determinism module exports
    det_init_path = src_path / 'rldk' / 'determinism' / '__init__.py'
    with open(det_init_path) as f:
        content = f.read()
    
    if 'generate_determinism_card' in content:
        print("✓ determinism module exports generate_determinism_card")
    else:
        print("❌ determinism module missing generate_determinism_card export")
        return False
    
    return True

def main():
    """Run import structure tests."""
    print("=== Import Structure Test (No Dependencies) ===")
    print()
    
    tests = [
        test_adapters_import_structure,
        test_package_structure,
        test_card_functions_exist,
        test_module_exports
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print()
            else:
                print(f"❌ {test.__name__} failed")
                print()
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
            print()
    
    print(f"=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("✅ All import structure tests passed!")
        print("The adapters import fix and relative paths are working correctly.")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())