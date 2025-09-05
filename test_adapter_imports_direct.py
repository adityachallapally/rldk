#!/usr/bin/env python3
"""Test adapter imports directly without triggering main package import."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(src_path))

def test_adapter_files_exist():
    """Test that adapter files exist in the ingest directory."""
    print("Testing adapter files exist...")
    
    adapter_files = ['base.py', 'trl.py', 'openrlhf.py', 'wandb.py', 'custom_jsonl.py']
    
    for filename in adapter_files:
        file_path = src_path / 'rldk' / 'ingest' / filename
        if file_path.exists():
            print(f"✓ {filename} exists")
        else:
            print(f"❌ {filename} missing")
            return False
    
    return True

def test_adapter_imports_syntax():
    """Test that adapter files have correct import syntax."""
    print("Testing adapter import syntax...")
    
    # Test TRL adapter
    trl_path = src_path / 'rldk' / 'ingest' / 'trl.py'
    with open(trl_path) as f:
        content = f.read()
    
    if 'from .base import BaseAdapter' in content:
        print("✓ TRL adapter imports base correctly")
    else:
        print("❌ TRL adapter import issue")
        return False
    
    # Test base adapter
    base_path = src_path / 'rldk' / 'ingest' / 'base.py'
    with open(base_path) as f:
        content = f.read()
    
    if 'from abc import ABC, abstractmethod' in content:
        print("✓ Base adapter imports correctly")
    else:
        print("❌ Base adapter import issue")
        return False
    
    return True

def test_ingest_imports():
    """Test that ingest.py has correct adapter imports."""
    print("Testing ingest.py adapter imports...")
    
    ingest_path = src_path / 'rldk' / 'ingest' / 'ingest.py'
    with open(ingest_path) as f:
        content = f.read()
    
    # Should import from local modules
    expected_imports = [
        'from .trl import TRLAdapter',
        'from .openrlhf import OpenRLHFAdapter',
        'from .wandb import WandBAdapter',
        'from .custom_jsonl import CustomJSONLAdapter'
    ]
    
    for import_line in expected_imports:
        if import_line in content:
            print(f"✓ {import_line}")
        else:
            print(f"❌ Missing: {import_line}")
            return False
    
    # Should not import from old adapters module
    if 'from ..adapters import' not in content:
        print("✓ No old adapters import")
    else:
        print("❌ Still has old adapters import")
        return False
    
    return True

def test_adapter_class_definitions():
    """Test that adapter classes are properly defined."""
    print("Testing adapter class definitions...")
    
    adapter_files = {
        'trl.py': 'TRLAdapter',
        'openrlhf.py': 'OpenRLHFAdapter', 
        'wandb.py': 'WandBAdapter',
        'custom_jsonl.py': 'CustomJSONLAdapter'
    }
    
    for filename, class_name in adapter_files.items():
        file_path = src_path / 'rldk' / 'ingest' / filename
        with open(file_path) as f:
            content = f.read()
        
        if f'class {class_name}' in content:
            print(f"✓ {class_name} class defined in {filename}")
        else:
            print(f"❌ {class_name} class not found in {filename}")
            return False
    
    return True

def main():
    """Run adapter import tests."""
    print("=== Adapter Import Test ===")
    print()
    
    tests = [
        test_adapter_files_exist,
        test_adapter_imports_syntax,
        test_ingest_imports,
        test_adapter_class_definitions
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
        print("✅ All adapter import tests passed!")
        print("The adapter files are properly located and have correct imports.")
        return 0
    else:
        print("❌ Some tests failed - there are adapter import issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())