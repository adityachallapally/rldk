#!/usr/bin/env python3
"""Test that the ingest module structure is correct and can be imported."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(src_path))

def test_ingest_module_structure():
    """Test that the ingest module has the correct structure."""
    print("Testing ingest module structure...")
    
    ingest_dir = src_path / 'rldk' / 'ingest'
    
    # Check required files exist
    required_files = [
        '__init__.py',
        'ingest.py', 
        'base.py',
        'trl.py',
        'openrlhf.py',
        'wandb.py',
        'custom_jsonl.py'
    ]
    
    for filename in required_files:
        file_path = ingest_dir / filename
        if file_path.exists():
            print(f"✓ {filename} exists")
        else:
            print(f"❌ {filename} missing")
            return False
    
    return True

def test_ingest_imports_syntax():
    """Test that ingest.py has correct import syntax."""
    print("Testing ingest.py import syntax...")
    
    ingest_path = src_path / 'rldk' / 'ingest' / 'ingest.py'
    with open(ingest_path) as f:
        content = f.read()
    
    # Check that it imports from local modules
    local_imports = [
        'from .trl import TRLAdapter',
        'from .openrlhf import OpenRLHFAdapter', 
        'from .wandb import WandBAdapter',
        'from .custom_jsonl import CustomJSONLAdapter'
    ]
    
    for import_line in local_imports:
        if import_line in content:
            print(f"✓ {import_line}")
        else:
            print(f"❌ Missing: {import_line}")
            return False
    
    # Check that it doesn't import from old adapters module
    if 'from ..adapters import' not in content:
        print("✓ No old adapters import")
    else:
        print("❌ Still has old adapters import")
        return False
    
    return True

def test_adapter_dependencies():
    """Test that adapter files have correct dependencies."""
    print("Testing adapter dependencies...")
    
    # Test that all adapters import from base
    adapter_files = ['trl.py', 'openrlhf.py', 'wandb.py', 'custom_jsonl.py']
    
    for filename in adapter_files:
        file_path = src_path / 'rldk' / 'ingest' / filename
        with open(file_path) as f:
            content = f.read()
        
        if 'from .base import BaseAdapter' in content:
            print(f"✓ {filename} imports BaseAdapter correctly")
        else:
            print(f"❌ {filename} missing BaseAdapter import")
            return False
    
    return True

def test_ingest_init_exports():
    """Test that ingest __init__.py exports the right functions."""
    print("Testing ingest __init__.py exports...")
    
    init_path = src_path / 'rldk' / 'ingest' / '__init__.py'
    with open(init_path) as f:
        content = f.read()
    
    # Should export main functions
    if 'ingest_runs' in content and 'ingest_runs_to_events' in content:
        print("✓ Main functions exported")
    else:
        print("❌ Main functions not exported")
        return False
    
    # Should export adapters
    adapter_exports = ['TRLAdapter', 'OpenRLHFAdapter', 'WandBAdapter', 'CustomJSONLAdapter']
    for adapter in adapter_exports:
        if adapter in content:
            print(f"✓ {adapter} exported")
        else:
            print(f"❌ {adapter} not exported")
            return False
    
    return True

def main():
    """Run ingest module structure tests."""
    print("=== Ingest Module Structure Test ===")
    print()
    
    tests = [
        test_ingest_module_structure,
        test_ingest_imports_syntax,
        test_adapter_dependencies,
        test_ingest_init_exports
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
        print("✅ All ingest module structure tests passed!")
        print("The adapter files are properly located and the ingest module is correctly structured.")
        return 0
    else:
        print("❌ Some tests failed - there are structural issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())