#!/usr/bin/env python3
"""Test with proper relative paths that work in any environment."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(src_path))

def test_adapters_import_fix():
    """Test that the adapters import fix works."""
    print("Testing adapters import fix...")
    
    try:
        from rldk.ingest import ingest_runs
        print("✓ ingest_runs imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

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

def test_card_functions():
    """Test that card functions can be imported."""
    print("Testing card functions...")
    
    try:
        from rldk.determinism import generate_determinism_card
        from rldk.diff import generate_drift_card
        from rldk.reward import generate_reward_card
        print("✓ All card functions imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Card import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_cli_imports():
    """Test that CLI can import all required functions."""
    print("Testing CLI imports...")
    
    try:
        from rldk.ingest import ingest_runs, ingest_runs_to_events
        from rldk.determinism import generate_determinism_card
        from rldk.diff import generate_drift_card
        from rldk.reward import generate_reward_card
        print("✓ All CLI imports successful")
        return True
    except ImportError as e:
        print(f"❌ CLI import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests with proper relative paths."""
    print("=== Fixed Paths Test ===")
    print()
    
    tests = [
        test_adapters_import_fix,
        test_package_structure,
        test_card_functions,
        test_cli_imports
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
        print("✅ All tests passed with proper relative paths!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())