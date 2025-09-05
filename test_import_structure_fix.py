#!/usr/bin/env python3
"""Test that the import structure is correct after the fix."""

from pathlib import Path

def test_ingest_init_exports():
    """Test that ingest __init__.py exports the function."""
    print("Testing ingest module exports...")
    
    init_path = Path('/workspace/src/rldk/ingest/__init__.py')
    with open(init_path) as f:
        content = f.read()
    
    # Check import statement
    if 'from .ingest import ingest_runs, ingest_runs_to_events' in content:
        print("✓ ingest_runs_to_events imported from .ingest")
    else:
        print("❌ ingest_runs_to_events not imported")
        return False
    
    # Check __all__ list
    if '"ingest_runs_to_events"' in content:
        print("✓ ingest_runs_to_events in __all__ list")
    else:
        print("❌ ingest_runs_to_events not in __all__ list")
        return False
    
    return True

def test_main_init_exports():
    """Test that main __init__.py exports the function."""
    print("Testing main package exports...")
    
    init_path = Path('/workspace/src/rldk/__init__.py')
    with open(init_path) as f:
        content = f.read()
    
    # Check import statement
    if 'from .ingest import ingest_runs, ingest_runs_to_events' in content:
        print("✓ ingest_runs_to_events imported from .ingest")
    else:
        print("❌ ingest_runs_to_events not imported")
        return False
    
    # Check __all__ list
    if '"ingest_runs_to_events"' in content:
        print("✓ ingest_runs_to_events in __all__ list")
    else:
        print("❌ ingest_runs_to_events not in __all__ list")
        return False
    
    return True

def test_cli_imports():
    """Test that CLI imports the function."""
    print("Testing CLI imports...")
    
    cli_path = Path('/workspace/src/rldk/cli.py')
    with open(cli_path) as f:
        content = f.read()
    
    # Check that CLI imports the function
    if 'from rldk.ingest import ingest_runs, ingest_runs_to_events' in content:
        print("✓ CLI imports ingest_runs_to_events")
    else:
        print("❌ CLI missing ingest_runs_to_events import")
        return False
    
    # Check that it's used in card commands
    if 'ingest_runs_to_events(run_a)' in content:
        print("✓ CLI uses ingest_runs_to_events in card commands")
    else:
        print("❌ CLI doesn't use ingest_runs_to_events")
        return False
    
    return True

def test_function_exists():
    """Test that the function exists in the source file."""
    print("Testing function exists in source...")
    
    ingest_path = Path('/workspace/src/rldk/ingest/ingest.py')
    with open(ingest_path) as f:
        content = f.read()
    
    if 'def ingest_runs_to_events(' in content:
        print("✓ ingest_runs_to_events function exists in ingest.py")
    else:
        print("❌ ingest_runs_to_events function not found in ingest.py")
        return False
    
    return True

def main():
    """Run import structure tests."""
    print("=== Import Structure Fix Test ===")
    print()
    
    tests = [
        test_ingest_init_exports,
        test_main_init_exports,
        test_cli_imports,
        test_function_exists
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
        print("The missing ingest_runs_to_events export has been fixed.")
        return 0
    else:
        print("❌ Some tests failed - there are still issues to fix")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())