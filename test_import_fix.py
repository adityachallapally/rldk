#!/usr/bin/env python3
"""Test that the missing import is now fixed."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(src_path))

def test_ingest_runs_to_events_import():
    """Test that ingest_runs_to_events can be imported."""
    print("Testing ingest_runs_to_events import...")
    
    try:
        # Test direct import from module
        from rldk.ingest import ingest_runs_to_events
        print("✓ Direct import from rldk.ingest works")
        
        # Test import from main package
        from rldk import ingest_runs_to_events
        print("✓ Import from main rldk package works")
        
        # Test that function exists and is callable
        if callable(ingest_runs_to_events):
            print("✓ Function is callable")
        else:
            print("❌ Function is not callable")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_cli_imports():
    """Test that CLI can import the function."""
    print("\nTesting CLI imports...")
    
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

def main():
    """Run import fix tests."""
    print("=== Import Fix Test ===")
    print()
    
    if test_ingest_runs_to_events_import():
        print()
        if test_cli_imports():
            print("\n✅ All import tests passed!")
            return 0
        else:
            print("\n❌ CLI import test failed")
            return 1
    else:
        print("\n❌ Import test failed")
        return 1

if __name__ == "__main__":
    from pathlib import Path
    sys.exit(main())