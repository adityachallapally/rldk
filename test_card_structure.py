#!/usr/bin/env python3
"""Test card generation structure without requiring dependencies."""

from pathlib import Path
import re

def test_drift_card_structure():
    """Test drift card file structure."""
    print("Testing drift card structure...")
    
    drift_path = Path('/workspace/src/rldk/diff/drift.py')
    if not drift_path.exists():
        print("❌ drift.py not found")
        return False
    
    with open(drift_path) as f:
        content = f.read()
    
    # Check function exists
    if 'def generate_drift_card(' in content:
        print("✓ generate_drift_card function exists")
    else:
        print("❌ generate_drift_card function not found")
        return False
    
    # Check correct import (should be from .diff, not ..diff.diff)
    if 'from .diff import first_divergence_events' in content:
        print("✓ Correct relative import for first_divergence_events")
    else:
        print("❌ Incorrect import for first_divergence_events")
        return False
    
    # Check function signature
    if 'events_a: List[Event]' in content and 'events_b: List[Event]' in content and 'run_a_path: str' in content and 'run_b_path: str' in content:
        print("✓ Correct function signature (5 parameters)")
    else:
        print("❌ Incorrect function signature")
        return False
    
    return True

def test_reward_card_structure():
    """Test reward card file structure."""
    print("Testing reward card structure...")
    
    reward_path = Path('/workspace/src/rldk/reward/reward.py')
    if not reward_path.exists():
        print("❌ reward.py not found")
        return False
    
    with open(reward_path) as f:
        content = f.read()
    
    # Check function exists
    if 'def generate_reward_card(' in content:
        print("✓ generate_reward_card function exists")
    else:
        print("❌ generate_reward_card function not found")
        return False
    
    # Check function signature (should be 3 parameters for single run)
    if 'events: List[Event]' in content and 'run_path: str' in content and 'output_dir: Optional[str]' in content:
        print("✓ Correct function signature (3 parameters, single run)")
    else:
        print("❌ Incorrect function signature")
        return False
    
    # Check no two-run parameters
    if 'events_a' not in content and 'events_b' not in content and 'run_a_path' not in content and 'run_b_path' not in content:
        print("✓ No two-run parameters (single run analysis)")
    else:
        print("❌ Contains two-run parameters")
        return False
    
    return True

def test_determinism_card_structure():
    """Test determinism card file structure."""
    print("Testing determinism card structure...")
    
    det_path = Path('/workspace/src/rldk/determinism/determinism.py')
    if not det_path.exists():
        print("❌ determinism.py not found")
        return False
    
    with open(det_path) as f:
        content = f.read()
    
    # Check function exists
    if 'def generate_determinism_card(' in content:
        print("✓ generate_determinism_card function exists")
    else:
        print("❌ generate_determinism_card function not found")
        return False
    
    # Check function signature (should be 3 parameters for single run)
    if 'events: List[Event]' in content and 'run_path: str' in content and 'output_dir: Optional[str]' in content:
        print("✓ Correct function signature (3 parameters, single run)")
    else:
        print("❌ Incorrect function signature")
        return False
    
    return True

def test_module_exports():
    """Test that modules export the functions correctly."""
    print("Testing module exports...")
    
    # Test diff module
    diff_init_path = Path('/workspace/src/rldk/diff/__init__.py')
    with open(diff_init_path) as f:
        content = f.read()
    
    if 'from .drift import generate_drift_card' in content and '"generate_drift_card"' in content:
        print("✓ diff module exports generate_drift_card")
    else:
        print("❌ diff module missing generate_drift_card export")
        return False
    
    # Test reward module
    reward_init_path = Path('/workspace/src/rldk/reward/__init__.py')
    with open(reward_init_path) as f:
        content = f.read()
    
    if 'from .reward import generate_reward_card' in content and '"generate_reward_card"' in content:
        print("✓ reward module exports generate_reward_card")
    else:
        print("❌ reward module missing generate_reward_card export")
        return False
    
    # Test determinism module
    det_init_path = Path('/workspace/src/rldk/determinism/__init__.py')
    with open(det_init_path) as f:
        content = f.read()
    
    if 'from .determinism import generate_determinism_card' in content and '"generate_determinism_card"' in content:
        print("✓ determinism module exports generate_determinism_card")
    else:
        print("❌ determinism module missing generate_determinism_card export")
        return False
    
    return True

def test_cli_imports():
    """Test that CLI imports are correct."""
    print("Testing CLI imports...")
    
    cli_path = Path('/workspace/src/rldk/cli.py')
    with open(cli_path) as f:
        content = f.read()
    
    # Check CLI imports
    expected_imports = [
        'from rldk.determinism import generate_determinism_card',
        'from rldk.diff import generate_drift_card',
        'from rldk.reward import generate_reward_card'
    ]
    
    for import_line in expected_imports:
        if import_line in content:
            print(f"✓ {import_line}")
        else:
            print(f"❌ Missing: {import_line}")
            return False
    
    return True

def main():
    """Run card structure tests."""
    print("=== Card Structure Test ===")
    print()
    
    tests = [
        test_drift_card_structure,
        test_reward_card_structure,
        test_determinism_card_structure,
        test_module_exports,
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
        print("✅ All card structure tests passed!")
        print("The card generation functions are properly structured and exported.")
        return 0
    else:
        print("❌ Some tests failed - there are structural issues")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())