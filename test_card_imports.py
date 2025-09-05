#!/usr/bin/env python3
"""Test card generation function imports."""

import sys
sys.path.insert(0, '/workspace/src')

def test_drift_card_import():
    """Test drift card import."""
    print("Testing drift card import...")
    
    try:
        from rldk.diff import generate_drift_card
        print("✓ generate_drift_card imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_reward_card_import():
    """Test reward card import."""
    print("Testing reward card import...")
    
    try:
        from rldk.reward import generate_reward_card
        print("✓ generate_reward_card imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_determinism_card_import():
    """Test determinism card import."""
    print("Testing determinism card import...")
    
    try:
        from rldk.determinism import generate_determinism_card
        print("✓ generate_determinism_card imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_cli_imports():
    """Test CLI imports."""
    print("Testing CLI imports...")
    
    try:
        # Test the exact imports used in CLI
        from rldk.determinism import generate_determinism_card
        from rldk.diff import generate_drift_card
        from rldk.reward import generate_reward_card
        print("✓ All CLI card imports successful")
        return True
    except ImportError as e:
        print(f"❌ CLI import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run card import tests."""
    print("=== Card Import Test ===")
    print()
    
    tests = [
        test_drift_card_import,
        test_reward_card_import,
        test_determinism_card_import,
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
        print("✅ All card import tests passed!")
        return 0
    else:
        print("❌ Some tests failed - there are import issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())