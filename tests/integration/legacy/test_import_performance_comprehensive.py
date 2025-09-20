#!/usr/bin/env python3
"""Comprehensive test for RLDK import performance improvements."""

import sys
import time


def test_import_time(module_name, max_time=2.0):
    """Test that a module imports within the time limit."""
    start_time = time.time()
    try:
        __import__(module_name)
        duration = time.time() - start_time
        print(f"✓ {module_name}: {duration:.2f}s")
        return duration <= max_time, duration
    except Exception as e:
        duration = time.time() - start_time
        print(f"✗ {module_name}: {duration:.2f}s - {e}")
        return False, duration

def test_lazy_loading():
    """Test that heavy dependencies are only loaded when needed."""
    print("\nTesting lazy loading behavior...")

    modules_to_clear = [mod for mod in sys.modules.keys() if 'torch' in mod or 'transformers' in mod]
    for mod in modules_to_clear:
        del sys.modules[mod]

    import rldk

    torch_loaded = any('torch' in mod for mod in sys.modules.keys())
    transformers_loaded = any('transformers' in mod for mod in sys.modules.keys())

    print("After importing rldk:")
    print(f"  torch loaded: {torch_loaded}")
    print(f"  transformers loaded: {transformers_loaded}")

    if torch_loaded or transformers_loaded:
        print("⚠️  WARNING: Heavy dependencies loaded eagerly")
        return False

    try:
        rldk.ModelTracker()
        print("✓ ModelTracker created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to create ModelTracker: {e}")
        return False

def main():
    print("RLDK Import Performance Test")
    print("=" * 50)

    success, duration = test_import_time("rldk", max_time=2.0)

    if success:
        print(f"\n🟢 SUCCESS: RLDK imports in {duration:.2f}s (target: <2s)")
    else:
        print(f"\n🔴 FAILED: RLDK import took {duration:.2f}s (target: <2s)")
        return 1

    lazy_success = test_lazy_loading()

    if lazy_success:
        print("🟢 SUCCESS: Lazy loading working correctly")
    else:
        print("🔴 FAILED: Lazy loading not working correctly")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
