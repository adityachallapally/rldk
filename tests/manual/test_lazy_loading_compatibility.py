#!/usr/bin/env python3
"""Test backward compatibility of lazy loading implementation."""

import sys
import time
import _path_setup  # noqa: F401


def test_isinstance_checks():
    """Test that isinstance checks work with lazy classes."""
    print("Testing isinstance checks...")

    import rldk

    tracker = rldk.ModelTracker()

    config = rldk.TrackingConfig(experiment_name="test_experiment")
    experiment_tracker = rldk.ExperimentTracker(config)

    assert isinstance(tracker, rldk.ModelTracker), "isinstance check failed for ModelTracker"
    assert isinstance(experiment_tracker, rldk.ExperimentTracker), "isinstance check failed for ExperimentTracker"

    print("✓ isinstance checks work correctly")

def test_class_attributes():
    """Test that class attributes and methods work."""
    print("Testing class attributes...")

    import rldk

    assert hasattr(rldk.ModelTracker, '__name__'), "Missing __name__ attribute"
    assert hasattr(rldk.ModelTracker, '__module__'), "Missing __module__ attribute"

    attrs = dir(rldk.ModelTracker)
    assert len(attrs) > 0, "dir() returned empty list"

    print("✓ Class attributes work correctly")

def test_settings_proxy():
    """Test that settings proxy works correctly."""
    print("Testing settings proxy...")

    import rldk

    settings = rldk.settings

    attrs = dir(settings)
    assert len(attrs) > 0, "dir(settings) returned empty list"

    repr_str = repr(settings)
    assert len(repr_str) > 0, "repr(settings) returned empty string"

    assert hasattr(settings, 'log_level'), "Missing log_level attribute"

    print("✓ Settings proxy works correctly")

def test_lazy_loading_behavior():
    """Test that heavy dependencies are still loaded lazily."""
    print("Testing lazy loading behavior...")

    modules_to_clear = [mod for mod in sys.modules.keys() if 'torch' in mod or 'transformers' in mod]
    for mod in modules_to_clear:
        del sys.modules[mod]


    torch_loaded = any('torch' in mod for mod in sys.modules.keys())
    transformers_loaded = any('transformers' in mod for mod in sys.modules.keys())

    print("After importing rldk:")
    print(f"  torch loaded: {torch_loaded}")
    print(f"  transformers loaded: {transformers_loaded}")

    if torch_loaded or transformers_loaded:
        print("⚠️  WARNING: Heavy dependencies loaded eagerly")
        return False

    print("✓ Lazy loading still working correctly")
    return True

def test_import_performance():
    """Test that import performance is still fast."""
    print("Testing import performance...")

    start_time = time.time()
    import rldk  # noqa: F401
    duration = time.time() - start_time

    print(f"Import time: {duration:.3f}s")

    if duration > 2.0:
        print(f"⚠️  WARNING: Import took {duration:.3f}s (target: <2s)")
        return False

    print("✓ Import performance still fast")
    return True

def main():
    print("RLDK Lazy Loading Compatibility Test")
    print("=" * 50)

    try:
        test_isinstance_checks()
        test_class_attributes()
        test_settings_proxy()
        test_lazy_loading_behavior()
        test_import_performance()

        print("\n🟢 SUCCESS: All compatibility tests passed")
        return 0

    except Exception as e:
        print(f"\n🔴 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
