#!/usr/bin/env python3
"""Comprehensive import validation for RLDK package layout cleanup (PR4)."""

import sys
import importlib
import pkgutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test basic RLDK imports."""
    print("Testing basic RLDK imports...")
    try:
        import rldk
        print(f"✅ RLDK imports successfully (version: {getattr(rldk, '__version__', 'unknown')})")
        return True
    except Exception as e:
        print(f"❌ Basic RLDK import failed: {e}")
        return False

def test_public_api_imports():
    """Test all public API imports from __init__.py."""
    print("\nTesting public API imports...")
    import rldk
    
    public_functions = [
        "ingest_runs", "ingest_runs_to_events", "first_divergence", "check",
        "bisect_commits", "health", "compare_models", "run", "replay",
        "scan_logs", "diff_checkpoints", "audit_environment",
        "generate_determinism_card", "generate_drift_card", "generate_reward_card",
        "write_json", "write_png", "mkdir_reports", "validate", "read_jsonl", "read_reward_head",
        "set_global_seed", "get_current_seed", "restore_seed_state", 
        "set_reproducible_environment", "validate_seed_consistency"
    ]
    
    public_classes = [
        "DivergenceReport", "DeterminismReport", "BisectResult", "RewardHealthReport",
        "EvalResult", "ReplayReport", "ComprehensivePPOForensics", "ComprehensivePPOMetrics",
        "ExperimentTracker", "TrackingConfig", "DatasetTracker", "ModelTracker",
        "EnvironmentTracker", "SeedTracker", "GitTracker", "BaseAdapter", "TRLAdapter",
        "OpenRLHFAdapter", "WandBAdapter", "CustomJSONLAdapter", "RLDKSettings", "ConfigSchema"
    ]
    
    failed = []
    
    for name in public_functions + public_classes + ["settings"]:
        try:
            obj = getattr(rldk, name)
            print(f"✅ {name}: {type(obj).__name__}")
        except AttributeError as e:
            print(f"❌ {name}: Missing from public API")
            failed.append(name)
        except Exception as e:
            print(f"❌ {name}: Error accessing - {e}")
            failed.append(name)
    
    if failed:
        print(f"\n❌ Failed public API imports: {failed}")
        return False
    else:
        print("✅ All public API imports successful")
        return True

def test_submodule_imports():
    """Test key submodule imports."""
    print("\nTesting key submodule imports...")
    
    test_imports = [
        ("rldk.tracking", "ExperimentTracker"),
        ("rldk.forensics", "ComprehensivePPOForensics"),
        ("rldk.determinism", "check"),
        ("rldk.adapters", "BaseAdapter"),
        ("rldk.adapters", "TRLAdapter"),
        ("rldk.adapters", "OpenRLHFAdapter"),
        ("rldk.config", "settings"),
        ("rldk.evals", "run"),
        ("rldk.replay", "replay"),
        ("rldk.cards", "generate_determinism_card"),
        ("rldk.utils.seed", "set_global_seed"),
        ("rldk.integrations.trl", "RLDKCallback"),
        ("rldk.integrations.openrlhf", "OpenRLHFCallback"),
    ]
    
    failed = []
    
    for module_name, symbol_name in test_imports:
        try:
            module = importlib.import_module(module_name)
            symbol = getattr(module, symbol_name)
            print(f"✅ {module_name}.{symbol_name}: {type(symbol).__name__}")
        except ImportError as e:
            print(f"❌ {module_name}: Import error - {e}")
            failed.append(module_name)
        except AttributeError as e:
            print(f"❌ {module_name}.{symbol_name}: Missing symbol - {e}")
            failed.append(f"{module_name}.{symbol_name}")
        except Exception as e:
            print(f"❌ {module_name}.{symbol_name}: Unexpected error - {e}")
            failed.append(f"{module_name}.{symbol_name}")
    
    if failed:
        print(f"\n❌ Failed submodule imports: {failed}")
        return False
    else:
        print("✅ All key submodule imports successful")
        return True

def walk_all_modules():
    """Walk through all modules in the package to find broken imports."""
    print("\nWalking all RLDK modules...")
    
    import rldk
    failed = []
    
    try:
        modules = [m.name for m in pkgutil.walk_packages(rldk.__path__, rldk.__name__ + '.')]
        print(f"Found {len(modules)} modules to test")
        
        for module_name in modules:
            try:
                importlib.import_module(module_name)
                print(f"✅ {module_name}")
            except Exception as e:
                print(f"❌ {module_name}: {type(e).__name__} - {e}")
                failed.append((module_name, type(e).__name__))
        
        if failed:
            print(f"\n❌ Failed module imports ({len(failed)}):")
            for module, error_type in failed:
                print(f"  - {module}: {error_type}")
            return False
        else:
            print(f"✅ All {len(modules)} modules imported successfully")
            return True
            
    except Exception as e:
        print(f"❌ Error during module walk: {e}")
        return False

def test_cli_imports():
    """Test CLI module imports."""
    print("\nTesting CLI imports...")
    
    try:
        import rldk.cli
        print("✅ CLI module imports successfully")
        if hasattr(rldk.cli, 'app') or hasattr(rldk.cli, 'typer'):
            print("✅ CLI has typer components")
        return True
    except Exception as e:
        print(f"❌ CLI import failed: {e}")
        return False

def main():
    """Run all import validation tests."""
    print("=" * 60)
    print("RLDK Package Import Validation (PR4)")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_public_api_imports,
        test_submodule_imports,
        test_cli_imports,
        walk_all_modules,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("✅ Package layout and imports are working correctly!")
        return 0
    else:
        print(f"❌ SOME TESTS FAILED ({passed}/{total})")
        print("❌ Package layout needs attention!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
