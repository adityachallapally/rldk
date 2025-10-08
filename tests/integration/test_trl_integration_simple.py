#!/usr/bin/env python3
"""Simplified test suite for RLDK TRL integration (no external dependencies)."""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_file_structure():
    """Test that all required files are in place."""
    print("📁 Testing file structure...")

    required_files = [
        "src/rldk/integrations/__init__.py",
        "src/rldk/integrations/trl/__init__.py",
        "src/rldk/integrations/trl/callbacks.py",
        "src/rldk/integrations/trl/monitors.py",
        "src/rldk/integrations/trl/dashboard.py",
        "examples/trl_integration/basic_ppo_integration.py",
        "examples/trl_integration/advanced_monitoring.py",
        "examples/trl_integration/custom_callbacks.py",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    print("✅ All required files present")
    return True


def test_code_syntax():
    """Test that the Python code has valid syntax."""
    print("\n🔍 Testing code syntax...")

    python_files = [
        "src/rldk/integrations/__init__.py",
        "src/rldk/integrations/trl/__init__.py",
        "src/rldk/integrations/trl/callbacks.py",
        "src/rldk/integrations/trl/monitors.py",
        "src/rldk/integrations/trl/dashboard.py",
    ]

    for file_path in python_files:
        try:
            with open(file_path) as f:
                code = f.read()

            # Try to compile the code to check syntax
            compile(code, file_path, 'exec')
            print(f"✅ {file_path} - syntax OK")

        except SyntaxError as e:
            print(f"❌ {file_path} - syntax error: {e}")
            return False
        except Exception as e:
            print(f"❌ {file_path} - error: {e}")
            return False

    print("✅ All Python files have valid syntax")
    return True


def test_import_structure():
    """Test that the import structure is correct."""
    print("\n📦 Testing import structure...")

    # Test that the main integration module can be imported
    try:
        # This will fail if there are import errors in the code
        import importlib.util

        # Test main integration init
        spec = importlib.util.spec_from_file_location(
            "rldk.integrations",
            "src/rldk/integrations/__init__.py"
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print("✅ src/rldk/integrations/__init__.py imports OK")

        # Test TRL integration init
        spec = importlib.util.spec_from_file_location(
            "rldk.integrations.trl",
            "src/rldk/integrations/trl/__init__.py"
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print("✅ src/rldk/integrations/trl/__init__.py imports OK")

        return True

    except Exception as e:
        print(f"❌ Import structure test failed: {e}")
        return False


def test_class_definitions():
    """Test that all required classes are defined."""
    print("\n🏗️  Testing class definitions...")

    # Read the callbacks file and check for class definitions
    callbacks_file = Path("src/rldk/integrations/trl/callbacks.py")
    if callbacks_file.exists():
        with open(callbacks_file) as f:
            content = f.read()

        required_classes = ["RLDKMetrics", "RLDKCallback", "RLDKMonitor"]
        for class_name in required_classes:
            if f"class {class_name}" in content:
                print(f"✅ {class_name} class defined")
            else:
                print(f"❌ {class_name} class missing")
                return False

    # Read the monitors file and check for class definitions
    monitors_file = Path("src/rldk/integrations/trl/monitors.py")
    if monitors_file.exists():
        with open(monitors_file) as f:
            content = f.read()

        required_classes = ["PPOMetrics", "PPOMonitor", "CheckpointMetrics", "CheckpointMonitor"]
        for class_name in required_classes:
            if f"class {class_name}" in content:
                print(f"✅ {class_name} class defined")
            else:
                print(f"❌ {class_name} class missing")
                return False

    # Read the dashboard file and check for class definitions
    dashboard_file = Path("src/rldk/integrations/trl/dashboard.py")
    if dashboard_file.exists():
        with open(dashboard_file) as f:
            content = f.read()

        required_classes = ["RLDKDashboard"]
        for class_name in required_classes:
            if f"class {class_name}" in content:
                print(f"✅ {class_name} class defined")
            else:
                print(f"❌ {class_name} class missing")
                return False

    print("✅ All required classes are defined")
    return True


def test_method_definitions():
    """Test that key methods are defined in the classes."""
    print("\n🔧 Testing method definitions...")

    # Check RLDKCallback methods
    callbacks_file = Path("src/rldk/integrations/trl/callbacks.py")
    if callbacks_file.exists():
        with open(callbacks_file) as f:
            content = f.read()

        required_methods = [
            "on_train_begin",
            "on_step_end",
            "on_log",
            "on_save",
            "on_train_end"
        ]

        for method in required_methods:
            if f"def {method}" in content:
                print(f"✅ RLDKCallback.{method} method defined")
            else:
                print(f"❌ RLDKCallback.{method} method missing")
                return False

    # Check PPOMonitor methods
    monitors_file = Path("src/rldk/integrations/trl/monitors.py")
    if monitors_file.exists():
        with open(monitors_file) as f:
            content = f.read()

        required_methods = [
            "on_step_end",
            "on_log",
            "_check_ppo_alerts",
            "_analyze_policy_health"
        ]

        for method in required_methods:
            if f"def {method}" in content:
                print(f"✅ PPOMonitor.{method} method defined")
            else:
                print(f"❌ PPOMonitor.{method} method missing")
                return False

    print("✅ All required methods are defined")
    return True


def test_example_files():
    """Test that example files contain expected content."""
    print("\n📚 Testing example files...")

    examples = [
        ("examples/trl_integration/basic_ppo_integration.py", [
            "RLDKCallback",
            "PPOMonitor",
            "test_basic_ppo_integration"
        ]),
        ("examples/trl_integration/advanced_monitoring.py", [
            "CustomRLDKCallback",
            "AdvancedPPOMonitor",
            "test_advanced_monitoring"
        ]),
        ("examples/trl_integration/custom_callbacks.py", [
            "RewardModelMonitor",
            "DataPipelineMonitor",
            "MemoryOptimizationMonitor"
        ])
    ]

    for file_path, expected_content in examples:
        if not Path(file_path).exists():
            print(f"❌ {file_path} missing")
            return False

        with open(file_path) as f:
            content = f.read()

        for expected in expected_content:
            if expected in content:
                print(f"✅ {file_path} contains {expected}")
            else:
                print(f"❌ {file_path} missing {expected}")
                return False

    print("✅ All example files contain expected content")
    return True


def test_dependencies():
    """Test that dependencies are properly declared."""
    print("\n📦 Testing dependencies...")

    # Check pyproject.toml for TRL dependency
    pyproject_file = Path("pyproject.toml")
    if pyproject_file.exists():
        with open(pyproject_file) as f:
            content = f.read()

        if "trl>=0.7.0" in content:
            print("✅ TRL dependency declared in pyproject.toml")
        else:
            print("❌ TRL dependency missing from pyproject.toml")
            return False

        if "accelerate>=0.20.0" in content:
            print("✅ Accelerate dependency declared in pyproject.toml")
        else:
            print("❌ Accelerate dependency missing from pyproject.toml")
            return False

    print("✅ Dependencies properly declared")
    return True


def main():
    """Run the simplified test suite."""
    print("🎯 RLDK TRL Integration Test Suite (Simplified)")
    print("=" * 60)

    tests = [
        ("File Structure", test_file_structure),
        ("Code Syntax", test_code_syntax),
        ("Import Structure", test_import_structure),
        ("Class Definitions", test_class_definitions),
        ("Method Definitions", test_method_definitions),
        ("Example Files", test_example_files),
        ("Dependencies", test_dependencies),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if success:
            passed += 1

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! RLDK TRL integration structure is correct.")
        print("\n🚀 Integration Features Implemented:")
        print("   ✅ RLDKCallback - Real-time training monitoring")
        print("   ✅ PPOMonitor - PPO-specific analytics")
        print("   ✅ CheckpointMonitor - Model health monitoring")
        print("   ✅ RLDKDashboard - Real-time visualization")
        print("   ✅ Custom callbacks for specialized monitoring")
        print("   ✅ Comprehensive examples and test suite")
        print("\n📋 Next Steps:")
        print("   1. Install dependencies: pip install trl accelerate")
        print("   2. Run full integration tests")
        print("   3. Start using with your TRL training!")
    else:
        print(f"\n❌ {total - passed} tests failed. Please check the output above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
