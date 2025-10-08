#!/usr/bin/env python3
"""
Script to check test coverage and ensure it meets the ≥80% requirement.
"""

import subprocess
import sys
from pathlib import Path


def check_coverage():
    """Check test coverage and ensure it meets requirements."""
    print("📊 Checking test coverage...")

    try:
        # Install coverage tools if not available
        try:
            import coverage
        except ImportError:
            print("Installing coverage tools...")
            subprocess.run([sys.executable, "-m", "pip", "install", "coverage", "pytest-cov"], check=True)

        # Run tests with coverage
        print("\n🧪 Running tests with coverage...")
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/unit/",
            "--cov=src/rldk",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--cov-report=html",
            "--cov-fail-under=80",
            "-v"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Coverage requirement met (≥80%)")
            print(result.stdout)
        else:
            print("❌ Coverage requirement not met (<80%)")
            print(result.stdout)
            print(result.stderr)
            return False

        # Generate detailed coverage report
        print("\n📈 Generating detailed coverage report...")
        subprocess.run([
            sys.executable, "-m", "coverage", "report", "--show-missing"
        ], check=True)

        # Generate HTML coverage report
        print("\n🌐 Generating HTML coverage report...")
        subprocess.run([
            sys.executable, "-m", "coverage", "html"
        ], check=True)

        print("\n📁 Coverage reports generated:")
        print("  - coverage.xml (for CI)")
        print("  - htmlcov/ (for local viewing)")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Coverage check failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def check_coverage_by_module():
    """Check coverage for specific modules."""
    print("\n🔍 Checking coverage by module...")

    modules = [
        "src/rldk/utils/seed.py",
        "src/rldk/utils/validation.py",
        "src/rldk/utils/error_handling.py",
        "src/rldk/utils/progress.py",
        "src/rldk/cli.py"
    ]

    coverage_issues = False

    for module in modules:
        if Path(module).exists():
            print(f"\n📊 Coverage for {module}:")

            # Find the corresponding test file (handle different naming conventions)
            module_name = Path(module).stem
            test_file_patterns = [
                f"tests/unit/test_{module_name}.py",
                f"tests/unit/test_utils_{module_name}.py",
                f"tests/unit/test_{module_name.replace('_', '')}.py"
            ]

            test_file = None
            for pattern in test_file_patterns:
                if Path(pattern).exists():
                    test_file = pattern
                    break

            if test_file is None:
                print(f"⚠️ No test file found for {module}")
                coverage_issues = True
                continue

            try:
                result = subprocess.run([
                    sys.executable, "-m", "pytest",
                    test_file,
                    f"--cov={module}",
                    "--cov-report=term-missing",
                    "--cov-fail-under=80",
                    "-v"
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    print(f"✅ {module} coverage OK")
                else:
                    print(f"⚠️ {module} coverage issues")
                    print(result.stdout)
                    coverage_issues = True

            except Exception as e:
                print(f"❌ Error checking {module}: {e}")
                coverage_issues = True
        else:
            print(f"⚠️ Module not found: {module}")
            coverage_issues = True

    return not coverage_issues


def main():
    """Main function."""
    print("🎯 RLDK Coverage Check")
    print("=" * 50)

    # Check overall coverage
    overall_success = check_coverage()

    # Check coverage by module
    module_success = check_coverage_by_module()

    if overall_success and module_success:
        print("\n🎉 All coverage checks passed!")
        sys.exit(0)
    else:
        print("\n💥 Some coverage checks failed!")
        if not overall_success:
            print("   - Overall coverage below 80%")
        if not module_success:
            print("   - Module-specific coverage issues detected")
        sys.exit(1)


if __name__ == "__main__":
    main()
