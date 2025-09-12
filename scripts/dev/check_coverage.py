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
    
    for module in modules:
        if Path(module).exists():
            print(f"\n📊 Coverage for {module}:")
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pytest",
                    f"tests/unit/test_{Path(module).stem}.py",
                    "--cov={module}",
                    "--cov-report=term-missing",
                    "-v"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ {module} coverage OK")
                else:
                    print(f"⚠️ {module} coverage issues")
                    print(result.stdout)
                    
            except Exception as e:
                print(f"❌ Error checking {module}: {e}")
        else:
            print(f"⚠️ Module not found: {module}")


def main():
    """Main function."""
    print("🎯 RLDK Coverage Check")
    print("=" * 50)
    
    # Check overall coverage
    success = check_coverage()
    
    # Check coverage by module
    check_coverage_by_module()
    
    if success:
        print("\n🎉 All coverage checks passed!")
        sys.exit(0)
    else:
        print("\n💥 Some coverage checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()