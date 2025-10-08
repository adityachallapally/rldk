#!/usr/bin/env python3
"""
Test script to verify dependency checking functionality.

This script tests the dependency checker utility and provides
a quick way to verify that all optional dependencies are available.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.dependency_checker import (
    DependencyChecker,
    check_all_optional_dependencies,
    check_streamlit_dependencies,
    check_wandb_dependencies,
)


def main():
    """Test dependency checking functionality."""
    print("🔍 Checking optional dependencies...")
    print("=" * 50)

    # Check all optional dependencies
    results = check_all_optional_dependencies()

    all_available = True
    for package, available in results.items():
        status = "✅" if available else "❌"
        print(f"{status} {package}")
        if not available:
            all_available = False

    print("\n" + "=" * 50)

    if all_available:
        print("🎉 All optional dependencies are available!")
        print("\nYou can use:")
        print("  - Monitoring dashboard: python monitor/app.py")
        print("  - Weights & Biases integration")
        print("  - All visualization features")
    else:
        print("⚠️  Some optional dependencies are missing.")
        print("\nMissing dependencies:")
        missing = [pkg for pkg, available in results.items() if not available]
        help_message = DependencyChecker.get_installation_help(missing)
        print(help_message)

    # Test specific dependency checks
    print("\n🧪 Testing specific dependency checks...")

    try:
        check_streamlit_dependencies()
        print("✅ Streamlit dependencies check passed")
    except ImportError as e:
        print(f"❌ Streamlit dependencies check failed: {e}")

    try:
        check_wandb_dependencies()
        print("✅ Weights & Biases dependencies check passed")
    except ImportError as e:
        print(f"❌ Weights & Biases dependencies check failed: {e}")

    return 0 if all_available else 1


if __name__ == "__main__":
    sys.exit(main())
