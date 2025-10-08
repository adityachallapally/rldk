"""
Dependency checker utility for graceful handling of missing optional dependencies.

This module provides utilities to check for optional dependencies and provide
helpful error messages when they are missing.
"""

import importlib
from typing import Dict, List, Optional, Tuple


class DependencyChecker:
    """Utility class for checking and handling missing dependencies."""

    # Define optional dependencies and their purposes
    OPTIONAL_DEPENDENCIES = {
        'streamlit': {
            'purpose': 'monitoring dashboard',
            'install_cmd': 'pip install streamlit',
            'break_system_cmd': 'pip install streamlit --break-system-packages'
        },
        'plotly': {
            'purpose': 'interactive visualizations in dashboard',
            'install_cmd': 'pip install plotly',
            'break_system_cmd': 'pip install plotly --break-system-packages'
        },
        'wandb': {
            'purpose': 'Weights & Biases integration',
            'install_cmd': 'pip install wandb',
            'break_system_cmd': 'pip install wandb --break-system-packages'
        }
    }

    @classmethod
    def check_dependency(cls, package_name: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a package is available.

        Args:
            package_name: Name of the package to check

        Returns:
            Tuple of (is_available, error_message)
        """
        try:
            importlib.import_module(package_name)
            return True, None
        except ImportError as e:
            return False, str(e)

    @classmethod
    def check_optional_dependencies(cls, packages: List[str]) -> Dict[str, Tuple[bool, Optional[str]]]:
        """
        Check multiple optional dependencies.

        Args:
            packages: List of package names to check

        Returns:
            Dictionary mapping package names to (is_available, error_message) tuples
        """
        results = {}
        for package in packages:
            results[package] = cls.check_dependency(package)
        return results

    @classmethod
    def get_installation_help(cls, missing_packages: List[str]) -> str:
        """
        Generate helpful installation instructions for missing packages.

        Args:
            missing_packages: List of missing package names

        Returns:
            Formatted help message with installation instructions
        """
        if not missing_packages:
            return ""

        help_lines = [
            "❌ Missing optional dependencies detected:",
            ""
        ]

        for package in missing_packages:
            if package in cls.OPTIONAL_DEPENDENCIES:
                dep_info = cls.OPTIONAL_DEPENDENCIES[package]
                help_lines.extend([
                    f"📦 {package} (for {dep_info['purpose']})",
                    f"   Standard: {dep_info['install_cmd']}",
                    f"   With --break-system-packages: {dep_info['break_system_cmd']}",
                    ""
                ])
            else:
                help_lines.extend([
                    f"📦 {package}",
                    f"   Install with: pip install {package}",
                    ""
                ])

        help_lines.extend([
            "💡 Tip: For systems with package conflicts, use --break-system-packages flag",
            "💡 Tip: All dependencies are included in pyproject.toml - try: pip install -e .",
            ""
        ])

        return "\n".join(help_lines)

    @classmethod
    def require_dependencies(cls, packages: List[str], feature_name: str = "this feature") -> None:
        """
        Require dependencies and raise helpful error if missing.

        Args:
            packages: List of required package names
            feature_name: Name of the feature requiring these dependencies

        Raises:
            ImportError: If any required packages are missing
        """
        missing_packages = []

        for package in packages:
            is_available, _ = cls.check_dependency(package)
            if not is_available:
                missing_packages.append(package)

        if missing_packages:
            help_message = cls.get_installation_help(missing_packages)
            error_message = f"""
🚫 Cannot use {feature_name} - missing required dependencies.

{help_message}
            """.strip()
            raise ImportError(error_message)


def check_streamlit_dependencies() -> None:
    """Check dependencies required for Streamlit dashboard."""
    DependencyChecker.require_dependencies(
        ['streamlit', 'plotly'],
        'monitoring dashboard'
    )


def check_wandb_dependencies() -> None:
    """Check dependencies required for Weights & Biases integration."""
    DependencyChecker.require_dependencies(
        ['wandb'],
        'Weights & Biases integration'
    )


def safe_import(module_name: str, fallback_message: str = None) -> Optional[object]:
    """
    Safely import a module with fallback message.

    Args:
        module_name: Name of module to import
        fallback_message: Custom message to show if import fails

    Returns:
        Imported module or None if import failed
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        if fallback_message:
            print(f"⚠️  {fallback_message}")
        return None


def check_all_optional_dependencies() -> Dict[str, bool]:
    """
    Check all optional dependencies and return availability status.

    Returns:
        Dictionary mapping package names to availability status
    """
    all_packages = list(DependencyChecker.OPTIONAL_DEPENDENCIES.keys())
    results = DependencyChecker.check_optional_dependencies(all_packages)
    return {package: available for package, (available, _) in results.items()}
