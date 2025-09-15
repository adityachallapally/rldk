"""Utility module for handling optional dependencies with friendly error messages."""

import importlib
from typing import Any, Dict, List, Optional, Tuple, Union


class OptionalImportError(ImportError):
    """Raised when an optional dependency is not available."""
    
    def __init__(self, package_name: str, extra_name: str, purpose: str = ""):
        self.package_name = package_name
        self.extra_name = extra_name
        self.purpose = purpose
        
        message = f"Optional dependency '{package_name}' is not installed."
        if extra_name:
            message += f" Install it with: pip install rldk[{extra_name}]"
        if purpose:
            message += f" (Required for: {purpose})"
            
        super().__init__(message)


def optional_import(
    module_name: str,
    extra_name: str = "",
    purpose: str = "",
    package_name: Optional[str] = None,
    min_version: Optional[str] = None
) -> Any:
    """
    Import an optional dependency with a friendly error message.
    
    Args:
        module_name: Name of the module to import
        extra_name: Name of the extra that provides this dependency
        purpose: Description of what this dependency is used for
        package_name: Name of the package (if different from module_name)
        min_version: Minimum version required (not enforced, just for error message)
    
    Returns:
        The imported module
        
    Raises:
        OptionalImportError: If the module cannot be imported
    """
    try:
        module = importlib.import_module(module_name)
        
        # Check version if specified
        if min_version and hasattr(module, '__version__'):
            from packaging import version
            if version.parse(module.__version__) < version.parse(min_version):
                raise OptionalImportError(
                    package_name or module_name,
                    extra_name,
                    f"{purpose} (requires version >= {min_version})"
                )
        
        return module
    except ImportError as e:
        raise OptionalImportError(
            package_name or module_name,
            extra_name,
            purpose
        ) from e


def optional_import_from(
    module_name: str,
    name: str,
    extra_name: str = "",
    purpose: str = "",
    package_name: Optional[str] = None
) -> Any:
    """
    Import a specific name from an optional dependency.
    
    Args:
        module_name: Name of the module to import from
        name: Name of the object to import
        extra_name: Name of the extra that provides this dependency
        purpose: Description of what this dependency is used for
        package_name: Name of the package (if different from module_name)
    
    Returns:
        The imported object
        
    Raises:
        OptionalImportError: If the module or object cannot be imported
    """
    try:
        module = importlib.import_module(module_name)
        return getattr(module, name)
    except (ImportError, AttributeError) as e:
        raise OptionalImportError(
            package_name or module_name,
            extra_name,
            f"{purpose} (importing {name})"
        ) from e


def check_optional_dependency(
    module_name: str,
    extra_name: str = "",
    purpose: str = ""
) -> bool:
    """
    Check if an optional dependency is available without importing it.
    
    Args:
        module_name: Name of the module to check
        extra_name: Name of the extra that provides this dependency
        purpose: Description of what this dependency is used for
    
    Returns:
        True if the module is available, False otherwise
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


# Common optional imports with predefined configurations
def import_torch():
    """Import PyTorch with RLHF extra."""
    return optional_import(
        "torch",
        "rlhf",
        "PyTorch operations and model handling"
    )


def import_transformers():
    """Import Transformers with RLHF extra."""
    return optional_import(
        "transformers",
        "rlhf",
        "Hugging Face Transformers integration"
    )


def import_datasets():
    """Import Datasets with RLHF extra."""
    return optional_import(
        "datasets",
        "rlhf",
        "Hugging Face Datasets integration"
    )


def import_trl():
    """Import TRL with RLHF extra."""
    return optional_import(
        "trl",
        "rlhf",
        "TRL (Transformers Reinforcement Learning) integration"
    )


def import_matplotlib():
    """Import Matplotlib with viz extra."""
    return optional_import(
        "matplotlib",
        "viz",
        "plotting and visualization"
    )


def import_plotly():
    """Import Plotly with viz extra."""
    return optional_import(
        "plotly",
        "viz",
        "interactive plotting and dashboards"
    )


def import_wandb():
    """Import WandB with tracking extra."""
    return optional_import(
        "wandb",
        "tracking",
        "Weights & Biases experiment tracking"
    )


def import_pyarrow():
    """Import PyArrow with io extra."""
    return optional_import(
        "pyarrow",
        "io",
        "Parquet file support"
    )


def import_streamlit():
    """Import Streamlit with io extra."""
    return optional_import(
        "streamlit",
        "io",
        "Streamlit web applications"
    )


def import_openrlhf():
    """Import OpenRLHF with openrlhf extra."""
    return optional_import(
        "openrlhf",
        "openrlhf",
        "OpenRLHF integration"
    )


def import_sklearn():
    """Import scikit-learn with rlhf extra."""
    return optional_import(
        "sklearn",
        "rlhf",
        "scikit-learn machine learning"
    )


# Convenience functions for common patterns
def get_optional_dependencies_info() -> Dict[str, Dict[str, str]]:
    """
    Get information about all optional dependencies.
    
    Returns:
        Dictionary mapping extra names to their dependency information
    """
    return {
        "viz": {
            "description": "Visualization and plotting capabilities",
            "dependencies": ["matplotlib", "seaborn", "plotly"],
            "install": "pip install rldk[viz]"
        },
        "rlhf": {
            "description": "RLHF and machine learning capabilities",
            "dependencies": ["torch", "transformers", "datasets", "trl", "accelerate", "detoxify", "vaderSentiment", "scikit-learn"],
            "install": "pip install rldk[rlhf]"
        },
        "io": {
            "description": "I/O and data processing capabilities",
            "dependencies": ["pyarrow", "streamlit"],
            "install": "pip install rldk[io]"
        },
        "tracking": {
            "description": "Experiment tracking and cloud integration",
            "dependencies": ["wandb", "flask"],
            "install": "pip install rldk[tracking]"
        },
        "openrlhf": {
            "description": "OpenRLHF integration",
            "dependencies": ["openrlhf"],
            "install": "pip install rldk[openrlhf]"
        },
        "all": {
            "description": "All optional dependencies",
            "dependencies": ["viz", "rlhf", "io", "tracking", "openrlhf"],
            "install": "pip install rldk[all]"
        }
    }


def suggest_install_command(missing_module: str) -> str:
    """
    Suggest the appropriate install command for a missing module.
    
    Args:
        missing_module: Name of the missing module
    
    Returns:
        Suggested pip install command
    """
    # Map modules to their extras
    module_to_extra = {
        "torch": "rlhf",
        "transformers": "rlhf", 
        "datasets": "rlhf",
        "trl": "rlhf",
        "accelerate": "rlhf",
        "detoxify": "rlhf",
        "vaderSentiment": "rlhf",
        "sklearn": "rlhf",
        "matplotlib": "viz",
        "seaborn": "viz",
        "plotly": "viz",
        "wandb": "tracking",
        "flask": "tracking",
        "pyarrow": "io",
        "streamlit": "io",
        "openrlhf": "openrlhf"
    }
    
    extra = module_to_extra.get(missing_module)
    if extra:
        return f"pip install rldk[{extra}]"
    else:
        return f"pip install {missing_module}"