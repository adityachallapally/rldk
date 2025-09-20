"""PyTorch compatibility utilities for handling version differences."""

from pathlib import Path
from typing import Any, Dict, Optional, Union


def safe_torch_load(
    f: Union[str, Path],
    map_location: Optional[Union[str, "torch.device"]] = None,
    **kwargs
) -> Any:
    """
    Load a PyTorch checkpoint with version compatibility handling.

    This function handles the weights_only parameter introduced in PyTorch 2.6+
    for security reasons. For older versions, it falls back to the original behavior.

    Args:
        f: Path to the checkpoint file
        map_location: Device to map tensors to
        **kwargs: Additional arguments passed to torch.load

    Returns:
        Loaded checkpoint data

    Raises:
        ValueError: If checkpoint loading fails
    """
    import torch  # Lazy import to avoid CLI hang
    
    try:
        if _supports_weights_only():
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False

        return torch.load(f, map_location=map_location, **kwargs)

    except Exception as e:
        if 'weights_only' in kwargs:
            try:
                kwargs_fallback = kwargs.copy()
                del kwargs_fallback['weights_only']
                return torch.load(f, map_location=map_location, **kwargs_fallback)
            except Exception:
                pass  # Fall through to original error

        raise ValueError(f"Failed to load checkpoint {f}: {e}")


def get_torch_version_info() -> Dict[str, Any]:
    """
    Get PyTorch version information for compatibility checks.

    Returns:
        Dictionary with version information
    """
    import torch  # Lazy import to avoid CLI hang
    
    version_str = torch.__version__
    major, minor, patch = _parse_version_string(version_str)

    return {
        'version_string': version_str,
        'major': major,
        'minor': minor,
        'patch': patch,
        'supports_weights_only': _supports_weights_only(),
    }


def _parse_version_string(version_str: str) -> tuple[int, int, int]:
    """
    Parse PyTorch version string robustly, handling non-standard formats.

    Args:
        version_str: Version string like '2.6.0+cu118' or '2.6.0.dev20240101'

    Returns:
        Tuple of (major, minor, patch) as integers
    """
    import re

    try:
        match = re.match(r'^(\d+)\.(\d+)(?:\.(\d+))?', version_str)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            patch = int(match.group(3)) if match.group(3) else 0
            return major, minor, patch
        else:
            parts = version_str.split('.')
            major = int(re.sub(r'[^\d]', '', parts[0])) if parts else 0
            minor = int(re.sub(r'[^\d]', '', parts[1])) if len(parts) > 1 else 0
            patch = int(re.sub(r'[^\d]', '', parts[2])) if len(parts) > 2 else 0
            return major, minor, patch
    except Exception:
        return 0, 0, 0


def _supports_weights_only() -> bool:
    """Check if current PyTorch version supports weights_only parameter."""
    try:
        import torch  # Lazy import to avoid CLI hang
        major, minor, _ = _parse_version_string(torch.__version__)
        return major > 2 or (major == 2 and minor >= 6)
    except Exception:
        return False
