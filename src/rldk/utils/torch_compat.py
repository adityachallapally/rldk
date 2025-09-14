"""PyTorch compatibility utilities for handling version differences."""

import torch
from typing import Any, Dict, Optional, Union
from pathlib import Path


def safe_torch_load(
    f: Union[str, Path], 
    map_location: Optional[Union[str, torch.device]] = None,
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
    try:
        torch_version = torch.__version__
        major, minor = map(int, torch_version.split('.')[:2])
        
        if major > 2 or (major == 2 and minor >= 6):
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
    version_str = torch.__version__
    parts = version_str.split('.')
    
    return {
        'version_string': version_str,
        'major': int(parts[0]),
        'minor': int(parts[1]) if len(parts) > 1 else 0,
        'patch': int(parts[2].split('+')[0]) if len(parts) > 2 else 0,
        'supports_weights_only': _supports_weights_only(),
    }


def _supports_weights_only() -> bool:
    """Check if current PyTorch version supports weights_only parameter."""
    try:
        version_str = torch.__version__
        major, minor = map(int, version_str.split('.')[:2])
        return major > 2 or (major == 2 and minor >= 6)
    except Exception:
        return False
