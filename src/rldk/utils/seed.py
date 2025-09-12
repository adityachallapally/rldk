"""Centralized seeding utilities for RLDK.

This module provides utilities for setting and managing random seeds across
all RLDK operations to ensure reproducible behavior. It handles seeding for
Python's random module, NumPy, and PyTorch.

Example:
    >>> from rldk.utils.seed import set_global_seed, get_global_seed
    >>> set_global_seed(42)
    >>> print(get_global_seed())
    42
"""

import random
import os
from typing import Optional

import numpy as np
import torch


# Default seed for RLDK operations
DEFAULT_SEED = 1337

# Global seed state
_global_seed: Optional[int] = None


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Set the global seed for all RLDK operations.
    
    This function sets seeds for Python's random module, NumPy, and PyTorch
    to ensure reproducible behavior across RLDK operations.
    
    Args:
        seed: The seed value to use. Defaults to DEFAULT_SEED (1337).
    """
    global _global_seed
    _global_seed = seed
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Set PyTorch seed
    torch.manual_seed(seed)
    
    # Set CUDA seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set environment variable for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_global_seed() -> Optional[int]:
    """
    Get the current global seed.
    
    Returns:
        The current global seed, or None if no seed has been set.
    """
    return _global_seed


def ensure_seeded() -> None:
    """
    Ensure that a global seed is set.
    
    If no global seed has been set, this function will set the default seed.
    """
    if _global_seed is None:
        set_global_seed()


def seeded_random_state(seed: Optional[int] = None) -> tuple:
    """
    Get a seeded random state for reproducible operations.
    
    Args:
        seed: Optional seed value. If None, uses the global seed.
        
    Returns:
        A tuple containing (python_random_state, numpy_random_state, torch_random_state)
    """
    if seed is None:
        seed = _global_seed or DEFAULT_SEED
    
    # Save current states
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    
    # Set new seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get new states
    new_python_state = random.getstate()
    new_numpy_state = np.random.get_state()
    new_torch_state = torch.get_rng_state()
    
    # Restore original states
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    
    return (new_python_state, new_numpy_state, new_torch_state)


def restore_random_state(states: tuple) -> None:
    """
    Restore random states from a previous seeded_random_state call.
    
    Args:
        states: Tuple of (python_state, numpy_state, torch_state) from seeded_random_state
    """
    python_state, numpy_state, torch_state = states
    
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)