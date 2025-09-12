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
from typing import Optional, Callable, Any
from functools import wraps
from contextlib import contextmanager

import numpy as np
import torch


# Default seed for RLDK operations
DEFAULT_SEED = 1337

# Global seed state
_global_seed: Optional[int] = None


def set_global_seed(seed: Optional[int] = DEFAULT_SEED) -> None:
    """
    Set the global seed for all RLDK operations.
    
    This function sets seeds for Python's random module, NumPy, and PyTorch
    to ensure reproducible behavior across RLDK operations.
    
    Args:
        seed: The seed value to use. If None, no seeding is performed.
              Defaults to DEFAULT_SEED (1337).
    """
    global _global_seed
    _global_seed = seed
    
    if seed is not None:
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


def ensure_seeded(func: Optional[Callable] = None) -> Any:
    """
    Ensure that a global seed is set.
    
    Can be used as a decorator or called directly.
    If no global seed has been set, this function will set the default seed.
    
    Args:
        func: Optional function to decorate. If provided, returns a decorated function.
              If None, sets the seed immediately.
    """
    if func is None:
        # Called directly
        if _global_seed is None:
            set_global_seed()
        return None
    
    # Used as decorator
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Save current state
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        
        try:
            if _global_seed is None:
                set_global_seed()
            return func(*args, **kwargs)
        finally:
            # Restore original state
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
    
    return wrapper


@contextmanager
def seeded_random_state(seed: Optional[int] = None):
    """
    Context manager for seeded random state operations.
    
    Temporarily sets a specific seed for reproducible operations within the context.
    The original random state is restored when exiting the context.
    
    Args:
        seed: Optional seed value. If None, uses the global seed.
        
    Yields:
        A tuple containing (python_random_state, numpy_random_state, torch_random_state)
    """
    if seed is None:
        seed = _global_seed or DEFAULT_SEED
    
    # Save current states
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    
    try:
        # Set new seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Get new states
        new_python_state = random.getstate()
        new_numpy_state = np.random.get_state()
        new_torch_state = torch.get_rng_state()
        
        yield (new_python_state, new_numpy_state, new_torch_state)
    
    finally:
        # Restore original states
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def restore_random_state(states) -> None:
    """
    Restore random states from a previous seeded_random_state call.
    
    Args:
        states: Either a tuple of (python_state, numpy_state, torch_state) from seeded_random_state
                or a dict with keys 'random', 'numpy', 'torch'
    """
    if isinstance(states, dict):
        python_state = states.get('random')
        numpy_state = states.get('numpy')
        torch_state = states.get('torch')
    else:
        python_state, numpy_state, torch_state = states
    
    if python_state is not None:
        random.setstate(python_state)
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    if torch_state is not None:
        torch.set_rng_state(torch_state)