"""Seed management utilities for reproducible RL experiments.

This module provides centralized seed handling for ensuring reproducibility
across different components of RL training, including Python, NumPy, PyTorch,
and CUDA random number generators.
"""

import random
import os
from typing import Optional, Dict, Any, Union
import numpy as np
import torch

from .error_handling import RLDKError


class SeedManager:
    """Centralized seed management for reproducible experiments.
    
    This class manages seeds across multiple random number generators to ensure
    reproducible behavior in RL experiments. It handles Python's random module,
    NumPy, PyTorch (CPU and CUDA), and can be extended for other libraries.
    
    Attributes:
        seed: The current seed value
        rng_state: Dictionary storing RNG states for different libraries
        deterministic: Whether deterministic mode is enabled
    """
    
    def __init__(self, default_seed: int = 1337):
        """Initialize the seed manager.
        
        Args:
            default_seed: Default seed value to use when no seed is specified
        """
        self.default_seed = default_seed
        self.seed = None
        self.rng_state = {}
        self.deterministic = False
        
        # Store original states for restoration
        self._original_states = {}
        self._backup_states()
    
    def _backup_states(self):
        """Backup original RNG states for restoration."""
        self._original_states = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
        }
        
        if torch.is_available():
            self._original_states['torch_cpu'] = torch.get_rng_state()
            if torch.cuda.is_available():
                self._original_states['torch_cuda'] = torch.cuda.get_rng_state()
    
    def set_global_seed(self, seed: Optional[int] = None, 
                       deterministic: bool = True) -> int:
        """Set a global seed for all random number generators.
        
        This function sets seeds for Python's random module, NumPy, PyTorch (CPU and CUDA),
        and enables deterministic behavior where possible.
        
        Args:
            seed: Seed value to use. If None, uses the default seed.
            deterministic: Whether to enable deterministic behavior
            
        Returns:
            The seed value that was set
            
        Raises:
            RLDKError: If seed validation fails
            
        Example:
            >>> seed_mgr = SeedManager()
            >>> actual_seed = seed_mgr.set_global_seed(42)
            >>> print(actual_seed)
            42
        """
        if seed is None:
            seed = self.default_seed
        
        # Validate seed
        if not isinstance(seed, int) or seed < 0:
            raise RLDKError(
                f"Seed must be a non-negative integer, got: {seed}",
                suggestion="Use a positive integer seed value",
                error_code="INVALID_SEED",
                details={"provided_seed": seed, "type": type(seed).__name__}
            )
        
        self.seed = seed
        self.deterministic = deterministic
        
        # Set Python random seed
        random.seed(seed)
        
        # Set NumPy seed
        np.random.seed(seed)
        
        # Set PyTorch seeds
        if torch.is_available():
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            
            # Enable deterministic behavior if requested
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        
        # Store current states
        self._store_current_states()
        
        return seed
    
    def _store_current_states(self):
        """Store current RNG states for later restoration."""
        self.rng_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
        }
        
        if torch.is_available():
            self.rng_state['torch_cpu'] = torch.get_rng_state()
            if torch.cuda.is_available():
                self.rng_state['torch_cuda'] = torch.cuda.get_rng_state()
    
    def restore_states(self):
        """Restore RNG states to their current checkpoint."""
        if not self.rng_state:
            raise RLDKError(
                "No RNG states to restore",
                suggestion="Call set_global_seed() first to create a checkpoint",
                error_code="NO_STATES_TO_RESTORE"
            )
        
        # Restore Python random state
        random.setstate(self.rng_state['python'])
        
        # Restore NumPy state
        np.random.set_state(self.rng_state['numpy'])
        
        # Restore PyTorch states
        if torch.is_available():
            torch.set_rng_state(self.rng_state['torch_cpu'])
            if torch.cuda.is_available() and 'torch_cuda' in self.rng_state:
                torch.cuda.set_rng_state(self.rng_state['torch_cuda'])
    
    def restore_original_states(self):
        """Restore RNG states to their original values (before any seed setting)."""
        # Restore Python random state
        random.setstate(self._original_states['python'])
        
        # Restore NumPy state
        np.random.set_state(self._original_states['numpy'])
        
        # Restore PyTorch states
        if torch.is_available():
            torch.set_rng_state(self._original_states['torch_cpu'])
            if torch.cuda.is_available() and 'torch_cuda' in self._original_states:
                torch.cuda.set_rng_state(self._original_states['torch_cuda'])
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of current RNG states.
        
        Returns:
            Dictionary containing state information for debugging
            
        Example:
            >>> seed_mgr = SeedManager()
            >>> seed_mgr.set_global_seed(42)
            >>> summary = seed_mgr.get_state_summary()
            >>> print(summary['seed'])
            42
        """
        summary = {
            'seed': self.seed,
            'deterministic': self.deterministic,
            'libraries': list(self.rng_state.keys()),
            'torch_available': torch.is_available(),
            'cuda_available': torch.cuda.is_available() if torch.is_available() else False,
        }
        
        if torch.is_available() and self.deterministic:
            summary['cudnn_deterministic'] = torch.backends.cudnn.deterministic
            summary['cudnn_benchmark'] = torch.backends.cudnn.benchmark
        
        return summary
    
    def set_environment_variables(self, seed: Optional[int] = None):
        """Set environment variables for reproducibility.
        
        This sets common environment variables that affect random number generation
        in various libraries and frameworks.
        
        Args:
            seed: Seed value to use. If None, uses the current seed.
            
        Example:
            >>> seed_mgr = SeedManager()
            >>> seed_mgr.set_global_seed(42)
            >>> seed_mgr.set_environment_variables()
        """
        if seed is None:
            seed = self.seed or self.default_seed
        
        # Set Python hash seed for consistent hashing
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Set tokenizers parallelism to false for determinism
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Set OMP threads for consistent parallel behavior
        os.environ['OMP_NUM_THREADS'] = '1'
        
        # Set MKL threads for consistent behavior
        os.environ['MKL_NUM_THREADS'] = '1'
        
        # Set CUDA launch blocking for debugging (optional)
        if torch.is_available() and torch.cuda.is_available():
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# Global seed manager instance
_global_seed_manager = SeedManager()


def set_global_seed(seed: Optional[int] = None, deterministic: bool = True) -> int:
    """Set a global seed for all random number generators.
    
    This is the main function for setting seeds in RLDK. It ensures reproducible
    behavior across Python, NumPy, PyTorch, and CUDA random number generators.
    
    Args:
        seed: Seed value to use. If None, uses the default seed (1337).
        deterministic: Whether to enable deterministic behavior for PyTorch
        
    Returns:
        The seed value that was set
        
    Raises:
        RLDKError: If seed validation fails
        
    Example:
        >>> import rldk
        >>> seed = rldk.set_global_seed(42)
        >>> print(f"Set seed to: {seed}")
        Set seed to: 42
        
        >>> # Use default seed
        >>> seed = rldk.set_global_seed()
        >>> print(f"Default seed: {seed}")
        Default seed: 1337
    """
    return _global_seed_manager.set_global_seed(seed, deterministic)


def get_current_seed() -> Optional[int]:
    """Get the currently set seed value.
    
    Returns:
        The current seed value, or None if no seed has been set
        
    Example:
        >>> import rldk
        >>> rldk.set_global_seed(42)
        >>> current_seed = rldk.get_current_seed()
        >>> print(current_seed)
        42
    """
    return _global_seed_manager.seed


def restore_seed_state():
    """Restore random number generator states to the last checkpoint.
    
    This restores the RNG states to the values they had when set_global_seed()
    was last called.
    
    Raises:
        RLDKError: If no seed state has been set
        
    Example:
        >>> import rldk
        >>> rldk.set_global_seed(42)
        >>> # ... some random operations ...
        >>> rldk.restore_seed_state()  # Back to state after set_global_seed(42)
    """
    _global_seed_manager.restore_states()


def restore_original_state():
    """Restore random number generator states to their original values.
    
    This restores the RNG states to the values they had before any RLDK seed
    management was applied.
    
    Example:
        >>> import rldk
        >>> rldk.set_global_seed(42)
        >>> # ... some operations ...
        >>> rldk.restore_original_state()  # Back to initial state
    """
    _global_seed_manager.restore_original_states()


def get_seed_state_summary() -> Dict[str, Any]:
    """Get a summary of the current seed state.
    
    Returns:
        Dictionary containing information about the current seed state
        
    Example:
        >>> import rldk
        >>> rldk.set_global_seed(42)
        >>> summary = rldk.get_seed_state_summary()
        >>> print(f"Current seed: {summary['seed']}")
        Current seed: 42
    """
    return _global_seed_manager.get_state_summary()


def set_reproducible_environment(seed: Optional[int] = None):
    """Set up a fully reproducible environment.
    
    This function sets the global seed and configures environment variables
    for maximum reproducibility across different libraries and frameworks.
    
    Args:
        seed: Seed value to use. If None, uses the default seed.
        
    Returns:
        The seed value that was set
        
    Example:
        >>> import rldk
        >>> seed = rldk.set_reproducible_environment(42)
        >>> print(f"Reproducible environment set with seed: {seed}")
        Reproducible environment set with seed: 42
    """
    # Set the global seed
    actual_seed = set_global_seed(seed, deterministic=True)
    
    # Set environment variables
    _global_seed_manager.set_environment_variables(actual_seed)
    
    return actual_seed


def validate_seed_consistency(expected_seed: int, tolerance: float = 1e-6) -> bool:
    """Validate that the current seed produces expected random values.
    
    This function generates a few random numbers and checks if they match
    expected values for the given seed. This is useful for testing seed
    consistency across different runs.
    
    Args:
        expected_seed: The seed that should produce the expected values
        tolerance: Tolerance for floating-point comparisons
        
    Returns:
        True if the seed produces expected values, False otherwise
        
    Example:
        >>> import rldk
        >>> rldk.set_global_seed(42)
        >>> is_consistent = rldk.validate_seed_consistency(42)
        >>> print(f"Seed consistency: {is_consistent}")
        Seed consistency: True
    """
    # Generate some test random numbers
    test_values = {
        'python': [random.random() for _ in range(3)],
        'numpy': np.random.random(3).tolist(),
    }
    
    if torch.is_available():
        test_values['torch'] = torch.rand(3).tolist()
    
    # Store current state
    _global_seed_manager._store_current_states()
    
    # Reset to expected seed and generate same values
    _global_seed_manager.set_global_seed(expected_seed)
    
    expected_values = {
        'python': [random.random() for _ in range(3)],
        'numpy': np.random.random(3).tolist(),
    }
    
    if torch.is_available():
        expected_values['torch'] = torch.rand(3).tolist()
    
    # Restore original state
    _global_seed_manager.restore_states()
    
    # Compare values
    for library in test_values:
        for i, (actual, expected) in enumerate(zip(test_values[library], expected_values[library])):
            if abs(actual - expected) > tolerance:
                return False
    
    return True


def create_seed_context(seed: int, deterministic: bool = True):
    """Create a context manager for temporary seed setting.
    
    This context manager temporarily sets a seed and restores the previous
    state when exiting the context.
    
    Args:
        seed: Seed value to use temporarily
        deterministic: Whether to enable deterministic behavior
        
    Returns:
        Context manager for seed management
        
    Example:
        >>> import rldk
        >>> rldk.set_global_seed(100)
        >>> with rldk.create_seed_context(42):
        ...     print(f"Temp seed: {rldk.get_current_seed()}")
        ...     # Random operations here use seed 42
        ... pass
        Temp seed: 42
        >>> print(f"Restored seed: {rldk.get_current_seed()}")
        Restored seed: 100
    """
    class SeedContext:
        def __init__(self, seed: int, deterministic: bool):
            self.seed = seed
            self.deterministic = deterministic
            self.original_states = None
        
        def __enter__(self):
            # Store current states
            self.original_states = _global_seed_manager.rng_state.copy()
            
            # Set new seed
            _global_seed_manager.set_global_seed(self.seed, self.deterministic)
            
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore original states
            if self.original_states:
                _global_seed_manager.rng_state = self.original_states
                _global_seed_manager.restore_states()
    
    return SeedContext(seed, deterministic)