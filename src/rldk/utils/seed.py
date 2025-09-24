"""Seed management utilities for reproducible RL experiments.

This module provides centralized seed handling for ensuring reproducibility
across different components of RL training, including Python, NumPy, PyTorch,
and CUDA random number generators.
"""

import os
import random
import threading
from typing import Any, Dict, Optional

import numpy as np

from .error_handling import RLDKError

# Conditional PyTorch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


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
        # Simple lock for thread safety in multi-threaded ML training scenarios
        self._lock = threading.Lock()

        # Store original states for restoration
        self._original_states = {}
        self._backup_states()

    def _backup_states(self):
        """Backup original RNG states for restoration."""
        self._original_states = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
        }

        if TORCH_AVAILABLE:
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
        with self._lock:
            if seed is None:
                seed = self.default_seed

            # Normalize booleans and integral floats
            if isinstance(seed, bool):
                seed = int(seed)
            elif isinstance(seed, float):
                if not seed.is_integer():
                    raise TypeError(
                        "Seed must be an integer value",
                    )
                seed = int(seed)

            if not isinstance(seed, int):
                raise TypeError("Seed must be an integer value")

            self.seed = seed
            self.deterministic = deterministic

            normalized_seed = seed if seed >= 0 else seed % (2**32)

            # Set Python random seed
            random.seed(normalized_seed)

            # Set NumPy seed
            np.random.seed(normalized_seed)

            # Store current states
            self._store_current_states()

        # Set PyTorch seeds outside lock to avoid deadlocks
        if TORCH_AVAILABLE and torch is not None:
            try:
                torch.manual_seed(normalized_seed)
            except Exception:
                pass

            cuda_module = getattr(torch, "cuda", None)
            if cuda_module is not None:
                try:
                    manual_seed = getattr(cuda_module, "manual_seed", None)
                    if manual_seed is not None:
                        manual_seed(normalized_seed)
                except Exception:
                    pass

                try:
                    manual_seed_all = getattr(cuda_module, "manual_seed_all", None)
                    if manual_seed_all is not None:
                        manual_seed_all(normalized_seed)
                except Exception:
                    pass

            if deterministic:
                try:
                    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
                except Exception:
                    pass

        return seed

    def _store_current_states(self):
        """Store current RNG states for later restoration.

        Note: This method should be called within a lock context.
        """
        self.rng_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
        }

        if TORCH_AVAILABLE:
            self.rng_state['torch_cpu'] = torch.get_rng_state()
            if torch.cuda.is_available():
                self.rng_state['torch_cuda'] = torch.cuda.get_rng_state()

    def restore_states(self):
        """Restore RNG states to their current checkpoint."""
        with self._lock:
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
            if TORCH_AVAILABLE:
                torch.set_rng_state(self.rng_state['torch_cpu'])
                if torch.cuda.is_available() and 'torch_cuda' in self.rng_state:
                    torch.cuda.set_rng_state(self.rng_state['torch_cuda'])

    def restore_original_states(self):
        """Restore RNG states to their original values (before any seed setting)."""
        with self._lock:
            # Restore Python random state
            random.setstate(self._original_states['python'])

            # Restore NumPy state
            np.random.set_state(self._original_states['numpy'])

            # Restore PyTorch states
            if TORCH_AVAILABLE and 'torch_cpu' in self._original_states:
                torch.set_rng_state(self._original_states['torch_cpu'])
                if torch.cuda.is_available() and 'torch_cuda' in self._original_states:
                    torch.cuda.set_rng_state(self._original_states['torch_cuda'])

    def reset_seed(self) -> None:
        """Clear tracked seed metadata and restore original RNG states."""
        self.restore_original_states()

        with self._lock:
            self.seed = None
            self.rng_state = {}
            self.deterministic = False

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
        with self._lock:
            summary = {
                'seed': self.seed,
                'global_seed': self.seed,
                'deterministic': self.deterministic,
                'libraries': list(self.rng_state.keys()),
                'torch_available': TORCH_AVAILABLE,
                'cuda_available': torch.cuda.is_available() if TORCH_AVAILABLE else False,
                'python_state': self.rng_state.get('python'),
                'numpy_state': self.rng_state.get('numpy'),
            }

            if TORCH_AVAILABLE:
                summary['torch_cpu_state'] = self.rng_state.get('torch_cpu')
                if torch.cuda.is_available():
                    summary['torch_cuda_state'] = self.rng_state.get('torch_cuda')

        if TORCH_AVAILABLE and self.deterministic:
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

        # Enforce deterministic kernel selection across popular accelerator stacks
        os.environ['CUDNN_DETERMINISTIC'] = 'true'
        os.environ['CUDNN_BENCHMARK'] = 'false'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

        # Set OMP threads for consistent parallel behavior
        os.environ['OMP_NUM_THREADS'] = '1'

        # Set MKL threads for consistent behavior
        os.environ['MKL_NUM_THREADS'] = '1'

        # Set CUDA launch blocking to promote synchronous kernel execution
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


def reset_global_seed() -> None:
    """Reset the tracked global seed and RNG checkpoints to their defaults."""
    _global_seed_manager.reset_seed()


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
    try:
        _global_seed_manager.restore_states()
    except RLDKError:
        # No checkpoint has been established yet; treat as no-op for callers
        return


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

    if TORCH_AVAILABLE:
        try:
            torch.use_deterministic_algorithms(True)
        except (AttributeError, RuntimeError):  # pragma: no cover - older torch versions
            pass

        try:
            torch.backends.cudnn.allow_tf32 = False
        except AttributeError:  # pragma: no cover - backend missing
            pass

        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except AttributeError:  # pragma: no cover - backend missing
            pass

        if hasattr(torch, "set_float32_matmul_precision"):
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:  # pragma: no cover - defensive guard
                pass

    return actual_seed


def validate_seed_consistency(expected_seed: int, tolerance: float = 1e-6) -> bool:
    """Validate that a seed produces consistent random values across runs.

    This function tests if the same seed produces identical random sequences
    across multiple runs, which is essential for reproducibility.

    Args:
        expected_seed: The seed to validate
        tolerance: Tolerance for floating-point comparisons

    Returns:
        True if the seed produces consistent values, False otherwise

    Example:
        >>> import rldk
        >>> is_consistent = rldk.validate_seed_consistency(42)
        >>> print(f"Seed consistency: {is_consistent}")
        Seed consistency: True
    """
    # Store original states before any modifications
    original_states = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
    }

    if TORCH_AVAILABLE:
        original_states['torch_cpu'] = torch.get_rng_state()
        if torch.cuda.is_available():
            original_states['torch_cuda'] = torch.cuda.get_rng_state()

    try:
        # Generate first set of values with the seed
        _global_seed_manager.set_global_seed(expected_seed)

        first_run_values = {
            'python': [random.random() for _ in range(3)],
            'numpy': np.random.random(3).tolist(),
        }

        if TORCH_AVAILABLE:
            first_run_values['torch'] = torch.rand(3).tolist()

        # Generate second set of values with the same seed
        _global_seed_manager.set_global_seed(expected_seed)

        second_run_values = {
            'python': [random.random() for _ in range(3)],
            'numpy': np.random.random(3).tolist(),
        }

        if TORCH_AVAILABLE:
            second_run_values['torch'] = torch.rand(3).tolist()

        # Compare values - they should be identical
        for library in first_run_values:
            for i, (first, second) in enumerate(zip(first_run_values[library], second_run_values[library])):
                if abs(first - second) > tolerance:
                    return False

        return True

    finally:
        # Always restore original states
        random.setstate(original_states['python'])
        np.random.set_state(original_states['numpy'])

        if TORCH_AVAILABLE:
            torch.set_rng_state(original_states['torch_cpu'])
            if torch.cuda.is_available() and 'torch_cuda' in original_states:
                torch.cuda.set_rng_state(original_states['torch_cuda'])


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
            self.original_seed = None
            self.original_states = None

        def __enter__(self):
            # Store current seed and states
            self.original_seed = _global_seed_manager.seed
            self.original_states = {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
            }

            if TORCH_AVAILABLE:
                self.original_states['torch_cpu'] = torch.get_rng_state()
                if torch.cuda.is_available():
                    self.original_states['torch_cuda'] = torch.cuda.get_rng_state()

            # Set new seed
            _global_seed_manager.set_global_seed(self.seed, self.deterministic)

            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore original states
            if self.original_states:
                random.setstate(self.original_states['python'])
                np.random.set_state(self.original_states['numpy'])

                if TORCH_AVAILABLE:
                    torch.set_rng_state(self.original_states['torch_cpu'])
                    if torch.cuda.is_available() and 'torch_cuda' in self.original_states:
                        torch.cuda.set_rng_state(self.original_states['torch_cuda'])

                # Restore original seed
                _global_seed_manager.seed = self.original_seed

    return SeedContext(seed, deterministic)
