"""Runtime utilities for RLDK including timeout and subprocess management."""

import signal
import subprocess
import signal
import subprocess
from contextlib import contextmanager
from functools import wraps
from typing import Callable, List, Union

from .error_handling import RLDKTimeoutError


def with_timeout(timeout_seconds: float):
    """Decorator to add timeout to operations.

    Args:
        timeout_seconds: Maximum time to wait for the operation to complete

    Returns:
        Decorated function that will raise RLDKTimeoutError if timeout is exceeded
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            def timeout_handler(signum, frame):
                raise RLDKTimeoutError(
                    f"Operation timed out after {timeout_seconds} seconds",
                    suggestion="Try increasing the timeout or optimizing the operation",
                    error_code="OPERATION_TIMEOUT",
                    details={"timeout_seconds": timeout_seconds}
                )

            import math

            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            if hasattr(signal, "setitimer"):
                signal.setitimer(signal.ITIMER_REAL, max(timeout_seconds, 0.001))
            else:
                timeout_int = max(1, math.ceil(timeout_seconds))
                signal.alarm(timeout_int)

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore old handler
                if hasattr(signal, "setitimer"):
                    signal.setitimer(signal.ITIMER_REAL, 0)
                else:
                    signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper
    return decorator


def run_with_timeout_subprocess(
    cmd: Union[str, List[str]],
    timeout_seconds: float = 300,
    **subprocess_kwargs
) -> subprocess.CompletedProcess:
    """Run a subprocess command with timeout.

    Args:
        cmd: Command to run (string or list of arguments)
        timeout_seconds: Maximum time to wait for command completion
        **subprocess_kwargs: Additional arguments passed to subprocess.run

    Returns:
        CompletedProcess result from subprocess.run

    Raises:
        RLDKTimeoutError: If the command times out
    """
    try:
        return subprocess.run(
            cmd,
            timeout=timeout_seconds,
            **subprocess_kwargs
        )
    except subprocess.TimeoutExpired as e:
        raise RLDKTimeoutError(
            f"Subprocess command timed out after {timeout_seconds} seconds",
            suggestion="Try increasing the timeout or optimizing the command",
            error_code="SUBPROCESS_TIMEOUT",
            details={
                "timeout_seconds": timeout_seconds,
                "command": cmd,
                "partial_output": getattr(e, 'stdout', ''),
                "partial_stderr": getattr(e, 'stderr', '')
            }
        ) from e


@contextmanager
def timeout_context(timeout_seconds: float):
    """Context manager for timeout operations.

    Args:
        timeout_seconds: Maximum time to wait

    Yields:
        None

    Raises:
        RLDKTimeoutError: If timeout is exceeded
    """
    def timeout_handler(signum, frame):
        raise RLDKTimeoutError(
            f"Operation timed out after {timeout_seconds} seconds",
            suggestion="Try increasing the timeout or optimizing the operation",
            error_code="OPERATION_TIMEOUT",
            details={"timeout_seconds": timeout_seconds}
        )

    # Set up timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout_seconds))

    try:
        yield
    finally:
        # Restore old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
