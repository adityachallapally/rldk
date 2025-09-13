"""Runtime utilities for RLDK with cross-platform timeout support."""

import os
import signal
import sys
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from functools import wraps
from typing import Callable, Any, Optional, Union, Dict, List

try:
    import psutil
except ImportError:
    psutil = None


class RLDKTimeoutError(Exception):
    """Raised when operations timeout."""
    pass


def with_timeout(seconds: int) -> Callable:
    """
    Decorator to add timeout to operations.
    
    On POSIX main thread: uses signal.SIGALRM for precise timeout control.
    Otherwise: uses concurrent.futures.ThreadPoolExecutor with soft cancellation.
    
    Args:
        seconds: Timeout in seconds
        
    Returns:
        Decorated function that raises RLDKTimeoutError on timeout
        
    Note:
        The ThreadPoolExecutor path provides soft cancellation - the function
        may continue running in the background after timeout.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Check if we're on POSIX and in the main thread
            if (os.name == 'posix' and 
                threading.current_thread() is threading.main_thread()):
                return _posix_timeout(func, seconds, *args, **kwargs)
            else:
                return _fallback_timeout(func, seconds, *args, **kwargs)
        return wrapper
    return decorator


def _posix_timeout(func: Callable, timeout_seconds: int, *args, **kwargs) -> Any:
    """POSIX timeout implementation using signal.SIGALRM."""
    def timeout_handler(signum, frame):
        raise RLDKTimeoutError(f"Operation timed out after {timeout_seconds} seconds")
    
    # Set up timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout_seconds))
    
    try:
        result = func(*args, **kwargs)
        return result
    finally:
        # Restore old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _fallback_timeout(func: Callable, timeout_seconds: int, *args, **kwargs) -> Any:
    """Fallback timeout implementation using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            raise RLDKTimeoutError(f"Operation timed out after {timeout_seconds} seconds")


def run_with_timeout_subprocess(
    argv: Union[List[str], str], 
    timeout: int, 
    cwd: Optional[str] = None, 
    env: Optional[Dict[str, str]] = None,
    shell: bool = False
) -> subprocess.CompletedProcess:
    """
    Run a subprocess with timeout and cross-platform process tree cleanup.
    
    Args:
        argv: Command and arguments as list or string (if shell=True)
        timeout: Timeout in seconds
        cwd: Working directory for the subprocess
        env: Environment variables for the subprocess
        shell: Whether to use shell execution
        
    Returns:
        CompletedProcess result
        
    Raises:
        RLDKTimeoutError: If the subprocess times out
        subprocess.CalledProcessError: If the subprocess fails
    """
    try:
        # Start the subprocess
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=shell
        )
        
        try:
            # Wait for completion with timeout
            stdout, stderr = process.communicate(timeout=timeout)
            return subprocess.CompletedProcess(
                argv, process.returncode, stdout, stderr
            )
        except subprocess.TimeoutExpired:
            # Kill the process tree cross-platform
            _kill_process_tree(process.pid)
            
            # Wait a bit for cleanup
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if still running
                process.kill()
                process.wait()
            
            raise RLDKTimeoutError(
                f"Subprocess timed out after {timeout} seconds. "
                f"Command: {' '.join(argv)}"
            )
            
    except FileNotFoundError as e:
        # Re-raise FileNotFoundError as-is since it's not a timeout issue
        raise e


def _kill_process_tree(pid: int) -> None:
    """
    Kill a process and all its children cross-platform.
    
    Args:
        pid: Process ID to kill
    """
    if psutil is not None:
        try:
            # Get the process
            parent = psutil.Process(pid)
            
            # Get all children
            children = parent.children(recursive=True)
            
            # Kill children first
            for child in children:
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Kill parent
            try:
                parent.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
        except psutil.NoSuchProcess:
            # Process already dead
            pass
        except Exception:
            # Fallback to basic kill
            _basic_kill_process(pid)
    else:
        # No psutil available, use basic kill
        _basic_kill_process(pid)


def _basic_kill_process(pid: int) -> None:
    """Basic process killing without psutil."""
    try:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], 
                         check=False, capture_output=True)
        else:
            subprocess.run(['kill', '-9', str(pid)], 
                         check=False, capture_output=True)
    except Exception:
        pass