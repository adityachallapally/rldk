"""Tests for runtime timeout utilities."""

import os
import sys
import time
import threading
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from rldk.utils.runtime import (
    RLDKTimeoutError, 
    with_timeout, 
    run_with_timeout_subprocess,
    _posix_timeout,
    _fallback_timeout
)


class TestRLDKTimeoutError:
    """Test RLDKTimeoutError exception."""
    
    def test_timeout_error_creation(self):
        """Test that RLDKTimeoutError can be created."""
        error = RLDKTimeoutError("Test timeout")
        assert str(error) == "Test timeout"
        assert isinstance(error, Exception)


class TestWithTimeoutDecorator:
    """Test the with_timeout decorator."""
    
    def test_successful_execution(self):
        """Test that successful function execution works."""
        @with_timeout(5)
        def quick_function():
            return "success"
        
        result = quick_function()
        assert result == "success"
    
    def test_timeout_on_sleeping_function(self):
        """Test that timeout works on a sleeping function."""
        @with_timeout(0.1)  # Very short timeout
        def sleeping_function():
            time.sleep(1)  # Sleep longer than timeout
            return "should not reach here"
        
        try:
            sleeping_function()
            assert False, "Expected RLDKTimeoutError but function completed"
        except RLDKTimeoutError:
            pass  # Expected
    
    def test_timeout_message(self):
        """Test that timeout error has correct message."""
        @with_timeout(0.1)
        def sleeping_function():
            time.sleep(1)
            return "should not reach here"
        
        with pytest.raises(RLDKTimeoutError) as exc_info:
            sleeping_function()
        
        assert "timed out after 0.1 seconds" in str(exc_info.value)
    
    def test_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""
        @with_timeout(5)
        def test_func(a, b=10):
            """Test function docstring."""
            return a + b
        
        assert test_func.__name__ == "test_func"
        assert test_func.__doc__ == "Test function docstring."
        assert test_func(1) == 11
        assert test_func(1, 2) == 3


class TestPosixTimeout:
    """Test POSIX timeout implementation."""
    
    @pytest.mark.skipif(os.name != 'posix', reason="POSIX-specific test")
    def test_posix_timeout_success(self):
        """Test POSIX timeout on successful function."""
        def quick_func():
            return "posix success"
        
        result = _posix_timeout(quick_func, 5)
        assert result == "posix success"
    
    @pytest.mark.skipif(os.name != 'posix', reason="POSIX-specific test")
    def test_posix_timeout_failure(self):
        """Test POSIX timeout on sleeping function."""
        def sleeping_func():
            time.sleep(1)
            return "should not reach here"
        
        with pytest.raises(RLDKTimeoutError):
            _posix_timeout(sleeping_func, 0.1)


class TestFallbackTimeout:
    """Test fallback timeout implementation."""
    
    def test_fallback_timeout_success(self):
        """Test fallback timeout on successful function."""
        def quick_func():
            return "fallback success"
        
        result = _fallback_timeout(quick_func, 5)
        assert result == "fallback success"
    
    def test_fallback_timeout_failure(self):
        """Test fallback timeout on sleeping function."""
        def sleeping_func():
            time.sleep(1)
            return "should not reach here"
        
        with pytest.raises(RLDKTimeoutError):
            _fallback_timeout(sleeping_func, 0.1)
    
    def test_fallback_timeout_soft_cancel(self):
        """Test that fallback timeout provides soft cancellation."""
        results = []
        
        def long_running_func():
            for i in range(10):
                time.sleep(0.1)
                results.append(i)
            return "completed"
        
        # This should timeout but the function may continue running
        with pytest.raises(RLDKTimeoutError):
            _fallback_timeout(long_running_func, 0.2)
        
        # Give it time to complete
        time.sleep(1)
        
        # The function may have continued running in the background
        # This demonstrates the "soft cancel" behavior
        assert len(results) > 0


class TestRunWithTimeoutSubprocess:
    """Test run_with_timeout_subprocess function."""
    
    def test_successful_command(self):
        """Test successful command execution."""
        result = run_with_timeout_subprocess(
            ["python", "-c", "print('hello')"],
            timeout=10
        )
        assert result.returncode == 0
        assert "hello" in result.stdout
    
    def test_command_timeout(self):
        """Test command timeout."""
        with pytest.raises(RLDKTimeoutError) as exc_info:
            run_with_timeout_subprocess(
                ["python", "-c", "import time; time.sleep(2)"],
                timeout=0.5
            )
        
        assert "timed out after 0.5 seconds" in str(exc_info.value)
    
    def test_command_with_cwd(self):
        """Test command with working directory."""
        result = run_with_timeout_subprocess(
            ["python", "-c", "import os; print(os.getcwd())"],
            timeout=10,
            cwd="/tmp"
        )
        assert result.returncode == 0
        assert "/tmp" in result.stdout
    
    def test_command_with_env(self):
        """Test command with environment variables."""
        env = {"TEST_VAR": "test_value"}
        result = run_with_timeout_subprocess(
            ["python", "-c", "import os; print(os.environ.get('TEST_VAR', 'not_found'))"],
            timeout=10,
            env=env
        )
        assert result.returncode == 0
        assert "test_value" in result.stdout
    
    def test_command_not_found(self):
        """Test command not found error."""
        with pytest.raises(RLDKTimeoutError) as exc_info:
            run_with_timeout_subprocess(
                ["nonexistent_command_12345"],
                timeout=10
            )
        
        assert "Command not found" in str(exc_info.value)
    
    def test_python_sleep_timeout(self):
        """Test Python sleep command with timeout (as specified in requirements)."""
        with pytest.raises(RLDKTimeoutError):
            run_with_timeout_subprocess(
                ["python", "-c", "import time; time.sleep(2)"],
                timeout=0.5
            )


class TestPlatformSpecificBehavior:
    """Test platform-specific behavior."""
    
    def test_decorator_platform_detection(self):
        """Test that decorator detects platform correctly."""
        # Mock platform detection
        with patch('rldk.utils.runtime.os.name', 'posix'), \
             patch('rldk.utils.runtime.threading.current_thread') as mock_thread, \
             patch('rldk.utils.runtime.threading.main_thread') as mock_main_thread:
            
            mock_thread.return_value = mock_main_thread.return_value
            
            @with_timeout(1)
            def test_func():
                return "posix"
            
            # Should use POSIX path
            with patch('rldk.utils.runtime._posix_timeout') as mock_posix:
                mock_posix.return_value = "posix_result"
                result = test_func()
                assert result == "posix_result"
                mock_posix.assert_called_once()
    
    def test_decorator_fallback_behavior(self):
        """Test that decorator falls back to ThreadPoolExecutor."""
        # Mock non-POSIX platform
        with patch('rldk.utils.runtime.os.name', 'nt'):
            @with_timeout(1)
            def test_func():
                return "fallback"
            
            # Should use fallback path
            with patch('rldk.utils.runtime._fallback_timeout') as mock_fallback:
                mock_fallback.return_value = "fallback_result"
                result = test_func()
                assert result == "fallback_result"
                mock_fallback.assert_called_once()
    
    def test_decorator_non_main_thread(self):
        """Test that decorator uses fallback in non-main thread."""
        with patch('rldk.utils.runtime.os.name', 'posix'), \
             patch('rldk.utils.runtime.threading.current_thread') as mock_current, \
             patch('rldk.utils.runtime.threading.main_thread') as mock_main:
            
            # Make current thread different from main thread
            mock_current.return_value = MagicMock()
            mock_main.return_value = MagicMock()
            
            @with_timeout(1)
            def test_func():
                return "non_main_thread"
            
            # Should use fallback path
            with patch('rldk.utils.runtime._fallback_timeout') as mock_fallback:
                mock_fallback.return_value = "fallback_result"
                result = test_func()
                assert result == "fallback_result"
                mock_fallback.assert_called_once()


class TestMonkeyPatching:
    """Test monkeypatching for platform and thread state."""
    
    def test_monkeypatch_platform_fallback(self):
        """Test monkeypatching platform to force fallback behavior."""
        # Force fallback behavior by mocking platform
        with patch('rldk.utils.runtime.os.name', 'nt'):
            @with_timeout(0.1)
            def sleeping_function():
                time.sleep(1)
                return "should not reach here"
            
            with pytest.raises(RLDKTimeoutError):
                sleeping_function()
    
    def test_monkeypatch_thread_state(self):
        """Test monkeypatching thread state to force fallback behavior."""
        # Force fallback behavior by mocking thread state
        with patch('rldk.utils.runtime.threading.current_thread') as mock_current, \
             patch('rldk.utils.runtime.threading.main_thread') as mock_main:
            
            # Make current thread different from main thread
            mock_current.return_value = MagicMock()
            mock_main.return_value = MagicMock()
            
            @with_timeout(0.1)
            def sleeping_function():
                time.sleep(1)
                return "should not reach here"
            
            with pytest.raises(RLDKTimeoutError):
                sleeping_function()


if __name__ == "__main__":
    pytest.main([__file__])