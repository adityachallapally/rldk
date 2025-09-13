"""Simple tests for runtime timeout utilities."""

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


def test_timeout_error_creation():
    """Test that RLDKTimeoutError can be created."""
    error = RLDKTimeoutError("Test timeout")
    assert str(error) == "Test timeout"
    assert isinstance(error, Exception)
    print("✓ RLDKTimeoutError creation test passed")


def test_successful_execution():
    """Test that successful function execution works."""
    @with_timeout(5)
    def quick_function():
        return "success"
    
    result = quick_function()
    assert result == "success"
    print("✓ Successful execution test passed")


def test_timeout_on_sleeping_function():
    """Test that timeout works on a sleeping function."""
    @with_timeout(0.1)  # Very short timeout
    def sleeping_function():
        time.sleep(1)  # Sleep longer than timeout
        return "should not reach here"
    
    try:
        sleeping_function()
        assert False, "Expected RLDKTimeoutError but function completed"
    except RLDKTimeoutError as e:
        assert "timed out after 0.1 seconds" in str(e)
        print("✓ Timeout on sleeping function test passed")


def test_fallback_timeout():
    """Test fallback timeout implementation."""
    def sleeping_func():
        time.sleep(1)
        return "should not reach here"
    
    try:
        _fallback_timeout(sleeping_func, 0.1)
        assert False, "Expected RLDKTimeoutError but function completed"
    except RLDKTimeoutError as e:
        assert "timed out after 0.1 seconds" in str(e)
        print("✓ Fallback timeout test passed")


def test_run_with_timeout_subprocess_success():
    """Test successful subprocess execution."""
    result = run_with_timeout_subprocess(
        ["python3", "-c", "print('hello')"],
        timeout=10
    )
    assert result.returncode == 0
    assert "hello" in result.stdout
    print("✓ Subprocess success test passed")


def test_run_with_timeout_subprocess_timeout():
    """Test subprocess timeout."""
    try:
        run_with_timeout_subprocess(
            ["python3", "-c", "import time; time.sleep(2)"],
            timeout=0.5
        )
        assert False, "Expected RLDKTimeoutError but subprocess completed"
    except RLDKTimeoutError as e:
        assert "timed out after 0.5 seconds" in str(e)
        print("✓ Subprocess timeout test passed")


def test_python_sleep_timeout():
    """Test Python sleep command with timeout (as specified in requirements)."""
    try:
        run_with_timeout_subprocess(
            ["python3", "-c", "import time; time.sleep(2)"],
            timeout=0.5
        )
        assert False, "Expected RLDKTimeoutError but subprocess completed"
    except RLDKTimeoutError as e:
        assert "timed out after 0.5 seconds" in str(e)
        print("✓ Python sleep timeout test passed")


def test_monkeypatch_platform_fallback():
    """Test monkeypatching platform to force fallback behavior."""
    with patch('rldk.utils.runtime.os.name', 'nt'):
        @with_timeout(0.1)
        def sleeping_function():
            time.sleep(1)
            return "should not reach here"
        
        try:
            sleeping_function()
            assert False, "Expected RLDKTimeoutError but function completed"
        except RLDKTimeoutError:
            print("✓ Monkeypatch platform fallback test passed")


def test_monkeypatch_thread_state():
    """Test monkeypatching thread state to force fallback behavior."""
    with patch('rldk.utils.runtime.threading.current_thread') as mock_current, \
         patch('rldk.utils.runtime.threading.main_thread') as mock_main:
        
        # Make current thread different from main thread
        mock_current.return_value = MagicMock()
        mock_main.return_value = MagicMock()
        
        @with_timeout(0.1)
        def sleeping_function():
            time.sleep(1)
            return "should not reach here"
        
        try:
            sleeping_function()
            assert False, "Expected RLDKTimeoutError but function completed"
        except RLDKTimeoutError:
            print("✓ Monkeypatch thread state test passed")


def main():
    """Run all tests."""
    print("Running runtime timeout tests...")
    
    try:
        test_timeout_error_creation()
        test_successful_execution()
        test_timeout_on_sleeping_function()
        test_fallback_timeout()
        test_run_with_timeout_subprocess_success()
        test_run_with_timeout_subprocess_timeout()
        test_python_sleep_timeout()
        test_monkeypatch_platform_fallback()
        test_monkeypatch_thread_state()
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)