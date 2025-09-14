#!/usr/bin/env python3
"""Unit tests for rldk.utils.progress module."""

import threading
import time
from unittest.mock import patch

import pytest

# Import the module under test
from rldk.utils.progress import (
    ProgressBar,
    Spinner,
    TaskContext,
    TaskTracker,
    create_batch_progress,
    create_download_progress,
    create_processing_progress,
    create_progress_callback,
    create_validation_progress,
    estimate_remaining_time,
    estimate_time_remaining,
    format_duration,
    format_time_remaining,
    print_operation_status,
    progress_bar,
    spinner,
    timed_operation,
    timed_operation_context,
    track_progress,
    track_tasks,
)


class TestProgressBar:
    """Test ProgressBar class."""

    def test_progress_bar_initialization(self):
        """Test ProgressBar initialization."""
        bar = ProgressBar(100, "Test Progress", 50, True)

        assert bar.total == 100
        assert bar.current == 0
        assert bar.description == "Test Progress"
        assert bar.width == 50
        assert bar.show_percentage is True

    def test_progress_bar_update(self):
        """Test ProgressBar update method."""
        bar = ProgressBar(100, "Test Progress")

        with patch('builtins.print') as mock_print:
            bar.update(10)

        assert bar.current == 10

        # Should print progress bar
        mock_print.assert_called()

    def test_progress_bar_update_with_description(self):
        """Test ProgressBar update with description change."""
        bar = ProgressBar(100, "Test Progress")

        with patch('builtins.print'):
            bar.update(10, "New Description")

        assert bar.description == "New Description"
        assert bar.current == 10

    def test_progress_bar_update_overflow(self):
        """Test ProgressBar update with overflow."""
        bar = ProgressBar(100, "Test Progress")

        with patch('builtins.print'):
            bar.update(150)  # More than total

        assert bar.current == 100  # Should be capped at total

    def test_progress_bar_set_progress(self):
        """Test ProgressBar set_progress method."""
        bar = ProgressBar(100, "Test Progress")

        with patch('builtins.print'):
            bar.set_progress(50)

        assert bar.current == 50

    def test_progress_bar_set_progress_overflow(self):
        """Test ProgressBar set_progress with overflow."""
        bar = ProgressBar(100, "Test Progress")

        with patch('builtins.print'):
            bar.set_progress(150)  # More than total

        assert bar.current == 100  # Should be capped at total

    def test_progress_bar_finish(self):
        """Test ProgressBar finish method."""
        bar = ProgressBar(100, "Test Progress")

        with patch('builtins.print') as mock_print:
            bar.finish()

        assert bar.current == 100

        # Should print final progress bar
        mock_print.assert_called()

    def test_progress_bar_finish_with_description(self):
        """Test ProgressBar finish with description change."""
        bar = ProgressBar(100, "Test Progress")

        with patch('builtins.print'):
            bar.finish("Finished")

        assert bar.description == "Finished"
        assert bar.current == 100

    def test_progress_bar_zero_total(self):
        """Test ProgressBar with zero total."""
        bar = ProgressBar(0, "Test Progress")

        with patch('builtins.print') as mock_print:
            bar.update(1)

        # Should not print anything when total is 0
        mock_print.assert_not_called()

    def test_progress_bar_thread_safety(self):
        """Test ProgressBar thread safety."""
        bar = ProgressBar(1000, "Test Progress")

        def update_progress():
            for _ in range(100):
                bar.update(1)

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=update_progress)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have updated progress correctly
        assert bar.current == 500  # 5 threads * 100 updates each


class TestSpinner:
    """Test Spinner class."""

    def test_spinner_initialization(self):
        """Test Spinner initialization."""
        spinner_obj = Spinner("Test Processing")

        assert spinner_obj.description == "Test Processing"
        assert spinner_obj.spinner_chars == "|/-\\"
        assert spinner_obj.current_char == 0
        assert spinner_obj.running is False

    def test_spinner_start_stop(self):
        """Test Spinner start and stop."""
        spinner_obj = Spinner("Test Processing")

        with patch('builtins.print') as mock_print:
            spinner_obj.start()
            time.sleep(0.2)  # Let it spin a bit
            spinner_obj.stop()

        # Should have printed spinner characters
        assert mock_print.call_count > 0

    def test_spinner_start_stop_with_message(self):
        """Test Spinner start and stop with message."""
        spinner_obj = Spinner("Test Processing")

        with patch('builtins.print') as mock_print:
            spinner_obj.start()
            time.sleep(0.1)
            spinner_obj.stop("Done")

        # Should have printed spinner characters and final message
        assert mock_print.call_count > 0

    def test_spinner_double_start(self):
        """Test Spinner double start."""
        spinner_obj = Spinner("Test Processing")

        with patch('builtins.print') as mock_print:
            spinner_obj.start()
            spinner_obj.start()  # Second start should be ignored
            time.sleep(0.1)
            spinner_obj.stop()

        # Should still work normally
        assert mock_print.call_count > 0

    def test_spinner_stop_when_not_running(self):
        """Test Spinner stop when not running."""
        spinner_obj = Spinner("Test Processing")

        with patch('builtins.print') as mock_print:
            spinner_obj.stop()  # Stop when not running

        # Should not print anything
        mock_print.assert_not_called()


class TestContextManagers:
    """Test context managers."""

    def test_progress_bar_context_manager(self):
        """Test progress_bar context manager."""
        with patch('builtins.print') as mock_print:
            with progress_bar(100, "Test Progress") as bar:
                bar.update(50)

        # Should have printed progress updates
        assert mock_print.call_count > 0

    def test_spinner_context_manager(self):
        """Test spinner context manager."""
        with patch('builtins.print') as mock_print:
            with spinner("Test Processing"):
                time.sleep(0.1)

        # Should have printed spinner characters
        assert mock_print.call_count > 0

    def test_timed_operation_context_success(self):
        """Test timed_operation_context with successful operation."""
        with patch('builtins.print') as mock_print:
            with timed_operation_context("Test Operation"):
                time.sleep(0.01)

        # Should print start and completion messages
        calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Starting Test Operation" in call for call in calls)
        assert any("completed" in call for call in calls)

    def test_timed_operation_context_failure(self):
        """Test timed_operation_context with failed operation."""
        with patch('builtins.print') as mock_print:
            with pytest.raises(ValueError):
                with timed_operation_context("Test Operation"):
                    raise ValueError("Test error")

        # Should print start and failure messages
        calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Starting Test Operation" in call for call in calls)
        assert any("failed" in call for call in calls)


class TestTrackProgress:
    """Test track_progress function."""

    def test_track_progress_with_total(self):
        """Test track_progress with known total."""
        items = [1, 2, 3, 4, 5]

        with patch('builtins.print') as mock_print:
            result = list(track_progress(items, total=5, description="Processing"))

        assert result == items
        assert mock_print.call_count > 0

    def test_track_progress_without_total(self):
        """Test track_progress without total."""
        items = [1, 2, 3, 4, 5]

        with patch('builtins.print') as mock_print:
            result = list(track_progress(items, description="Processing"))

        assert result == items
        assert mock_print.call_count > 0

    def test_track_progress_with_spinner(self):
        """Test track_progress with spinner for unknown length."""
        def generator():
            yield from range(3)

        with patch('builtins.print') as mock_print:
            result = list(track_progress(generator(), description="Processing"))

        assert result == [0, 1, 2]
        assert mock_print.call_count > 0


class TestTimedOperation:
    """Test timed_operation decorator."""

    def test_timed_operation_success(self):
        """Test timed_operation decorator with successful operation."""
        @timed_operation("Test Operation")
        def test_func():
            return "success"

        with patch('builtins.print') as mock_print:
            result = test_func()

        assert result == "success"

        # Should print start and completion messages
        calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Starting Test Operation" in call for call in calls)
        assert any("completed" in call for call in calls)

    def test_timed_operation_failure(self):
        """Test timed_operation decorator with failed operation."""
        @timed_operation("Test Operation")
        def test_func():
            raise ValueError("Test error")

        with patch('builtins.print') as mock_print:
            with pytest.raises(ValueError, match="Test error"):
                test_func()

        # Should print start and failure messages
        calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Starting Test Operation" in call for call in calls)
        assert any("failed" in call for call in calls)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_estimate_remaining_time(self):
        """Test estimate_remaining_time function."""
        start_time = time.time() - 10  # 10 seconds ago
        current = 50
        total = 100

        eta = estimate_remaining_time(current, total, start_time)

        # Should be approximately 10 seconds (half done, 10 seconds elapsed)
        assert 8 <= eta <= 12

    def test_estimate_remaining_time_zero_current(self):
        """Test estimate_remaining_time with zero current."""
        start_time = time.time()
        current = 0
        total = 100

        eta = estimate_remaining_time(current, total, start_time)

        # Should be infinity when no progress made
        assert eta == float('inf')

    def test_format_duration_seconds(self):
        """Test format_duration with seconds."""
        result = format_duration(30.5)
        assert result == "30.5s"

    def test_format_duration_minutes(self):
        """Test format_duration with minutes."""
        result = format_duration(90)
        assert result == "1.5m"

    def test_format_duration_hours(self):
        """Test format_duration with hours."""
        result = format_duration(7200)
        assert result == "2.0h"

    def test_print_operation_status(self):
        """Test print_operation_status function."""
        with patch('builtins.print') as mock_print:
            print_operation_status("Test Operation", "success", "Details")

        mock_print.assert_called_once_with("✅ Test Operation: Details")

    def test_print_operation_status_no_details(self):
        """Test print_operation_status without details."""
        with patch('builtins.print') as mock_print:
            print_operation_status("Test Operation", "success")

        mock_print.assert_called_once_with("✅ Test Operation")

    def test_print_operation_status_unknown_status(self):
        """Test print_operation_status with unknown status."""
        with patch('builtins.print') as mock_print:
            print_operation_status("Test Operation", "unknown")

        mock_print.assert_called_once_with("• Test Operation")


class TestProgressCallbacks:
    """Test progress callback functions."""

    def test_create_progress_callback(self):
        """Test create_progress_callback function."""
        callback = create_progress_callback(100, "Test Progress")

        with patch('builtins.print') as mock_print:
            callback(50)

        # Should have printed progress
        assert mock_print.call_count > 0

    def test_create_download_progress(self):
        """Test create_download_progress function."""
        callback = create_download_progress(1024, "Download")

        with patch('builtins.print') as mock_print:
            callback(512)

        # Should have printed download progress
        assert mock_print.call_count > 0

    def test_create_processing_progress(self):
        """Test create_processing_progress function."""
        callback = create_processing_progress(100, "Processing")

        with patch('builtins.print') as mock_print:
            callback(50, "item1")

        # Should have printed processing progress
        assert mock_print.call_count > 0

    def test_create_batch_progress(self):
        """Test create_batch_progress function."""
        callback = create_batch_progress(10, "Batches")

        with patch('builtins.print') as mock_print:
            callback(5, 32, 160)

        # Should have printed batch progress
        assert mock_print.call_count > 0

    def test_create_validation_progress(self):
        """Test create_validation_progress function."""
        callback = create_validation_progress(100, "Validation")

        with patch('builtins.print') as mock_print:
            callback(50, "item1", True)
            callback(51, "item2", False)

        # Should have printed validation progress
        assert mock_print.call_count > 0


class TestTaskTracker:
    """Test TaskTracker class."""

    def test_task_tracker_initialization(self):
        """Test TaskTracker initialization."""
        tracker = TaskTracker(10, "Test Tasks")

        assert tracker.total_tasks == 10
        assert tracker.completed_tasks == 0
        assert tracker.failed_tasks == 0
        assert tracker.description == "Test Tasks"
        assert tracker.task_results == {}

    def test_task_tracker_complete_task_success(self):
        """Test TaskTracker complete_task with success."""
        tracker = TaskTracker(10, "Test Tasks")

        with patch('builtins.print') as mock_print:
            tracker.complete_task("task1", success=True, result="result1")

        assert tracker.completed_tasks == 1
        assert tracker.failed_tasks == 0
        assert "task1" in tracker.task_results
        assert tracker.task_results["task1"]["success"] is True
        assert tracker.task_results["task1"]["result"] == "result1"

        # Should have printed status
        assert mock_print.call_count > 0

    def test_task_tracker_complete_task_failure(self):
        """Test TaskTracker complete_task with failure."""
        tracker = TaskTracker(10, "Test Tasks")

        with patch('builtins.print') as mock_print:
            tracker.complete_task("task1", success=False, result="error1")

        assert tracker.completed_tasks == 0
        assert tracker.failed_tasks == 1
        assert "task1" in tracker.task_results
        assert tracker.task_results["task1"]["success"] is False
        assert tracker.task_results["task1"]["result"] == "error1"

        # Should have printed status
        assert mock_print.call_count > 0

    def test_task_tracker_get_summary(self):
        """Test TaskTracker get_summary method."""
        tracker = TaskTracker(10, "Test Tasks")

        tracker.complete_task("task1", success=True)
        tracker.complete_task("task2", success=False)

        summary = tracker.get_summary()

        assert summary["total_tasks"] == 10
        assert summary["completed_tasks"] == 1
        assert summary["failed_tasks"] == 1
        assert summary["success_rate"] == 0.5
        assert "elapsed_time" in summary
        assert "task_results" in summary

    def test_task_tracker_thread_safety(self):
        """Test TaskTracker thread safety."""
        tracker = TaskTracker(100, "Test Tasks")

        def complete_tasks():
            for i in range(50):
                tracker.complete_task(f"task{i}", success=True)

        threads = []
        for _ in range(2):
            thread = threading.Thread(target=complete_tasks)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have completed all tasks correctly
        assert tracker.completed_tasks == 100
        assert tracker.failed_tasks == 0


class TestTaskContext:
    """Test TaskContext class."""

    def test_task_context_success(self):
        """Test TaskContext with successful operation."""
        tracker = TaskTracker(10, "Test Tasks")

        with patch('builtins.print'):
            with tracker.start_task("task1") as context:
                assert isinstance(context, TaskContext)
                assert context.task_name == "task1"

        # Should have completed the task successfully
        assert tracker.completed_tasks == 1
        assert tracker.failed_tasks == 0
        assert "task1" in tracker.task_results
        assert tracker.task_results["task1"]["success"] is True

    def test_task_context_failure(self):
        """Test TaskContext with failed operation."""
        tracker = TaskTracker(10, "Test Tasks")

        with patch('builtins.print'):
            with pytest.raises(ValueError):
                with tracker.start_task("task1"):
                    raise ValueError("Test error")

        # Should have marked the task as failed
        assert tracker.completed_tasks == 0
        assert tracker.failed_tasks == 1
        assert "task1" in tracker.task_results
        assert tracker.task_results["task1"]["success"] is False
        assert "Test error" in str(tracker.task_results["task1"]["result"])


class TestTrackTasks:
    """Test track_tasks context manager."""

    def test_track_tasks_success(self):
        """Test track_tasks with successful operations."""
        with patch('builtins.print') as mock_print:
            with track_tasks(3, "Test Tasks") as tracker:
                tracker.complete_task("task1", success=True)
                tracker.complete_task("task2", success=True)
                tracker.complete_task("task3", success=True)

        # Should have printed completion message
        calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("completed: 3/3 successful" in call for call in calls)

    def test_track_tasks_with_failures(self):
        """Test track_tasks with some failures."""
        with patch('builtins.print') as mock_print:
            with track_tasks(3, "Test Tasks") as tracker:
                tracker.complete_task("task1", success=True)
                tracker.complete_task("task2", success=False)
                tracker.complete_task("task3", success=True)

        # Should have printed completion message
        calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("completed: 2/3 successful" in call for call in calls)


class TestTimeEstimation:
    """Test time estimation functions."""

    def test_estimate_time_remaining(self):
        """Test estimate_time_remaining function."""
        start_time = time.time() - 10  # 10 seconds ago
        completed = 50
        total = 100

        eta = estimate_time_remaining(completed, total, start_time)

        # Should be approximately 10 seconds (half done, 10 seconds elapsed)
        assert 8 <= eta <= 12

    def test_estimate_time_remaining_zero_completed(self):
        """Test estimate_time_remaining with zero completed."""
        start_time = time.time()
        completed = 0
        total = 100

        eta = estimate_time_remaining(completed, total, start_time)

        # Should be infinity when no progress made
        assert eta == float('inf')

    def test_format_time_remaining_seconds(self):
        """Test format_time_remaining with seconds."""
        result = format_time_remaining(30)
        assert result == "30s"

    def test_format_time_remaining_minutes(self):
        """Test format_time_remaining with minutes."""
        result = format_time_remaining(90)
        assert result == "1.5m"

    def test_format_time_remaining_hours(self):
        """Test format_time_remaining with hours."""
        result = format_time_remaining(7200)
        assert result == "2.0h"

    def test_format_time_remaining_infinity(self):
        """Test format_time_remaining with infinity."""
        result = format_time_remaining(float('inf'))
        assert result == "Unknown"


class TestProgressIntegration:
    """Test progress integration scenarios."""

    def test_progress_bar_with_eta_calculation(self):
        """Test progress bar with ETA calculation."""
        bar = ProgressBar(100, "Test Progress")

        # Simulate some time passing
        time.sleep(0.1)

        with patch('builtins.print') as mock_print:
            bar.update(50)

        # Should have printed ETA
        calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("ETA:" in call for call in calls)

    def test_spinner_with_long_operation(self):
        """Test spinner with long operation."""
        spinner_obj = Spinner("Long Operation")

        with patch('builtins.print') as mock_print:
            spinner_obj.start()
            time.sleep(0.2)  # Let it spin
            spinner_obj.stop("Done")

        # Should have printed multiple spinner characters
        assert mock_print.call_count > 1

    def test_task_tracker_with_mixed_results(self):
        """Test task tracker with mixed success/failure results."""
        tracker = TaskTracker(5, "Mixed Tasks")

        with patch('builtins.print') as mock_print:
            tracker.complete_task("task1", success=True)
            tracker.complete_task("task2", success=False)
            tracker.complete_task("task3", success=True)
            tracker.complete_task("task4", success=False)
            tracker.complete_task("task5", success=True)

        summary = tracker.get_summary()
        assert summary["completed_tasks"] == 3
        assert summary["failed_tasks"] == 2
        assert summary["success_rate"] == 0.6

        # Should have printed status updates
        assert mock_print.call_count > 0

    def test_progress_bar_percentage_vs_count(self):
        """Test progress bar with percentage vs count display."""
        bar1 = ProgressBar(100, "Percentage", show_percentage=True)
        bar2 = ProgressBar(100, "Count", show_percentage=False)

        with patch('builtins.print') as mock_print:
            bar1.update(50)
            bar2.update(50)

        # Should have printed different formats
        calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("%" in call for call in calls)
        assert any("/" in call for call in calls)


if __name__ == "__main__":
    pytest.main([__file__])
