"""Progress indication utilities for RLDK."""

import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional


class ProgressBar:
    """Simple progress bar for terminal output."""

    def __init__(self, total: int, description: str = "Progress",
                 width: int = 50, show_percentage: bool = True):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.show_percentage = show_percentage
        self.start_time = time.time()
        self._lock = threading.Lock()

    def update(self, increment: int = 1, description: Optional[str] = None):
        """Update progress by increment."""
        with self._lock:
            self.current = min(self.current + increment, self.total)
            if description:
                self.description = description
            self._display()

    def set_progress(self, current: int, description: Optional[str] = None):
        """Set absolute progress."""
        with self._lock:
            self.current = min(current, self.total)
            if description:
                self.description = description
            self._display()

    def _display(self):
        """Display the progress bar."""
        if self.total == 0:
            return

        # Calculate progress
        progress = self.current / self.total
        filled_width = int(self.width * progress)
        bar = "█" * filled_width + "░" * (self.width - filled_width)

        # Calculate ETA
        elapsed = time.time() - self.start_time
        if progress > 0:
            eta = elapsed * (1 - progress) / progress
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: --"

        # Format percentage
        if self.show_percentage:
            pct_str = f"{progress * 100:.1f}%"
        else:
            pct_str = f"{self.current}/{self.total}"

        # Print progress bar
        print(f"\r{self.description}: |{bar}| {pct_str} {eta_str}", end="", flush=True)

        if self.current >= self.total:
            print()  # New line when complete

    def finish(self, description: Optional[str] = None):
        """Finish the progress bar."""
        with self._lock:
            self.current = self.total
            if description:
                self.description = description
            self._display()


class Spinner:
    """Simple spinner for indeterminate progress."""

    def __init__(self, description: str = "Processing"):
        self.description = description
        self.spinner_chars = "|/-\\"
        self.current_char = 0
        self.running = False
        self.thread = None
        self._lock = threading.Lock()

    def start(self):
        """Start the spinner."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()

    def stop(self, message: Optional[str] = None):
        """Stop the spinner."""
        if not self.running:
            return

        self.running = False
        if self.thread:
            self.thread.join()

        if message:
            print(f"\r{self.description}: {message}")
        else:
            print(f"\r{self.description}: Complete")

    def _spin(self):
        """Spin the spinner."""
        while self.running:
            with self._lock:
                char = self.spinner_chars[self.current_char]
                print(f"\r{self.description}: {char}", end="", flush=True)
                self.current_char = (self.current_char + 1) % len(self.spinner_chars)
            time.sleep(0.1)


@contextmanager
def progress_bar(total: int, description: str = "Progress",
                width: int = 50, show_percentage: bool = True):
    """Context manager for progress bar."""
    bar = ProgressBar(total, description, width, show_percentage)
    try:
        yield bar
    finally:
        bar.finish()


@contextmanager
def spinner(description: str = "Processing"):
    """Context manager for spinner."""
    spinner_obj = Spinner(description)
    try:
        spinner_obj.start()
        yield spinner_obj
    finally:
        spinner_obj.stop()


def track_progress(iterable: Iterator[Any], total: Optional[int] = None,
                  description: str = "Processing") -> Iterator[Any]:
    """Track progress of an iterator."""
    if total is None:
        # Try to get length if possible
        try:
            total = len(iterable)
        except TypeError:
            total = None

    if total is not None:
        with progress_bar(total, description) as bar:
            for i, item in enumerate(iterable):
                bar.update(1)
                yield item
    else:
        with spinner(description):
            for item in iterable:
                yield item


def timed_operation(operation_name: str):
    """Decorator to time operations and show progress."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            print(f"🔄 Starting {operation_name}...")

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                print(f"✅ {operation_name} completed in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"❌ {operation_name} failed after {elapsed:.2f}s: {e}")
                raise

        return wrapper
    return decorator


@contextmanager
def timed_operation_context(operation_name: str):
    """Context manager to time operations and show progress."""
    start_time = time.time()
    print(f"🔄 Starting {operation_name}...")

    try:
        yield
        elapsed = time.time() - start_time
        print(f"✅ {operation_name} completed in {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {operation_name} failed after {elapsed:.2f}s: {e}")
        raise


def estimate_remaining_time(current: int, total: int, start_time: float) -> float:
    """Estimate remaining time based on current progress."""
    if current == 0:
        return float('inf')

    elapsed = time.time() - start_time
    progress = current / total
    return elapsed * (1 - progress) / progress


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_operation_status(operation: str, status: str, details: Optional[str] = None):
    """Print operation status with appropriate emoji."""
    status_icons = {
        "start": "🔄",
        "progress": "⏳",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "info": "ℹ️"
    }

    icon = status_icons.get(status, "•")
    message = f"{icon} {operation}"

    if details:
        message += f": {details}"

    print(message)


def create_progress_callback(total: int, description: str = "Progress") -> Callable[[int], None]:
    """Create a progress callback function."""
    bar = ProgressBar(total, description)

    def callback(current: int):
        bar.set_progress(current)

    return callback


class TaskTracker:
    """Track multiple tasks with progress indication."""

    def __init__(self, total_tasks: int, description: str = "Tasks"):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.description = description
        self.start_time = time.time()
        self.task_results = {}
        self._lock = threading.Lock()

    def start_task(self, task_name: str) -> 'TaskContext':
        """Start tracking a specific task."""
        return TaskContext(self, task_name)

    def complete_task(self, task_name: str, success: bool = True, result: Any = None):
        """Mark a task as completed."""
        with self._lock:
            if success:
                self.completed_tasks += 1
            else:
                self.failed_tasks += 1

            self.task_results[task_name] = {
                "success": success,
                "result": result,
                "completed_at": time.time()
            }

            self._display_status()

    def _display_status(self):
        """Display current status."""
        total_completed = self.completed_tasks + self.failed_tasks
        elapsed = time.time() - self.start_time

        if total_completed > 0:
            avg_time = elapsed / total_completed
            remaining = (self.total_tasks - total_completed) * avg_time
            eta_str = f"ETA: {remaining:.1f}s"
        else:
            eta_str = "ETA: --"

        status = f"{self.description}: {total_completed}/{self.total_tasks} completed"
        if self.failed_tasks > 0:
            status += f" ({self.failed_tasks} failed)"

        print(f"\r{status} - {eta_str}", end="", flush=True)

        if total_completed >= self.total_tasks:
            print("")  # New line when complete

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of task completion."""
        total_completed = self.completed_tasks + self.failed_tasks
        elapsed = time.time() - self.start_time

        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.completed_tasks / total_completed if total_completed > 0 else 0,
            "elapsed_time": elapsed,
            "task_results": self.task_results
        }


class TaskContext:
    """Context manager for individual task tracking."""

    def __init__(self, tracker: TaskTracker, task_name: str):
        self.tracker = tracker
        self.task_name = task_name
        self.start_time = time.time()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        result = exc_val if not success else None
        self.tracker.complete_task(self.task_name, success, result)


@contextmanager
def track_tasks(total_tasks: int, description: str = "Tasks"):
    """Context manager for tracking multiple tasks."""
    tracker = TaskTracker(total_tasks, description)
    try:
        yield tracker
    finally:
        summary = tracker.get_summary()
        print(f"\n✅ {description} completed: {summary['completed_tasks']}/{summary['total_tasks']} successful")


def estimate_time_remaining(completed: int, total: int, start_time: float) -> float:
    """Estimate remaining time based on current progress."""
    if completed == 0:
        return float('inf')

    elapsed = time.time() - start_time
    progress = completed / total
    return elapsed * (1 - progress) / progress


def format_time_remaining(seconds: float) -> str:
    """Format remaining time in human-readable format."""
    if seconds == float('inf'):
        return "Unknown"
    elif seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_download_progress(total_bytes: int, description: str = "Download") -> Callable[[int], None]:
    """Create a progress callback for downloads."""
    def format_bytes(bytes_val: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f}{unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f}TB"

    start_time = time.time()
    downloaded = 0

    def callback(bytes_downloaded: int):
        nonlocal downloaded
        downloaded = bytes_downloaded

        elapsed = time.time() - start_time
        if downloaded > 0 and elapsed > 0:
            rate = downloaded / elapsed
            eta = (total_bytes - downloaded) / rate if rate > 0 else 0
            eta_str = f"ETA: {format_time_remaining(eta)}"
        else:
            eta_str = "ETA: --"

        progress = downloaded / total_bytes
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)

        print(f"\r{description}: |{bar}| {format_bytes(downloaded)}/{format_bytes(total_bytes)} {eta_str}",
              end="", flush=True)

    return callback


def create_processing_progress(total_items: int, description: str = "Processing") -> Callable[[int, str], None]:
    """Create a progress callback for processing items."""
    bar = ProgressBar(total_items, description)

    def callback(processed: int, current_item: str = ""):
        bar.set_progress(processed, f"{description} {current_item}")

    return callback


def create_batch_progress(total_batches: int, description: str = "Batches") -> Callable[[int, int, int], None]:
    """Create a progress callback for batch processing."""
    bar = ProgressBar(total_batches, description)

    def callback(batch_num: int, batch_size: int, total_processed: int):
        bar.set_progress(batch_num, f"{description} {batch_num}/{total_batches} (size: {batch_size})")

    return callback


def create_validation_progress(total_items: int, description: str = "Validation") -> Callable[[int, str, bool], None]:
    """Create a progress callback for validation."""
    bar = ProgressBar(total_items, description)
    validated = 0
    failed = 0

    def callback(processed: int, current_item: str, success: bool):
        nonlocal validated, failed
        if success:
            validated += 1
        else:
            failed += 1

        status = f"{description} {current_item} (✓{validated} ✗{failed})"
        bar.set_progress(processed, status)

    return callback
