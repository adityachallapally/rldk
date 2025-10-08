"""
Profiler hooks and registry system.

This module provides a hook system for integrating profiler functionality
into training loops and other components.
"""

import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional


class ProfilerHooks:
    """Registry for profiler hooks."""

    def __init__(self):
        self.hooks: Dict[str, List[Callable]] = {
            "before_step": [],
            "after_step": [],
            "before_forward": [],
            "after_forward": [],
            "before_backward": [],
            "after_backward": [],
            "before_optimizer_step": [],
            "after_optimizer_step": [],
            "anomaly_detection": [],
            "function_timing": []
        }

    def register_hook(self, event: str, hook: Callable):
        """
        Register a hook for a specific event.

        Args:
            event: Event name to hook into
            hook: Function to call when event occurs
        """
        if event not in self.hooks:
            self.hooks[event] = []
        self.hooks[event].append(hook)

    def call_hooks(self, event: str, *args, **kwargs):
        """Call all hooks registered for an event."""
        if event in self.hooks:
            for hook in self.hooks[event]:
                try:
                    hook(*args, **kwargs)
                except Exception as e:
                    print(f"Warning: Hook {hook.__name__} failed with error: {e}")

    def clear_hooks(self, event: Optional[str] = None):
        """Clear hooks for an event or all events."""
        if event is None:
            for event_name in self.hooks:
                self.hooks[event_name] = []
        elif event in self.hooks:
            self.hooks[event] = []


# Global profiler hooks registry
profiler_registry = ProfilerHooks()


def profiler_hook(event: str):
    """
    Decorator to register a function as a profiler hook.

    Args:
        event: Event name to hook into
    """
    def decorator(func):
        profiler_registry.register_hook(event, func)
        return func
    return decorator


def time_function(func_name: str = None):
    """
    Decorator to time function execution and log to profiler.

    Args:
        func_name: Name to use for timing (defaults to function name)
    """
    def decorator(func):
        name = func_name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                profiler_registry.call_hooks("function_timing", {
                    "function": name,
                    "duration": duration,
                    "timestamp": time.time()
                })

        return wrapper
    return decorator


class StepProfiler:
    """Profiler for individual training steps."""

    def __init__(self, hooks: ProfilerHooks = None):
        self.hooks = hooks or profiler_registry
        self.step_count = 0
        self.step_times: List[float] = []

    def start_step(self):
        """Start profiling a training step."""
        self.step_start_time = time.time()
        self.hooks.call_hooks("before_step", self.step_count)

    def end_step(self, model=None, optimizer=None, loss=None, **kwargs):
        """End profiling a training step.

        Args:
            model: The model being trained (optional)
            optimizer: The optimizer being used (optional)
            loss: The loss value for this step (optional)
            **kwargs: Additional training context parameters
        """
        if hasattr(self, 'step_start_time'):
            step_duration = time.time() - self.step_start_time
            self.step_times.append(step_duration)

            # Pass training context to hooks
            hook_kwargs = {
                'model': model,
                'optimizer': optimizer,
                'loss': loss,
                **kwargs
            }
            self.hooks.call_hooks("after_step", self.step_count, step_duration, **hook_kwargs)
            self.step_count += 1

    def forward_pass(self):
        """Profile forward pass."""
        self.hooks.call_hooks("before_forward", self.step_count)
        # This would be called around the actual forward pass
        self.hooks.call_hooks("after_forward", self.step_count)

    def backward_pass(self):
        """Profile backward pass."""
        self.hooks.call_hooks("before_backward", self.step_count)
        # This would be called around the actual backward pass
        self.hooks.call_hooks("after_backward", self.step_count)

    def optimizer_step(self):
        """Profile optimizer step."""
        self.hooks.call_hooks("before_optimizer_step", self.step_count)
        # This would be called around the actual optimizer step
        self.hooks.call_hooks("after_optimizer_step", self.step_count)

    def get_average_step_time(self) -> float:
        """Get average step time."""
        return sum(self.step_times) / len(self.step_times) if self.step_times else 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get step profiler summary."""
        return {
            "step_count": self.step_count,
            "average_step_time": self.get_average_step_time(),
            "total_time": sum(self.step_times),
            "step_times": self.step_times
        }


class AnomalyDetectionHook:
    """Hook for integrating anomaly detection into training loops."""

    def __init__(self, anomaly_detector=None):
        """
        Initialize anomaly detection hook.

        Args:
            anomaly_detector: AdvancedAnomalyDetector instance
        """
        self.anomaly_detector = anomaly_detector
        self.current_step_data = {}

    def before_step(self, step: int, **kwargs):
        """Called before each training step."""
        self.current_step_data = {
            'step': step,
            'start_time': time.time()
        }

    def after_step(self, step: int, step_duration: float, **kwargs):
        """Called after each training step."""
        if self.anomaly_detector and hasattr(self, 'current_step_data'):
            # Extract training data from kwargs or context
            model = kwargs.get('model')
            optimizer = kwargs.get('optimizer')
            loss = kwargs.get('loss')
            batch_size = kwargs.get('batch_size', 1)
            rewards = kwargs.get('rewards')
            predictions = kwargs.get('predictions')

            if model is not None and optimizer is not None and loss is not None:
                alerts = self.anomaly_detector.analyze_training_step(
                    model=model,
                    optimizer=optimizer,
                    loss=loss,
                    batch_size=batch_size,
                    rewards=rewards,
                    predictions=predictions
                )

                # Call anomaly detection hooks
                if alerts:
                    profiler_registry.call_hooks("anomaly_detection", alerts, step)

    def register_with_profiler(self):
        """Register this hook with the profiler registry."""
        profiler_registry.register_hook("before_step", self.before_step)
        profiler_registry.register_hook("after_step", self.after_step)
