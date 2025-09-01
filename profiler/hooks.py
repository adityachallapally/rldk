"""
Profiler hooks and registry system.

This module provides a hook system for integrating profiler functionality
into training loops and other components.
"""

import time
from typing import Dict, List, Callable, Any, Optional
from functools import wraps


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
            "after_optimizer_step": []
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
    
    def end_step(self):
        """End profiling a training step."""
        if hasattr(self, 'step_start_time'):
            step_duration = time.time() - self.step_start_time
            self.step_times.append(step_duration)
            self.hooks.call_hooks("after_step", self.step_count, step_duration)
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