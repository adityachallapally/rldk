"""
PyTorch profiler wrapper for RLHF training.

This module provides a high-level interface to PyTorch's profiler
with additional functionality for training-specific profiling.
"""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.profiler


class TorchProfiler:
    """Wrapper around PyTorch profiler with training-specific features."""

    def __init__(
        self,
        output_dir: str,
        warmup_steps: int = 1,
        active_steps: int = 3,
        repeat: int = 1,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = True
    ):
        """
        Initialize the PyTorch profiler.

        Args:
            output_dir: Directory to save profiler outputs
            warmup_steps: Number of warmup steps before profiling
            active_steps: Number of active profiling steps
            repeat: Number of profiling cycles
            record_shapes: Whether to record tensor shapes
            profile_memory: Whether to profile memory usage
            with_stack: Whether to record stack traces
        """
        self.output_dir = Path(output_dir)
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.repeat = repeat
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack

        self.profiler: Optional[torch.profiler.profile] = None
        self.step_count = 0
        self.is_active = False

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        """Start profiling."""
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        try:
            # Check if profiler is already running to avoid Kineto warnings
            if hasattr(torch.profiler, '_current_profiler') and torch.profiler._current_profiler is not None:
                print("Warning: Profiler already running, stopping existing profiler first")
                try:
                    torch.profiler._current_profiler.__exit__(None, None, None)
                except Exception as e:
                    print(f"Warning: Error stopping existing profiler: {e}")
                    pass

            self.profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    wait=0,
                    warmup=self.warmup_steps,
                    active=self.active_steps,
                    repeat=self.repeat
                ),
                on_trace_ready=self._on_trace_ready,
                record_shapes=self.record_shapes,
                profile_memory=self.profile_memory,
                with_stack=self.with_stack
            )

            # Start the profiler context
            self.profiler.__enter__()
            self.is_active = True
            self.step_count = 0
            print(f"TorchProfiler started successfully - is_active: {self.is_active}")
        except Exception as e:
            print(f"Warning: Failed to start profiler: {e}")
            self.profiler = None
            self.is_active = False

    def step(self):
        """Step the profiler (call at each training step)."""
        if self.profiler is not None:
            try:
                self.profiler.step()
                self.step_count += 1
            except Exception as e:
                print(f"Warning: Error stepping profiler: {e}")
                # Disable profiler on error
                self.profiler = None
                self.is_active = False

    def stop(self):
        """Stop profiling and save results."""
        if self.profiler is not None:
            try:
                # Only stop if profiler is actually active
                if self.is_active:
                    self.profiler.__exit__(None, None, None)

                    # If trace_ready callback wasn't triggered, manually save memory stats
                    if not (self.output_dir / "memory_stats.json").exists():
                        self._save_basic_memory_stats()
            except Exception as e:
                print(f"Warning: Error stopping profiler: {e}")
            finally:
                self.profiler = None
                self.is_active = False

    def _on_trace_ready(self, prof):
        """Callback when trace is ready."""
        try:
            # Save Chrome trace
            trace_path = self.output_dir / "trace.json"
            prof.export_chrome_trace(str(trace_path))

            # Save operation statistics
            self._save_operation_stats(prof)

            # Save memory usage
            self._save_memory_stats(prof)
        except Exception as e:
            print(f"Warning: Error saving profiler trace data: {e}")

    def _save_operation_stats(self, prof):
        """Save operation statistics to CSV."""
        op_stats_path = self.output_dir / "op_stats.csv"

        try:
            with open(op_stats_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'Name', 'Self CPU Time (μs)', 'CPU Time (μs)',
                    'Self CUDA Time (μs)', 'CUDA Time (μs)', 'Count',
                    'Self CPU Memory (bytes)', 'CPU Memory (bytes)',
                    'Self CUDA Memory (bytes)', 'CUDA Memory (bytes)'
                ])

                for event in prof.events():
                    try:
                        # Use new device attributes if available, otherwise fall back to cuda attributes
                        self_device_time = getattr(event, 'self_device_time_total', 0)
                        device_time = getattr(event, 'device_time_total', 0)
                        self_device_memory = getattr(event, 'self_device_memory_usage', 0)
                        device_memory = getattr(event, 'device_memory_usage', 0)

                        writer.writerow([
                            event.name,
                            getattr(event, 'self_cpu_time_total', 0),
                            getattr(event, 'cpu_time_total', 0),
                            self_device_time,
                            device_time,
                            getattr(event, 'count', 0),
                            getattr(event, 'self_cpu_memory_usage', 0),
                            getattr(event, 'cpu_memory_usage', 0),
                            self_device_memory,
                            device_memory
                        ])
                    except Exception as e:
                        print(f"Warning: Error processing profiler event: {e}")
                        continue
        except Exception as e:
            print(f"Warning: Error saving operation statistics: {e}")

    def _save_memory_stats(self, prof):
        """Save memory usage statistics."""
        memory_stats_path = self.output_dir / "memory_stats.json"

        try:
            # Get memory usage from profiler events, using current PyTorch API
            peak_cpu_memory = 0
            peak_cuda_memory = 0

            for event in prof.events():
                try:
                    # Use device_memory_usage instead of cuda_memory_usage for current PyTorch versions
                    if hasattr(event, 'device_memory_usage'):
                        peak_cuda_memory = max(peak_cuda_memory, event.device_memory_usage)
                    elif hasattr(event, 'cuda_memory_usage'):
                        peak_cuda_memory = max(peak_cuda_memory, event.cuda_memory_usage)

                    if hasattr(event, 'cpu_memory_usage'):
                        peak_cpu_memory = max(peak_cpu_memory, event.cpu_memory_usage)
                except Exception as e:
                    print(f"Warning: Error processing memory event: {e}")
                    continue

            memory_data = {
                "step_count": self.step_count,
                "peak_memory_usage": {
                    "cpu": peak_cpu_memory,
                    "cuda": peak_cuda_memory
                },
                "timestamp": time.time()
            }

            with open(memory_stats_path, 'w') as f:
                json.dump(memory_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Error saving memory statistics: {e}")

    def _save_basic_memory_stats(self):
        """Save basic memory statistics when trace_ready callback wasn't triggered."""
        memory_stats_path = self.output_dir / "memory_stats.json"

        try:
            memory_data = {
                "step_count": self.step_count,
                "peak_memory_usage": {
                    "cpu": 0,  # Will be updated if we can get actual values
                    "cuda": 0
                },
                "timestamp": time.time(),
                "note": "Basic memory stats - trace_ready callback not triggered"
            }

            with open(memory_stats_path, 'w') as f:
                json.dump(memory_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Error saving basic memory statistics: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get profiler summary."""
        return {
            "is_active": self.is_active,
            "step_count": self.step_count,
            "output_dir": str(self.output_dir),
            "artifacts": {
                "trace_json": (self.output_dir / "trace.json").exists(),
                "op_stats_csv": (self.output_dir / "op_stats.csv").exists(),
                "memory_stats_json": (self.output_dir / "memory_stats.json").exists()
            }
        }
