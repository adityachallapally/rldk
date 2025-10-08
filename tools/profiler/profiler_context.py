"""
Profiler context manager for stage-level profiling.

This module provides context managers for profiling specific stages
of the training process.
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


class ProfilerContext:
    """Context manager for profiling training stages."""

    def __init__(self, output_dir: str, model_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize profiler context.

        Args:
            output_dir: Directory to save stage profiling data
            model_name: Name of the model being profiled (optional)
            config: Configuration dictionary (optional)
        """
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.config = config
        self.stages: Dict[str, List[float]] = {}
        self.current_stage: Optional[str] = None
        self.start_time: Optional[float] = None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def stage(self, stage_name: str):
        """
        Context manager for profiling a training stage.

        Args:
            stage_name: Name of the stage to profile
        """
        self.start_stage(stage_name)
        try:
            # Add custom event to PyTorch profiler if available
            if hasattr(torch.profiler, 'record_function'):
                with torch.profiler.record_function(f"stage_{stage_name}"):
                    yield
            else:
                yield
        finally:
            self.end_stage()

    def start_stage(self, stage_name: str):
        """Start profiling a stage."""
        if self.current_stage is not None:
            self.end_stage()

        self.current_stage = stage_name
        self.start_time = time.time()

    def end_stage(self):
        """End profiling the current stage."""
        if self.current_stage is not None and self.start_time is not None:
            duration = time.time() - self.start_time
            if self.current_stage not in self.stages:
                self.stages[self.current_stage] = []
            self.stages[self.current_stage].append(duration)

        self.current_stage = None
        self.start_time = None

    def save_stage_times(self):
        """Save stage timing data to JSON."""
        stage_times_path = self.output_dir / "stage_times.json"

        # Derive total_steps from the recorded stage timing data
        # Use the maximum number of recorded timings across all stages
        total_steps = 0
        if self.stages:
            # Find the stage with the most recorded timings
            max_timings = max(len(timings) for timings in self.stages.values())
            total_steps = max_timings

        stage_data = {
            "stage_times": self.stages,
            "average_times": self.get_average_times(),
            "total_stages": len(self.stages),
            "total_steps": total_steps,
            "timestamp": time.time()
        }

        if self.model_name is not None:
            stage_data["model_name"] = self.model_name

        if self.config is not None:
            stage_data["config"] = self.config

        with open(stage_times_path, 'w') as f:
            json.dump(stage_data, f, indent=2)

    def get_average_times(self) -> Dict[str, float]:
        """Get average times for each stage."""
        return {
            stage: sum(times) / len(times) if times else 0.0
            for stage, times in self.stages.items()
        }

    def get_stage_times(self) -> Dict[str, List[float]]:
        """Get all stage times."""
        return self.stages.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get profiler context summary."""
        # Derive total_steps from the recorded stage timing data
        # Use the maximum number of recorded timings across all stages
        total_steps = 0
        if self.stages:
            # Find the stage with the most recorded timings
            max_timings = max(len(timings) for timings in self.stages.values())
            total_steps = max_timings

        summary = {
            "output_dir": str(self.output_dir),
            "stages": list(self.stages.keys()),
            "average_times": self.get_average_times(),
            "total_stages": len(self.stages),
            "total_steps": total_steps
        }

        if self.model_name is not None:
            summary["model_name"] = self.model_name

        if self.config is not None:
            summary["config"] = self.config

        return summary
