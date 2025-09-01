"""
Profiler context manager for stage-level profiling.

This module provides context managers for profiling specific stages
of the training process.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager


class ProfilerContext:
    """Context manager for profiling training stages."""
    
    def __init__(self, output_dir: str):
        """
        Initialize profiler context.
        
        Args:
            output_dir: Directory to save stage profiling data
        """
        self.output_dir = Path(output_dir)
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
        
        stage_data = {
            "stage_times": self.stages,
            "average_times": self.get_average_times(),
            "total_stages": len(self.stages),
            "timestamp": time.time()
        }
        
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
        return {
            "output_dir": str(self.output_dir),
            "stages": list(self.stages.keys()),
            "average_times": self.get_average_times(),
            "total_stages": len(self.stages)
        }