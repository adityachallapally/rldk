"""
Core profiler infrastructure for RLHF training.

This module provides the ProfilerManager class that integrates with PyTorch's
profiler to collect performance metrics during training runs.
"""

import json
import csv
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import torch
import torch.profiler


class StageTimer:
    """Timer for tracking training stage durations."""
    
    def __init__(self):
        self.stages: Dict[str, List[float]] = {}
        self.current_stage: Optional[str] = None
        self.start_time: Optional[float] = None
    
    def start_stage(self, stage_name: str):
        """Start timing a training stage."""
        if self.current_stage is not None:
            self.end_stage()
        
        self.current_stage = stage_name
        self.start_time = time.time()
    
    def end_stage(self):
        """End timing the current stage."""
        if self.current_stage is not None and self.start_time is not None:
            duration = time.time() - self.start_time
            if self.current_stage not in self.stages:
                self.stages[self.current_stage] = []
            self.stages[self.current_stage].append(duration)
        
        self.current_stage = None
        self.start_time = None
    
    def get_stage_times(self) -> Dict[str, List[float]]:
        """Get all recorded stage times."""
        return self.stages.copy()
    
    def get_average_times(self) -> Dict[str, float]:
        """Get average times for each stage."""
        return {
            stage: sum(times) / len(times) if times else 0.0
            for stage, times in self.stages.items()
        }


class ProfilerManager:
    """Manager for PyTorch profiler integration during training."""
    
    def __init__(self, output_dir: str, enabled: bool = True):
        """
        Initialize the profiler manager.
        
        Args:
            output_dir: Directory to save profiler artifacts
            enabled: Whether profiling is enabled
        """
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.stage_timer = StageTimer()
        self.profiler: Optional[torch.profiler.profile] = None
        self.profiler_step = 0
        
        # Create output directory if it doesn't exist
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def start_profiling(self, warmup_steps: int = 5, active_steps: int = 10, repeat: int = 1):
        """Start PyTorch profiling."""
        if not self.enabled:
            return
        
        # Build activities list, filtering out None
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        try:
            self.profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    wait=0,
                    warmup=warmup_steps,
                    active=active_steps,
                    repeat=repeat
                ),
                on_trace_ready=self._on_trace_ready,
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
        except Exception as e:
            print(f"Warning: Failed to start profiler: {e}")
            self.profiler = None
    
    def step_profiler(self):
        """Step the profiler (call this at each training step)."""
        if self.profiler is not None:
            try:
                self.profiler.step()
                self.profiler_step += 1
            except Exception as e:
                print(f"Warning: Error stepping profiler: {e}")
                # Disable profiler on error
                self.profiler = None
    
    def stop_profiling(self):
        """Stop profiling and save results."""
        if self.profiler is not None:
            try:
                self.profiler.stop()
            except Exception as e:
                print(f"Warning: Error stopping profiler: {e}")
            finally:
                self.profiler = None
    
    def _on_trace_ready(self, prof):
        """Callback when trace is ready to be saved."""
        if not self.enabled:
            return
        
        # Save Chrome trace
        trace_path = self.output_dir / "trace.json"
        prof.export_chrome_trace(str(trace_path))
        
        # Save operation statistics
        self._save_op_stats(prof)
    
    def _save_op_stats(self, prof):
        """Save operation statistics to CSV."""
        op_stats_path = self.output_dir / "op_stats.csv"
        
        with open(op_stats_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Name', 'Self CPU Time (μs)', 'CPU Time (μs)', 'Self CUDA Time (μs)', 'CUDA Time (μs)', 'Count'])
            
            for event in prof.events():
                writer.writerow([
                    event.name,
                    event.self_cpu_time_total,
                    event.cpu_time_total,
                    event.self_cuda_time_total,
                    event.cuda_time_total,
                    event.count
                ])
    
    def start_stage(self, stage_name: str):
        """Start timing a training stage."""
        self.stage_timer.start_stage(stage_name)
    
    def end_stage(self):
        """End timing the current stage."""
        self.stage_timer.end_stage()
    
    @contextmanager
    def stage(self, stage_name: str):
        """Context manager for timing a stage."""
        self.start_stage(stage_name)
        try:
            yield
        finally:
            self.end_stage()
    
    def save_stage_times(self):
        """Save stage timing data to JSON."""
        if not self.enabled:
            return
        
        stage_times_path = self.output_dir / "stage_times.json"
        stage_data = {
            "stage_times": self.stage_timer.get_stage_times(),
            "average_times": self.stage_timer.get_average_times(),
            "total_steps": self.profiler_step
        }
        
        with open(stage_times_path, 'w') as f:
            json.dump(stage_data, f, indent=2)
    
    def get_profiler_summary(self) -> Dict[str, Any]:
        """Get a summary of profiler data."""
        if not self.enabled:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "output_dir": str(self.output_dir),
            "stage_times": self.stage_timer.get_average_times(),
            "total_steps": self.profiler_step,
            "artifacts": {
                "trace_json": (self.output_dir / "trace.json").exists(),
                "op_stats_csv": (self.output_dir / "op_stats.csv").exists(),
                "stage_times_json": (self.output_dir / "stage_times.json").exists()
            }
        }
    
    def cleanup(self):
        """Cleanup profiler resources."""
        self.stop_profiling()
        self.stage_timer.end_stage()  # End any active stage
        self.save_stage_times()