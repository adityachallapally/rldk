"""
Profiler report generation utilities.

This module provides functionality to generate comprehensive profiler reports
from collected profiling data.
"""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class ProfilerReport:
    """Generate profiler reports from collected data."""

    def __init__(self, profiler_dir: str):
        """
        Initialize profiler report generator.

        Args:
            profiler_dir: Directory containing profiler artifacts
        """
        self.profiler_dir = Path(profiler_dir)

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive profiler report."""
        report = {
            "timestamp": time.time(),
            "profiler_dir": str(self.profiler_dir),
            "artifacts": self._check_artifacts(),
            "stage_times": self._load_stage_times(),
            "operation_stats": self._load_operation_stats(),
            "memory_stats": self._load_memory_stats(),
            "summary": {}
        }

        # Generate summary statistics
        report["summary"] = self._generate_summary(report)

        return report

    def _check_artifacts(self) -> Dict[str, bool]:
        """Check which profiler artifacts exist."""
        artifacts = {
            "trace_json": (self.profiler_dir / "trace.json").exists(),
            "op_stats_csv": (self.profiler_dir / "op_stats.csv").exists(),
            "stage_times_json": (self.profiler_dir / "stage_times.json").exists(),
            "memory_stats_json": (self.profiler_dir / "memory_stats.json").exists()
        }
        return artifacts

    def _load_stage_times(self) -> Optional[Dict[str, Any]]:
        """Load stage timing data."""
        stage_times_path = self.profiler_dir / "stage_times.json"
        if stage_times_path.exists():
            with open(stage_times_path) as f:
                return json.load(f)
        return None

    def _load_operation_stats(self) -> Optional[List[Dict[str, Any]]]:
        """Load operation statistics."""
        op_stats_path = self.profiler_dir / "op_stats.csv"
        if op_stats_path.exists():
            stats = []
            with open(op_stats_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stats.append(row)
            return stats
        return None

    def _load_memory_stats(self) -> Optional[Dict[str, Any]]:
        """Load memory statistics."""
        memory_stats_path = self.profiler_dir / "memory_stats.json"
        if memory_stats_path.exists():
            with open(memory_stats_path) as f:
                return json.load(f)
        return None

    def _generate_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from the report."""
        summary = {
            "total_artifacts": sum(report["artifacts"].values()),
            "profiling_successful": report["artifacts"]["trace_json"]
        }

        # Stage timing summary
        if report["stage_times"]:
            stage_data = report["stage_times"]
            summary["stage_summary"] = {
                "total_stages": len(stage_data.get("stage_times", {})),
                "average_times": stage_data.get("average_times", {}),
                "total_steps": stage_data.get("total_steps", 0)
            }

        # Operation statistics summary
        if report["operation_stats"]:
            ops = report["operation_stats"]
            summary["operation_summary"] = {
                "total_operations": len(ops),
                "top_operations": self._get_top_operations(ops, 5)
            }

        # Memory summary
        if report["memory_stats"]:
            memory_data = report["memory_stats"]
            summary["memory_summary"] = {
                "peak_cpu_memory": memory_data.get("peak_memory_usage", {}).get("cpu", 0),
                "peak_cuda_memory": memory_data.get("peak_memory_usage", {}).get("cuda", 0),
                "step_count": memory_data.get("step_count", 0)
            }

        return summary

    def _get_top_operations(self, ops: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
        """Get top N operations by CPU time."""
        try:
            # Sort by CPU time (convert to float, handle missing values)
            sorted_ops = sorted(
                ops,
                key=lambda x: float(x.get("CPU Time (μs)", 0)),
                reverse=True
            )
            return sorted_ops[:n]
        except (ValueError, TypeError):
            return []

    def save_report(self, output_path: Optional[str] = None) -> str:
        """Save profiler report to JSON file."""
        if output_path is None:
            output_path = self.profiler_dir / "profiler_report.json"
        else:
            output_path = Path(output_path)

        report = self.generate_report()

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        return str(output_path)

    def print_summary(self):
        """Print a human-readable summary of the profiler report."""
        report = self.generate_report()
        summary = report["summary"]

        print("=" * 50)
        print("PROFILER REPORT SUMMARY")
        print("=" * 50)

        print(f"Profiler Directory: {report['profiler_dir']}")
        print(f"Total Artifacts: {summary['total_artifacts']}/4")
        print(f"Profiling Successful: {summary['profiling_successful']}")

        if "stage_summary" in summary:
            stage_summary = summary["stage_summary"]
            print("\nStage Timing:")
            print(f"  Total Stages: {stage_summary['total_stages']}")
            print(f"  Total Steps: {stage_summary['total_steps']}")
            if stage_summary["average_times"]:
                print("  Average Times:")
                for stage, time_val in stage_summary["average_times"].items():
                    print(f"    {stage}: {time_val:.4f}s")

        if "operation_summary" in summary:
            op_summary = summary["operation_summary"]
            print("\nOperation Statistics:")
            print(f"  Total Operations: {op_summary['total_operations']}")
            if op_summary["top_operations"]:
                print("  Top Operations by CPU Time:")
                for i, op in enumerate(op_summary["top_operations"], 1):
                    name = op.get("Name", "Unknown")
                    cpu_time = op.get("CPU Time (μs)", "0")
                    print(f"    {i}. {name}: {cpu_time}μs")

        if "memory_summary" in summary:
            mem_summary = summary["memory_summary"]
            print("\nMemory Usage:")
            print(f"  Peak CPU Memory: {mem_summary['peak_cpu_memory']} bytes")
            print(f"  Peak CUDA Memory: {mem_summary['peak_cuda_memory']} bytes")
            print(f"  Steps Profiled: {mem_summary['step_count']}")

        print("=" * 50)
