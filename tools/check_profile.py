#!/usr/bin/env python3
"""
Profiler validation and checking tool.

This script validates profiler artifacts and provides detailed analysis
of profiler results.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from profiler.report import ProfilerReport


def check_profiler_artifacts(profiler_dir: str) -> dict:
    """
    Check profiler artifacts in a directory.

    Args:
        profiler_dir: Directory containing profiler artifacts

    Returns:
        Dictionary with validation results
    """
    profiler_path = Path(profiler_dir)

    if not profiler_path.exists():
        return {
            "valid": False,
            "error": f"Profiler directory does not exist: {profiler_dir}"
        }

    # Expected artifacts (check multiple locations)
    expected_artifacts = {
        "trace.json": "Chrome trace file",
        "op_stats.csv": "Operation statistics CSV",
        "stage_times.json": "Stage timing data",
        "memory_stats.json": "Memory usage statistics"
    }

    # Also check subdirectories
    subdirs_to_check = ["", "torch_profiler", "profiler_context"]

    results = {
        "valid": True,
        "profiler_dir": str(profiler_path),
        "artifacts": {},
        "errors": [],
        "warnings": []
    }

    # Check each artifact in all possible locations
    for artifact, description in expected_artifacts.items():
        found = False
        artifact_path = None

        # Check main directory and subdirectories
        for subdir in subdirs_to_check:
            if subdir:
                check_path = profiler_path / subdir / artifact
            else:
                check_path = profiler_path / artifact

            if check_path.exists():
                found = True
                artifact_path = check_path
                break

        results["artifacts"][artifact] = {
            "exists": found,
            "description": description,
            "path": str(artifact_path) if artifact_path else "Not found"
        }

        if found:
            # Additional validation for specific artifacts
            try:
                if artifact.endswith('.json'):
                    with open(artifact_path) as f:
                        json.load(f)  # Validate JSON
                elif artifact.endswith('.csv'):
                    pd.read_csv(artifact_path)  # Validate CSV
            except Exception as e:
                results["warnings"].append(f"Artifact {artifact} exists but has validation issues: {e}")
        else:
            results["warnings"].append(f"Missing artifact: {artifact} ({description})")

    # Check if at least some profiler data exists (minimum requirement)
    has_stage_times = results["artifacts"]["stage_times.json"]["exists"]
    has_trace = results["artifacts"]["trace.json"]["exists"]

    if not has_stage_times and not has_trace:
        results["valid"] = False
        results["errors"].append("No profiler data found - profiling may have failed")
    elif not has_trace:
        # Stage timing is sufficient for basic profiling
        results["warnings"].append("Chrome trace not available - only stage timing data collected")

    return results


def analyze_profiler_data(profiler_dir: str) -> dict:
    """
    Analyze profiler data and provide insights.

    Args:
        profiler_dir: Directory containing profiler artifacts

    Returns:
        Dictionary with analysis results
    """
    profiler_path = Path(profiler_dir)
    analysis = {
        "profiler_dir": str(profiler_path),
        "analysis": {},
        "recommendations": []
    }

    # Analyze stage times (check multiple locations)
    stage_times_path = None
    for subdir in ["", "profiler_context"]:
        if subdir:
            check_path = profiler_path / subdir / "stage_times.json"
        else:
            check_path = profiler_path / "stage_times.json"
        if check_path.exists():
            stage_times_path = check_path
            break

    if stage_times_path and stage_times_path.exists():
        with open(stage_times_path) as f:
            stage_data = json.load(f)

        if "average_times" in stage_data:
            avg_times = stage_data["average_times"]
            analysis["analysis"]["stage_times"] = avg_times

            # Find slowest stage
            if avg_times:
                slowest_stage = max(avg_times.items(), key=lambda x: x[1])
                analysis["recommendations"].append(
                    f"Slowest stage: {slowest_stage[0]} ({slowest_stage[1]:.4f}s) - consider optimization"
                )

    # Analyze operation statistics (check multiple locations)
    op_stats_path = None
    for subdir in ["", "torch_profiler"]:
        if subdir:
            check_path = profiler_path / subdir / "op_stats.csv"
        else:
            check_path = profiler_path / "op_stats.csv"
        if check_path.exists():
            op_stats_path = check_path
            break

    if op_stats_path and op_stats_path.exists():
        try:
            df = pd.read_csv(op_stats_path)

            if not df.empty and "CPU Time (μs)" in df.columns:
                df["CPU Time (μs)"] = pd.to_numeric(df["CPU Time (μs)"], errors='coerce')

                # Top operations by CPU time
                top_ops = df.nlargest(5, "CPU Time (μs)")
                analysis["analysis"]["top_operations"] = top_ops[["Name", "CPU Time (μs)"]].to_dict('records')

                # Total CPU time
                total_cpu_time = df["CPU Time (μs)"].sum()
                analysis["analysis"]["total_cpu_time"] = total_cpu_time

                # Memory usage analysis
                if "CUDA Memory (bytes)" in df.columns:
                    df["CUDA Memory (bytes)"] = pd.to_numeric(df["CUDA Memory (bytes)"], errors='coerce')
                    peak_memory = df["CUDA Memory (bytes)"].max()
                    analysis["analysis"]["peak_cuda_memory"] = peak_memory

                    if peak_memory > 1e9:  # > 1GB
                        analysis["recommendations"].append(
                            f"High CUDA memory usage detected: {peak_memory/1e9:.2f}GB - consider memory optimization"
                        )
        except Exception as e:
            analysis["recommendations"].append(f"Could not analyze operation statistics: {e}")

    # Analyze memory statistics (check multiple locations)
    memory_stats_path = None
    for subdir in ["", "torch_profiler"]:
        if subdir:
            check_path = profiler_path / subdir / "memory_stats.json"
        else:
            check_path = profiler_path / "memory_stats.json"
        if check_path.exists():
            memory_stats_path = check_path
            break

    if memory_stats_path and memory_stats_path.exists():
        with open(memory_stats_path) as f:
            memory_data = json.load(f)

        analysis["analysis"]["memory_stats"] = memory_data

        peak_cuda = memory_data.get("peak_memory_usage", {}).get("cuda", 0)
        if peak_cuda > 1e9:  # > 1GB
            analysis["recommendations"].append(
                f"Peak CUDA memory usage: {peak_cuda/1e9:.2f}GB - monitor for memory leaks"
            )

    return analysis


def print_validation_report(validation_results: dict):
    """Print a formatted validation report."""
    print("=" * 60)
    print("PROFILER VALIDATION REPORT")
    print("=" * 60)

    print(f"Profiler Directory: {validation_results['profiler_dir']}")
    print(f"Overall Status: {'✅ VALID' if validation_results['valid'] else '❌ INVALID'}")

    print("\n📁 ARTIFACTS:")
    for artifact, info in validation_results["artifacts"].items():
        status = "✅" if info["exists"] else "❌"
        print(f"  {status} {artifact}: {info['description']}")

    if validation_results["errors"]:
        print("\n❌ ERRORS:")
        for error in validation_results["errors"]:
            print(f"  • {error}")

    if validation_results["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in validation_results["warnings"]:
            print(f"  • {warning}")


def print_analysis_report(analysis_results: dict):
    """Print a formatted analysis report."""
    print("\n" + "=" * 60)
    print("PROFILER ANALYSIS REPORT")
    print("=" * 60)

    print(f"Profiler Directory: {analysis_results['profiler_dir']}")

    if "stage_times" in analysis_results["analysis"]:
        print("\n⏱️  STAGE TIMING:")
        stage_times = analysis_results["analysis"]["stage_times"]
        for stage, time_val in stage_times.items():
            print(f"  • {stage}: {time_val:.4f}s")

    if "top_operations" in analysis_results["analysis"]:
        print("\n🔧 TOP OPERATIONS (by CPU time):")
        top_ops = analysis_results["analysis"]["top_operations"]
        for i, op in enumerate(top_ops, 1):
            name = op["Name"]
            cpu_time = op["CPU Time (μs)"]
            print(f"  {i}. {name}: {cpu_time}μs")

    if "total_cpu_time" in analysis_results["analysis"]:
        total_time = analysis_results["analysis"]["total_cpu_time"]
        print(f"\n📊 Total CPU Time: {total_time}μs ({total_time/1e6:.2f}ms)")

    if "peak_cuda_memory" in analysis_results["analysis"]:
        peak_memory = analysis_results["analysis"]["peak_cuda_memory"]
        print(f"💾 Peak CUDA Memory: {peak_memory:,} bytes ({peak_memory/1e9:.2f}GB)")

    if analysis_results["recommendations"]:
        print("\n💡 RECOMMENDATIONS:")
        for rec in analysis_results["recommendations"]:
            print(f"  • {rec}")


def main():
    """Main function for profiler validation tool."""
    parser = argparse.ArgumentParser(description="Profiler validation and analysis tool")
    parser.add_argument("profiler_dir", help="Directory containing profiler artifacts")
    parser.add_argument("--analysis", action="store_true", help="Perform detailed analysis")
    parser.add_argument("--report", action="store_true", help="Generate profiler report")
    parser.add_argument("--output", help="Output file for report (default: stdout)")

    args = parser.parse_args()

    # Validate profiler artifacts
    print("Validating profiler artifacts...")
    validation_results = check_profiler_artifacts(args.profiler_dir)
    print_validation_report(validation_results)

    # Perform analysis if requested
    if args.analysis:
        print("\nPerforming detailed analysis...")
        analysis_results = analyze_profiler_data(args.profiler_dir)
        print_analysis_report(analysis_results)

    # Generate profiler report if requested
    if args.report:
        print("\nGenerating profiler report...")
        try:
            profiler_report = ProfilerReport(args.profiler_dir)
            report_path = profiler_report.save_report()
            print(f"Profiler report saved to: {report_path}")

            if not args.output:
                profiler_report.print_summary()
        except Exception as e:
            print(f"Error generating profiler report: {e}")

    # Exit with appropriate code
    if not validation_results["valid"]:
        sys.exit(1)
    else:
        print("\n✅ Profiler validation completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
