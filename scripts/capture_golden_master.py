#!/usr/bin/env python3
"""Capture golden master for RL Debug Kit.

This script runs every public CLI command and key programmatic entry points
with small synthetic inputs, then writes:
- all stdout one liners per command
- all JSON artifacts produced by the commands
- a summary JSON that includes command exit codes, artifact checksums, artifact JSON Schemas
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema
from jsonschema import Draft7Validator

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from normalize import get_normalized_checksum, normalize_json, normalize_text

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set deterministic execution
import random

import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from rldk import (
    EvalResult,
    RewardHealthReport,
    bisect_commits,
    check,
    first_divergence,
    health,
    ingest_runs,
    run,
)
from rldk.tracking import ExperimentTracker, TrackingConfig


@dataclass
class CommandResult:
    """Result of running a command."""
    command: str
    exit_code: int
    stdout: str
    stderr: str
    artifacts: List[str]
    artifact_checksums: Dict[str, str]
    artifact_schemas: Dict[str, Dict[str, Any]]
    duration: float


@dataclass
class GoldenMasterSummary:
    """Summary of golden master capture."""
    version: str
    timestamp: str
    commands: List[CommandResult]
    total_commands: int
    successful_commands: int
    failed_commands: int


def create_synthetic_data() -> Dict[str, Path]:
    """Create synthetic data for testing."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create synthetic training run data
    run_data = {
        "step": [1, 2, 3, 4, 5],
        "loss": [0.5, 0.4, 0.3, 0.25, 0.2],
        "reward_scalar": [0.1, 0.15, 0.2, 0.25, 0.3],
        "global_step": [1, 2, 3, 4, 5],
        "reward_mean": [0.1, 0.15, 0.2, 0.25, 0.3],
    }

    # Create run A
    run_a_path = temp_dir / "run_a"
    run_a_path.mkdir()
    with open(run_a_path / "metrics.jsonl", "w") as f:
        for i in range(len(run_data["step"])):
            record = {k: v[i] for k, v in run_data.items()}
            f.write(json.dumps(record) + "\n")

    # Create run B (slightly different)
    run_b_path = temp_dir / "run_b"
    run_b_path.mkdir()
    with open(run_b_path / "metrics.jsonl", "w") as f:
        for i in range(len(run_data["step"])):
            record = {k: v[i] for k, v in run_data.items()}
            # Make run B slightly different
            record["loss"] = record["loss"] + 0.01
            record["reward_scalar"] = record["reward_scalar"] + 0.005
            f.write(json.dumps(record) + "\n")

    # Create prompts file for reward drift
    prompts_path = temp_dir / "prompts.jsonl"
    with open(prompts_path, "w") as f:
        prompts = [
            {"text": "This is a test prompt."},
            {"text": "Another test prompt for evaluation."},
            {"text": "Third prompt to test reward models."},
        ]
        for prompt in prompts:
            f.write(json.dumps(prompt) + "\n")

    # Create synthetic model directories for reward drift
    model_a_path = temp_dir / "model_a"
    model_a_path.mkdir()
    with open(model_a_path / "config.json", "w") as f:
        json.dump({"model_type": "test", "version": "1.0"}, f)

    model_b_path = temp_dir / "model_b"
    model_b_path.mkdir()
    with open(model_b_path / "config.json", "w") as f:
        json.dump({"model_type": "test", "version": "1.1"}, f)

    return {
        "temp_dir": temp_dir,
        "run_a": run_a_path,
        "run_b": run_b_path,
        "prompts": prompts_path,
        "model_a": model_a_path,
        "model_b": model_b_path,
    }


def run_cli_command(cmd: List[str], cwd: Optional[Path] = None) -> CommandResult:
    """Run a CLI command and capture results."""
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=30,  # 30 second timeout
        )
        duration = time.time() - start_time

        return CommandResult(
            command=" ".join(cmd),
            exit_code=result.returncode,
            stdout=normalize_text(result.stdout.strip()),
            stderr=normalize_text(result.stderr.strip()),
            artifacts=[],
            artifact_checksums={},
            artifact_schemas={},
            duration=duration,
        )
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return CommandResult(
            command=" ".join(cmd),
            exit_code=-1,
            stdout="",
            stderr=normalize_text("Command timed out after 30 seconds"),
            artifacts=[],
            artifact_checksums={},
            artifact_schemas={},
            duration=duration,
        )
    except Exception as e:
        duration = time.time() - start_time
        return CommandResult(
            command=" ".join(cmd),
            exit_code=-1,
            stdout="",
            stderr=normalize_text(str(e)),
            artifacts=[],
            artifact_checksums={},
            artifact_schemas={},
            duration=duration,
        )


def calculate_file_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file using normalized content."""
    if not file_path.exists():
        return ""

    with open(file_path, "rb") as f:
        content = f.read()

    # Try to parse as JSON first, then fall back to text normalization
    try:
        data = json.loads(content.decode('utf-8'))
        return get_normalized_checksum(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Not JSON, normalize as text
        return get_normalized_checksum(content.decode('utf-8', errors='ignore'))


def infer_json_schema(data: Any) -> Dict[str, Any]:
    """Infer JSON schema from data."""
    if isinstance(data, dict):
        schema = {"type": "object", "properties": {}, "required": []}
        for key, value in data.items():
            schema["properties"][key] = infer_json_schema(value)
            schema["required"].append(key)
        return schema
    elif isinstance(data, list):
        if data:
            schema = {"type": "array", "items": infer_json_schema(data[0])}
        else:
            schema = {"type": "array", "items": {}}
        return schema
    elif isinstance(data, (int, float)):
        return {"type": "number"}
    elif isinstance(data, bool):
        return {"type": "boolean"}
    elif isinstance(data, str):
        return {"type": "string"}
    else:
        return {"type": "null"}


def run_programmatic_tests(data_paths: Dict[str, Path]) -> List[CommandResult]:
    """Run programmatic entry point tests."""
    results = []

    # Test 1: ingest_runs
    try:
        start_time = time.time()
        df = ingest_runs(str(data_paths["run_a"]))
        duration = time.time() - start_time

        # Convert to JSON for artifact
        temp_file = data_paths["temp_dir"] / "ingest_result.json"
        df.to_json(temp_file, orient="records")

        results.append(CommandResult(
            command="ingest_runs(run_a)",
            exit_code=0,
            stdout=normalize_text(f"Successfully ingested {len(df)} records"),
            stderr="",
            artifacts=[str(temp_file)],
            artifact_checksums={str(temp_file): calculate_file_checksum(temp_file)},
            artifact_schemas={str(temp_file): infer_json_schema(json.loads(df.to_json(orient="records")))},
            duration=duration,
        ))
    except Exception as e:
        results.append(CommandResult(
            command="ingest_runs(run_a)",
            exit_code=1,
            stdout="",
            stderr=normalize_text(str(e)),
            artifacts=[],
            artifact_checksums={},
            artifact_schemas={},
            duration=0,
        ))

    # Test 2: first_divergence
    try:
        start_time = time.time()
        df_a = ingest_runs(str(data_paths["run_a"]))
        df_b = ingest_runs(str(data_paths["run_b"]))
        report = first_divergence(df_a, df_b, ["loss", "reward_scalar"], 2, 3, 1.0, "temp_diff")
        duration = time.time() - start_time

        # Create artifact
        temp_file = data_paths["temp_dir"] / "diff_result.json"
        with open(temp_file, "w") as f:
            json.dump({
                "diverged": report.diverged,
                "first_step": report.first_step,
                "tripped_signals": report.tripped_signals,
            }, f)

        results.append(CommandResult(
            command="first_divergence(run_a, run_b)",
            exit_code=0,
            stdout=normalize_text(f"Divergence analysis complete, diverged: {report.diverged}"),
            stderr="",
            artifacts=[str(temp_file)],
            artifact_checksums={str(temp_file): calculate_file_checksum(temp_file)},
            artifact_schemas={str(temp_file): infer_json_schema(json.load(open(temp_file)))},
            duration=duration,
        ))
    except Exception as e:
        results.append(CommandResult(
            command="first_divergence(run_a, run_b)",
            exit_code=1,
            stdout="",
            stderr=normalize_text(str(e)),
            artifacts=[],
            artifact_checksums={},
            artifact_schemas={},
            duration=0,
        ))

    # Test 3: check determinism
    try:
        start_time = time.time()
        report = check(
            "python -c 'import torch; print(torch.randn(1).item())'",
            ["loss"],
            None,
            2,
            "cpu"
        )
        duration = time.time() - start_time

        # Create artifact
        temp_file = data_paths["temp_dir"] / "determinism_result.json"
        with open(temp_file, "w") as f:
            json.dump({
                "passed": report.passed,
                "culprit": report.culprit,
                "fixes": report.fixes,
            }, f)

        results.append(CommandResult(
            command="check_determinism",
            exit_code=0,
            stdout=normalize_text(f"Determinism check complete, passed: {report.passed}"),
            stderr="",
            artifacts=[str(temp_file)],
            artifact_checksums={str(temp_file): calculate_file_checksum(temp_file)},
            artifact_schemas={str(temp_file): infer_json_schema(json.load(open(temp_file)))},
            duration=duration,
        ))
    except Exception as e:
        results.append(CommandResult(
            command="check_determinism",
            exit_code=1,
            stdout="",
            stderr=normalize_text(str(e)),
            artifacts=[],
            artifact_checksums={},
            artifact_schemas={},
            duration=0,
        ))

    # Test 4: reward health
    try:
        start_time = time.time()
        run_data = ingest_runs(str(data_paths["run_a"]))
        health_report = health(
            run_data=run_data,
            reference_data=None,
            reward_col="reward_mean",
            step_col="step",
        )
        duration = time.time() - start_time

        # Create artifact
        temp_file = data_paths["temp_dir"] / "reward_health_result.json"
        with open(temp_file, "w") as f:
            json.dump({
                "passed": health_report.passed,
                "drift_detected": health_report.drift_detected,
                "calibration_score": health_report.calibration_score,
            }, f)

        results.append(CommandResult(
            command="reward_health(run_a)",
            exit_code=0,
            stdout=normalize_text(f"Reward health analysis complete, passed: {health_report.passed}"),
            stderr="",
            artifacts=[str(temp_file)],
            artifact_checksums={str(temp_file): calculate_file_checksum(temp_file)},
            artifact_schemas={str(temp_file): infer_json_schema(json.load(open(temp_file)))},
            duration=duration,
        ))
    except Exception as e:
        results.append(CommandResult(
            command="reward_health(run_a)",
            exit_code=1,
            stdout="",
            stderr=normalize_text(str(e)),
            artifacts=[],
            artifact_checksums={},
            artifact_schemas={},
            duration=0,
        ))

    # Test 5: evaluation
    try:
        start_time = time.time()
        run_data = ingest_runs(str(data_paths["run_a"]))
        eval_result = run(
            run_data=run_data,
            suite="quick",
            seed=42,
            sample_size=3,
            output_dir=str(data_paths["temp_dir"] / "eval_output"),
        )
        duration = time.time() - start_time

        # Create artifact
        temp_file = data_paths["temp_dir"] / "eval_result.json"
        with open(temp_file, "w") as f:
            json.dump({
                "sample_size": eval_result.sample_size,
                "seed": eval_result.seed,
                "scores": eval_result.scores,
            }, f)

        results.append(CommandResult(
            command="eval_quick(run_a)",
            exit_code=0,
            stdout=normalize_text(f"Evaluation complete, sample_size: {eval_result.sample_size}"),
            stderr="",
            artifacts=[str(temp_file)],
            artifact_checksums={str(temp_file): calculate_file_checksum(temp_file)},
            artifact_schemas={str(temp_file): infer_json_schema(json.load(open(temp_file)))},
            duration=duration,
        ))
    except Exception as e:
        results.append(CommandResult(
            command="eval_quick(run_a)",
            exit_code=1,
            stdout="",
            stderr=normalize_text(str(e)),
            artifacts=[],
            artifact_checksums={},
            artifact_schemas={},
            duration=0,
        ))

    return results


def run_cli_tests(data_paths: Dict[str, Path]) -> List[CommandResult]:
    """Run CLI command tests."""
    results = []

    # Test 1: version
    results.append(run_cli_command(["rldk", "version"]))

    # Test 2: help
    results.append(run_cli_command(["rldk", "--help"]))

    # Test 3: ingest
    results.append(run_cli_command([
        "rldk", "ingest", str(data_paths["run_a"]), "--output", str(data_paths["temp_dir"] / "ingest_output.jsonl")
    ]))

    # Test 4: diff
    results.append(run_cli_command([
        "rldk", "diff",
        "--a", str(data_paths["run_a"]),
        "--b", str(data_paths["run_b"]),
        "--signals", "loss,reward_scalar",
        "--output-dir", str(data_paths["temp_dir"] / "diff_output")
    ]))

    # Test 5: check-determinism
    results.append(run_cli_command([
        "rldk", "check-determinism",
        "--cmd", "python -c 'import torch; print(torch.randn(1).item())'",
        "--compare", "loss",
        "--replicas", "2",
        "--output-dir", str(data_paths["temp_dir"] / "determinism_output")
    ]))

    # Test 6: reward-health
    results.append(run_cli_command([
        "rldk", "reward-health",
        "--run", str(data_paths["run_a"]),
        "--output-dir", str(data_paths["temp_dir"] / "reward_health_output")
    ]))

    # Test 7: eval
    results.append(run_cli_command([
        "rldk", "eval",
        "--run", str(data_paths["run_a"]),
        "--suite", "quick",
        "--output-dir", str(data_paths["temp_dir"] / "eval_output")
    ]))

    # Test 8: track
    results.append(run_cli_command([
        "rldk", "track",
        "test_experiment",
        "--output-dir", str(data_paths["temp_dir"] / "tracking_output"),
        "--no-wandb"
    ]))

    # Test 9: forensics compare-runs
    results.append(run_cli_command([
        "rldk", "forensics", "compare-runs",
        str(data_paths["run_a"]),
        str(data_paths["run_b"])
    ]))

    # Test 10: reward reward-drift
    results.append(run_cli_command([
        "rldk", "reward", "reward-drift",
        str(data_paths["model_a"]),
        str(data_paths["model_b"]),
        "--prompts", str(data_paths["prompts"])
    ]))

    return results


def capture_golden_master(output_dir: Path) -> GoldenMasterSummary:
    """Capture golden master by running all commands and programmatic entry points."""
    print("Creating synthetic test data...")
    data_paths = create_synthetic_data()

    print("Running CLI command tests...")
    cli_results = run_cli_tests(data_paths)

    print("Running programmatic entry point tests...")
    prog_results = run_programmatic_tests(data_paths)

    all_results = cli_results + prog_results

    # Create summary
    summary = GoldenMasterSummary(
        version="1.0",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        commands=all_results,
        total_commands=len(all_results),
        successful_commands=len([r for r in all_results if r.exit_code == 0]),
        failed_commands=len([r for r in all_results if r.exit_code != 0]),
    )

    # Write results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write summary
    with open(output_dir / "golden_master_summary.json", "w") as f:
        json.dump(asdict(summary), f, indent=2, default=str)

    # Write stdout logs
    with open(output_dir / "stdout_logs.txt", "w") as f:
        for result in all_results:
            f.write(f"=== {result.command} ===\n")
            f.write(f"Exit Code: {result.exit_code}\n")
            f.write(f"Duration: {result.duration:.2f}s\n")
            f.write(f"STDOUT:\n{result.stdout}\n")
            f.write(f"STDERR:\n{result.stderr}\n")
            f.write("\n")

    # Copy artifacts
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    for result in all_results:
        for artifact_path in result.artifacts:
            if Path(artifact_path).exists():
                dest_path = artifacts_dir / Path(artifact_path).name
                shutil.copy2(artifact_path, dest_path)

    # Clean up temp data
    shutil.rmtree(data_paths["temp_dir"])

    return summary


def main():
    """Main entry point."""
    output_dir = Path("golden_master_output")

    print("Starting golden master capture...")
    summary = capture_golden_master(output_dir)

    print("\nGolden master capture complete!")
    print(f"Total commands: {summary.total_commands}")
    print(f"Successful: {summary.successful_commands}")
    print(f"Failed: {summary.failed_commands}")
    print(f"Output directory: {output_dir}")

    # Create zip file
    import zipfile
    zip_path = Path("golden_master.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in output_dir.rglob('*'):
            if file_path.is_file():
                zipf.write(file_path, file_path.relative_to(output_dir))

    print(f"Created zip file: {zip_path}")


if __name__ == "__main__":
    main()
