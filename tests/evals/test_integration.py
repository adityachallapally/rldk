"""Integration tests for CLI column mapping."""

import json
import tempfile
from pathlib import Path
import pandas as pd
import pytest
from typer.testing import CliRunner
from rldk.cli import app


def test_cli_column_mapping_key_value():
    """Test CLI with key=value column mapping format."""
    df = pd.DataFrame({
        "global_step": [1, 2, 3, 4],
        "reward": [0.1, 0.2, 0.15, 0.25],
        "kl": [0.04, 0.05, 0.06, 0.05],
        "entropy": [5.2, 5.1, 5.0, 4.9]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')
        temp_file = Path(f.name)
    
    try:
        runner = CliRunner()
        result = runner.invoke(app, [
            "evals", "evaluate", str(temp_file),
            "--suite", "training_metrics",
            "--column-mapping", "global_step=step"
        ])
        
        assert result.exit_code == 0
        assert "evaluation suite completed" in result.stdout.lower() or "evaluation: " in result.stdout.lower()
    finally:
        temp_file.unlink()


def test_cli_column_mapping_json():
    """Test CLI with JSON column mapping format."""
    df = pd.DataFrame({
        "global_step": [1, 2, 3, 4],
        "reward": [0.1, 0.2, 0.15, 0.25],
        "kl": [0.04, 0.05, 0.06, 0.05],
        "entropy": [5.2, 5.1, 5.0, 4.9]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')
        temp_file = Path(f.name)
    
    try:
        runner = CliRunner()
        result = runner.invoke(app, [
            "evals", "evaluate", str(temp_file),
            "--suite", "training_metrics",
            "--column-mapping", '{"global_step":"step"}'
        ])
        
        assert result.exit_code == 0
        assert "evaluation suite completed" in result.stdout.lower() or "evaluation: " in result.stdout.lower()
    finally:
        temp_file.unlink()


def test_cli_column_mapping_for_standard_eval_dataset():
    """Ensure evaluation datasets with column mappings still bypass the normalizer."""
    records = [
        {
            "step": 1,
            "input": "prompt 1",
            "output": "response 1",
            "events": "[]",
            "reward_mean": 0.1,
        },
        {
            "step": 2,
            "input": "prompt 2",
            "output": "response 2",
            "events": "[]",
            "reward_mean": 0.2,
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
        dataset_path = Path(handle.name)

    try:
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "evals",
                "evaluate",
                str(dataset_path),
                "--suite",
                "quick",
                "--column-mapping",
                "prompt=input,completion=output",
            ],
        )

        assert result.exit_code == 0
    finally:
        dataset_path.unlink()
