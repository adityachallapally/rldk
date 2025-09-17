"""Tests for Phase A and Phase B reward CLI features."""

import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from rldk.testing.cli_detect import detect_reward_drift_cmd, detect_reward_health_cmd


class TestPhaseABRewardCLI:
    """Test Phase A and Phase B reward CLI functionality."""

    @pytest.fixture
    def fixtures_dir(self):
        """Get path to test fixtures."""
        return Path(__file__).parent.parent / "fixtures" / "phase_ab"

    def test_reward_health_cli_on_jsonl(self, fixtures_dir):
        """Test 4: Call reward health command with stream_small.jsonl."""
        health_cmd = detect_reward_health_cmd()
        stream_path = fixtures_dir / "stream_small.jsonl"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            outdir = Path(temp_dir) / "output"
            
            cmd_parts = health_cmd.split() + ["--run", str(stream_path), "--output-dir", str(outdir), "--field-map", '{"reward": "reward_mean"}']
            
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0, f"Command failed: {result.stderr}"
            
            json_files = list(outdir.glob("*.json"))
            assert len(json_files) > 0, "No JSON report found"
            
            report_path = json_files[0]
            with open(report_path) as f:
                report = json.load(f)
            
            health_fields = ["health", "health_score", "score", "overall", "calibration_score"]
            found_field = None
            for field in health_fields:
                if field in report:
                    found_field = field
                    break
            
            assert found_field is not None, f"No health field found in {list(report.keys())}"
            health_value = report[found_field]
            assert isinstance(health_value, (int, float))
            assert not pd.isna(health_value)

    def test_reward_health_cli_on_csv(self, fixtures_dir):
        """Test 5: Call reward health command with table_small.csv."""
        health_cmd = detect_reward_health_cmd()
        csv_path = fixtures_dir / "table_small.csv"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            outdir = Path(temp_dir) / "output"
            
            cmd_parts = health_cmd.split() + ["--run", str(csv_path), "--output-dir", str(outdir)]
            
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0, f"Command failed: {result.stderr}"
            
            json_files = list(outdir.glob("*.json"))
            assert len(json_files) > 0, "No JSON report found"

    def test_reward_health_missing_reward(self):
        """Test 6: Create JSONL with only kl and loss, expect clear error."""
        health_cmd = detect_reward_health_cmd()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = Path(temp_dir) / "no_reward.jsonl"
            outdir = Path(temp_dir) / "output"
            
            records = [
                {"time": 1.0, "step": 1, "name": "kl", "value": 0.1},
                {"time": 2.0, "step": 2, "name": "loss", "value": 0.5},
            ]
            
            with open(jsonl_path, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
            
            cmd_parts = health_cmd.split() + ["--run", str(jsonl_path), "--output-dir", str(outdir)]
            
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode != 0, "Expected command to fail"
            error_output = (result.stderr or result.stdout).lower()
            assert "reward" in error_output
            assert any(hint in error_output for hint in ["preset", "field", "map"])

    def test_reward_drift_score_file_mode(self, fixtures_dir):
        """Test 7: Call reward drift command with scores_a and scores_b."""
        drift_cmd = detect_reward_drift_cmd()
        scores_a_path = fixtures_dir / "scores_a.jsonl"
        scores_b_path = fixtures_dir / "scores_b.jsonl"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cmd_parts = drift_cmd.split() + [
                "--scores-a", str(scores_a_path),
                "--scores-b", str(scores_b_path)
            ]
            
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=temp_dir
            )
            
            assert result.returncode == 0, f"Command failed: {result.stderr}"
            
            report_path = Path(temp_dir) / "rldk_reports/reward_drift.json"
            assert report_path.exists(), "Drift report not found"
            
            with open(report_path) as f:
                report = json.load(f)
            
            drift_fields = ["drift", "drift_magnitude", "effect_size"]
            found_field = None
            for field in drift_fields:
                if field in report:
                    found_field = field
                    break
            
            assert found_field is not None, f"No drift field found in {list(report.keys())}"
            drift_value = report[found_field]
            assert isinstance(drift_value, (int, float))
            assert not pd.isna(drift_value)
