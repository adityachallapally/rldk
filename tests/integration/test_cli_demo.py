"""Integration tests for CLI demo functionality."""

import pytest
import subprocess
import tempfile
import json
from pathlib import Path
import pandas as pd

from rldk.ingest import ingest_runs


class TestCLIDemo:
    """Test CLI demo command functionality."""

    def test_demo_command_basic(self):
        """Test basic demo command execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run demo command
            result = subprocess.run(
                ["python", "-m", "rldk", "demo", "--out", temp_dir, "--steps", "10", "--variants", "2"],
                capture_output=True,
                text=True,
                cwd="/workspace"
            )
            
            assert result.returncode == 0, f"Demo command failed: {result.stderr}"
            
            # Check that files were created
            temp_path = Path(temp_dir)
            ppo_files = list(temp_path.glob("ppo_run_*.jsonl"))
            grpo_files = list(temp_path.glob("grpo_run_*.jsonl"))
            
            assert len(ppo_files) == 2, f"Expected 2 PPO files, got {len(ppo_files)}"
            assert len(grpo_files) == 2, f"Expected 2 GRPO files, got {len(grpo_files)}"
            
            # Check that files contain data
            for file_path in ppo_files + grpo_files:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    assert len(lines) == 10, f"Expected 10 lines in {file_path}, got {len(lines)}"
                    
                    # Check first line is valid JSON
                    first_data = json.loads(lines[0])
                    assert "step" in first_data
                    assert "reward_mean" in first_data

    def test_demo_command_with_seed(self):
        """Test demo command with specific seed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run demo command with specific seed
            result = subprocess.run(
                ["python", "-m", "rldk", "demo", "--out", temp_dir, "--seed", "42", "--steps", "5"],
                capture_output=True,
                text=True,
                cwd="/workspace"
            )
            
            assert result.returncode == 0, f"Demo command failed: {result.stderr}"
            
            # Check that files were created
            temp_path = Path(temp_dir)
            ppo_files = list(temp_path.glob("ppo_run_*.jsonl"))
            
            assert len(ppo_files) >= 1
            
            # Check that data is deterministic (same seed should produce same data)
            with open(ppo_files[0], 'r') as f:
                first_run_data = [json.loads(line) for line in f.readlines()]
            
            # Run again with same seed
            with tempfile.TemporaryDirectory() as temp_dir2:
                result2 = subprocess.run(
                    ["python", "-m", "rldk", "demo", "--out", temp_dir2, "--seed", "42", "--steps", "5"],
                    capture_output=True,
                    text=True,
                    cwd="/workspace"
                )
                
                assert result2.returncode == 0
                
                temp_path2 = Path(temp_dir2)
                ppo_files2 = list(temp_path2.glob("ppo_run_*.jsonl"))
                
                with open(ppo_files2[0], 'r') as f:
                    second_run_data = [json.loads(line) for line in f.readlines()]
                
                # Data should be identical
                assert first_run_data == second_run_data

    def test_fixture_generation(self):
        """Test that fixture data is generated."""
        # Run demo command
        result = subprocess.run(
            ["python", "-m", "rldk", "demo", "--steps", "5"],
            capture_output=True,
            text=True,
            cwd="/workspace"
        )
        
        assert result.returncode == 0, f"Demo command failed: {result.stderr}"
        
        # Check that fixture was created
        fixture_path = Path("tests/fixtures/minirun/run.jsonl")
        assert fixture_path.exists(), "Fixture file was not created"
        
        # Check fixture content
        with open(fixture_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 5, f"Expected 5 lines in fixture, got {len(lines)}"
            
            # Check first line
            first_data = json.loads(lines[0])
            assert first_data["step"] == 1
            assert "reward_mean" in first_data
            assert "pass_rate" in first_data

    def test_ingest_demo_data(self):
        """Test that demo data can be ingested."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate demo data
            result = subprocess.run(
                ["python", "-m", "rldk", "demo", "--out", temp_dir, "--steps", "5"],
                capture_output=True,
                text=True,
                cwd="/workspace"
            )
            
            assert result.returncode == 0
            
            # Test ingestion
            temp_path = Path(temp_dir)
            ppo_files = list(temp_path.glob("ppo_run_*.jsonl"))
            
            assert len(ppo_files) >= 1
            
            # Test auto-detection
            df = ingest_runs(ppo_files[0])
            assert len(df) == 5
            assert df["step"].tolist() == [1, 2, 3, 4, 5]
            
            # Test explicit adapter
            df2 = ingest_runs(ppo_files[0], adapter_hint="demo_jsonl")
            assert len(df2) == 5

    def test_ingest_fixture_data(self):
        """Test that fixture data can be ingested."""
        # Generate fixture data
        result = subprocess.run(
            ["python", "-m", "rldk", "demo", "--steps", "5"],
            capture_output=True,
            text=True,
            cwd="/workspace"
        )
        
        assert result.returncode == 0
        
        # Test ingestion of fixture
        fixture_path = Path("tests/fixtures/minirun/run.jsonl")
        assert fixture_path.exists()
        
        df = ingest_runs(fixture_path)
        assert len(df) == 5
        assert df["step"].tolist() == [1, 2, 3, 4, 5]

    def test_demo_help(self):
        """Test that demo command shows help."""
        result = subprocess.run(
            ["python", "-m", "rldk", "demo", "--help"],
            capture_output=True,
            text=True,
            cwd="/workspace"
        )
        
        assert result.returncode == 0
        assert "Generate synthetic PPO and GRPO training runs" in result.stdout
        assert "--out" in result.stdout
        assert "--seed" in result.stdout
        assert "--steps" in result.stdout
        assert "--variants" in result.stdout


class TestCLIIntegration:
    """Test CLI integration with demo data."""

    def test_forensics_log_scan_demo_data(self):
        """Test forensics log-scan with demo data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate demo data
            result = subprocess.run(
                ["python", "-m", "rldk", "demo", "--out", temp_dir, "--steps", "50"],
                capture_output=True,
                text=True,
                cwd="/workspace"
            )
            
            assert result.returncode == 0
            
            # Run forensics log-scan
            result = subprocess.run(
                ["python", "-m", "rldk", "forensics", "log-scan", temp_dir],
                capture_output=True,
                text=True,
                cwd="/workspace"
            )
            
            # Should not fail (even if no anomalies found)
            assert result.returncode == 0

    def test_forensics_compare_runs_demo_data(self):
        """Test forensics compare-runs with demo data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate demo data
            result = subprocess.run(
                ["python", "-m", "rldk", "demo", "--out", temp_dir, "--steps", "20"],
                capture_output=True,
                text=True,
                cwd="/workspace"
            )
            
            assert result.returncode == 0
            
            # Run forensics compare-runs
            result = subprocess.run(
                ["python", "-m", "rldk", "forensics", "compare-runs", temp_dir, temp_dir],
                capture_output=True,
                text=True,
                cwd="/workspace"
            )
            
            # Should not fail
            assert result.returncode == 0

    def test_diff_command_demo_data(self):
        """Test diff command with demo data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate demo data
            result = subprocess.run(
                ["python", "-m", "rldk", "demo", "--out", temp_dir, "--steps", "20"],
                capture_output=True,
                text=True,
                cwd="/workspace"
            )
            
            assert result.returncode == 0
            
            # Run diff command
            result = subprocess.run(
                ["python", "-m", "rldk", "diff", "--a", temp_dir, "--b", temp_dir, "--signals", "loss,reward_mean"],
                capture_output=True,
                text=True,
                cwd="/workspace"
            )
            
            # Should not fail
            assert result.returncode == 0

    def test_check_determinism_demo_data(self):
        """Test check-determinism with demo data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate demo data
            result = subprocess.run(
                ["python", "-m", "rldk", "demo", "--out", temp_dir, "--steps", "10"],
                capture_output=True,
                text=True,
                cwd="/workspace"
            )
            
            assert result.returncode == 0
            
            # Create a simple deterministic script
            script_path = temp_dir + "/deterministic_script.py"
            with open(script_path, 'w') as f:
                f.write("""
import json
import random
random.seed(42)
for i in range(5):
    print(json.dumps({"step": i, "loss": 1.0 - i*0.1}))
""")
            
            # Run check-determinism
            result = subprocess.run(
                ["python", "-m", "rldk", "check-determinism", "--cmd", f"python {script_path}", "--compare", "loss", "--replicas", "2"],
                capture_output=True,
                text=True,
                cwd="/workspace"
            )
            
            # Should not fail
            assert result.returncode == 0

    def test_ingest_command_demo_data(self):
        """Test ingest command with demo data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate demo data
            result = subprocess.run(
                ["python", "-m", "rldk", "demo", "--out", temp_dir, "--steps", "10"],
                capture_output=True,
                text=True,
                cwd="/workspace"
            )
            
            assert result.returncode == 0
            
            # Run ingest command
            result = subprocess.run(
                ["python", "-m", "rldk", "ingest", temp_dir, "--adapter", "demo_jsonl", "--output", temp_dir + "/output.jsonl"],
                capture_output=True,
                text=True,
                cwd="/workspace"
            )
            
            # Should not fail
            assert result.returncode == 0
            
            # Check output file was created
            output_path = Path(temp_dir) / "output.jsonl"
            assert output_path.exists()
            
            # Check output content
            with open(output_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) > 0