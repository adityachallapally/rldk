"""Test compare runs functionality."""

import json
import tempfile
from pathlib import Path

from rldk.forensics.log_scan import scan_logs


def test_compare_runs_clean_vs_doctored():
    """Test comparing clean vs doctored runs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create clean logs
        clean_dir = Path(temp_dir) / "clean"
        clean_dir.mkdir()

        clean_logs = []
        for step in range(100):
            log = {
                "step": step,
                "kl": 0.05 + 0.01 * (step % 10) / 10,
                "kl_coef": 0.1,
                "entropy": 2.0,
                "advantage_mean": 0.0,
                "advantage_std": 1.0,
                "grad_norm_policy": 0.5,
                "grad_norm_value": 0.3,
            }
            clean_logs.append(log)

        with open(clean_dir / "training.jsonl", "w") as f:
            for log in clean_logs:
                f.write(json.dumps(log) + "\n")

        # Create doctored logs with KL spike
        doctored_dir = Path(temp_dir) / "doctored"
        doctored_dir.mkdir()

        doctored_logs = []
        for step in range(100):
            if step < 50:
                kl = 0.05
            else:
                kl = 0.5  # KL spike

            log = {
                "step": step,
                "kl": kl,
                "kl_coef": 0.1,
                "entropy": 2.0,
                "advantage_mean": 0.0,
                "advantage_std": 1.0,
                "grad_norm_policy": 0.5,
                "grad_norm_value": 0.3,
            }
            doctored_logs.append(log)

        with open(doctored_dir / "training.jsonl", "w") as f:
            for log in doctored_logs:
                f.write(json.dumps(log) + "\n")

        # Scan both runs
        clean_scan = scan_logs(str(clean_dir))
        doctored_scan = scan_logs(str(doctored_dir))

        # Clean run should have no anomalies
        assert len(clean_scan["rules_fired"]) == 0

        # Doctored run should have KL spike
        kl_spike_rules = [
            r for r in doctored_scan["rules_fired"] if r["rule"] == "kl_spike"
        ]
        assert len(kl_spike_rules) > 0

        # Check that earliest step is detected
        assert clean_scan["earliest_step"] == 0
        assert doctored_scan["earliest_step"] == 0


def test_compare_runs_empty_logs():
    """Test comparing runs with empty logs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create empty logs
        empty_dir = Path(temp_dir) / "empty"
        empty_dir.mkdir()

        with open(empty_dir / "training.jsonl", "w"):
            pass  # Empty file

        # Scan empty run
        scan = scan_logs(str(empty_dir))

        # Should handle empty logs gracefully
        assert len(scan["rules_fired"]) == 0
        assert scan["earliest_step"] is None
