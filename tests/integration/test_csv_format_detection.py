"""Test CSV format detection and processing."""

import csv
import tempfile
from pathlib import Path

from rldk.forensics.log_scan import detect_and_read_logs, scan_logs


def test_csv_file_detection():
    """Test that CSV files are properly detected and processed."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "training_data.csv"

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'reward', 'kl', 'entropy'])
            writer.writerow([0, 0.1, 0.05, 2.0])
            writer.writerow([1, 0.2, 0.06, 1.9])
            writer.writerow([2, 0.3, 0.04, 2.1])

        events = list(detect_and_read_logs(csv_file))

        assert len(events) == 3
        assert events[0]['step'] == 0
        assert events[0]['reward'] == 0.1
        assert events[1]['step'] == 1
        assert events[2]['step'] == 2


def test_csv_directory_detection():
    """Test CSV files in directories are detected."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "metrics.csv"

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'reward', 'loss'])
            writer.writerow([0, 0.5, 1.2])
            writer.writerow([1, 0.6, 1.1])

        events = list(detect_and_read_logs(Path(temp_dir)))

        assert len(events) == 2
        assert events[0]['step'] == 0
        assert events[1]['step'] == 1


def test_csv_scan_logs_integration():
    """Test that CSV files work with full scan_logs pipeline."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "ppo_training.csv"

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'kl', 'kl_coef', 'entropy', 'advantage_mean', 'advantage_std'])
            for step in range(50):
                kl = 0.05 + (0.01 * step / 50)  # Gradual KL increase
                writer.writerow([step, kl, 0.1, 2.0, 0.0, 1.0])

        result = scan_logs(str(csv_file))

        assert 'rules_fired' in result
        assert 'earliest_step' in result
        assert result['earliest_step'] == 0


def test_csv_error_handling():
    """Test error handling for malformed CSV files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "malformed.csv"

        with open(csv_file, 'w') as f:
            f.write("step,reward\n")
            f.write("0,0.1\n")
            f.write("invalid,data,too,many,columns\n")

        try:
            list(detect_and_read_logs(csv_file))
            raise AssertionError("Expected ValueError for malformed CSV")
        except ValueError as e:
            assert "Failed to parse CSV file" in str(e)
            assert "Error tokenizing data" in str(e)


def test_mixed_format_directory():
    """Test directory with mixed file formats."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = Path(temp_dir) / "data.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'reward'])
            writer.writerow([0, 0.1])

        jsonl_file = Path(temp_dir) / "data.jsonl"
        with open(jsonl_file, 'w') as f:
            f.write('{"step": 1, "reward": 0.2}\n')

        events = list(detect_and_read_logs(Path(temp_dir)))

        assert len(events) >= 1, "Should process at least one file from mixed format directory"

        steps = [event.get('step') for event in events if 'step' in event]
        assert len(steps) >= 1, "Should have at least one event with step data"
        assert any(step in [0, 1] for step in steps), "Should process data from either CSV (step=0) or JSONL (step=1) file"

        rewards = [event.get('reward') for event in events if 'reward' in event]
        assert len(rewards) >= 1, "Should have at least one event with reward data"
        assert any(reward in [0.1, 0.2] for reward in rewards), "Should process reward data from either CSV (0.1) or JSONL (0.2) file"
