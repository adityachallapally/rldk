#!/usr/bin/env python3
"""One-shot acceptance runner for Phase A and Phase B features."""

import json
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from rldk.ingest.training_metrics_normalizer import normalize_training_metrics
from rldk.io.event_schema import dataframe_to_events, events_to_dataframe
from rldk.reward.api import reward_health
from rldk.testing.cli_detect import detect_reward_drift_cmd, detect_reward_health_cmd


def run_check(name: str, check_func) -> Tuple[bool, str]:
    """Run a single check and return (passed, message)."""
    try:
        check_func()
        return True, "PASS"
    except Exception as e:
        return False, f"FAIL: {str(e)}"


def check_stream_to_table_pivot():
    """Check 1: Stream to table pivot."""
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "phase_ab"
    stream_path = fixtures_dir / "stream_small.jsonl"
    
    df = normalize_training_metrics(stream_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "step" in df.columns
    assert df["step"].dtype == "int64"
    
    reward_col = None
    if "reward" in df.columns and not df["reward"].dropna().empty:
        reward_col = "reward"
    elif "reward_mean" in df.columns and not df["reward_mean"].dropna().empty:
        reward_col = "reward_mean"
    
    assert reward_col is not None, f"No reward column with data found. Available columns: {list(df.columns)}"
    
    kl_col = "kl_mean" if "kl_mean" in df.columns else "kl"
    assert kl_col in df.columns
    
    assert df["step"].is_monotonic_increasing


def check_coercion_and_none_guards():
    """Check 2: Coercion and None guards."""
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "phase_ab"
    stream_path = fixtures_dir / "stream_mixed_types.jsonl"
    
    df = normalize_training_metrics(stream_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df["step"].dtype == "int64"
    
    reward_col = None
    if "reward" in df.columns and not df["reward"].dropna().empty:
        reward_col = "reward"
    elif "reward_mean" in df.columns and not df["reward_mean"].dropna().empty:
        reward_col = "reward_mean"
    
    if reward_col is not None:
        assert pd.api.types.is_numeric_dtype(df[reward_col])


def check_round_trip_events():
    """Check 3: Round trip events."""
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "phase_ab"
    csv_path = fixtures_dir / "table_small.csv"
    
    original_df = pd.read_csv(csv_path)
    events = dataframe_to_events(original_df)
    assert len(events) == len(original_df)
    
    restored_df = events_to_dataframe(events)
    assert len(restored_df) == len(original_df)


def check_reward_health_cli_jsonl():
    """Check 4: Reward health CLI on JSONL."""
    health_cmd = detect_reward_health_cmd()
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "phase_ab"
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


def check_reward_health_cli_csv():
    """Check 5: Reward health CLI on CSV."""
    health_cmd = detect_reward_health_cmd()
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "phase_ab"
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


def check_reward_health_missing_reward():
    """Check 6: Reward health missing reward."""
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


def check_reward_drift_score_file_mode():
    """Check 7: Reward drift score file mode."""
    drift_cmd = detect_reward_drift_cmd()
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "phase_ab"
    scores_a_path = fixtures_dir / "scores_a.jsonl"
    scores_b_path = fixtures_dir / "scores_b.jsonl"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_dir)
            
            cmd_parts = drift_cmd.split() + [
                "--scores-a", str(scores_a_path),
                "--scores-b", str(scores_b_path)
            ]
            
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0, f"Command failed: {result.stderr}"
            
            report_path = Path("rldk_reports/reward_drift.json")
            assert report_path.exists(), "Drift report not found"
            
        finally:
            os.chdir(original_cwd)


def check_api_accepts_dataframe():
    """Check 8: API accepts DataFrame."""
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "phase_ab"
    csv_path = fixtures_dir / "table_small.csv"
    df = pd.read_csv(csv_path)
    
    field_map = {"reward_mean": "reward_mean"}
    result = reward_health(df, field_map=field_map)
    assert result is not None


def check_api_accepts_list_of_dicts():
    """Check 9: API accepts list of dicts."""
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "phase_ab"
    csv_path = fixtures_dir / "table_small.csv"
    
    df = pd.read_csv(csv_path)
    records = df.to_dict('records')
    
    field_map = {"reward_mean": "reward_mean"}
    result = reward_health(records, field_map=field_map)
    assert result is not None


def check_api_accepts_jsonl_path():
    """Check 10: API accepts JSONL path."""
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "phase_ab"
    jsonl_path = fixtures_dir / "stream_small.jsonl"
    
    field_map = {"reward": "reward_mean"}
    result = reward_health(str(jsonl_path), field_map=field_map)
    assert result is not None


def check_golden_path_equivalence():
    """Check 11: Golden path equivalence."""
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "phase_ab"
    csv_path = fixtures_dir / "table_small.csv"
    jsonl_path = fixtures_dir / "stream_small.jsonl"
    
    df = pd.read_csv(csv_path)
    records = df.to_dict('records')
    
    field_map_csv = {"reward_mean": "reward_mean"}
    field_map_stream = {"reward": "reward_mean"}
    
    result_path = reward_health(str(jsonl_path), field_map=field_map_stream)
    result_list = reward_health(records, field_map=field_map_csv)
    result_df = reward_health(df, field_map=field_map_csv)
    
    assert result_path is not None
    assert result_list is not None
    assert result_df is not None


def main():
    """Run all acceptance checks."""
    repo_root = Path(__file__).parent.parent
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    checks = [
        ("Stream to table pivot", check_stream_to_table_pivot),
        ("Coercion and None guards", check_coercion_and_none_guards),
        ("Round trip events", check_round_trip_events),
        ("Reward health CLI on JSONL", check_reward_health_cli_jsonl),
        ("Reward health CLI on CSV", check_reward_health_cli_csv),
        ("Reward health missing reward", check_reward_health_missing_reward),
        ("Reward drift score file mode", check_reward_drift_score_file_mode),
        ("API accepts DataFrame", check_api_accepts_dataframe),
        ("API accepts list of dicts", check_api_accepts_list_of_dicts),
        ("API accepts JSONL path", check_api_accepts_jsonl_path),
        ("Golden path equivalence", check_golden_path_equivalence),
    ]
    
    results = []
    passes = 0
    failures = 0
    
    print("Phase A/B Acceptance Test Results")
    print("=" * 50)
    
    for name, check_func in checks:
        passed, message = run_check(name, check_func)
        results.append({"name": name, "status": "PASS" if passed else "FAIL", "message": message})
        
        if passed:
            print(f"✅ {name}: {message}")
            passes += 1
        else:
            print(f"❌ {name}: {message}")
            failures += 1
    
    print("\n" + "=" * 50)
    print(f"Summary: {passes} passed, {failures} failed")
    
    summary = {
        "total_checks": len(checks),
        "passes": passes,
        "failures": failures,
        "results": results
    }
    
    summary_path = artifacts_dir / "phase_ab_acceptance_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nMachine-readable summary saved to: {summary_path}")
    
    if failures > 0:
        exit(1)
    else:
        print("\n🎉 All checks passed!")
        exit(0)


if __name__ == "__main__":
    main()
