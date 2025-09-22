"""Integration tests for the GRPO adapter ingestion path."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rldk.ingest import ingest_runs


FIXTURES = Path("test_artifacts/logs_grpo")


def _assert_canonical_columns(df: pd.DataFrame) -> None:
    canonical_columns = ["step", "reward_mean", "reward_std", "kl_mean", "entropy_mean"]
    for column in canonical_columns:
        assert column in df.columns, f"Missing canonical column {column}"
        assert df[column].notna().any(), f"Canonical column {column} should contain values"


def _assert_grpo_specific_columns(df: pd.DataFrame) -> None:
    grpo_columns = [
        "advantage_mean",
        "advantage_std",
        "grad_norm_policy",
        "grad_norm_value",
        "kl_coef",
    ]
    for column in grpo_columns:
        assert column in df.columns, f"Missing GRPO specific column {column}"
        assert df[column].notna().any(), f"Expected GRPO column {column} to contain values"


def test_grpo_ingest_seed_directory() -> None:
    """Each seeded run should ingest with canonical and GRPO-specific metrics."""
    seed_path = FIXTURES / "seed_1"
    df = ingest_runs(seed_path, adapter_hint="grpo")

    assert not df.empty
    _assert_canonical_columns(df)
    _assert_grpo_specific_columns(df)
    assert df["seed"].nunique() == 1
    assert set(df["seed"].dropna()) == {1}


def test_grpo_ingest_seed_run_file() -> None:
    """Loading a specific run.jsonl file should preserve canonical and extra metrics."""
    run_path = FIXTURES / "seed_2" / "run.jsonl"
    df = ingest_runs(run_path, adapter_hint="grpo")

    assert not df.empty
    _assert_canonical_columns(df)
    _assert_grpo_specific_columns(df)
    assert df["run_id"].iloc[0] == "seed_2"


def test_grpo_ingest_multiple_seeds() -> None:
    """Aggregating the directory should keep seed-level metadata available."""
    df = ingest_runs(FIXTURES, adapter_hint="grpo")

    assert not df.empty
    _assert_canonical_columns(df)
    _assert_grpo_specific_columns(df)
    assert df["seed"].nunique() == 3
    assert set(df["run_id"].dropna()) == {"seed_1", "seed_2", "seed_3"}
