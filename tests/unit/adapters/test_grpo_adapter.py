import json

import pandas as pd
import pytest

from rldk.adapters import GRPOAdapter


def test_grpo_adapter_maps_aliases_and_sanitizes(tmp_path):
    run_dir = tmp_path / "grpo"
    run_dir.mkdir()
    run_file = run_dir / "run.jsonl"

    record = {
        "step": 1,
        "reward": 1.5,
        "reward_stddev": 0.2,
        "policy_kl": 0.3,
        "policy_entropy": 2.5,
        "accept_rate": 0.75,
        "adv_mean": 0.1,
        "advantage_stddev": 0.4,
        "policy_grad_norm": 0.5,
        "critic_grad_norm": 0.6,
        "training_phase": "eval",
        "run": "run-123",
        "seed": 7,
        "custom_numeric": 5,
        "custom_flag": True,
        "prompt": "should be dropped",
        "response_text": "should be dropped",
        "metadata_text": "should be dropped",
        "labels": "should be dropped",
        "nested_payload": {"prompt": "nested"},
    }

    run_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    adapter = GRPOAdapter(run_dir)
    assert adapter.can_handle()

    df = adapter.load()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1

    row = df.iloc[0]
    assert row["reward_mean"] == pytest.approx(1.5)
    assert row["reward_std"] == pytest.approx(0.2)
    assert row["kl_mean"] == pytest.approx(0.3)
    assert row["entropy_mean"] == pytest.approx(2.5)
    assert row["acceptance_rate"] == pytest.approx(0.75)
    assert row["advantage_mean"] == pytest.approx(0.1)
    assert row["advantage_std"] == pytest.approx(0.4)
    assert row["grad_norm_policy"] == pytest.approx(0.5)
    assert row["grad_norm_value"] == pytest.approx(0.6)
    assert row["phase"] == "eval"
    assert row["run_id"] == "run-123"
    assert row["seed"] == 7
    assert row["custom_numeric"] == 5
    assert bool(row["custom_flag"]) is True

    for dropped in ["prompt", "response_text", "metadata_text", "labels", "nested_payload"]:
        assert dropped not in df.columns
