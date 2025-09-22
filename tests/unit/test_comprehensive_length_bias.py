import math
from datetime import datetime

import pytest

from src.rldk.forensics.comprehensive_ppo_forensics import ComprehensivePPOForensics
from src.rldk.monitor.engine import Event, MonitorEngine, load_rules
from src.rldk.monitor.presets import get_rule_preset


@pytest.fixture
def response_batch() -> list[dict[str, float | str]]:
    return [
        {"response": "short", "length": 4, "reward": 0.2},
        {"response": "medium", "length": 8, "reward": 0.35},
        {"response": "longer", "length": 12, "reward": 0.55},
        {"response": "longest", "length": 16, "reward": 0.7},
        {"response": "mega", "length": 20, "reward": 0.85},
        {"response": "ultra", "length": 24, "reward": 1.05},
    ]


def test_length_bias_detection_enabled(response_batch: list[dict[str, float | str]]) -> None:
    forensics = ComprehensivePPOForensics(
        enable_length_bias_detection=True,
        length_bias_threshold=0.3,
    )

    metrics = forensics.update(
        step=1,
        kl=0.1,
        kl_coef=0.2,
        entropy=2.0,
        reward_mean=0.5,
        reward_std=0.2,
        response_data=response_batch,
    )

    analysis = forensics.get_length_bias_analysis()
    assert analysis["detected"] is True
    assert analysis["metrics"]["valid_sample_count"] == len(response_batch)

    assert metrics.length_bias_detected is True
    assert metrics.length_bias_score is not None and metrics.length_bias_score > 0.3
    assert math.isclose(
        metrics.length_bias_threshold or 0.0,
        forensics.length_bias_threshold,
        rel_tol=1e-6,
    )
    assert metrics.length_reward_correlation_abs is not None
    assert metrics.length_reward_spearman_abs is not None


def test_length_bias_detection_disabled(response_batch: list[dict[str, float | str]]) -> None:
    forensics = ComprehensivePPOForensics(enable_length_bias_detection=False)

    metrics = forensics.update(
        step=5,
        kl=0.05,
        kl_coef=0.2,
        entropy=2.3,
        reward_mean=0.4,
        reward_std=0.1,
        response_data=response_batch,
    )

    assert forensics.get_length_bias_analysis() == {}
    assert metrics.length_bias_score is None
    assert metrics.length_bias_detected is False
    assert metrics.length_bias_metrics is None


def test_length_bias_rule_presets_register() -> None:
    safe_preset = get_rule_preset("ppo_safe")
    assert safe_preset is not None
    ids = {rule["id"] for rule in safe_preset["rules"]}
    assert "ppo_length_bias_severity" in ids
    assert "ppo_length_bias_corr" in ids
    assert "ppo_length_bias_rank_corr" in ids

    strict_preset = get_rule_preset("ppo_strict")
    assert strict_preset is not None
    strict_ids = {rule["id"] for rule in strict_preset["rules"]}
    assert "ppo_strict_length_bias_severity" in strict_ids
    assert "ppo_strict_length_bias_corr" in strict_ids
    assert "ppo_strict_length_bias_rank_corr" in strict_ids

    length_bias = get_rule_preset("length_bias")
    assert length_bias is not None
    assert {rule["id"] for rule in length_bias["rules"]} == {
        "length_bias_severity_gate",
        "length_bias_corr_guard",
        "length_bias_rank_corr_guard",
    }


def test_length_bias_preset_fires_alerts() -> None:
    rules = load_rules("length_bias")
    engine = MonitorEngine(rules)

    meta = {
        "length_bias_threshold": 0.4,
        "length_bias_corr_threshold": 0.25,
    }

    alerts = []
    for idx in range(1, 8):
        alerts.extend(
            engine.process_event(
                Event(
                    time=datetime.utcnow().isoformat() + "Z",
                    step=idx,
                    name="length_bias_score",
                    value=0.6,
                    run_id="run-1",
                    tags={},
                    meta=meta,
                )
            )
        )

    assert any(alert.rule_id == "length_bias_severity_gate" for alert in alerts)

    corr_alerts = []
    for idx in range(20, 27):
        corr_alerts.extend(
            engine.process_event(
                Event(
                    time=datetime.utcnow().isoformat() + "Z",
                    step=idx,
                    name="length_reward_correlation_abs",
                    value=0.5,
                    run_id="run-1",
                    tags={},
                    meta=meta,
                )
            )
        )
    assert any(alert.rule_id == "length_bias_corr_guard" for alert in corr_alerts)
