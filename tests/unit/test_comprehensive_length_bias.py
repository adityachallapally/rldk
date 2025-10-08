import math
from datetime import datetime
from types import SimpleNamespace

import pytest

from src.rldk.forensics.comprehensive_ppo_forensics import (
    ComprehensivePPOForensics,
    ComprehensivePPOMetrics,
)
from src.rldk.integrations.trl import monitors as trl_monitors
from src.rldk.integrations.trl.monitors import (
    ComprehensivePPOMonitor,
    TRAINER_API_AVAILABLE,
)
from src.rldk.monitor.engine import Event, MonitorEngine, load_rules
from src.rldk.monitor.presets import get_rule_preset


pytestmark = pytest.mark.skipif(
    not TRAINER_API_AVAILABLE,
    reason="Transformers trainer callbacks unavailable",
)

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


class DummyForensics:
    """Lightweight forensics stub for monitor interaction tests."""

    def __init__(self) -> None:
        self.enable_length_bias_detection = True
        self.length_bias_threshold = 0.3
        self.length_bias_corr_threshold = 0.2
        self.current_metrics = ComprehensivePPOMetrics()
        self.update_calls: list[dict[str, object]] = []

    def update(self, **kwargs):  # type: ignore[no-untyped-def]
        self.update_calls.append(kwargs)
        return ComprehensivePPOMetrics()

    def get_anomalies(self):  # type: ignore[no-untyped-def]
        return []

    def get_health_summary(self):  # type: ignore[no-untyped-def]
        return {}

    def save_analysis(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return None


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


def _make_dummy_state(step: int = 1) -> SimpleNamespace:
    return SimpleNamespace(global_step=step)


def _make_logs() -> dict[str, float]:
    return {
        "ppo/rewards/mean": 0.5,
        "ppo/rewards/std": 0.1,
        "ppo/policy/kl_mean": 0.05,
        "ppo/policy/kl_coef": 1.0,
        "ppo/policy/entropy": 1.2,
    }


def test_comprehensive_monitor_wires_response_payloads(
    response_batch: list[dict[str, float | str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trl_monitors, "TRL_AVAILABLE", True)

    monitor = ComprehensivePPOMonitor(enable_length_bias_detection=True, log_interval=1)
    monitor.forensics = DummyForensics()

    args = SimpleNamespace()
    control = SimpleNamespace()
    state = _make_dummy_state(1)

    monitor.on_log(args, state, control, _make_logs(), response_data=response_batch)

    assert monitor.forensics.update_calls, "Monitor did not invoke forensics update"
    recorded = monitor.forensics.update_calls[-1]["response_data"]
    assert isinstance(recorded, list)
    assert len(recorded or []) == len(response_batch)
    for record, original in zip(recorded or [], response_batch):
        assert pytest.approx(float(original["reward"])) == record["reward"]
        assert pytest.approx(float(original["length"])) == record.get("length")
        assert record.get("response") == original["response"]


def test_comprehensive_monitor_deduplicates_batch_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trl_monitors, "TRL_AVAILABLE", True)

    monitor = ComprehensivePPOMonitor(enable_length_bias_detection=True, log_interval=1)
    monitor.forensics = DummyForensics()

    args = SimpleNamespace()
    control = SimpleNamespace()
    state = _make_dummy_state(2)

    batch_payload = {
        "responses": ["short", "long"],
        "rewards": [0.2, 0.8],
        "response_lengths": [4, 12],
    }

    monitor.on_step_end(args, state, control, batch=batch_payload)
    monitor.on_log(args, state, control, _make_logs(), batch=batch_payload)

    recorded = monitor.forensics.update_calls[-1]["response_data"]
    assert isinstance(recorded, list)
    assert len(recorded or []) == 2
    assert recorded == [
        {"reward": pytest.approx(0.2), "response": "short", "length": pytest.approx(4.0)},
        {"reward": pytest.approx(0.8), "response": "long", "length": pytest.approx(12.0)},
    ]


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
