"""Tests for GRPO monitor presets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import pytest

from rldk.monitor import Alert, ActionExecutor, Event, MonitorEngine, load_rules
from rldk.monitor.presets import get_field_map_preset, get_rule_preset

_FIXTURE_PATH = Path("test_artifacts/logs_grpo/seed_1/run.jsonl")


class _NoopExecutor(ActionExecutor):
    """Capture alerts without side effects for testing."""

    def execute(self, action, rule, event, window):  # type: ignore[override]
        return Alert(
            rule_id=rule.id,
            action=action.kind,
            event=event,
            window_size=rule.window_size,
            window_kind=rule.window_kind,
            message=None,
            status="success",
        )


def _iter_fixture_events(
    metric_names: Sequence[str],
    *,
    field_map: dict[str, str] | None = None,
    extra_metrics: dict[str, Sequence[float]] | None = None,
) -> Iterable[Event]:
    with _FIXTURE_PATH.open() as handle:
        last_step = 0
        for line in handle:
            payload = json.loads(line)
            step = int(payload["step"])
            last_step = max(last_step, step)
            timestamp = "2024-01-01T00:00:00Z"
            for metric in metric_names:
                value = payload.get(metric)
                if value is None:
                    continue
                canonical = field_map.get(metric, metric) if field_map else metric
                yield Event(
                    time=timestamp,
                    step=step,
                    name=canonical,
                    value=float(value),
                    run_id="seed_1",
                )
    if not extra_metrics:
        return

    timestamp = "2024-01-02T00:00:00Z"
    current_step = last_step
    for metric, values in extra_metrics.items():
        canonical = field_map.get(metric, metric) if field_map else metric
        for value in values:
            current_step += 1
            yield Event(
                time=timestamp,
                step=current_step,
                name=canonical,
                value=float(value),
                run_id="seed_1",
            )


def _drain_events(engine: MonitorEngine, events: Iterable[Event]) -> set[str]:
    fired: set[str] = set()
    for event in events:
        for alert in engine.process_event(event):
            fired.add(alert.rule_id)
    return fired


def _gradient_events(
    pairs: Sequence[Tuple[str, float]],
    *,
    start_step: int,
    run_id: str,
    timestamp: str = "2024-01-02T00:00:00Z",
) -> list[Event]:
    events: list[Event] = []
    step = start_step
    last_policy: float | None = None
    last_value: float | None = None
    for name, value in pairs:
        step += 1
        meta: dict[str, float] = {}
        if name == "grad_norm_policy":
            if last_value is not None:
                ratio = float(value / (last_value + 1e-6))
                inverse_ratio = float(last_value / (value + 1e-6))
                meta["value_grad_norm"] = float(last_value)
                meta["policy_over_value"] = ratio
                meta["value_over_policy"] = inverse_ratio
            last_policy = float(value)
        elif name == "grad_norm_value":
            if last_policy is not None:
                ratio = float(last_policy / (value + 1e-6))
                inverse_ratio = float(value / (last_policy + 1e-6))
                meta["policy_grad_norm"] = float(last_policy)
                meta["policy_over_value"] = ratio
                meta["value_over_policy"] = inverse_ratio
            last_value = float(value)
        events.append(
            Event(
                time=timestamp,
                step=step,
                name=name,
                value=float(value),
                run_id=run_id,
                meta=meta,
            )
        )
    return events


def _reward_health_events(total_steps: int = 140) -> list[Event]:
    run_id = "reward_health"
    timestamp = "2024-02-01T00:00:00Z"
    events: list[Event] = []
    for step in range(1, total_steps + 1):
        progress = step / total_steps
        gold_value = 0.6 - 0.2 * progress
        reward_value = 0.1 + 0.9 * progress
        kl_value = 0.22 + 0.08 * progress
        events.append(
            Event(
                time=timestamp,
                step=step,
                name="gold_metric",
                value=float(gold_value),
                run_id=run_id,
            )
        )
        events.append(
            Event(
                time=timestamp,
                step=step,
                name="reward_mean",
                value=float(reward_value),
                run_id=run_id,
            )
        )
        events.append(
            Event(
                time=timestamp,
                step=step,
                name="kl",
                value=float(kl_value),
                run_id=run_id,
            )
        )
        events.append(
            Event(
                time=timestamp,
                step=step,
                name="epoch",
                value=float(step),
                run_id=run_id,
            )
        )
    return events


def test_grpo_safe_preset_triggers_on_fixture_anomalies() -> None:
    preset = get_rule_preset("grpo_safe")
    assert preset is not None
    rules = load_rules(preset)
    engine = MonitorEngine(rules, action_executor=_NoopExecutor())

    metrics = ["kl", "kl_coef", "reward_mean"]
    field_map = get_field_map_preset("grpo") or {}
    extra_metrics = {
        "diversity/pass_at_1": [
            0.42,
            0.4,
            0.35,
            0.28,
            0.24,
            0.22,
            0.21,
            0.2,
            0.19,
            0.18,
            0.17,
            0.16,
        ],
        "distinct_4": [
            0.24,
            0.2,
            0.16,
            0.12,
            0.1,
            0.09,
            0.08,
            0.07,
            0.06,
            0.05,
            0.045,
            0.04,
        ],
        "self_bleu": [
            0.72,
            0.75,
            0.8,
            0.86,
            0.9,
            0.92,
            0.93,
            0.94,
            0.95,
            0.952,
        ],
        "output_entropy": [
            1.6,
            1.48,
            1.4,
            1.32,
            1.24,
            1.18,
            1.14,
            1.1,
            1.05,
            1.0,
        ],
    }
    fired = _drain_events(
        engine,
        _iter_fixture_events(
            metrics,
            field_map=field_map,
            extra_metrics=extra_metrics,
        ),
    )

    gradient_sequence = [
        ("grad_norm_policy", 2.8),
        ("grad_norm_value", 2.5),
        ("grad_norm_policy", 3.1),
        ("grad_norm_value", 2.4),
        ("grad_norm_policy", 3.0),
        ("grad_norm_value", 2.3),
        ("grad_norm_policy", 8.6),
        ("grad_norm_policy", 8.9),
        ("grad_norm_value", 2.0),
        ("grad_norm_policy", 9.3),
        ("grad_norm_value", 0.35),
    ]
    fired.update(
        _drain_events(
            engine,
            _gradient_events(
                gradient_sequence,
                start_step=2000,
                run_id="seed_1",
            ),
        )
    )

    assert "grpo_safe_kl_spike" in fired
    assert "grpo_safe_kl_coef_stall" in fired
    assert "grpo_safe_reward_saturation" in fired
    assert "grpo_safe_diversity_pass_floor" in fired
    assert "grpo_safe_diversity_distinct_collapse" in fired
    assert "grpo_safe_self_bleu_spike" in fired
    assert "grpo_safe_output_entropy_floor" in fired
    assert "grpo_safe_policy_grad_spike" in fired
    assert "grpo_safe_grad_ratio_imbalance" in fired


@pytest.mark.parametrize(
    "metric, values, expected_rule",
    [
        (
            "entropy",
            [1.7] * 8,
            "grpo_strict_entropy_floor",
        ),
        (
            "advantage_std",
            [0.3] * 6,
            "grpo_strict_advantage_collapse",
        ),
        (
            "acceptance_rate",
            [0.5, 0.55, 0.58, 0.6, 0.62, 0.1, 0.85, 0.12, 0.8, 0.15],
            "grpo_strict_acceptance_swings",
        ),
        (
            "diversity_pass_at_1",
            [
                0.42,
                0.38,
                0.34,
                0.3,
                0.27,
                0.24,
                0.22,
                0.21,
                0.2,
                0.19,
            ],
            "grpo_strict_diversity_pass_floor",
        ),
        (
            "diversity_distinct_4",
            [0.26, 0.22, 0.2, 0.18, 0.15, 0.13, 0.12, 0.1, 0.08],
            "grpo_strict_diversity_distinct_collapse",
        ),
        (
            "diversity_self_bleu",
            [0.7, 0.75, 0.8, 0.85, 0.87, 0.89, 0.9, 0.91, 0.92],
            "grpo_strict_self_bleu_spike",
        ),
        (
            "diversity_output_entropy",
            [1.55, 1.5, 1.46, 1.42, 1.36, 1.3, 1.24, 1.18, 1.12],
            "grpo_strict_output_entropy_floor",
        ),
    ],
)
def test_grpo_strict_preset_flags_strict_failures(
    metric: str, values: Sequence[float], expected_rule: str
) -> None:
    preset = get_rule_preset("grpo_strict")
    assert preset is not None
    rules = load_rules(preset)
    engine = MonitorEngine(rules, action_executor=_NoopExecutor())

    baseline_metrics = ["kl", "kl_coef", "reward_mean", "entropy", "advantage_std"]
    fired = _drain_events(engine, _iter_fixture_events(baseline_metrics))

    for offset, value in enumerate(values, start=1):
        event = Event(
            time="2024-01-02T00:00:00Z",
            step=1500 + offset,
            name=metric,
            value=float(value),
            run_id="seed_1",
        )
        for alert in engine.process_event(event):
            fired.add(alert.rule_id)

    gradient_sequence = [
        ("grad_norm_policy", 2.6),
        ("grad_norm_value", 2.3),
        ("grad_norm_policy", 2.9),
        ("grad_norm_value", 2.2),
        ("grad_norm_policy", 6.7),
        ("grad_norm_policy", 6.9),
        ("grad_norm_value", 1.8),
        ("grad_norm_policy", 7.2),
        ("grad_norm_value", 0.55),
    ]
    for event in _gradient_events(
        gradient_sequence,
        start_step=2500,
        run_id="seed_1",
    ):
        for alert in engine.process_event(event):
            fired.add(alert.rule_id)

    assert "grpo_strict_kl_spike" in fired
    assert "grpo_strict_kl_coef_stall" in fired
    assert "grpo_strict_reward_saturation" in fired
    assert expected_rule in fired
    assert "grpo_strict_policy_grad_spike" in fired
    assert "grpo_strict_grad_ratio_imbalance" in fired


def test_grpo_safe_kl_spike_triggers_after_step_reset() -> None:
    preset = get_rule_preset("grpo_safe")
    assert preset is not None
    rules = load_rules(preset)
    engine = MonitorEngine(rules, action_executor=_NoopExecutor())

    def make_event(step: int, value: float) -> Event:
        return Event(
            time="2024-01-03T00:00:00Z",
            step=step,
            name="kl",
            value=value,
            run_id="step_reset",
        )

    fired_steps: list[int] = []

    # Prime the rule to satisfy the configured grace period.
    for step in range(1, 13):
        engine.process_event(make_event(step, 0.1))

    # First spike with monotonically increasing steps.
    for step in (13, 14):
        alerts = engine.process_event(make_event(step, 0.45))
        fired_steps.extend(
            alert.event.step
            for alert in alerts
            if alert.rule_id == "grpo_safe_kl_spike"
        )

    assert 14 in fired_steps

    # Step counter resets; buffer and grace counters should restart.
    post_reset_fired_steps: list[int] = []

    for step, value in ((4, 0.5), (5, 0.55)):
        alerts = engine.process_event(make_event(step, value))
        post_reset_fired_steps.extend(
            alert.event.step
            for alert in alerts
            if alert.rule_id == "grpo_safe_kl_spike"
        )

    # Insufficient post-reset events to satisfy grace period.
    assert not post_reset_fired_steps

    for step in range(6, 16):
        alerts = engine.process_event(make_event(step, 0.1))
        post_reset_fired_steps.extend(
            alert.event.step
            for alert in alerts
            if alert.rule_id == "grpo_safe_kl_spike"
        )

    assert not post_reset_fired_steps

    for step, value in ((16, 0.45), (17, 0.55)):
        alerts = engine.process_event(make_event(step, value))
        post_reset_fired_steps.extend(
            alert.event.step
            for alert in alerts
            if alert.rule_id == "grpo_safe_kl_spike"
        )

    assert post_reset_fired_steps and set(post_reset_fired_steps) == {17}


def test_reward_health_presets_emit_synthetic_alerts() -> None:
    safe_preset = get_rule_preset("grpo_safe")
    strict_preset = get_rule_preset("grpo_strict")
    assert safe_preset is not None
    assert strict_preset is not None

    safe_engine = MonitorEngine(
        load_rules(safe_preset),
        action_executor=_NoopExecutor(),
        reward_health_window=120,
    )
    strict_engine = MonitorEngine(
        load_rules(strict_preset),
        action_executor=_NoopExecutor(),
        reward_health_window=120,
    )

    events = _reward_health_events(total_steps=140)

    safe_fired = _drain_events(safe_engine, events)
    strict_fired = _drain_events(strict_engine, events)

    assert "grpo_safe_reward_health_overoptimization" in safe_fired
    assert "grpo_safe_reward_health_label_leakage" in safe_fired

    assert "grpo_strict_reward_health_overoptimization" in strict_fired
    assert "grpo_strict_reward_health_label_leakage" in strict_fired
