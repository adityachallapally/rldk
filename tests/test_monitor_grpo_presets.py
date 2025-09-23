"""Tests for GRPO monitor presets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

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

    assert "grpo_safe_kl_spike" in fired
    assert "grpo_safe_kl_coef_stall" in fired
    assert "grpo_safe_reward_saturation" in fired
    assert "grpo_safe_diversity_pass_floor" in fired
    assert "grpo_safe_diversity_distinct_collapse" in fired
    assert "grpo_safe_self_bleu_spike" in fired
    assert "grpo_safe_output_entropy_floor" in fired


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

    assert "grpo_strict_kl_spike" in fired
    assert "grpo_strict_kl_coef_stall" in fired
    assert "grpo_strict_reward_saturation" in fired
    assert expected_rule in fired


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
