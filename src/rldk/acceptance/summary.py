"""Summary generation for fullscale acceptance runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableSequence, Sequence

PathLike = str | Path

ALERT_COUNT_THRESHOLD = 3
WARNING_LEVEL = 1
SEVERITY_RANK = {
    "debug": 0,
    "trace": 0,
    "info": 0,
    "notice": 0,
    "warning": WARNING_LEVEL,
    "error": 2,
    "critical": 2,
    "fatal": 3,
}


@dataclass
class AcceptanceSummary:
    """Result of evaluating fullscale acceptance gating."""

    ok: bool
    lines: list[str]

    @property
    def text(self) -> str:
        """Join the summary lines for convenience."""

        return "\n".join(self.lines)


def _load_json_lines(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def summarize_from_artifacts(root: PathLike) -> AcceptanceSummary:
    """Load acceptance artifacts from ``root`` and evaluate their summary."""

    root_path = Path(root)
    run_events = _load_json_lines(root_path / "run.jsonl")
    baseline_events = _load_json_lines(root_path / "baseline.jsonl")

    alerts_path = root_path / "monitor_alerts.jsonl"
    alert_records = _load_json_lines(alerts_path) if alerts_path.exists() else []

    monitor_report_path = root_path / "monitor_report.json"
    monitor_report = _load_json(monitor_report_path) if monitor_report_path.exists() else None

    reward_summary_path = root_path / "reward_health" / "reward_health_summary.json"
    reward_summary = (
        _load_json(reward_summary_path) if reward_summary_path.exists() else None
    )

    diff_report_path = root_path / "diff" / "diff_report.json"
    diff_report = _load_json(diff_report_path) if diff_report_path.exists() else None

    determinism_card_path = root_path / "determinism_report" / "determinism_card.json"
    determinism_card = (
        _load_json(determinism_card_path) if determinism_card_path.exists() else None
    )

    reward_card_path = root_path / "cards" / "reward_card.json"
    reward_card = _load_json(reward_card_path) if reward_card_path.exists() else None

    return summarize_fullscale_acceptance(
        run_events,
        baseline_events,
        alerts=alert_records,
        monitor_report=monitor_report,
        reward_summary=reward_summary,
        diff_report=diff_report,
        determinism_card=determinism_card,
        reward_card=reward_card,
        monitor_report_present=monitor_report_path.exists(),
        reward_summary_present=reward_summary_path.exists(),
        diff_report_present=diff_report_path.exists(),
        determinism_card_present=determinism_card_path.exists(),
        reward_card_present=reward_card_path.exists(),
    )


def summarize_fullscale_acceptance(
    run_events: Sequence[Mapping[str, object]],
    baseline_events: Sequence[Mapping[str, object]],
    *,
    alerts: Sequence[Mapping[str, object]] | None = None,
    monitor_report: Mapping[str, object] | None = None,
    reward_summary: Mapping[str, object] | None = None,
    diff_report: Mapping[str, object] | None = None,
    determinism_card: Mapping[str, object] | None = None,
    reward_card: Mapping[str, object] | None = None,
    monitor_report_present: bool | None = None,
    reward_summary_present: bool | None = None,
    diff_report_present: bool | None = None,
    determinism_card_present: bool | None = None,
    reward_card_present: bool | None = None,
) -> AcceptanceSummary:
    """Produce the acceptance summary and gating result.

    Parameters allow callers to provide in-memory structures or the loaded contents of the
    JSON/JSONL files that the fullscale acceptance script generates. ``*_present`` flags can
    be supplied to distinguish between explicit ``None`` payloads and a genuinely missing
    file. When omitted, ``None`` payloads are treated as missing artifacts.
    """

    summary_lines: MutableSequence[str] = ["# Fullscale Acceptance Summary", ""]
    status_ok = True

    summary_lines.append(f"- Training events: {len(run_events)} entries in run.jsonl")
    summary_lines.append(f"- Baseline events: {len(baseline_events)} entries in baseline.jsonl")

    if len(run_events) <= 1000:
        status_ok = False
        summary_lines.append("  - ❌ Expected >1000 events in run.jsonl")
    if len(baseline_events) <= 1000:
        status_ok = False
        summary_lines.append("  - ❌ Expected >1000 events in baseline.jsonl")

    reward_series = [
        event.get("value")
        for event in run_events
        if event.get("name") == "reward_mean" and isinstance(event.get("value"), (int, float))
    ]
    if reward_series:
        delta = reward_series[-1] - reward_series[0]
        summary_lines.append(f"- Reward mean delta: {delta:.3f}")
        if abs(delta) < 0.02:
            status_ok = False
            summary_lines.append("  - ❌ Reward mean did not change enough over run")
    else:
        status_ok = False
        summary_lines.append("- ❌ reward_mean series missing from run events")

    alert_records: Iterable[Mapping[str, object]] = alerts or []
    alert_records_list = list(alert_records)
    summary_lines.append(f"- Monitor alerts fired: {len(alert_records_list)}")
    if not alert_records_list:
        summary_lines.append("  - ℹ️ No monitor alerts were emitted during the run")
    else:
        if len(alert_records_list) > ALERT_COUNT_THRESHOLD:
            status_ok = False
            summary_lines.append(
                f"  - ❌ Alert count {len(alert_records_list)} exceeded threshold {ALERT_COUNT_THRESHOLD}"
            )
        severities = [
            str(record.get("severity", "warning")).lower() for record in alert_records_list
        ]
        high_severity = [
            severity
            for severity in severities
            if SEVERITY_RANK.get(severity, WARNING_LEVEL) >= WARNING_LEVEL
        ]
        if high_severity:
            status_ok = False
            summary_lines.append(
                "  - ❌ Monitor emitted warning-or-higher severity alerts: "
                + ", ".join(sorted(set(high_severity)))
            )

    monitor_report_available = (
        monitor_report_present if monitor_report_present is not None else monitor_report is not None
    )
    if monitor_report_available and monitor_report is not None:
        summary_lines.append(f"- Monitor report status: {monitor_report.get('status', 'unknown')}")
    else:
        status_ok = False
        summary_lines.append("- ❌ Monitor report missing")

    reward_summary_available = (
        reward_summary_present if reward_summary_present is not None else reward_summary is not None
    )
    if reward_summary_available and reward_summary is not None:
        summary_lines.append(
            f"- Reward health verdict: {reward_summary.get('overall_status', 'unknown')}"
        )
        if not reward_summary.get("passed", False):
            status_ok = False
            summary_lines.append("  - ❌ Reward health gate failed (passed flag false)")
    else:
        status_ok = False
        summary_lines.append("- ❌ Reward health summary missing")

    diff_report_available = (
        diff_report_present if diff_report_present is not None else diff_report is not None
    )
    if diff_report_available and diff_report is not None:
        summary_lines.append(
            f"- Diff verdict: {diff_report.get('summary', {}).get('verdict', 'unknown')}"
        )
    else:
        status_ok = False
        summary_lines.append("- ❌ Diff report missing")

    determinism_card_available = (
        determinism_card_present
        if determinism_card_present is not None
        else determinism_card is not None
    )
    if determinism_card_available and determinism_card is not None:
        summary_lines.append(
            f"- Determinism check passed: {bool(determinism_card.get('passed', False))}"
        )
        if not determinism_card.get("passed", False):
            status_ok = False
            summary_lines.append("  - ❌ Determinism check failed")
    else:
        status_ok = False
        summary_lines.append("- ❌ Determinism card missing")

    reward_card_available = (
        reward_card_present if reward_card_present is not None else reward_card is not None
    )
    if reward_card_available and reward_card is not None:
        summary_lines.append(
            f"- Reward card status: {'HEALTHY' if reward_card.get('passed') else 'ISSUES'}"
        )
        if not reward_card.get("passed", False):
            status_ok = False
            summary_lines.append("  - ❌ Reward card gate failed (passed flag false)")
    else:
        status_ok = False
        summary_lines.append("- ❌ Reward card missing")

    summary_lines.append("")
    summary_lines.append("## Overall Result")
    summary_lines.append("PASS" if status_ok else "FAIL")

    return AcceptanceSummary(ok=status_ok, lines=list(summary_lines))
