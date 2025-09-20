"""Compute guardrail quality metrics (lead time, precision, and recall)."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Iterable, Sequence


def _expand_alert_paths(patterns: Sequence[str]) -> list[Path]:
    """Expand glob patterns and explicit paths into a sorted list of files."""

    resolved: set[Path] = set()
    for pattern in patterns:
        matches = [Path(match) for match in Path().glob(pattern)]
        if matches:
            resolved.update(path for path in matches if path.is_file())
            continue
        path = Path(pattern)
        if path.is_file():
            resolved.add(path)
        else:
            raise FileNotFoundError(f"Alerts file not found: {pattern}")

    if not resolved:
        raise FileNotFoundError("No alerts files found for the provided patterns")

    return sorted(resolved)


def _load_alert_steps(path: Path, rule_ids: set[str] | None) -> set[int]:
    """Load unique alert steps from a JSONL file, optionally filtering by rule id."""

    steps: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record_raw = line.strip()
            if not record_raw:
                continue
            try:
                record = json.loads(record_raw)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if rule_ids and record.get("rule_id") not in rule_ids:
                continue
            if "step" not in record:
                raise ValueError(f"Missing 'step' field in {path}:{line_number}")
            steps.add(int(record["step"]))
    return steps


def _mean(values: Iterable[float]) -> float:
    payload = list(values)
    if not payload:
        raise ValueError("Cannot compute mean of empty values")
    return statistics.fmean(payload)


def _std(values: Iterable[float]) -> float:
    payload = list(values)
    if not payload:
        raise ValueError("Cannot compute std of empty values")
    if len(payload) == 1:
        return 0.0
    return statistics.pstdev(payload)


def compute_metrics(alert_paths: Sequence[Path], *, window_start: int, window_end: int, rule_ids: set[str] | None) -> dict:
    """Compute per-seed metrics and aggregate summary statistics."""

    if window_end < window_start:
        raise ValueError("window_end must be greater than or equal to window_start")

    window_steps = set(range(window_start, window_end + 1))
    window_size = len(window_steps)

    per_seed: list[dict] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    lead_times: list[float] = []

    for path in alert_paths:
        alert_steps = _load_alert_steps(path, rule_ids)
        true_positive_steps = alert_steps & window_steps
        false_positive_steps = alert_steps - window_steps

        total_alerts = len(alert_steps)
        precision = len(true_positive_steps) / total_alerts if total_alerts else 0.0
        recall = len(true_positive_steps) / window_size if window_size else 0.0

        lead_time = None
        if true_positive_steps:
            first_alert = min(true_positive_steps)
            lead_time = max(0, window_start - first_alert)
            lead_times.append(float(lead_time))

        per_seed.append(
            {
                "path": str(path),
                "alert_steps": sorted(alert_steps),
                "true_positive_steps": sorted(true_positive_steps),
                "false_positive_steps": sorted(false_positive_steps),
                "precision": precision,
                "recall": recall,
                "lead_time": lead_time,
            }
        )
        precision_values.append(precision)
        recall_values.append(recall)

    summary: dict[str, object] = {
        "window_start": window_start,
        "window_end": window_end,
        "window_size": window_size,
        "seed_count": len(per_seed),
        "per_seed": per_seed,
        "rule_ids": sorted(rule_ids) if rule_ids else None,
    }

    if lead_times:
        summary["lead_time_mean"] = _mean(lead_times)
        summary["lead_time_std"] = _std(lead_times) if len(lead_times) > 1 else 0.0
    else:
        summary["lead_time_mean"] = None
        summary["lead_time_std"] = None

    summary["precision_mean"] = _mean(precision_values)
    summary["precision_std"] = _std(precision_values) if len(precision_values) > 1 else 0.0
    summary["recall_mean"] = _mean(recall_values)
    summary["recall_std"] = _std(recall_values) if len(recall_values) > 1 else 0.0
    summary["seeds_with_alerts"] = len(lead_times)

    return summary


def _format_float(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.3f}"
    return f"{value:.0f}"


def write_markdown(summary: dict, path: Path, label: str) -> None:
    """Write a compact two-row Markdown table summarizing the metrics."""

    headers = [
        "Run",
        "Lead time mean",
        "Lead time std",
        "Precision mean",
        "Precision std",
        "Recall mean",
        "Recall std",
        "Seeds",
        "Seeds with alerts",
    ]

    row = [
        label,
        _format_float(summary.get("lead_time_mean")),
        _format_float(summary.get("lead_time_std")),
        _format_float(summary.get("precision_mean")),
        _format_float(summary.get("precision_std")),
        _format_float(summary.get("recall_mean")),
        _format_float(summary.get("recall_std")),
        _format_float(summary.get("seed_count")),
        _format_float(summary.get("seeds_with_alerts")),
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("|" + "---|" * len(headers) + "\n")
        handle.write("| " + " | ".join(row) + " |\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alerts",
        nargs="+",
        required=True,
        help="One or more alerts.jsonl paths or glob patterns",
    )
    parser.add_argument("--window-start", type=int, required=True)
    parser.add_argument("--window-end", type=int, required=True)
    parser.add_argument(
        "--rule-id",
        action="append",
        dest="rule_ids",
        default=None,
        help="Filter to alerts with the provided rule id (can repeat)",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--label", default="summary", help="Label for the markdown table row")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rule_ids = set(args.rule_ids) if args.rule_ids else None
    alert_paths = _expand_alert_paths(args.alerts)
    summary = compute_metrics(
        alert_paths,
        window_start=args.window_start,
        window_end=args.window_end,
        rule_ids=rule_ids,
    )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
    else:
        print(json.dumps(summary, indent=2))

    if args.output_markdown:
        write_markdown(summary, args.output_markdown, args.label)


if __name__ == "__main__":
    main()
