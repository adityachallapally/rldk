"""One-shot acceptance runner for Phase A/B coverage."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rldk.ingest.stream_normalizer import stream_jsonl_to_dataframe
from rldk.ingest.training_metrics_normalizer import standardize_training_metrics
from rldk.io.event_schema import dataframe_to_events, events_to_dataframe
from rldk.reward.api import HealthAnalysisResult, reward_health
from rldk.testing.cli_detect import (
    detect_reward_drift_cmd,
    detect_reward_health_cmd,
)

FIXTURES = ROOT / "tests" / "fixtures" / "phase_ab"
ACCEPTANCE_DIR = ROOT / "artifacts" / "phase_ab_acceptance"
ACCEPTANCE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = ACCEPTANCE_DIR / "summary.json"

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str | None = None


def _require_column(df: pd.DataFrame, options: Sequence[str]) -> str:
    for name in options:
        if name in df.columns:
            return name
    raise AssertionError(f"Expected one of {options}, got columns {list(df.columns)}")


def _run_subprocess(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def _find_numeric(payload: dict, candidates: Sequence[str]) -> float:
    for key in candidates:
        value = payload.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
    raise AssertionError(f"No numeric field found in {candidates}; payload keys: {sorted(payload)}")


def _extract_health(result: HealthAnalysisResult) -> float:
    value = float(result.report.calibration_score)
    if not np.isfinite(value):
        raise AssertionError("Calibration score is not finite")
    return value


class PhaseABAcceptance:
    def __init__(self) -> None:
        self.health_cmd = detect_reward_health_cmd()
        self.drift_cmd = detect_reward_drift_cmd()

    def run(self) -> List[CheckResult]:
        checks: list[tuple[str, Callable[[], None]]] = [
            ("Stream JSONL to table pivot", self.check_stream_pivot),
            ("Schema standardizer coercion", self.check_standardizer),
            ("Round trip normalized events", self.check_round_trip),
            ("Reward health CLI on JSONL", self.check_reward_health_jsonl),
            ("Reward health CLI on CSV", self.check_reward_health_csv),
            ("Reward health CLI missing reward", self.check_reward_health_missing),
            ("Reward drift CLI score mode", self.check_reward_drift_scores),
            ("Reward health API accepts DataFrame", self.check_api_dataframe),
            ("Reward health API accepts list of dicts", self.check_api_records),
            ("Reward health API accepts JSONL path", self.check_api_path),
            ("Reward health API equivalence", self.check_api_equivalence),
        ]

        results: List[CheckResult] = []
        for name, fn in checks:
            try:
                fn()
            except Exception as exc:  # pragma: no cover - exercised in CI
                results.append(CheckResult(name=name, passed=False, details=str(exc)))
            else:
                results.append(CheckResult(name=name, passed=True))
        return results

    def check_stream_pivot(self) -> None:
        df = stream_jsonl_to_dataframe(FIXTURES / "stream_small.jsonl")
        if df.empty:
            raise AssertionError("Normalized DataFrame is empty")
        if not df["step"].is_monotonic_increasing:
            raise AssertionError("Steps are not sorted ascending")
        reward_col = _require_column(df, ["reward_mean", "reward"])
        kl_col = _require_column(df, ["kl_mean", "kl"])
        if not np.issubdtype(df["step"].dtype, np.integer):
            raise AssertionError("Step column is not integer typed")
        if df[reward_col].dtype.kind not in {"f", "i"}:
            raise AssertionError("Reward column is not numeric")
        if df[kl_col].dtype.kind not in {"f", "i"}:
            raise AssertionError("KL column is not numeric")
        if all(column in {"step", reward_col, kl_col} for column in df.columns):
            raise AssertionError("Expected passthrough metrics beyond canonical columns")

    def check_standardizer(self) -> None:
        mixed = stream_jsonl_to_dataframe(FIXTURES / "stream_mixed_types.jsonl")
        reward_col = _require_column(mixed, ["reward_mean", "reward"])
        mixed[reward_col] = mixed[reward_col].astype("string")
        if "kl_mean" in mixed.columns:
            mixed["kl_mean"] = mixed["kl_mean"].astype("string")
        invalid_row = {column: None for column in mixed.columns}
        invalid_row["step"] = "bad-step"
        invalid_row[reward_col] = "0.123"
        mixed = pd.concat([mixed, pd.DataFrame([invalid_row])], ignore_index=True)

        standardized = standardize_training_metrics(mixed)
        canonical_reward = _require_column(standardized, ["reward_mean", "reward"])
        if not is_numeric_dtype(standardized[canonical_reward]):
            raise AssertionError("Reward column was not coerced to numeric")
        if "kl_mean" in standardized.columns and not is_numeric_dtype(standardized["kl_mean"]):
            raise AssertionError("KL column was not coerced to numeric")
        if len(standardized) != len(mixed) - 1:
            raise AssertionError("Invalid step row was not dropped")
        reward_mean = standardized[canonical_reward].dropna().mean()
        if not np.isfinite(reward_mean):
            raise AssertionError("Reward mean is not finite")

    def check_round_trip(self) -> None:
        table_df = pd.read_csv(FIXTURES / "table_small.csv")
        events = dataframe_to_events(table_df)
        events.append({"step": "oops", "metrics": {"reward_mean": "oops"}})
        events.append("invalid")  # type: ignore[arg-type]
        round_trip_df = events_to_dataframe(events)
        reward_col = _require_column(round_trip_df, ["reward_mean", "reward"])
        kl_col = _require_column(round_trip_df, ["kl_mean", "kl"])
        merged = round_trip_df.merge(table_df, on="step", suffixes=("_actual", "_expected"))
        if merged.empty:
            raise AssertionError("Round-trip merge is empty")
        np.testing.assert_allclose(
            merged[f"{reward_col}_actual"], merged[f"{reward_col}_expected"], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            merged[f"{kl_col}_actual"], merged[f"{kl_col}_expected"], rtol=1e-6, atol=1e-6
        )

    def _run_reward_health(self, source: Path, name: str) -> dict:
        output_dir = ACCEPTANCE_DIR / name
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [*self.health_cmd, "--run", str(source), "--output-dir", str(output_dir)]
        result = _run_subprocess(command, cwd=output_dir)
        if result.returncode != 0:
            raise AssertionError(result.stderr or result.stdout)
        summary_path = output_dir / "reward_health_summary.json"
        if not summary_path.exists():
            raise AssertionError("Reward health summary JSON not found")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        _find_numeric(
            payload,
            [
                "health",
                "health_score",
                "score",
                "overall",
                "overall_score",
                "calibration_score",
                "label_leakage_risk",
            ],
        )
        return payload

    def check_reward_health_jsonl(self) -> None:
        self._run_reward_health(FIXTURES / "stream_small.jsonl", "reward_health_jsonl")

    def check_reward_health_csv(self) -> None:
        self._run_reward_health(FIXTURES / "table_small.csv", "reward_health_csv")

    def check_reward_health_missing(self) -> None:
        out_dir = ACCEPTANCE_DIR / "reward_health_missing"
        out_dir.mkdir(parents=True, exist_ok=True)
        source_path = out_dir / "no_reward.jsonl"
        records = [
            {"time": 1.0, "step": 1, "name": "kl_mean", "value": 0.1},
            {"time": 1.5, "step": 1, "name": "loss", "value": 0.2},
        ]
        with source_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        command = [*self.health_cmd, "--run", str(source_path), "--output-dir", str(out_dir)]
        result = _run_subprocess(command, cwd=out_dir)
        if result.returncode == 0:
            raise AssertionError("Expected failure when reward column is missing")
        stderr = (result.stderr or result.stdout).lower()
        if "reward" not in stderr or ("preset" not in stderr and "field-map" not in stderr and "field map" not in stderr):
            raise AssertionError("Missing reward error message did not mention presets or field maps")

    def check_reward_drift_scores(self) -> None:
        out_dir = ACCEPTANCE_DIR / "reward_drift_scores"
        out_dir.mkdir(parents=True, exist_ok=True)
        command = [
            *self.drift_cmd,
            "--scores-a",
            str(FIXTURES / "scores_a.jsonl"),
            "--scores-b",
            str(FIXTURES / "scores_b.jsonl"),
        ]
        result = _run_subprocess(command, cwd=out_dir)
        if result.returncode != 0:
            raise AssertionError(result.stderr or result.stdout)
        report_path = out_dir / "rldk_reports" / "reward_drift.json"
        if not report_path.exists():
            raise AssertionError("Reward drift report not generated")
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        _find_numeric(payload, ["drift", "drift_magnitude", "effect_size"])

    def check_api_dataframe(self) -> None:
        df = stream_jsonl_to_dataframe(FIXTURES / "stream_small.jsonl")
        result = reward_health(df)
        _extract_health(result)
        self._write_api_result("api_dataframe", result)

    def check_api_records(self) -> None:
        df = stream_jsonl_to_dataframe(FIXTURES / "stream_small.jsonl")
        records = df.to_dict(orient="records")
        result = reward_health(records)
        _extract_health(result)
        self._write_api_result("api_records", result)

    def check_api_path(self) -> None:
        result = reward_health(FIXTURES / "stream_small.jsonl")
        _extract_health(result)
        self._write_api_result("api_path", result)

    def check_api_equivalence(self) -> None:
        df = stream_jsonl_to_dataframe(FIXTURES / "stream_small.jsonl")
        records = df.to_dict(orient="records")
        results = [reward_health(df), reward_health(records), reward_health(FIXTURES / "stream_small.jsonl")]
        scores = [_extract_health(item) for item in results]
        if max(scores) - min(scores) > 1e-9:
            raise AssertionError("Reward health scores differ across input modes")

    def _write_api_result(self, name: str, result: HealthAnalysisResult) -> None:
        out_dir = ACCEPTANCE_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "calibration_score": result.report.calibration_score,
            "label_leakage_risk": result.report.label_leakage_risk,
            "passed": result.report.passed,
        }
        (out_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    runner = PhaseABAcceptance()
    results = runner.run()

    passes = sum(result.passed for result in results)
    failures = len(results) - passes

    for result in results:
        color = GREEN if result.passed else RED
        status = "PASS" if result.passed else "FAIL"
        message = f" - {result.details}" if (result.details and not result.passed) else ""
        print(f"{color}{status:<5}{RESET} {result.name}{message}")

    summary = {
        "passed": passes,
        "failed": failures,
        "results": [
            {
                "name": result.name,
                "status": "pass" if result.passed else "fail",
                "details": result.details,
            }
            for result in results
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
