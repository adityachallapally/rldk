#!/usr/bin/env python3
"""Assemble reports for the RLDK stability micro-benchmark."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

ENV = os.environ.copy()
existing_pythonpath = ENV.get("PYTHONPATH")
env_path = str(SRC_PATH)
if existing_pythonpath:
    ENV["PYTHONPATH"] = f"{env_path}:{existing_pythonpath}"
else:
    ENV["PYTHONPATH"] = env_path

from rldk.diff import first_divergence
from rldk.ingest import normalize_training_metrics_source


RULE_CATEGORY_MAP = {
    "ppo_high_kl_guard": "kl_spike",
    "grpo_safe_kl_spike": "kl_spike",
    "ppo_reward_freefall": "reward_instability",
    "grpo_safe_reward_saturation": "reward_instability",
    "ppo_grad_norm_spike": "gradient_spike",
    "grpo_safe_advantage_collapse": "advantage_collapse",
    "grpo_safe_acceptance_swings": "acceptance_swing",
    "grpo_safe_entropy_floor": "entropy_floor",
    "ppo_kl_drift_detection": "kl_drift_monitor",
}


@dataclass
class RunMetadata:
    algorithm: str
    model: str
    task: str
    seed: int
    steps: int
    run_dir: Path
    run_name: str
    dataset_path: Path
    run_file: Path

    @classmethod
    def from_path(cls, metadata_path: Path) -> "RunMetadata":
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        algorithm = payload["algorithm"]
        model = payload["model"]
        task = payload["task"]
        seed = int(payload["seed"])
        steps = int(payload.get("steps", 0))
        run_dir = metadata_path.parent
        dataset = run_dir / "eval_dataset.jsonl"
        run_file_value = payload.get("run_file")
        run_file = Path(run_file_value) if run_file_value else run_dir / "run.jsonl"
        run_name = slugify(algorithm, task, model, f"seed{seed}")
        return cls(
            algorithm=algorithm,
            model=model,
            task=task,
            seed=seed,
            steps=steps,
            run_dir=run_dir,
            run_name=run_name,
            dataset_path=dataset,
            run_file=run_file,
        )


def slugify(*parts: str) -> str:
    normalized: List[str] = []
    for part in parts:
        if not part:
            continue
        normalized.append(
            part.replace("/", "_").replace(" ", "-").replace("..", ".")
        )
    return "_".join(normalized)


def run_cli(args: Sequence[str], *, cwd: Optional[Path] = None) -> None:
    print(f"➡️  Running: {' '.join(shlex.quote(str(arg)) for arg in args)}")
    subprocess.run(args, check=True, cwd=cwd, env=ENV)


def discover_runs(base_dir: Path) -> List[RunMetadata]:
    runs: List[RunMetadata] = []
    for metadata_path in sorted(base_dir.rglob("metadata.json")):
        if "cards" in metadata_path.parts or "determinism" in metadata_path.parts:
            continue
        runs.append(RunMetadata.from_path(metadata_path))
    return runs


def aggregate_alerts(base_dir: Path, runs: Sequence[RunMetadata]) -> Tuple[Dict[str, List[dict]], Dict[str, Set[str]]]:
    alerts_by_run: Dict[str, List[dict]] = {}
    categories_by_run: Dict[str, Set[str]] = {}
    aggregated_path = base_dir / "alerts.jsonl"
    with aggregated_path.open("w", encoding="utf-8") as aggregate_handle:
        for run in runs:
            alerts_path = run.run_dir / "alerts.jsonl"
            alerts: List[dict] = []
            categories: Set[str] = set()
            if alerts_path.exists():
                for line in alerts_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    record["run"] = run.run_name
                    alerts.append(record)
                    rule_id = record.get("rule_id")
                    if isinstance(rule_id, str):
                        categories.add(RULE_CATEGORY_MAP.get(rule_id, rule_id))
                    aggregate_handle.write(json.dumps(record) + "\n")
            alerts_by_run[run.run_name] = alerts
            categories_by_run[run.run_name] = categories
    return alerts_by_run, categories_by_run


def aggregate_monitor_reports(base_dir: Path, runs: Sequence[RunMetadata]) -> None:
    report_payload: Dict[str, dict] = {}
    for run in runs:
        report_path = run.run_dir / "report.json"
        if report_path.exists():
            try:
                report_payload[run.run_name] = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    if report_payload:
        with (base_dir / "report.json").open("w", encoding="utf-8") as handle:
            json.dump(report_payload, handle, indent=2)


def run_kl_drift_cards(base_dir: Path, runs: Sequence[RunMetadata]) -> Dict[str, dict]:
    cards_dir = base_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    drift_results: Dict[str, dict] = {}
    for run in runs:
        output_dir = cards_dir / run.run_name / "kl_drift"
        output_dir.mkdir(parents=True, exist_ok=True)
        source_path = run.run_file if run.run_file.exists() else run.run_dir
        cmd = [
            sys.executable,
            "-m",
            "rldk.cli",
            "forensics",
            "kl-drift",
            str(source_path),
            "--output-dir",
            str(output_dir),
        ]
        if source_path.is_file() and source_path.suffix.lower() == ".jsonl":
            cmd.extend(["--kl-col", "kl", "--kl-coef-col", "kl_coef"])
        try:
            run_cli(cmd)
        except subprocess.CalledProcessError as exc:
            print(f"⚠️  KL drift analysis failed for {run.run_name}: {exc}")
            continue
        card_candidates = sorted(output_dir.glob("*.json"))
        payload = {}
        for candidate in card_candidates:
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "analysis" in data:
                payload = data
                break
        if payload:
            drift_results[run.run_name] = payload
    return drift_results


def generate_reward_cards(base_dir: Path, runs: Sequence[RunMetadata]) -> Dict[str, dict]:
    cards_dir = base_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, dict] = {}
    for run in runs:
        output_dir = cards_dir / run.run_name / "reward"
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "rldk.cli",
            "card",
            "reward",
            str(run.run_dir),
            "--output-dir",
            str(output_dir),
        ]
        try:
            run_cli(cmd)
        except subprocess.CalledProcessError as exc:
            print(f"⚠️  Reward card generation failed for {run.run_name}: {exc}")
            continue
        json_path = output_dir / "reward_card.json"
        if json_path.exists():
            try:
                results[run.run_name] = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return results


def generate_determinism_cards(base_dir: Path, runs: Sequence[RunMetadata]) -> Dict[str, dict]:
    cards_dir = base_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, dict] = {}
    for run in runs:
        output_dir = cards_dir / run.run_name / "determinism"
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "rldk.cli",
            "card",
            "determinism",
            str(run.run_dir),
            "--output-dir",
            str(output_dir),
        ]
        try:
            run_cli(cmd)
        except subprocess.CalledProcessError as exc:
            print(f"⚠️  Determinism card generation failed for {run.run_name}: {exc}")
            continue
        json_path = output_dir / "determinism_card.json"
        if json_path.exists():
            try:
                results[run.run_name] = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return results


def generate_drift_cards(
    base_dir: Path,
    pairs: Dict[Tuple[str, str, int], Dict[str, RunMetadata]],
) -> Dict[str, dict]:
    cards_dir = base_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, dict] = {}
    for key, pair in pairs.items():
        if "ppo" not in pair or "grpo" not in pair:
            continue
        tag = slugify("drift_card", key[0], key[1], f"seed{key[2]}")
        output_dir = cards_dir / tag / "drift"
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "rldk.cli",
            "card",
            "drift",
            str(pair["ppo"].run_dir),
            str(pair["grpo"].run_dir),
            "--output-dir",
            str(output_dir),
        ]
        try:
            run_cli(cmd)
        except subprocess.CalledProcessError as exc:
            print(f"⚠️  Drift card generation failed for {tag}: {exc}")
            continue
        json_path = output_dir / "drift_card.json"
        if json_path.exists():
            try:
                results[tag] = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return results


def run_compare_runs(base_dir: Path, pairs: Dict[Tuple[str, str, int], Dict[str, RunMetadata]]) -> Dict[str, dict]:
    compare_dir = base_dir / "cards"
    compare_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, dict] = {}
    for key, pair in pairs.items():
        if "ppo" not in pair or "grpo" not in pair:
            continue
        tag = slugify("compare", key[0], key[1], f"seed{key[2]}")
        workspace_dir = compare_dir / tag / "compare_runs"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "rldk.cli",
            "forensics",
            "compare-runs",
            str(pair["ppo"].run_dir),
            str(pair["grpo"].run_dir),
        ]
        try:
            run_cli(cmd, cwd=workspace_dir)
        except subprocess.CalledProcessError as exc:
            print(f"⚠️  Run comparison failed for {tag}: {exc}")
            continue
        comparison_path = workspace_dir / "rldk_reports" / "run_comparison.json"
        if comparison_path.exists():
            dest = workspace_dir / f"{tag}_run_comparison.json"
            dest.write_text(comparison_path.read_text(encoding="utf-8"), encoding="utf-8")
            try:
                results[tag] = json.loads(dest.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            try:
                comparison_path.unlink()
            except OSError:
                pass
    return results


def run_evaluations(runs: Sequence[RunMetadata]) -> Dict[str, dict]:
    eval_results: Dict[str, dict] = {}
    for run in runs:
        if not run.dataset_path.exists():
            continue
        output_file = run.run_dir / "eval_results.json"
        cmd = [
            sys.executable,
            "-m",
            "rldk.cli",
            "evals",
            "evaluate",
            str(run.dataset_path),
            "--suite",
            "comprehensive",
            "--output",
            str(output_file),
            "--min-samples",
            "0",
        ]
        try:
            run_cli(cmd)
        except subprocess.CalledProcessError as exc:
            print(f"⚠️  Evaluation failed for {run.run_name}: {exc}")
        if output_file.exists():
            try:
                eval_results[run.run_name] = json.loads(output_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return eval_results


def run_determinism_checks(base_dir: Path, runs: Sequence[RunMetadata]) -> Dict[str, dict]:
    by_algo_task_model_seed: Dict[Tuple[str, str, str, int], RunMetadata] = {}
    for run in runs:
        key = (run.algorithm, run.task, run.model, run.seed)
        by_algo_task_model_seed.setdefault(key, run)
    results: Dict[str, dict] = {}
    for (algorithm, task, model, seed), run in sorted(by_algo_task_model_seed.items()):
        output_dir = base_dir / "determinism" / slugify(algorithm, task, model, f"seed{seed}")
        output_dir.mkdir(parents=True, exist_ok=True)
        train_script = REPO_ROOT / "benchmarks" / "stability_micro" / f"{algorithm}_train.py"
        command = shlex.join(
            [
                sys.executable,
                str(train_script),
                "--model",
                model,
                "--task",
                task,
                "--seed",
                str(seed),
            ]
        )
        cmd = [
            sys.executable,
            "-m",
            "rldk.cli",
            "check-determinism",
            "--cmd",
            command,
            "--compare",
            "kl,reward",
            "--replicas",
            "3",
            "--stride",
            "50",
            "--output-dir",
            str(output_dir),
        ]
        try:
            run_cli(cmd)
        except subprocess.CalledProcessError as exc:
            print(f"⚠️  Determinism check failed for {algorithm}_{task}: {exc}")
            continue
        card_path = output_dir / "determinism_card.json"
        if card_path.exists():
            try:
                key_name = slugify(algorithm, task, model, f"seed{seed}")
                results[key_name] = json.loads(card_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return results


def run_diff_and_divergence(base_dir: Path, pairs: Dict[Tuple[str, str, int], Dict[str, RunMetadata]]) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    diff_reports: Dict[str, dict] = {}
    divergence_reports: Dict[str, dict] = {}
    for key, pair in pairs.items():
        if "ppo" not in pair or "grpo" not in pair:
            continue
        tag = slugify("diff", key[0], key[1], f"seed{key[2]}")
        diff_dir = base_dir / "diffs" / tag
        diff_dir.mkdir(parents=True, exist_ok=True)
        run_a_path = pair["ppo"].run_file if pair["ppo"].run_file.exists() else pair["ppo"].run_dir
        run_b_path = pair["grpo"].run_file if pair["grpo"].run_file.exists() else pair["grpo"].run_dir
        cmd = [
            sys.executable,
            "-m",
            "rldk.cli",
            "diff",
            "--a",
            str(run_a_path),
            "--b",
            str(run_b_path),
            "--signals",
            "kl",
            "--signals",
            "reward",
            "--signals",
            "entropy",
            "--output-dir",
            str(diff_dir),
        ]
        diff_succeeded = True
        try:
            run_cli(cmd)
        except subprocess.CalledProcessError as exc:
            print(f"⚠️  Diff command failed for {tag}: {exc}")
            diff_succeeded = False
        report_path = diff_dir / "diff_report.json"
        if diff_succeeded and report_path.exists():
            try:
                diff_reports[tag] = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        if not diff_succeeded:
            continue
        try:
            df_ppo = normalize_training_metrics_source(str(run_a_path))
            df_grpo = normalize_training_metrics_source(str(run_b_path))
        except Exception as exc:
            print(f"⚠️  Failed to normalize runs for {tag}: {exc}")
            continue
        try:
            report = first_divergence(
                df_ppo,
                df_grpo,
                signals=["kl", "reward", "entropy"],
                output_dir=str(diff_dir),
            )
        except Exception as exc:
            print(f"⚠️  First divergence analysis failed for {tag}: {exc}")
            continue
        divergence_reports[tag] = {
            "diverged": report.diverged,
            "first_step": report.first_step,
            "signals": report.tripped_signals,
            "notes": report.notes,
            "report_path": report.report_path,
            "events_csv_path": report.events_csv_path,
        }
    if divergence_reports:
        with (base_dir / "first_divergence.json").open("w", encoding="utf-8") as handle:
            json.dump(divergence_reports, handle, indent=2)
    return diff_reports, divergence_reports


def summarize_shared_detectors(
    pairs: Dict[Tuple[str, str, int], Dict[str, RunMetadata]],
    categories: Dict[str, Set[str]],
) -> Dict[str, List[str]]:
    summary: Dict[str, List[str]] = {}
    for key, pair in pairs.items():
        if "ppo" not in pair or "grpo" not in pair:
            continue
        ppo_name = pair["ppo"].run_name
        grpo_name = pair["grpo"].run_name
        shared = sorted(categories.get(ppo_name, set()) & categories.get(grpo_name, set()))
        summary[slugify(key[0], key[1], f"seed{key[2]}")] = shared
    return summary


def build_summary_markdown(
    base_dir: Path,
    runs: Sequence[RunMetadata],
    alert_categories: Dict[str, Set[str]],
    shared_detectors: Dict[str, List[str]],
    kl_drift: Dict[str, dict],
    reward_cards: Dict[str, dict],
    determinism_cards: Dict[str, dict],
    eval_results: Dict[str, dict],
    determinism_checks: Dict[str, dict],
    drift_cards: Dict[str, dict],
    comparison_reports: Dict[str, dict],
    diff_reports: Dict[str, dict],
    divergence_reports: Dict[str, dict],
) -> str:
    lines: List[str] = []
    lines.append(f"# RLDK Stability Micro-Benchmark ({base_dir.name})")
    lines.append("")
    lines.append("## Overview")
    lines.append(
        f"- Total runs: **{len(runs)}** across algorithms {sorted({r.algorithm for r in runs})}"
    )
    lines.append(f"- Tasks: {', '.join(sorted({r.task for r in runs}))}")
    lines.append(f"- Models: {', '.join(sorted({r.model for r in runs}))}")
    lines.append("")

    lines.append("## Detector activations")
    for run in runs:
        categories = ", ".join(sorted(alert_categories.get(run.run_name, set()))) or "none"
        lines.append(f"- **{run.run_name}** → {categories}")
    lines.append("")

    lines.append("### Shared detectors by task/model/seed")
    for key, shared in sorted(shared_detectors.items()):
        descriptor = key.replace("_seed", " seed ")
        if shared:
            lines.append(f"- {descriptor}: {', '.join(shared)}")
        else:
            lines.append(f"- {descriptor}: _no shared monitor alerts_")
    lines.append("")

    if kl_drift:
        lines.append("## KL drift forensics")
        for run_name, payload in kl_drift.items():
            analysis = payload.get("analysis", {})
            detected = analysis.get("detected", False)
            score = analysis.get("score")
            lines.append(
                f"- {run_name}: {'detected' if detected else 'stable'}"
                + (f" (score={score:.3f})" if isinstance(score, (int, float)) else "")
            )
        lines.append("")

    if reward_cards:
        lines.append("## Reward card health")
        for run_name, card in reward_cards.items():
            status = "HEALTHY" if card.get("passed") else "ISSUES"
            calibration = card.get("calibration_score")
            calibration_str = (
                f"{float(calibration):.2f}" if isinstance(calibration, (int, float)) else "n/a"
            )
            drift = "yes" if card.get("drift_detected") else "no"
            lines.append(
                f"- {run_name}: status={status}, calibration={calibration_str}, drift={drift}"
            )
        lines.append("")

    if determinism_cards:
        lines.append("## Determinism cards")
        for run_name, card in determinism_cards.items():
            status = "PASS" if card.get("passed") else "FAIL"
            replicas = card.get("replicas")
            hints = len(card.get("nondeterminism_hints", []))
            lines.append(
                f"- {run_name}: status={status}, replicas={replicas}, hints={hints}"
            )
        lines.append("")

    if eval_results:
        lines.append("## Evaluation suite highlights")
        for run_name, payload in eval_results.items():
            summary = payload.get("summary", {})
            overall = summary.get("overall_score")
            available = summary.get("available_fraction")
            if isinstance(overall, (int, float)):
                overall_str = f"{overall:.3f}"
            else:
                overall_str = "n/a"
            if isinstance(available, (int, float)):
                available_str = f"{available:.2f}"
            else:
                available_str = "n/a"
            lines.append(
                f"- {run_name}: overall_score={overall_str}, available_fraction={available_str}"
            )
        lines.append("")

    if determinism_checks:
        lines.append("## Determinism checks")
        for key, report in determinism_checks.items():
            status = "pass" if report.get("passed") else "mismatch"
            mismatch_count = len(report.get("mismatches", []))
            lines.append(f"- {key}: {status} (mismatches={mismatch_count})")
        lines.append("")

    if drift_cards:
        lines.append("## Drift cards")
        for key, card in drift_cards.items():
            diverged = "yes" if card.get("diverged") else "no"
            first_step = card.get("first_step")
            signals = ", ".join(card.get("tripped_signals", []))
            lines.append(
                f"- {key}: diverged={diverged}"
                + (f", first_step={first_step}" if first_step else "")
                + (f", signals={signals}" if signals else "")
            )
        lines.append("")

    if comparison_reports:
        lines.append("## Run comparison anomalies")
        for key, report in comparison_reports.items():
            run_a = len(report.get("run_a", {}).get("anomalies", []))
            run_b = len(report.get("run_b", {}).get("anomalies", []))
            earliest = report.get("earliest_divergent_step")
            lines.append(
                f"- {key}: run_a_anomalies={run_a}, run_b_anomalies={run_b}"
                + (f", earliest_divergent_step={earliest}" if earliest else "")
            )
        lines.append("")

    if diff_reports:
        lines.append("## Run diffs and first divergence")
        for key, report in diff_reports.items():
            summary = report.get("summary", {})
            verdict = summary.get("verdict", "unknown")
            max_delta = summary.get("max_abs_delta")
            divergence = divergence_reports.get(key, {})
            diverged = divergence.get("diverged")
            first_step = divergence.get("first_step")
            lines.append(
                f"- {key}: diff verdict={verdict}"
                + (f", max_abs_delta={max_delta}" if max_delta is not None else "")
                + (f"; divergence={'yes' if diverged else 'no'}"
                   + (f" at step {first_step}" if first_step else "")
                   if divergence else "")
            )
        lines.append("")

    lines.append("## Key artifacts")
    lines.append("- Aggregated alerts: `alerts.jsonl`")
    lines.append("- Aggregated monitor report: `report.json`")
    lines.append("- Cards directory: `cards/`")
    lines.append("- Diff reports: `diffs/`")
    lines.append("- Determinism diagnostics: `determinism/`")
    lines.append("- First divergence summary: `first_divergence.json`")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the stability micro-benchmark report")
    parser.add_argument("--base-dir", type=Path, required=True, help="Benchmark output directory")
    args = parser.parse_args(argv)

    base_dir = args.base_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(base_dir)
    if not runs:
        print("⚠️  No runs discovered; skipping report generation.")
        return

    _, categories = aggregate_alerts(base_dir, runs)
    aggregate_monitor_reports(base_dir, runs)
    kl_drift = run_kl_drift_cards(base_dir, runs)
    reward_cards = generate_reward_cards(base_dir, runs)
    determinism_cards = generate_determinism_cards(base_dir, runs)

    pairs: Dict[Tuple[str, str, int], Dict[str, RunMetadata]] = defaultdict(dict)
    for run in runs:
        pairs[(run.task, run.model, run.seed)][run.algorithm] = run

    drift_cards = generate_drift_cards(base_dir, pairs)
    comparison_reports = run_compare_runs(base_dir, pairs)
    eval_results = run_evaluations(runs)
    determinism_checks = run_determinism_checks(base_dir, runs)
    diff_reports, divergence_reports = run_diff_and_divergence(base_dir, pairs)
    shared_detectors = summarize_shared_detectors(pairs, categories)

    summary_md = build_summary_markdown(
        base_dir,
        runs,
        categories,
        shared_detectors,
        kl_drift,
        reward_cards,
        determinism_cards,
        eval_results,
        determinism_checks,
        drift_cards,
        comparison_reports,
        diff_reports,
        divergence_reports,
    )
    (base_dir / "SUMMARY.md").write_text(summary_md, encoding="utf-8")
    print(f"📄 Summary written to {base_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
