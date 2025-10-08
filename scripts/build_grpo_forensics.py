#!/usr/bin/env python3
"""Generate GRPO forensic reports and detector comparison tables."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rldk.io.naming_conventions import FileNamingConventions  # noqa: E402
REPORTS_DIR = REPO_ROOT / "rldk_reports"
BLOG_TABLE_PATH = REPO_ROOT / "docs" / "assets" / "blog_catch_failures" / "detector_matrix.md"
CSV_OUTPUT_PATH = REPORTS_DIR / "detector_matrix.csv"


def run_cli(args: Iterable[str]) -> None:
    """Run an rldk CLI command."""

    args = list(args)
    print("Running:", " ".join(args))
    env = os.environ.copy()
    pythonpath = str(REPO_ROOT / "src")
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    subprocess.run(args, check=True, cwd=REPO_ROOT, env=env)


def copy_report(src: Path, dest: Path) -> None:
    """Copy a report, replacing the destination if needed."""

    if not src.exists():
        raise FileNotFoundError(f"Expected report not found: {src}")

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        dest.unlink()

    shutil.copy2(src, dest)


def move_report(src: Path, dest: Path) -> None:
    """Move a report, replacing the destination if needed."""

    if not src.exists():
        raise FileNotFoundError(f"Expected report not found: {src}")

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        dest.unlink()

    src.rename(dest)


def collect_rules(report_path: Path) -> Set[str]:
    """Extract fired rule identifiers from a scan report."""

    with open(report_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rules = payload.get("rules_fired", [])
    return {rule.get("rule", "") for rule in rules if rule.get("rule")}


def write_detector_matrix(ppo_rules: Set[str], grpo_rules: Set[str]) -> None:
    """Write detector comparison outputs to CSV and markdown files."""

    detectors = sorted(ppo_rules.union(grpo_rules))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_OUTPUT_PATH, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["detector_id", "fires_on_ppo", "fires_on_grpo"])
        for detector in detectors:
            writer.writerow(
                [
                    detector,
                    "yes" if detector in ppo_rules else "no",
                    "yes" if detector in grpo_rules else "no",
                ]
            )

    BLOG_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BLOG_TABLE_PATH, "w", encoding="utf-8") as handle:
        handle.write("| Detector id | Fires on PPO? | Fires on GRPO? |\n")
        handle.write("|-------------|----------------|-----------------|\n")
        for detector in detectors:
            handle.write(
                f"| {detector} | {'Yes' if detector in ppo_rules else 'No'} | "
                f"{'Yes' if detector in grpo_rules else 'No'} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ppo-log",
        default="test_artifacts/logs_doctored_kl_spike/training.jsonl",
        help="Path to PPO log file for comparison",
    )
    parser.add_argument(
        "--grpo-run",
        default="test_artifacts/logs_grpo/seed_1",
        help="Path to GRPO run directory",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to invoke the RLDK CLI",
    )

    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ppo_target = REPO_ROOT / args.ppo_log
    grpo_target = REPO_ROOT / args.grpo_run

    scan_filename = FileNamingConventions.get_filename("ppo_scan")
    card_filename = FileNamingConventions.get_filename("determinism_card")
    scan_path = REPORTS_DIR / scan_filename
    card_path = REPORTS_DIR / card_filename

    if not ppo_target.exists():
        raise FileNotFoundError(f"PPO log not found at {ppo_target}")
    if not grpo_target.exists():
        raise FileNotFoundError(f"GRPO run not found at {grpo_target}")

    scan_backup = scan_path.read_bytes() if scan_path.exists() else None
    card_backup = card_path.read_bytes() if card_path.exists() else None

    # Run PPO log scan and rename report
    run_cli([args.python, "-m", "rldk.cli", "forensics", "log-scan", str(ppo_target)])
    copy_report(scan_path, REPORTS_DIR / "ppo_log_scan.json")

    # Run GRPO log scan and rename report
    run_cli([args.python, "-m", "rldk.cli", "forensics", "log-scan", str(grpo_target)])
    copy_report(scan_path, REPORTS_DIR / "grpo_log_scan_seed1.json")

    # Run GRPO doctor diagnostics and rename outputs
    run_cli([args.python, "-m", "rldk.cli", "forensics", "doctor", str(grpo_target)])
    copy_report(scan_path, REPORTS_DIR / "grpo_doctor_log_scan_seed1.json")
    copy_report(card_path, REPORTS_DIR / "grpo_determinism_card_seed1.json")
    lock_path = REPO_ROOT / "rldk.lock"
    if lock_path.exists():
        move_report(lock_path, REPORTS_DIR / "grpo_seed1.lock")

    # Load reports and build comparison matrix
    ppo_rules = collect_rules(REPORTS_DIR / "ppo_log_scan.json")
    grpo_rules = collect_rules(REPORTS_DIR / "grpo_log_scan_seed1.json")

    write_detector_matrix(ppo_rules, grpo_rules)
    print(f"Detector matrix written to {CSV_OUTPUT_PATH} and {BLOG_TABLE_PATH}")

    # Restore default reports for downstream docs/tests
    if scan_backup is not None:
        scan_path.write_bytes(scan_backup)
    else:
        copy_report(REPORTS_DIR / "ppo_log_scan.json", scan_path)

    if card_backup is not None:
        card_path.write_bytes(card_backup)


if __name__ == "__main__":
    main()
