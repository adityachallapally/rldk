"""Helpers for discovering reward CLI command spellings."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


_PREFIX: Sequence[str] | None = None
_ENV = os.environ.copy()
_SRC_DIR = Path(__file__).resolve().parents[2]
_ENV["PYTHONPATH"] = (
    str(_SRC_DIR)
    if not _ENV.get("PYTHONPATH")
    else str(_SRC_DIR) + os.pathsep + _ENV["PYTHONPATH"]
)
os.environ["PYTHONPATH"] = _ENV["PYTHONPATH"]


def _ensure_prefix() -> Sequence[str]:
    global _PREFIX
    if _PREFIX is not None:
        return _PREFIX

    candidates = [["rldk"], [sys.executable, "-m", "rldk.cli"]]
    errors: List[str] = []
    for candidate in candidates:
        try:
            result = subprocess.run(
                [*candidate, "--help"],
                check=False,
                capture_output=True,
                text=True,
                env=_ENV,
            )
        except FileNotFoundError:
            errors.append(" ".join(candidate))
            continue
        if result.returncode == 0:
            _PREFIX = candidate
            return candidate
        errors.append(" ".join(candidate))

    message = (
        "The 'rldk' CLI is not available. Checked: "
        + ", ".join(errors)
        + ". Install the package in editable mode before running acceptance tests."
    )
    raise RuntimeError(message)


def _run_help(extra: Sequence[str]) -> str | None:
    prefix = _ensure_prefix()
    result = subprocess.run(
        [*prefix, *extra, "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=_ENV,
    )
    if result.returncode != 0:
        return None
    return "".join(filter(None, [result.stdout, result.stderr]))


def _candidate_extras() -> List[List[str]]:
    extras: List[List[str]] = []

    reward_help = _run_help(["reward"])
    if reward_help:
        if "reward-health" in reward_help:
            health_help = _run_help(["reward", "reward-health"])
            if health_help and "run" in health_help:
                extras.append(["reward", "reward-health", "run"])
            extras.append(["reward", "reward-health"])
        if "health" in reward_help:
            extras.append(["reward", "health"])
        if "reward-drift" in reward_help:
            extras.append(["reward", "reward-drift"])
        if "drift" in reward_help:
            extras.append(["reward", "drift"])

    root_help = _run_help([])
    if root_help:
        if "reward-health" in root_help:
            extras.append(["reward-health"])
        if "reward-drift" in root_help:
            extras.append(["reward-drift"])

    extras.extend(
        [
            ["reward-health"],
            ["reward", "health"],
            ["reward", "reward-health"],
            ["reward", "reward-health", "run"],
            ["reward-drift"],
            ["reward", "reward-drift"],
            ["reward", "drift"],
        ]
    )

    deduped = {}
    for extra in extras:
        deduped.setdefault(tuple(extra), list(extra))

    preference = [
        ("reward-health",),
        ("reward", "health"),
        ("reward", "reward-health"),
        ("reward", "reward-health", "run"),
        ("reward", "reward-drift"),
        ("reward", "drift"),
        ("reward-drift",),
    ]

    ordered: List[List[str]] = []
    for key in preference:
        if key in deduped:
            ordered.append(deduped.pop(key))
    ordered.extend(deduped.values())
    return ordered


def _select_command(extras: Iterable[Sequence[str]], kind: str) -> List[str]:
    prefix = _ensure_prefix()
    extras_list = list(extras)
    tried = []
    for extra in extras_list:
        help_text = _run_help(extra)
        tried.append(" ".join([*prefix, *extra]))
        if help_text is not None:
            return [*prefix, *extra]
    if extras_list:
        # Fall back to the first candidate even if --help fails; some Typer versions
        # raise during help generation but still support the command itself.
        return [*prefix, *extras_list[0]]
    raise RuntimeError(
        f"Unable to detect an rldk {kind} command. No candidate commands were available."
    )


def detect_reward_health_cmd() -> Sequence[str]:
    """Return a runnable reward health CLI command."""

    extras = [extra for extra in _candidate_extras() if "health" in extra[-1] or "reward-health" in extra]
    return _select_command(extras, "reward health")


def detect_reward_drift_cmd() -> Sequence[str]:
    """Return a runnable reward drift CLI command."""

    extras = [extra for extra in _candidate_extras() if "drift" in extra[-1] or "reward-drift" in extra]
    return _select_command(extras, "reward drift")


__all__ = ["detect_reward_health_cmd", "detect_reward_drift_cmd"]
