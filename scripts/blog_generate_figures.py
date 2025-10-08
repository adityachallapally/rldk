#!/usr/bin/env python3
"""Generate blog-ready plots from the demo artifacts."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path("docs/assets/blog_catch_failures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_monitor_trace():
    data_path = Path("artifacts/run.jsonl")
    alerts_path = Path("artifacts/alerts.jsonl")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing monitor run file: {data_path}")

    df = pd.read_json(data_path, lines=True)
    kl = df[df["name"] == "kl"]

    alert_steps = []
    if alerts_path.exists():
        with alerts_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                alert = json.loads(line)
                if alert.get("metric") == "kl":
                    alert_steps.append((alert.get("step"), alert.get("value")))

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(kl["step"], kl["value"], label="KL divergence", color="#1f77b4")
    ax.axhline(
        0.35,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.2,
        label="KL guardrail (5 step window)",
    )

    if alert_steps:
        steps, values = zip(*alert_steps)
        ax.scatter(steps, values, color="#d62728", marker="x", s=70, label="Auto-stop alerts")

    ax.set_title("RLDK monitor caught runaway KL in real time")
    ax.set_xlabel("Step")
    ax.set_ylabel("KL value")
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)

    output = OUTPUT_DIR / "monitor_kl_trace.png"
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def plot_forensics_spike():
    log_path = Path("test_artifacts/logs_doctored_kl_spike/training.jsonl")
    if not log_path.exists():
        raise FileNotFoundError(f"Missing doctored log: {log_path}")

    df = pd.read_json(log_path, lines=True)
    kl = df[["step", "kl"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(kl["step"], kl["kl"], color="#9467bd", linewidth=1.3, label="KL trace")
    spike_mask = kl["step"].between(780, 820)
    ax.fill_between(kl["step"], kl["kl"], where=spike_mask, color="#c5b0d5", alpha=0.6, label="Spike window")

    ax.set_title("Comprehensive PPO log scan highlights KL spikes")
    ax.set_xlabel("Step")
    ax.set_ylabel("KL value")
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)

    output = OUTPUT_DIR / "forensics_kl_spike.png"
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def plot_determinism_mismatches():
    nondet_card = OUTPUT_DIR / "determinism_nondet" / "determinism_card.json"
    det_card = OUTPUT_DIR / "determinism_det" / "determinism_card.json"
    if not nondet_card.exists() or not det_card.exists():
        raise FileNotFoundError("Determinism cards missing; run the determinism demo first.")

    with nondet_card.open("r", encoding="utf-8") as fp:
        nondet = json.load(fp)

    diff_by_step = {}
    for mismatch in nondet.get("mismatches", []):
        step = mismatch.get("step")
        diff = mismatch.get("difference", 0.0)
        diff_by_step[step] = max(diff_by_step.get(step, 0.0), diff)

    if not diff_by_step:
        raise ValueError("No mismatches recorded in nondeterministic run")

    steps = sorted(diff_by_step.keys())
    diffs = [diff_by_step[s] for s in steps]

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, diffs, color="#e377c2", linewidth=1.2, label="Non-deterministic run")
    ax.axhline(0, color="#2ca02c", linewidth=1.2, label="Deterministic run")

    ax.set_title("Determinism checker pinpoints replica drift")
    ax.set_xlabel("Step")
    ax.set_ylabel("Max |Δ reward_mean| across replicas")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")

    output = OUTPUT_DIR / "determinism_mismatch.png"
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def main():
    outputs = {
        "monitor": str(plot_monitor_trace()),
        "forensics": str(plot_forensics_spike()),
        "determinism": str(plot_determinism_mismatches()),
    }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
