#!/usr/bin/env python3
"""Simulate PPO-like metrics with optional nondeterministic noise.

This script is designed for the blog demo to showcase RLDK's determinism checker.
It produces a JSONL metrics stream compatible with RLDK by writing to the path
pointed to by the RLDK_METRICS_PATH environment variable.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np


def build_noise_sources(mode: str, seed: int):
    """Return callables that provide noise for scalar and vector metrics."""
    if mode == "deterministic":
        scalar_rng = random.Random(seed)
        vector_rng = np.random.default_rng(seed)

        def scalar_noise() -> float:
            return scalar_rng.random()

        def vector_noise(size: int) -> np.ndarray:
            return vector_rng.normal(0.0, 1.0, size)

    else:
        sys_rng = random.SystemRandom()
        # Seed NumPy from high-entropy source to avoid replica alignment
        entropy_seed = int.from_bytes(os.urandom(16), "big") % (2**32)
        vector_rng = np.random.default_rng(entropy_seed)

        def scalar_noise() -> float:
            return sys_rng.random()

        def vector_noise(size: int) -> np.ndarray:
            return vector_rng.normal(0.0, 1.0, size)

    return scalar_noise, vector_noise


def simulate_metrics(mode: str, steps: int, sleep: float) -> list[dict]:
    """Generate synthetic PPO metrics."""
    scalar_noise, vector_noise = build_noise_sources(mode, seed=1234)
    metrics = []

    for step in range(steps):
        # Base trajectories mimic a stabilizing PPO run
        kl_base = 0.04 + 0.01 * math.sin(step / 30)
        loss_base = 0.8 * math.exp(-step / 120)
        reward_trend = 1.0 - math.exp(-step / 180)

        # Inject structured noise so the traces look organic
        kl = kl_base + 0.004 * scalar_noise()
        policy_loss = loss_base + 0.02 * (scalar_noise() - 0.5)
        reward_mean = reward_trend + 0.01 * (scalar_noise() - 0.5)

        grad_samples = vector_noise(64)
        policy_grad_norm = float(np.linalg.norm(grad_samples) / math.sqrt(64))

        metrics.append(
            {
                "step": step,
                "kl": kl,
                "policy_loss": policy_loss,
                "reward_mean": reward_mean,
                "policy_grad_norm": policy_grad_norm,
            }
        )

        time.sleep(sleep)

    return metrics


def write_metrics(metrics: list[dict], path: Path) -> None:
    """Write metrics to JSONL format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for record in metrics:
            fp.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PPO-style metrics for determinism demo")
    parser.add_argument("--mode", choices=["deterministic", "nondet"], default="deterministic")
    parser.add_argument("--steps", type=int, default=240, help="Number of training steps to simulate")
    parser.add_argument("--sleep", type=float, default=0.02, help="Pause (s) between steps to emulate work")
    args = parser.parse_args()

    metrics_path = Path(os.environ.get("RLDK_METRICS_PATH", "artifacts/blog_determinism_metrics.jsonl"))
    metrics = simulate_metrics(args.mode, args.steps, args.sleep)
    write_metrics(metrics, metrics_path)

    summary = {
        "mode": args.mode,
        "steps": len(metrics),
        "metrics_path": str(metrics_path.resolve()),
        "kl_peak": max(m["kl"] for m in metrics),
        "reward_final": metrics[-1]["reward_mean"],
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
