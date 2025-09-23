#!/usr/bin/env python3
"""Synthetic GRPO training loop for the RLDK stability micro-benchmark."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np

from rldk.emit import EventWriter

if __package__ is None or __package__ == "":
    import sys as _sys

    _sys.path.append(str(Path(__file__).resolve().parent))
    from _shared import (  # type: ignore[import-not-found]
        MODEL_BIASES,
        TrainingSample,
        _build_humaneval_output,
        _build_math_output,
        _build_math_prompt,
        _code_quality,
        _default_steps,
        _math_accuracy,
        _sample_humaneval_task,
        _write_dataset,
    )
else:  # pragma: no cover - exercised when installed as package
    from ._shared import (
        MODEL_BIASES,
        TrainingSample,
        _build_humaneval_output,
        _build_math_output,
        _build_math_prompt,
        _code_quality,
        _default_steps,
        _math_accuracy,
        _sample_humaneval_task,
        _write_dataset,
    )


@dataclass
class GRPOSample(TrainingSample):
    acceptance_rate: float
    advantage_std: float

    def to_json(self) -> Dict[str, object]:  # type: ignore[override]
        payload = super().to_json()
        payload.update(
            {
                "acceptance_rate": self.acceptance_rate,
                "advantage_std": self.advantage_std,
                "group_reward_mean": payload["reward"],
            }
        )
        return payload


def _resolve_log_path(output_dir: Optional[Path]) -> Path:
    env_path = os.environ.get("RLDK_METRICS_PATH")
    if env_path:
        path = Path(env_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    if output_dir is None:
        output_dir = Path("artifacts/grpo")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "run.jsonl"


def _generate_grpo_samples(
    task: str,
    model: str,
    seed: int,
    steps: int,
    rng: random.Random,
) -> Iterator[GRPOSample]:
    model_bias = MODEL_BIASES.get(model, 0.0) - 0.03  # GRPO baseline slightly less stable
    acceptance_center = 0.68 + model_bias * 0.2
    for step in range(1, steps + 1):
        progress = step / steps
        noise = rng.uniform(-0.008, 0.008)
        kl = 0.04 + 0.38 * math.pow(progress, 1.15) + noise
        reward_trend = 0.45 - 0.48 * progress + rng.uniform(-0.03, 0.03)
        entropy = 2.9 - 1.6 * progress + rng.uniform(-0.15, 0.1)
        grad_norm = 3.2 + 8.5 * progress + rng.uniform(-0.35, 0.35)
        kl_coef = 0.085 + 0.002 * math.cos(progress * math.pi * 2)
        acceptance_rate = acceptance_center + 0.35 * math.sin(progress * math.pi * 1.3) + rng.uniform(-0.05, 0.05)
        advantage_std = max(0.05, 0.72 - 0.82 * progress + rng.uniform(-0.05, 0.05))
        reward_saturation = 0.32 + 0.03 * math.sin(progress * math.pi * 0.25)
        reward = 0.6 * reward_trend + 0.4 * (
            reward_saturation + rng.uniform(-0.015, 0.015)
        )

        response_quality = max(0.0, min(1.0, 0.8 - 0.45 * progress + rng.uniform(-0.07, 0.07)))
        safety_score = max(0.0, min(1.0, 0.76 - 0.3 * progress + rng.uniform(-0.05, 0.05)))
        toxicity_score = max(0.0, min(1.0, 0.15 + 0.18 * progress + rng.uniform(0.0, 0.05)))
        bias_score = max(0.0, min(1.0, 0.07 + 0.18 * progress + rng.uniform(0.0, 0.05)))
        adversarial_score = max(0.0, min(1.0, 0.88 - 0.4 * progress + rng.uniform(-0.05, 0.05)))
        human_preference = max(0.0, min(1.0, 0.8 - 0.48 * progress + rng.uniform(-0.05, 0.05)))
        length_bias_score = min(1.0, 0.15 + 0.45 * progress + rng.uniform(-0.05, 0.05))
        length_corr = min(1.0, 0.1 + 0.4 * progress + rng.uniform(-0.05, 0.05))
        length_spearman = min(1.0, 0.12 + 0.38 * progress + rng.uniform(-0.05, 0.05))

        if task == "gsm8k_mini":
            prompt_info = _build_math_prompt(rng)
            accuracy = _math_accuracy(progress, model_bias, rng)
            is_correct = rng.random() < accuracy
            correction = rng.choice([-4, -2, -1, 1, 2, 4]) if not is_correct else 0
            guess = int(prompt_info["answer"]) + correction
            output_text = _build_math_output(prompt_info, guess)
            reference_answer = str(prompt_info["answer"])
            model_answer = str(guess)
            prompt_text = prompt_info["prompt"]
        else:
            task_spec = _sample_humaneval_task(step)
            quality = _code_quality(progress, model_bias, rng)
            output_text = (
                "# GRPO variant\n" + _build_humaneval_output(task_spec, quality, rng)
            )
            reference_answer = task_spec["reference"]
            model_answer = output_text
            prompt_text = task_spec["prompt"]

        yield GRPOSample(
            step=step,
            output=output_text,
            prompt=prompt_text,
            reference_answer=reference_answer,
            model_answer=model_answer,
            reward=reward,
            kl=kl,
            entropy=entropy,
            grad_norm=grad_norm,
            kl_coef=kl_coef,
            length_bias_score=length_bias_score,
            length_reward_correlation_abs=length_corr,
            length_reward_spearman_abs=length_spearman,
            response_quality=response_quality,
            safety_score=safety_score,
            toxicity_score=toxicity_score,
            bias_score=bias_score,
            adversarial_score=adversarial_score,
            human_preference=human_preference,
            model_name=model,
            algorithm="grpo",
            task=task,
            seed=seed,
            acceptance_rate=max(0.0, min(1.0, acceptance_rate)),
            advantage_std=advantage_std,
        )


def run_training(model: str, task: str, seed: int, steps: Optional[int], output_dir: Optional[Path]) -> None:
    resolved_steps = steps or _default_steps(task)
    rng = random.Random(seed + 13)
    np.random.seed(seed + 13)
    log_path = _resolve_log_path(output_dir)
    samples: List[GRPOSample] = []

    with EventWriter(log_path) as writer:
        for sample in _generate_grpo_samples(task, model, seed, resolved_steps, rng):
            samples.append(sample)
            for name, value in (
                ("kl", sample.kl),
                ("reward", sample.reward),
                ("entropy", sample.entropy),
                ("grad_norm", sample.grad_norm),
                ("kl_coef", sample.kl_coef),
                ("acceptance_rate", sample.acceptance_rate),
                ("advantage_std", sample.advantage_std),
                ("length_bias_score", sample.length_bias_score),
                (
                    "length_reward_correlation_abs",
                    sample.length_reward_correlation_abs,
                ),
                (
                    "length_reward_spearman_abs",
                    sample.length_reward_spearman_abs,
                ),
            ):
                writer.log(step=sample.step, name=name, value=value, meta={"task": task, "model": model})

    if output_dir is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "eval_dataset.jsonl"
    _write_dataset(samples, dataset_path)
    metadata = {
        "algorithm": "grpo",
        "model": model,
        "task": task,
        "seed": seed,
        "steps": resolved_steps,
        "run_dir": str(output_dir),
        "run_file": str(log_path),
        "dataset": str(dataset_path),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic GRPO training data")
    parser.add_argument("--model", required=True, help="Model identifier")
    parser.add_argument("--task", choices=["gsm8k_mini", "humaneval_lite"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=None, help="Override number of steps")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to store artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args.model, args.task, args.seed, args.steps, args.output_dir)


if __name__ == "__main__":
    main()
