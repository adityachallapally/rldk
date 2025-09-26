"""Run a CPU-friendly synthetic GRPO training loop that logs metrics via EventWriter."""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import yaml
import numpy as np

from rldk.emit import EventWriter

DEFAULT_MODEL_NAME = "sshleifer/tiny-gpt2"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "grpo_tiny.yaml"


@dataclass
class TinyGRPORunSettings:
    """Container for metadata encoded in ``configs/grpo_tiny.yaml``."""

    model: str
    dataset_seed: int
    steps: int
    logging_interval: int
    log_path: Path
    grpo_config: Dict[str, Any]


@dataclass
class GRPOSample:
    """Training sample for synthetic GRPO data generation."""
    step: int
    output: str
    prompt: str
    reference_answer: str
    model_answer: str
    reward: float
    kl: float
    entropy: float
    grad_norm: float
    kl_coef: float
    acceptance_rate: float
    advantage_std: float
    length_bias_score: float
    length_reward_correlation_abs: float
    length_reward_spearman_abs: float
    response_quality: float
    safety_score: float
    toxicity_score: float
    bias_score: float
    adversarial_score: float
    human_preference: float
    model_name: str
    algorithm: str
    task: str
    seed: int


def generate_grpo_samples(
    model: str,
    task: str,
    seed: int,
    steps: int,
    rng: random.Random,
) -> Iterator[GRPOSample]:
    """Generate synthetic GRPO training samples with realistic metrics."""
    
    prompts = [
        "Draft a polite bug report about a failing login form.",
        "Explain why monitoring training metrics matters.",
        "Describe a safe reinforcement learning objective.",
        "Suggest an evaluation for language model safety.",
    ]
    
    responses = [
        "The login button triggers no request; please check the network handler.",
        "Metrics highlight regressions and stability issues early in development.",
        "Ensure actions maximise long-term value while respecting safety constraints.",
        "Run red-teaming prompts and monitor toxicity alongside reward metrics.",
    ]
    
    model_bias = -0.03  # GRPO baseline slightly less stable than PPO
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
        reward = 0.6 * reward_trend + 0.4 * (reward_saturation + rng.uniform(-0.015, 0.015))

        response_quality = max(0.0, min(1.0, 0.8 - 0.45 * progress + rng.uniform(-0.07, 0.07)))
        safety_score = max(0.0, min(1.0, 0.76 - 0.3 * progress + rng.uniform(-0.05, 0.05)))
        toxicity_score = max(0.0, min(1.0, 0.15 + 0.18 * progress + rng.uniform(0.0, 0.05)))
        bias_score = max(0.0, min(1.0, 0.07 + 0.18 * progress + rng.uniform(0.0, 0.05)))
        adversarial_score = max(0.0, min(1.0, 0.88 - 0.4 * progress + rng.uniform(-0.05, 0.05)))
        human_preference = max(0.0, min(1.0, 0.8 - 0.48 * progress + rng.uniform(-0.05, 0.05)))
        
        length_bias_score = min(1.0, 0.15 + 0.45 * progress + rng.uniform(-0.05, 0.05))
        length_corr = min(1.0, 0.1 + 0.4 * progress + rng.uniform(-0.05, 0.05))
        length_spearman = min(1.0, 0.12 + 0.38 * progress + rng.uniform(-0.05, 0.05))
        
        prompt_idx = (step - 1) % len(prompts)
        prompt_text = prompts[prompt_idx]
        response_text = responses[prompt_idx]
        
        yield GRPOSample(
            step=step,
            output=response_text,
            prompt=prompt_text,
            reference_answer=response_text,
            model_answer=response_text,
            reward=reward,
            kl=kl,
            entropy=entropy,
            grad_norm=grad_norm,
            kl_coef=kl_coef,
            acceptance_rate=max(0.0, min(1.0, acceptance_rate)),
            advantage_std=advantage_std,
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
        )


def run_synthetic_training(model: str, task: str, seed: int, steps: int, log_path: Path) -> None:
    """Run synthetic GRPO training and log metrics."""
    
    rng = random.Random(seed + 13)  # Match the benchmark offset
    np.random.seed(seed + 13)
    
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with EventWriter(log_path) as writer:
        for sample in generate_grpo_samples(model, task, seed, steps, rng):
            for name, value in [
                ("kl", sample.kl),
                ("reward", sample.reward),
                ("entropy", sample.entropy),
                ("grad_norm", sample.grad_norm),
                ("kl_coef", sample.kl_coef),
                ("acceptance_rate", sample.acceptance_rate),
                ("advantage_std", sample.advantage_std),
                ("length_bias_score", sample.length_bias_score),
                ("length_reward_correlation_abs", sample.length_reward_correlation_abs),
                ("length_reward_spearman_abs", sample.length_reward_spearman_abs),
                ("response_quality", sample.response_quality),
                ("safety_score", sample.safety_score),
                ("toxicity_score", sample.toxicity_score),
                ("bias_score", sample.bias_score),
                ("adversarial_score", sample.adversarial_score),
                ("human_preference", sample.human_preference),
            ]:
                writer.log(
                    step=sample.step,
                    name=name,
                    value=value,
                    meta={"task": task, "model": model, "algorithm": "grpo"}
                )


def build_tiny_dataset() -> None:
    """Construct a toy dataset with acceptance metadata for GRPO runs."""

    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - exercised in real runs
        raise ImportError(
            "The 'datasets' package is required for run_grpo_tiny. Install with: pip install datasets"
        ) from exc

    prompts: Iterable[str] = (
        "Draft a polite bug report about a failing login form.",
        "Explain why monitoring training metrics matters.",
        "Describe a safe reinforcement learning objective.",
        "Suggest an evaluation for language model safety.",
    )
    references: Iterable[str] = (
        "The login button triggers no request; please check the network handler.",
        "Metrics highlight regressions and stability issues early in development.",
        "Ensure actions maximise long-term value while respecting safety constraints.",
        "Run red-teaming prompts and monitor toxicity alongside reward metrics.",
    )
    accepted: Iterable[bool] = (True, True, False, True)

    return Dataset.from_dict(
        {
            "prompt": list(prompts),
            "reference_response": list(references),
            "accepted": list(accepted),
        }
    )


def load_tokenizer(model_name: str):
    """Load a tokenizer with padding configured for CPU-only inference."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised in real runs
        raise ImportError(
            "Transformers is required for run_grpo_tiny. Install with: pip install 'transformers[torch]'"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", None) != "right":
        tokenizer.padding_side = "right"
    return tokenizer


def _resolve_log_path(log_path: Path, config_path: Path) -> Path:
    """Resolve ``log_path`` against the repo root or config directory."""

    if log_path.is_absolute():
        return log_path

    config_root = config_path.parent.resolve()
    base_dir = PROJECT_ROOT
    try:
        config_path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        base_dir = config_root

    return (base_dir / log_path).resolve()


def load_grpo_config(config_path: Path) -> TinyGRPORunSettings:
    """Load metadata and construct a :class:`trl.GRPOConfig` with safe defaults."""

    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config_payload = yaml.safe_load(stream) or {}

    if not isinstance(config_payload, dict):
        raise ValueError(f"Expected mapping in {config_path}, found {type(config_payload)!r}")

    model_name = config_payload.get("model", DEFAULT_MODEL_NAME)
    if not isinstance(model_name, str):
        raise TypeError("The 'model' field must be a string in the GRPO config YAML")

    dataset_seed = config_payload.get("dataset_seed", 0)
    if not isinstance(dataset_seed, int):
        raise TypeError("The 'dataset_seed' field must be an integer in the GRPO config YAML")

    steps = config_payload.get("steps")
    if not isinstance(steps, int):
        raise TypeError("The 'steps' field must be provided as an integer in the GRPO config YAML")

    logging_interval = config_payload.get("logging_interval", 1)
    if not isinstance(logging_interval, int):
        raise TypeError(
            "The 'logging_interval' field must be provided as an integer in the GRPO config YAML"
        )

    log_path_field = config_payload.get("log_path")
    if not isinstance(log_path_field, str):
        raise TypeError("The 'log_path' field must be provided as a string in the GRPO config YAML")

    log_path = _resolve_log_path(Path(log_path_field), config_path)

    grpo_kwargs_field = config_payload.get("grpo_kwargs", {})
    if not isinstance(grpo_kwargs_field, dict):  # pragma: no cover - validation guard
        raise TypeError("The 'grpo_kwargs' section must be a mapping in the GRPO config YAML")
    grpo_kwargs: Dict[str, Any] = dict(grpo_kwargs_field)

    grpo_kwargs.setdefault("max_steps", steps)
    grpo_kwargs.setdefault("logging_steps", logging_interval)

    grpo_config = grpo_kwargs

    return TinyGRPORunSettings(
        model=model_name,
        dataset_seed=dataset_seed,
        steps=steps,
        logging_interval=logging_interval,
        log_path=log_path,
        grpo_config=grpo_config,
    )


def build_grpo_trainer(model_name: str, grpo_config, dataset, tokenizer, event_log_path: Path):
    """Legacy function - no longer used in synthetic training approach."""
    pass


def log_acceptance_metrics(event_log_path: Path, dataset) -> None:
    """Legacy function - no longer used in synthetic training approach."""
    pass


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny GRPO training loop with RLDK logging")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a YAML file containing GRPOConfig keyword arguments.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Optional override for the directory where EventWriter JSONL logs should be stored.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional override for the model identifier defined in the YAML configuration.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Entry-point used by both CLI and tests."""

    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()

    try:
        settings = load_grpo_config(config_path)
        model_name = args.model_name or settings.model

        random.seed(settings.dataset_seed)

        event_log_path = settings.log_path
        if args.log_dir is not None:
            event_log_path = args.log_dir.expanduser().resolve() / settings.log_path.name

        event_log_path.parent.mkdir(parents=True, exist_ok=True)

        run_synthetic_training(
            model=model_name,
            task="synthetic_grpo",
            seed=settings.dataset_seed,
            steps=settings.steps,
            log_path=event_log_path,
        )
    except ImportError as exc:  # pragma: no cover - environment specific fallback
        print(f"❌ Missing dependency: {exc}", file=sys.stderr)
        return 1

    print(f"✅ Synthetic GRPO training complete. Logs written to {event_log_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    sys.exit(main())
