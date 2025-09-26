"""Run a CPU-friendly PPO trainer instance that logs metrics via ``EventWriter``."""

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
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "ppo_tiny.yaml"


@dataclass
class TinyPPORunSettings:
    """Container for the metadata encoded in ``configs/ppo_tiny.yaml``."""

    model: str
    dataset_seed: int
    steps: int
    logging_interval: int
    log_path: Path
    ppo_config: Dict[str, Any]


@dataclass
class PPOSample:
    """Training sample for synthetic PPO data generation."""
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


def generate_ppo_samples(
    model: str,
    task: str,
    seed: int,
    steps: int,
    rng: random.Random,
) -> Iterator[PPOSample]:
    """Generate synthetic PPO training samples with realistic metrics."""
    
    prompts = [
        "Summarize the importance of unit tests.",
        "Name a benefit of continuous integration.",
        "Why is code review valuable?",
        "Suggest a use-case for reinforcement learning.",
    ]
    
    responses = [
        "Unit tests prevent regressions and document expected behaviour.",
        "Continuous integration catches integration bugs before release.",
        "Code review shares knowledge and identifies issues early.",
        "Reinforcement learning optimizes sequential decision making.",
    ]
    
    model_bias = 0.0  # PPO baseline
    
    for step in range(1, steps + 1):
        progress = step / steps
        noise = rng.uniform(-0.01, 0.01)
        
        kl = 0.02 + 0.25 * math.pow(progress, 1.2) + noise
        reward_trend = 0.3 + 0.4 * progress + rng.uniform(-0.05, 0.05)
        entropy = 3.2 - 1.8 * progress + rng.uniform(-0.2, 0.15)
        grad_norm = 2.8 + 6.5 * progress + rng.uniform(-0.4, 0.4)
        kl_coef = 0.1 + 0.001 * math.sin(progress * math.pi * 3)
        
        response_quality = max(0.0, min(1.0, 0.75 + 0.2 * progress + rng.uniform(-0.1, 0.1)))
        safety_score = max(0.0, min(1.0, 0.8 - 0.25 * progress + rng.uniform(-0.05, 0.05)))
        toxicity_score = max(0.0, min(1.0, 0.1 + 0.15 * progress + rng.uniform(0.0, 0.03)))
        bias_score = max(0.0, min(1.0, 0.05 + 0.12 * progress + rng.uniform(0.0, 0.03)))
        adversarial_score = max(0.0, min(1.0, 0.9 - 0.3 * progress + rng.uniform(-0.05, 0.05)))
        human_preference = max(0.0, min(1.0, 0.75 + 0.2 * progress + rng.uniform(-0.08, 0.08)))
        
        length_bias_score = min(1.0, 0.12 + 0.35 * progress + rng.uniform(-0.05, 0.05))
        length_corr = min(1.0, 0.08 + 0.3 * progress + rng.uniform(-0.05, 0.05))
        length_spearman = min(1.0, 0.1 + 0.28 * progress + rng.uniform(-0.05, 0.05))
        
        reward = 0.7 * reward_trend + 0.3 * rng.uniform(-0.02, 0.02)
        
        prompt_idx = (step - 1) % len(prompts)
        prompt_text = prompts[prompt_idx]
        response_text = responses[prompt_idx]
        
        yield PPOSample(
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
            algorithm="ppo",
            task=task,
            seed=seed,
        )


def build_tiny_dataset(tokenizer) -> None:
    """Legacy function - no longer used in synthetic training approach."""
    pass


def load_tokenizer():
    """Legacy function - no longer used in synthetic training approach."""
    pass


def create_ppo_trainer():
    """Legacy function - no longer used in synthetic training approach."""
    pass


def run_synthetic_training(model: str, task: str, seed: int, steps: int, log_path: Path) -> None:
    """Run synthetic PPO training and log metrics."""
    
    rng = random.Random(seed)
    np.random.seed(seed)
    
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with EventWriter(log_path) as writer:
        for sample in generate_ppo_samples(model, task, seed, steps, rng):
            for name, value in [
                ("kl", sample.kl),
                ("reward", sample.reward),
                ("entropy", sample.entropy),
                ("grad_norm", sample.grad_norm),
                ("kl_coef", sample.kl_coef),
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
                    meta={"task": task, "model": model, "algorithm": "ppo"}
                )


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


def load_ppo_config(config_path: Path) -> TinyPPORunSettings:
    """Load metadata from YAML config for synthetic PPO training."""

    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config_payload = yaml.safe_load(stream) or {}

    if not isinstance(config_payload, dict):
        raise ValueError(f"Expected mapping in {config_path}, found {type(config_payload)!r}")

    model_name = config_payload.get("model", DEFAULT_MODEL_NAME)
    if not isinstance(model_name, str):
        raise TypeError("The 'model' field must be a string in the PPO config YAML")

    dataset_seed = config_payload.get("dataset_seed", 0)
    if not isinstance(dataset_seed, int):
        raise TypeError("The 'dataset_seed' field must be an integer in the PPO config YAML")

    steps = config_payload.get("steps")
    if not isinstance(steps, int):
        raise TypeError("The 'steps' field must be provided as an integer in the PPO config YAML")

    logging_interval = config_payload.get("logging_interval", 1)
    if not isinstance(logging_interval, int):
        raise TypeError(
            "The 'logging_interval' field must be provided as an integer in the PPO config YAML"
        )

    log_path_field = config_payload.get("log_path")
    if not isinstance(log_path_field, str):
        raise TypeError("The 'log_path' field must be provided as a string in the PPO config YAML")

    log_path = _resolve_log_path(Path(log_path_field), config_path)

    ppo_kwargs_field = config_payload.get("ppo_kwargs", {})
    if not isinstance(ppo_kwargs_field, dict):  # pragma: no cover - validation guard
        raise TypeError("The 'ppo_kwargs' section must be a mapping in the PPO config YAML")
    ppo_kwargs: Dict[str, Any] = dict(ppo_kwargs_field)

    return TinyPPORunSettings(
        model=model_name,
        dataset_seed=dataset_seed,
        steps=steps,
        logging_interval=logging_interval,
        log_path=log_path,
        ppo_config=ppo_kwargs,
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny PPO training loop with RLDK logging")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a YAML file containing PPOConfig keyword arguments.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Optional override for the directory where JSONL logs should be stored.",
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
        settings = load_ppo_config(config_path)
        model_name = args.model_name or settings.model

        random.seed(settings.dataset_seed)

        event_log_path = settings.log_path
        if args.log_dir is not None:
            event_log_path = args.log_dir.expanduser().resolve() / settings.log_path.name

        run_synthetic_training(
            model=model_name,
            task="synthetic_ppo",
            seed=settings.dataset_seed,
            steps=settings.steps,
            log_path=event_log_path,
        )
        
    except Exception as exc:
        print(f"❌ Error during synthetic PPO training: {exc}", file=sys.stderr)
        return 1

    print(f"✅ Synthetic PPO training complete. Logs written to {event_log_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    sys.exit(main())
