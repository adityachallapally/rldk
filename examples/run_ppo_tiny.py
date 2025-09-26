"""Run a CPU-friendly PPO trainer instance that logs metrics via ``EventWriter``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import yaml

from rldk.integrations.trl import create_ppo_trainer

DEFAULT_MODEL_NAME = "sshleifer/tiny-gpt2"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "ppo_tiny.yaml"
DEFAULT_LOG_DIR = Path("artifacts/ppo_tiny")


def build_tiny_dataset() -> "Dataset":
    """Construct a tiny prompt/response dataset for PPO warm-up runs."""

    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - exercised in real runs
        raise ImportError(
            "The 'datasets' package is required for run_ppo_tiny. Install with: pip install datasets"
        ) from exc

    prompts: Iterable[str] = (
        "Summarize the importance of unit tests.",
        "Name a benefit of continuous integration.",
        "Why is code review valuable?",
        "Suggest a use-case for reinforcement learning.",
    )
    responses: Iterable[str] = (
        "Unit tests prevent regressions and document expected behaviour.",
        "Continuous integration catches integration bugs before release.",
        "Code review shares knowledge and identifies issues early.",
        "Reinforcement learning optimizes sequential decision making.",
    )

    return Dataset.from_dict({"prompt": list(prompts), "response": list(responses)})


def load_tokenizer(model_name: str):
    """Load a tokenizer for the provided model with safe padding defaults."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised in real runs
        raise ImportError(
            "Transformers is required for run_ppo_tiny. Install with: pip install 'transformers[torch]'"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", None) != "right":
        tokenizer.padding_side = "right"
    return tokenizer


def load_ppo_config(config_path: Path):
    """Load a :class:`trl.PPOConfig` from YAML."""

    with config_path.open("r", encoding="utf-8") as stream:
        config_payload = yaml.safe_load(stream) or {}

    if not isinstance(config_payload, dict):
        raise ValueError(f"Expected mapping in {config_path}, found {type(config_payload)!r}")

    try:
        from trl import PPOConfig
    except ImportError as exc:  # pragma: no cover - exercised in real runs
        raise ImportError("TRL is required for run_ppo_tiny. Install with: pip install trl") from exc

    return PPOConfig(**config_payload)


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
        default=DEFAULT_LOG_DIR,
        help="Directory where EventWriter JSONL logs should be stored.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model identifier to use for tokenizer and policy/value weights.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Entry-point used by both CLI and tests."""

    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    log_dir = args.log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    event_log_path = log_dir / "run.jsonl"

    try:
        ppo_config = load_ppo_config(config_path)
        dataset = build_tiny_dataset()
        tokenizer = load_tokenizer(args.model_name)
        trainer = create_ppo_trainer(
            model_name=args.model_name,
            ppo_config=ppo_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
            event_log_path=event_log_path,
        )
        trainer.train()
    except ImportError as exc:  # pragma: no cover - environment specific fallback
        print(f"❌ Missing dependency: {exc}", file=sys.stderr)
        return 1

    print(f"✅ Tiny PPO run complete. Logs written to {event_log_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    sys.exit(main())
