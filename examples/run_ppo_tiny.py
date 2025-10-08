"""Run a CPU-friendly PPO trainer instance that logs metrics via ``EventWriter``."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from rldk.integrations.trl import create_ppo_trainer, tokenize_text_column

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
    ppo_config: "PPOConfig"


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
    """Load metadata and create a :class:`trl.PPOConfig` from YAML."""

    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config_payload = yaml.safe_load(stream) or {}

    if not isinstance(config_payload, dict):
        raise ValueError(f"Expected mapping in {config_path}, found {type(config_payload)!r}")

    try:
        from trl import PPOConfig
    except ImportError as exc:  # pragma: no cover - exercised in real runs
        raise ImportError("TRL is required for run_ppo_tiny. Install with: pip install trl") from exc

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

    ppo_kwargs.setdefault("max_steps", steps)
    ppo_kwargs.setdefault("logging_steps", logging_interval)

    ppo_config = PPOConfig(**ppo_kwargs)

    return TinyPPORunSettings(
        model=model_name,
        dataset_seed=dataset_seed,
        steps=steps,
        logging_interval=logging_interval,
        log_path=log_path,
        ppo_config=ppo_config,
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

        event_log_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = build_tiny_dataset()
        tokenizer = load_tokenizer(model_name)
        dataset = tokenize_text_column(
            dataset,
            tokenizer,
            text_column="prompt",
            padding=True,
            truncation=True,
            keep_original=False,
            desc="Tokenizing PPO prompts",
        )
        if hasattr(dataset, "remove_columns"):
            if "response" in getattr(dataset, "column_names", []):
                dataset = dataset.remove_columns(["response"])
        else:  # pragma: no cover - exercised in unit tests with simple stubs
            for record in dataset:
                record.pop("response", None)
        trainer = create_ppo_trainer(
            model_name=model_name,
            ppo_config=settings.ppo_config,
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
