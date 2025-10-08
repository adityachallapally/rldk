"""Run a CPU-friendly GRPO trainer instance that mirrors metrics to EventWriter."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from rldk.emit import EventWriter
from rldk.integrations.trl import EventWriterCallback, create_grpo_config, fix_generation_config

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
    grpo_config: "GRPOConfig"


def build_tiny_dataset() -> "Dataset":
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

    grpo_config = create_grpo_config(**grpo_kwargs)

    return TinyGRPORunSettings(
        model=model_name,
        dataset_seed=dataset_seed,
        steps=steps,
        logging_interval=logging_interval,
        log_path=log_path,
        grpo_config=grpo_config,
    )


def build_grpo_trainer(
    model_name: str,
    grpo_config,
    dataset,
    tokenizer,
    event_log_path: Path,
):
    """Initialise TRL's GRPO trainer with EventWriter logging enabled."""

    try:
        from trl import AutoModelForCausalLMWithValueHead, GRPOTrainer
    except ImportError as exc:  # pragma: no cover - exercised in real runs
        raise ImportError("TRL is required for run_grpo_tiny. Install with: pip install trl") from exc

    # For GRPO, we use the model as both policy and reward function
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    model = fix_generation_config(model, tokenizer)

    callback = EventWriterCallback(event_log_path, run_id=getattr(grpo_config, "run_name", None))

    trainer = GRPOTrainer(
        args=grpo_config,
        model=model,
        reward_funcs=model,  # Use the same model as reward function
        processing_class=tokenizer,
        train_dataset=dataset,
        callbacks=[callback],
    )
    return trainer


def log_acceptance_metrics(event_log_path: Path, dataset) -> None:
    """Append a summary acceptance rate entry to the JSONL log."""

    accepted_flags = [bool(row.get("accepted", False)) for row in dataset]
    acceptance_rate = sum(accepted_flags) / len(accepted_flags) if accepted_flags else 0.0

    with EventWriter(event_log_path) as writer:
        writer.log(
            step=0,
            name="acceptance_rate",
            value=float(acceptance_rate),
            tags={"phase": "summary", "trainer": "grpo_tiny"},
            meta={"source": "run_grpo_tiny"},
        )


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

        dataset = build_tiny_dataset()
        tokenizer = load_tokenizer(model_name)
        trainer = build_grpo_trainer(
            model_name=model_name,
            grpo_config=settings.grpo_config,
            dataset=dataset,
            tokenizer=tokenizer,
            event_log_path=event_log_path,
        )
        trainer.train()
        log_acceptance_metrics(event_log_path, dataset)
    except ImportError as exc:  # pragma: no cover - environment specific fallback
        print(f"❌ Missing dependency: {exc}", file=sys.stderr)
        return 1

    print(f"✅ Tiny GRPO run complete. Logs written to {event_log_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    sys.exit(main())
