"""Run a CPU-friendly GRPO trainer instance that mirrors metrics to EventWriter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import yaml

from rldk.emit import EventWriter
from rldk.integrations.trl import EventWriterCallback, create_grpo_config

DEFAULT_MODEL_NAME = "sshleifer/tiny-gpt2"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "grpo_tiny.yaml"
DEFAULT_LOG_DIR = Path("artifacts/grpo_tiny")


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


def load_grpo_config(config_path: Path):
    """Load a :class:`trl.GRPOConfig` using the helper that applies CPU defaults."""

    with config_path.open("r", encoding="utf-8") as stream:
        config_payload = yaml.safe_load(stream) or {}

    if not isinstance(config_payload, dict):
        raise ValueError(f"Expected mapping in {config_path}, found {type(config_payload)!r}")

    return create_grpo_config(**config_payload)


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

    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

    callback = EventWriterCallback(event_log_path, run_id=getattr(grpo_config, "run_name", None))

    trainer = GRPOTrainer(
        args=grpo_config,
        model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
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
        grpo_config = load_grpo_config(config_path)
        dataset = build_tiny_dataset()
        tokenizer = load_tokenizer(args.model_name)
        trainer = build_grpo_trainer(
            model_name=args.model_name,
            grpo_config=grpo_config,
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
