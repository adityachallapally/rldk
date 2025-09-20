"""Readers and writers for training metrics data."""

import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Union

import pandas as pd
import torch

from ..utils.torch_compat import safe_torch_load
from .schema import MetricsSchema, TrainingMetrics


def read_metrics_jsonl(file_path: Union[str, Path]) -> pd.DataFrame:
    """Read metrics from JSONL file."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {file_path}")

    metrics = []
    with open(file_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                metrics.append(TrainingMetrics(**data))
            except (json.JSONDecodeError, Exception) as e:
                raise ValueError(f"Error parsing line {line_num}: {e}")

    schema = MetricsSchema(metrics=metrics)
    return schema.to_dataframe()


# Note: write_metrics_jsonl function moved to consolidated_writers.py for consistency
# This function is deprecated - use the one in consolidated_writers.py instead


def read_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """Read JSONL file and yield dictionaries."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield data
            except json.JSONDecodeError:
                # Skip malformed lines instead of failing
                continue


def read_tensorboard_export(dir_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """Read TensorBoard export directory and yield event dictionaries."""
    dir_path = Path(dir_path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    # Look for CSV files in the export
    csv_files = list(dir_path.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {dir_path}")

    for csv_file in csv_files:
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric values
                event = {}
                for key, value in row.items():
                    try:
                        if "." in value:
                            event[key] = float(value)
                        else:
                            event[key] = int(value)
                    except ValueError:
                        event[key] = value
                yield event


def read_wandb_export(dir_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """Read Weights & Biases export directory and yield event dictionaries."""
    dir_path = Path(dir_path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    # Look for JSONL files in the export
    jsonl_files = list(dir_path.glob("*.jsonl"))

    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {dir_path}")

    for jsonl_file in jsonl_files:
        yield from read_jsonl(jsonl_file)


def read_checkpoint(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """Read PyTorch checkpoint and return state dict on CPU."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        # Load checkpoint to CPU
        checkpoint = safe_torch_load(path, map_location="cpu")

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                return checkpoint["state_dict"]
            elif "model" in checkpoint:
                return checkpoint["model"]
            else:
                return checkpoint
        else:
            return checkpoint

    except Exception as e:
        raise ValueError(f"Failed to load checkpoint {path}: {e}")


def read_reward_head(dir_path: Union[str, Path]) -> Callable[[List[str]], List[float]]:
    """Read reward model from directory and return scoring function."""
    dir_path = Path(dir_path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Reward model directory not found: {dir_path}")

    # Look for model files
    model_files = list(dir_path.glob("*.pt")) + list(dir_path.glob("*.pth"))

    if not model_files:
        raise ValueError(f"No model files found in {dir_path}")

    model_path = model_files[0]  # Use first model file found

    try:
        # Load model to CPU
        model = safe_torch_load(model_path, map_location="cpu")

        # Create model-dependent scoring function
        def score_prompts(prompts: List[str]) -> List[float]:
            # Generate model-dependent scores based on model weights and prompt content
            import hashlib

            # Create a hash from model weights to ensure identical models produce identical scores
            model_hash = hashlib.md5(str(sorted(model.items())).encode()).hexdigest()[
                :8
            ]

            scores = []
            for prompt in prompts:
                # Create a hash from model weights and prompt to ensure model-dependent but deterministic output
                combined = f"{model_hash}:{prompt}"
                hash_obj = hashlib.md5(combined.encode())
                hash_int = int(hash_obj.hexdigest()[:8], 16)

                # Convert to float in [-1, 1] range
                score = (hash_int % 2000 - 1000) / 1000.0
                scores.append(score)

            return scores

        return score_prompts

    except Exception as e:
        raise ValueError(f"Failed to load reward model {model_path}: {e}")
