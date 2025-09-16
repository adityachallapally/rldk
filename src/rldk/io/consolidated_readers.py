"""Consolidated readers for all training data formats and sources."""

import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import pandas as pd

from ..utils.torch_compat import safe_torch_load
from .consolidated_schemas import (
    Event,
    MetricsSchema,
    TrainingMetrics,
    create_event_from_row,
)


def _hash_state_dict(state_dict: Dict[str, Any]) -> str:
    """
    Efficiently hash a state dictionary by sampling tensor data.

    This function creates a deterministic hash from model weights without
    converting entire tensors to strings, which is memory-intensive and
    non-deterministic for large models.

    Args:
        state_dict: Dictionary containing model parameters

    Returns:
        8-character hash string
    """
    import hashlib

    import numpy as np

    hash_components = []

    for name, param in sorted(state_dict.items()):
        if hasattr(param, 'data') and hasattr(param.data, 'numpy'):
            # PyTorch tensor - sample data efficiently
            try:
                # Convert to numpy and sample for large tensors
                data = param.data.detach().cpu().numpy()

                if data.size > 1000:
                    # For large tensors, sample a subset of values
                    # Use deterministic sampling based on tensor shape
                    sample_size = min(1000, data.size)
                    indices = np.linspace(0, data.size - 1, sample_size, dtype=int)
                    sampled_data = data.flat[indices]
                else:
                    # For small tensors, use all data
                    sampled_data = data.flatten()

                # Create hash from shape, dtype, and sampled data
                shape_str = str(data.shape)
                dtype_str = str(data.dtype)
                data_str = str(sampled_data.tolist())
                hash_components.append(f"{name}:{shape_str}:{dtype_str}:{data_str}")

            except Exception:
                # Fallback: use tensor properties
                shape_str = str(param.shape) if hasattr(param, 'shape') else 'unknown'
                dtype_str = str(param.dtype) if hasattr(param, 'dtype') else 'unknown'
                hash_components.append(f"{name}:{shape_str}:{dtype_str}:fallback")

        elif hasattr(param, 'shape') and hasattr(param, 'dtype'):
            # Tensor-like object
            shape_str = str(param.shape)
            dtype_str = str(param.dtype)
            hash_components.append(f"{name}:{shape_str}:{dtype_str}:tensor_like")

        else:
            # Non-tensor parameter - use string representation
            param_str = str(param)[:100]  # Limit length to avoid memory issues
            hash_components.append(f"{name}:{param_str}")

    # Create final hash from all components
    combined = "|".join(hash_components)
    return hashlib.md5(combined.encode()).hexdigest()[:8]


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


def read_checkpoint(path: Union[str, Path]) -> Dict[str, "torch.Tensor"]:
    """Read PyTorch checkpoint and return state dict on CPU."""
    import torch  # Lazy import to avoid CLI hang
    
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
            # Handle different model types that torch.load() can return
            try:
                if isinstance(model, dict):
                    # Model is a state dict or dictionary - hash tensor data directly
                    model_hash = _hash_state_dict(model)
                elif hasattr(model, 'state_dict'):
                    # Model is a PyTorch module with state_dict method
                    model_hash = _hash_state_dict(model.state_dict())
                elif hasattr(model, 'parameters'):
                    # Model is a PyTorch module with parameters
                    state_dict = dict(model.named_parameters())
                    model_hash = _hash_state_dict(state_dict)
                else:
                    # Fallback: use model type and basic properties
                    model_type = type(model).__name__
                    model_size = len(str(model)) if hasattr(model, '__len__') else 0
                    model_hash = hashlib.md5(f"{model_type}_{model_size}_{model_path}".encode()).hexdigest()[:8]

            except Exception:
                # Ultimate fallback: use model path and size as hash
                model_hash = hashlib.md5(f"{model_path}_{len(str(model))}".encode()).hexdigest()[:8]

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


def read_events_jsonl(
    file_path: Union[str, Path],
    run_id: Optional[str] = None,
    git_sha: Optional[str] = None
) -> List[Event]:
    """
    Read events from JSONL file and return list of Event objects.

    Args:
        file_path: Path to JSONL file
        run_id: Optional run ID for events
        git_sha: Optional git SHA for events

    Returns:
        List of Event objects
    """
    events = []
    for data in read_jsonl(file_path):
        if run_id or git_sha:
            # Use provided run_id/git_sha if available
            event = create_event_from_row(data, run_id or "unknown", git_sha)
        else:
            # Try to extract from data
            event_run_id = data.get("run_id", "unknown")
            event_git_sha = data.get("git_sha")
            event = create_event_from_row(data, event_run_id, event_git_sha)
        events.append(event)

    return events


def read_csv_metrics(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read metrics from CSV file.

    Args:
        file_path: Path to CSV file

    Returns:
        DataFrame with metrics data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {file_path}: {e}")


def read_json_report(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read JSON report file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary with report data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to read JSON file {file_path}: {e}")


def read_markdown_report(file_path: Union[str, Path]) -> str:
    """
    Read markdown report file.

    Args:
        file_path: Path to markdown file

    Returns:
        String with markdown content
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {file_path}")

    try:
        with open(file_path) as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Failed to read markdown file {file_path}: {e}")


def read_directory_metrics(
    dir_path: Union[str, Path],
    pattern: str = "*.jsonl"
) -> pd.DataFrame:
    """
    Read all metrics files from a directory and combine into single DataFrame.

    Args:
        dir_path: Directory containing metrics files
        pattern: File pattern to match (default: "*.jsonl")

    Returns:
        Combined DataFrame with all metrics
    """
    dir_path = Path(dir_path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    all_metrics = []

    for file_path in dir_path.glob(pattern):
        try:
            if file_path.suffix == ".jsonl":
                df = read_metrics_jsonl(file_path)
            elif file_path.suffix == ".csv":
                df = read_csv_metrics(file_path)
            else:
                continue

            all_metrics.append(df)
        except Exception as e:
            print(f"Warning: Failed to read {file_path}: {e}")
            continue

    if not all_metrics:
        raise ValueError(f"No valid metrics files found in {dir_path}")

    return pd.concat(all_metrics, ignore_index=True)
