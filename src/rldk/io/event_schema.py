"""Normalized event schema for RL training runs."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import json


@dataclass
class Event:
    """Normalized event object representing a single training step."""
    
    step: int
    wall_time: float
    metrics: Dict[str, float]
    rng: Dict[str, Any]
    data_slice: Dict[str, Any]
    model_info: Dict[str, Any]
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "step": self.step,
            "wall_time": self.wall_time,
            "metrics": self.metrics,
            "rng": self.rng,
            "data_slice": self.data_slice,
            "model_info": self.model_info,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        return cls(
            step=data["step"],
            wall_time=data["wall_time"],
            metrics=data["metrics"],
            rng=data["rng"],
            data_slice=data["data_slice"],
            model_info=data["model_info"],
            notes=data.get("notes", [])
        )
    
    def to_json(self) -> str:
        """Serialize event to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        """Create event from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


def create_event_from_row(
    row: Dict[str, Any],
    run_id: str,
    git_sha: Optional[str] = None
) -> Event:
    """
    Create an Event object from a training data row.
    
    Args:
        row: Dictionary containing training step data
        run_id: Unique identifier for the training run
        git_sha: Git commit SHA if available
    
    Returns:
        Event object with normalized data
    """
    # Extract metrics from the row
    metrics = {}
    metric_fields = [
        'reward_mean', 'reward_std', 'kl_mean', 'entropy_mean', 
        'clip_frac', 'grad_norm', 'lr', 'loss'
    ]
    
    for field in metric_fields:
        if field in row and row[field] is not None:
            metrics[field] = float(row[field])
    
    # Extract RNG information
    rng = {
        "seed": row.get("seed"),
        "python_hash_seed": row.get("python_hash_seed"),
        "torch_seed": row.get("torch_seed"),
        "numpy_seed": row.get("numpy_seed"),
        "random_seed": row.get("random_seed")
    }
    
    # Extract data slice information
    data_slice = {
        "tokens_in": row.get("tokens_in"),
        "tokens_out": row.get("tokens_out"),
        "batch_size": row.get("batch_size"),
        "sequence_length": row.get("sequence_length")
    }
    
    # Extract model information
    model_info = {
        "run_id": run_id,
        "git_sha": git_sha,
        "phase": row.get("phase", "train"),
        "model_name": row.get("model_name"),
        "model_size": row.get("model_size"),
        "optimizer": row.get("optimizer"),
        "scheduler": row.get("scheduler")
    }
    
    # Generate notes based on data quality
    notes = []
    if row.get("clip_frac", 0) > 0.2:
        notes.append("High clipping fraction detected")
    if row.get("grad_norm", 0) > 10.0:
        notes.append("Large gradient norm detected")
    if row.get("kl_mean", 0) > 0.2:
        notes.append("High KL divergence detected")
    
    return Event(
        step=int(row["step"]),
        wall_time=float(row.get("wall_time", 0)),
        metrics=metrics,
        rng=rng,
        data_slice=data_slice,
        model_info=model_info,
        notes=notes
    )


def events_to_dataframe(events: List[Event]) -> 'pd.DataFrame':
    """
    Convert a list of events to a pandas DataFrame.
    
    Args:
        events: List of Event objects
    
    Returns:
        DataFrame with flattened event data
    """
    import pandas as pd
    
    rows = []
    for event in events:
        row = {
            "step": event.step,
            "wall_time": event.wall_time,
            "run_id": event.model_info["run_id"],
            "git_sha": event.model_info.get("git_sha"),
            "phase": event.model_info.get("phase", "train"),
            "tokens_in": event.data_slice.get("tokens_in"),
            "tokens_out": event.data_slice.get("tokens_out"),
            "seed": event.rng.get("seed"),
            "notes": "; ".join(event.notes) if event.notes else None
        }
        
        # Add metrics
        for key, value in event.metrics.items():
            row[key] = value
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def dataframe_to_events(df: 'pd.DataFrame', run_id: str, git_sha: Optional[str] = None) -> List[Event]:
    """
    Convert a pandas DataFrame to a list of Event objects.
    
    Args:
        df: DataFrame with training data
        run_id: Unique identifier for the training run
        git_sha: Git commit SHA if available
    
    Returns:
        List of Event objects
    """
    events = []
    for _, row in df.iterrows():
        event = create_event_from_row(row.to_dict(), run_id, git_sha)
        events.append(event)
    
    return events