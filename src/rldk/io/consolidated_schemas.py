"""Consolidated schema definitions for all RL Debug Kit artifacts and data structures."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import jsonschema
import pandas as pd
from pydantic import BaseModel, Field

# ============================================================================
# Pydantic Schemas for Training Data
# ============================================================================

class TrainingMetrics(BaseModel):
    """Schema for individual training metrics record."""

    step: int = Field(..., description="Training step")
    phase: Optional[str] = Field(
        None, description="Training phase (e.g., 'train', 'eval')"
    )
    reward_mean: Optional[float] = Field(None, description="Mean reward")
    reward_std: Optional[float] = Field(None, description="Reward standard deviation")
    kl_mean: Optional[float] = Field(None, description="Mean KL divergence")
    entropy_mean: Optional[float] = Field(None, description="Mean entropy")
    clip_frac: Optional[float] = Field(None, description="Clipping fraction")
    grad_norm: Optional[float] = Field(None, description="Gradient norm")
    lr: Optional[float] = Field(None, description="Learning rate")
    loss: Optional[float] = Field(None, description="Training loss")
    tokens_in: Optional[int] = Field(None, description="Input tokens")
    tokens_out: Optional[int] = Field(None, description="Output tokens")
    wall_time: Optional[float] = Field(None, description="Wall time in seconds")
    seed: Optional[int] = Field(None, description="Random seed")
    run_id: Optional[str] = Field(None, description="Run identifier")
    git_sha: Optional[str] = Field(None, description="Git commit SHA")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame([self.model_dump()])


class MetricsSchema(BaseModel):
    """Schema for a collection of training metrics."""

    metrics: List[TrainingMetrics]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame([m.model_dump() for m in self.metrics])

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "MetricsSchema":
        """Create from pandas DataFrame."""
        metrics = []
        for _, row in df.iterrows():
            # Handle NaN values by converting to None
            row_dict = row.to_dict()
            for key, value in row_dict.items():
                if pd.isna(value):
                    row_dict[key] = None
            metrics.append(TrainingMetrics(**row_dict))
        return cls(metrics=metrics)


# ============================================================================
# Event Schema for Normalized Training Data
# ============================================================================

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
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            step=data["step"],
            wall_time=data["wall_time"],
            metrics=data["metrics"],
            rng=data["rng"],
            data_slice=data["data_slice"],
            model_info=data["model_info"],
            notes=data.get("notes", []),
        )

    def to_json(self) -> str:
        """Serialize event to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Event":
        """Create event from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


# ============================================================================
# JSON Schemas for Report Validation
# ============================================================================

def validate(schema: Dict[str, Any], obj: Dict[str, Any]) -> None:
    """Validate object against JSON schema."""
    try:
        jsonschema.validate(instance=obj, schema=schema)
    except jsonschema.ValidationError as e:
        raise ValueError(f"Schema validation failed: {e}")


# Report Card Schemas
DeterminismCardV1 = {
    "type": "object",
    "required": ["version", "rng", "flags", "nondeterminism_hints", "pass"],
    "properties": {
        "version": {"type": "string"},
        "rng": {
            "type": "object",
            "properties": {
                "python": {"oneOf": [{"type": "integer"}, {"type": "null"}]},
                "torch": {"oneOf": [{"type": "integer"}, {"type": "null"}]},
            },
        },
        "flags": {
            "type": "object",
            "required": [
                "cudnn_deterministic",
                "cudnn_benchmark",
                "tokenizers_parallelism",
            ],
            "properties": {
                "cudnn_deterministic": {"type": "boolean"},
                "cudnn_benchmark": {"type": "boolean"},
                "tokenizers_parallelism": {
                    "oneOf": [{"type": "string"}, {"type": "null"}]
                },
            },
        },
        "nondeterminism_hints": {"type": "array", "items": {"type": "string"}},
        "pass": {"type": "boolean"},
    },
}

DeterminismCardV2 = {
    "type": "object",
    "required": [
        "version",
        "run_id",
        "generated_at",
        "passed",
        "replicas",
        "metrics_compared",
        "replica_variance",
        "rng_map",
        "mismatches",
        "fixes",
        "nondeterminism_hints",
        "flags",
    ],
    "properties": {
        "version": {"type": "string"},
        "run_id": {"type": "string"},
        "generated_at": {"type": "string"},
        "passed": {"type": "boolean"},
        "replicas": {"type": "integer"},
        "metrics_compared": {"type": "array", "items": {"type": "string"}},
        "replica_variance": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
        "rng_map": {
            "type": "object",
            "properties": {
                "python_hash_seed": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                "torch_deterministic": {"type": "boolean"},
                "torch_seed": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                "numpy_seed": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                "random_seed": {"oneOf": [{"type": "string"}, {"type": "null"}]},
            },
        },
        "mismatches": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["step", "metric", "replica_1", "replica_2", "variance"],
                "properties": {
                    "step": {"type": "integer"},
                    "metric": {"type": "string"},
                    "replica_1": {"type": "number"},
                    "replica_2": {"type": "number"},
                    "variance": {"type": "number"},
                },
            },
        },
        "fixes": {"type": "array", "items": {"type": "string"}},
        "nondeterminism_hints": {"type": "array", "items": {"type": "string"}},
        "flags": {
            "type": "object",
            "required": [
                "cudnn_deterministic",
                "cudnn_benchmark",
                "tokenizers_parallelism",
            ],
            "properties": {
                "cudnn_deterministic": {"type": "boolean"},
                "cudnn_benchmark": {"type": "boolean"},
                "tokenizers_parallelism": {
                    "oneOf": [{"type": "string"}, {"type": "null"}]
                },
            },
        },
    },
}

DriftCardV1 = {
    "type": "object",
    "required": [
        "version",
        "run_a",
        "run_b",
        "generated_at",
        "diverged",
        "first_step",
        "tripped_signals",
        "suspected_causes",
        "repro",
        "details",
        "notes",
    ],
    "properties": {
        "version": {"type": "string"},
        "run_a": {"type": "string"},
        "run_b": {"type": "string"},
        "generated_at": {"type": "string"},
        "diverged": {"type": "boolean"},
        "first_step": {"oneOf": [{"type": "integer"}, {"type": "null"}]},
        "tripped_signals": {"type": "array", "items": {"type": "string"}},
        "suspected_causes": {"type": "array", "items": {"type": "string"}},
        "repro": {
            "type": "object",
            "required": ["command", "changes"],
            "properties": {
                "command": {"type": "string"},
                "changes": {"type": "array", "items": {"type": "string"}},
            },
        },
        "details": {
            "type": "object",
            "properties": {
                "kl_divergence": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
                "reward_drift": {
                    "type": "object",
                    "required": ["correlation", "mae"],
                    "properties": {
                        "correlation": {"type": "number"},
                        "mae": {"type": "number"},
                    },
                },
                "metric_correlations": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
                "drift_patterns": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
}

RewardCardV1 = {
    "type": "object",
    "required": [
        "version",
        "run_id",
        "generated_at",
        "passed",
        "drift_detected",
        "calibration_score",
        "saturation_detected",
        "shortcut_signals",
        "label_noise",
        "metrics",
        "slice_analysis",
        "recommendations",
    ],
    "properties": {
        "version": {"type": "string"},
        "run_id": {"type": "string"},
        "generated_at": {"type": "string"},
        "passed": {"type": "boolean"},
        "drift_detected": {"type": "boolean"},
        "calibration_score": {"type": "number"},
        "saturation_detected": {"type": "boolean"},
        "shortcut_signals": {"type": "array", "items": {"type": "string"}},
        "label_noise": {"type": "number"},
        "metrics": {
            "type": "object",
            "required": ["correlation", "mae", "l2_distance"],
            "properties": {
                "correlation": {"type": "number"},
                "mae": {"type": "number"},
                "l2_distance": {"type": "number"},
            },
        },
        "slice_analysis": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["delta_mean", "n_samples"],
                "properties": {
                    "delta_mean": {"type": "number"},
                    "n_samples": {"type": "integer"},
                },
            },
        },
        "recommendations": {"type": "array", "items": {"type": "string"}},
    },
}

# Analysis Report Schemas
PPOScanReportV1 = {
    "type": "object",
    "required": ["version", "rules_fired", "stats"],
    "properties": {
        "version": {"type": "string"},
        "rules_fired": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["rule", "description"],
                "properties": {
                    "rule": {"type": "string"},
                    "description": {"type": "string"},
                    "step_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
            },
        },
        "earliest_step": {"oneOf": [{"type": "integer"}, {"type": "null"}]},
        "stats": {
            "type": "object",
            "properties": {
                "kl_median": {"type": "number"},
                "grad_ratio_median": {"type": "number"},
                "entropy_trend": {"type": "string"},
            },
        },
    },
}

CkptDiffReportV1 = {
    "type": "object",
    "required": ["version", "summary", "top_movers"],
    "properties": {
        "version": {"type": "string"},
        "summary": {
            "type": "object",
            "required": ["num_params", "num_common_params", "num_only_in_a", "num_only_in_b", "avg_cosine", "l2_p05", "l2_p50", "l2_p95"],
            "properties": {
                "num_params": {"type": "integer"},
                "num_common_params": {"type": "integer"},
                "num_only_in_a": {"type": "integer"},
                "num_only_in_b": {"type": "integer"},
                "avg_cosine": {"type": "number"},
                "l2_p05": {"type": "number"},
                "l2_p50": {"type": "number"},
                "l2_p95": {"type": "number"},
            },
        },
        "top_movers": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "l2", "cosine"],
                "properties": {
                    "name": {"type": "string"},
                    "l2": {"type": "number"},
                    "cosine": {"type": "number"},
                    "note": {"type": "string"},
                },
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
}

RewardDriftReportV1 = {
    "type": "object",
    "required": [
        "version",
        "pearson",
        "spearman",
        "mae_z",
        "l2_z",
        "sign_flip_rate",
        "drift_magnitude",
        "effect_size",
        "confidence_summary",
        "slice_deltas",
    ],
    "properties": {
        "version": {"type": "string"},
        "pearson": {"type": "number"},
        "spearman": {"type": "number"},
        "mae_z": {"type": "number"},
        "l2_z": {"type": "number"},
        "sign_flip_rate": {"type": "number"},
        "drift_magnitude": {"type": "number"},
        "effect_size": {"type": "number"},
        "confidence_summary": {"type": "string"},
        "slice_deltas": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["delta_mean", "n"],
                "properties": {
                    "delta_mean": {"type": "number"},
                    "n": {"type": "integer"},
                },
            },
        },
    },
}

# ============================================================================
# Artifact Schemas (from scripts/artifact_schemas.py)
# ============================================================================

ARTIFACT_SCHEMAS = {
    "ingest_result": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "step": {"type": "number"},
                "loss": {"type": "number"},
                "reward_mean": {"type": "number"},
                "phase": {"type": "string"},
                "run_id": {"type": "string"},
            },
            "required": ["step", "loss", "reward_mean"],
            "additionalProperties": True,
        },
    },

    "diff_result": {
        "type": "object",
        "properties": {
            "diverged": {"type": "boolean"},
            "first_step": {"type": ["number", "null"]},
            "tripped_signals": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["diverged", "first_step", "tripped_signals"],
        "additionalProperties": True,
    },

    "determinism_result": {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "culprit": {"type": ["string", "null"]},
            "fixes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["passed", "culprit", "fixes"],
        "additionalProperties": True,
    },

    "reward_health_result": {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "drift_detected": {"type": "boolean"},
            "calibration_score": {"type": "number"},
        },
        "required": ["passed", "drift_detected", "calibration_score"],
        "additionalProperties": True,
    },

    "eval_result": {
        "type": "object",
        "properties": {
            "sample_size": {"type": "number"},
            "seed": {"type": "number"},
            "scores": {"type": "object"},
        },
        "required": ["sample_size", "seed", "scores"],
        "additionalProperties": True,
    },

    "golden_master_summary": {
        "type": "object",
        "properties": {
            "version": {"type": "string"},
            "timestamp": {"type": "string"},
            "total_commands": {"type": "number"},
            "successful_commands": {"type": "number"},
            "failed_commands": {"type": "number"},
            "commands": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "exit_code": {"type": "number"},
                        "stdout": {"type": "string"},
                        "stderr": {"type": "string"},
                        "artifacts": {"type": "array", "items": {"type": "string"}},
                        "artifact_checksums": {"type": "object"},
                        "artifact_schemas": {"type": "object"},
                        "duration": {"type": "number"},
                    },
                    "required": [
                        "command", "exit_code", "stdout", "stderr",
                        "artifacts", "artifact_checksums", "artifact_schemas", "duration"
                    ],
                },
            },
        },
        "required": ["version", "timestamp", "total_commands", "successful_commands", "failed_commands", "commands"],
        "additionalProperties": True,
    },

    "determinism_card": {
        "type": "object",
        "properties": {
            "version": {"type": "string"},
            "passed": {"type": "boolean"},
            "culprit": {"type": ["string", "null"]},
            "fixes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["version", "passed", "culprit", "fixes"],
        "additionalProperties": True,
    },

    "drift_card": {
        "type": "object",
        "properties": {
            "version": {"type": "string"},
            "diverged": {"type": "boolean"},
            "first_step": {"type": ["number", "null"]},
            "signals_monitored": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["version", "diverged", "first_step", "signals_monitored"],
        "additionalProperties": True,
    },

    "reward_card": {
        "type": "object",
        "properties": {
            "version": {"type": "string"},
            "passed": {"type": "boolean"},
            "calibration_score": {"type": "number"},
            "drift_detected": {"type": "boolean"},
        },
        "required": ["version", "passed", "calibration_score", "drift_detected"],
        "additionalProperties": True,
    },

    "diff_report": {
        "type": "object",
        "properties": {
            "version": {"type": "string"},
            "diverged": {"type": "boolean"},
            "first_step": {"type": ["number", "null"]},
            "tripped_signals": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["version", "diverged", "first_step", "tripped_signals"],
        "additionalProperties": True,
    },

    "reward_health_summary": {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "drift_detected": {"type": "boolean"},
            "calibration_score": {"type": "number"},
            "saturation_issues": {"type": "array"},
        },
        "required": ["passed", "drift_detected", "calibration_score", "saturation_issues"],
        "additionalProperties": True,
    },

    "eval_summary": {
        "type": "object",
        "properties": {
            "suite": {"type": "string"},
            "sample_size": {"type": "number"},
            "seed": {"type": "number"},
            "scores": {"type": "object"},
        },
        "required": ["suite", "sample_size", "seed", "scores"],
        "additionalProperties": True,
    },

    "run_comparison": {
        "type": "object",
        "properties": {
            "version": {"type": "string"},
            "run_a": {"type": "object"},
            "run_b": {"type": "object"},
            "earliest_divergent_step": {"type": ["number", "null"]},
        },
        "required": ["version", "run_a", "run_b", "earliest_divergent_step"],
        "additionalProperties": True,
    },

    "reward_drift": {
        "type": "object",
        "properties": {
            "pearson": {"type": "number"},
            "spearman": {"type": "number"},
            "mae_z": {"type": "number"},
        },
        "required": ["pearson", "spearman", "mae_z"],
        "additionalProperties": True,
    },

    "ckpt_diff": {
        "type": "object",
        "properties": {
            "total_params": {"type": "number"},
            "diff_count": {"type": "number"},
            "max_diff": {"type": "number"},
            "top_movers": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["total_params", "diff_count", "max_diff", "top_movers"],
        "additionalProperties": True,
    },

    "ppo_scan": {
        "type": "object",
        "properties": {
            "rules_fired": {"type": "array", "items": {"type": "object"}},
            "earliest_step": {"type": ["number", "null"]},
            "anomaly_count": {"type": "number"},
        },
        "required": ["rules_fired", "earliest_step", "anomaly_count"],
        "additionalProperties": True,
    },

    "env_audit": {
        "type": "object",
        "properties": {
            "deterministic": {"type": "boolean"},
            "reproducible": {"type": "boolean"},
            "issues": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["deterministic", "reproducible", "issues"],
        "additionalProperties": True,
    },

    "replay_comparison": {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "original_seed": {"type": "number"},
            "replay_seed": {"type": "number"},
            "mismatches": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["passed", "original_seed", "replay_seed", "mismatches"],
        "additionalProperties": True,
    },

    "tracking_data": {
        "type": "object",
        "properties": {
            "experiment_id": {"type": "string"},
            "experiment_name": {"type": "string"},
            "timestamp": {"type": "string"},
            "environment": {"type": "object"},
        },
        "required": ["experiment_id", "experiment_name", "timestamp", "environment"],
        "additionalProperties": True,
    },
}

# ============================================================================
# Schema Utility Functions
# ============================================================================

def get_schema_for_artifact(artifact_name: str) -> dict:
    """Get the JSON schema for a specific artifact type."""
    return ARTIFACT_SCHEMAS.get(artifact_name, {})


def validate_artifact(artifact_name: str, data: dict) -> bool:
    """Validate artifact data against its schema."""
    from jsonschema import Draft7Validator

    schema = get_schema_for_artifact(artifact_name)
    if not schema:
        return True  # No schema defined, assume valid

    try:
        Draft7Validator(schema).validate(data)
        return True
    except jsonschema.ValidationError:
        return False


def list_artifact_types() -> list:
    """List all available artifact types."""
    return list(ARTIFACT_SCHEMAS.keys())


# ============================================================================
# Event Creation Utilities
# ============================================================================

def create_event_from_row(
    row: Dict[str, Any], run_id: str, git_sha: Optional[str] = None
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
        "reward_mean",
        "reward_std",
        "kl_mean",
        "entropy_mean",
        "clip_frac",
        "grad_norm",
        "lr",
        "loss",
        # Network metrics
        "network_bandwidth",
        "network_latency",
        "bandwidth_mbps",
        "latency_ms",
        "bandwidth_upload_mbps",
        "bandwidth_download_mbps",
        "total_bandwidth_mbps",
        "allreduce_bandwidth",
        "broadcast_bandwidth",
        "gather_bandwidth",
        "scatter_bandwidth",
        "packet_loss_percent",
        "network_errors",
        "dns_resolution_ms",
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
        "random_seed": row.get("random_seed"),
    }

    # Extract data slice information
    data_slice = {
        "tokens_in": row.get("tokens_in"),
        "tokens_out": row.get("tokens_out"),
        "batch_size": row.get("batch_size"),
        "sequence_length": row.get("sequence_length"),
    }

    # Extract model information
    model_info = {
        "run_id": run_id,
        "git_sha": git_sha,
        "phase": row.get("phase", "train"),
        "model_name": row.get("model_name"),
        "model_size": row.get("model_size"),
        "optimizer": row.get("optimizer"),
        "scheduler": row.get("scheduler"),
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
        notes=notes,
    )


def events_to_dataframe(events: List[Event]) -> pd.DataFrame:
    """
    Convert a list of events to a pandas DataFrame.

    Args:
        events: List of Event objects

    Returns:
        DataFrame with flattened event data
    """
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
            "notes": "; ".join(event.notes) if event.notes else None,
        }

        # Add metrics
        for key, value in event.metrics.items():
            row[key] = value

        rows.append(row)

    return pd.DataFrame(rows)


def dataframe_to_events(
    df: pd.DataFrame, run_id: str, git_sha: Optional[str] = None
) -> List[Event]:
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
