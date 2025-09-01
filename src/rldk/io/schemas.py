"""JSON schemas for report validation."""

import jsonschema
from typing import Dict, Any


def validate(schema: Dict[str, Any], obj: Dict[str, Any]) -> None:
    """Validate object against JSON schema."""
    try:
        jsonschema.validate(instance=obj, schema=schema)
    except jsonschema.ValidationError as e:
        raise ValueError(f"Schema validation failed: {e}")


# Schema definitions
DeterminismCardV1 = {
    "type": "object",
    "required": ["version", "rng", "flags", "nondeterminism_hints", "pass"],
    "properties": {
        "version": {"type": "string"},
        "rng": {
            "type": "object",
            "properties": {
                "python": {"type": "integer"},
                "torch": {"type": "integer"}
            }
        },
        "flags": {
            "type": "object",
            "required": ["cudnn_deterministic", "cudnn_benchmark", "tokenizers_parallelism"],
            "properties": {
                "cudnn_deterministic": {"type": "boolean"},
                "cudnn_benchmark": {"type": "boolean"},
                "tokenizers_parallelism": {"oneOf": [{"type": "string"}, {"type": "null"}]}
            }
        },
        "nondeterminism_hints": {
            "type": "array",
            "items": {"type": "string"}
        },
        "pass": {"type": "boolean"}
    }
}


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
                        "maxItems": 2
                    }
                }
            }
        },
        "earliest_step": {"type": "integer"},
        "stats": {
            "type": "object",
            "properties": {
                "kl_median": {"type": "number"},
                "grad_ratio_median": {"type": "number"},
                "entropy_trend": {"type": "string"}
            }
        }
    }
}


CkptDiffReportV1 = {
    "type": "object",
    "required": ["version", "summary", "top_movers"],
    "properties": {
        "version": {"type": "string"},
        "summary": {
            "type": "object",
            "required": ["num_params", "avg_cosine", "l2_p05", "l2_p50", "l2_p95"],
            "properties": {
                "num_params": {"type": "integer"},
                "avg_cosine": {"type": "number"},
                "l2_p05": {"type": "number"},
                "l2_p50": {"type": "number"},
                "l2_p95": {"type": "number"}
            }
        },
        "top_movers": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "l2", "cosine"],
                "properties": {
                    "name": {"type": "string"},
                    "l2": {"type": "number"},
                    "cosine": {"type": "number"}
                }
            }
        },
        "notes": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}


RewardDriftReportV1 = {
    "type": "object",
    "required": ["version", "pearson", "spearman", "mae_z", "l2_z", "sign_flip_rate", "slice_deltas"],
    "properties": {
        "version": {"type": "string"},
        "pearson": {"type": "number"},
        "spearman": {"type": "number"},
        "mae_z": {"type": "number"},
        "l2_z": {"type": "number"},
        "sign_flip_rate": {"type": "number"},
        "slice_deltas": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["delta_mean", "n"],
                "properties": {
                    "delta_mean": {"type": "number"},
                    "n": {"type": "integer"}
                }
            }
        }
    }
}