"""JSON schemas for report validation."""

from typing import Any, Dict

import jsonschema


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


# Card schemas
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
