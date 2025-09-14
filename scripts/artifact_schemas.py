#!/usr/bin/env python3
"""JSON Schemas for RL Debug Kit artifacts.

This module defines minimal JSON schemas for each artifact type,
with only required fields included.
"""

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


def get_schema_for_artifact(artifact_name: str) -> dict:
    """Get the JSON schema for a specific artifact type."""
    return ARTIFACT_SCHEMAS.get(artifact_name, {})


def validate_artifact(artifact_name: str, data: dict) -> bool:
    """Validate artifact data against its schema."""
    import jsonschema
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


if __name__ == "__main__":
    print("Available artifact schemas:")
    for artifact_type in list_artifact_types():
        print(f"  - {artifact_type}")

    print(f"\nTotal schemas: {len(ARTIFACT_SCHEMAS)}")
