"""Consolidated IO utilities for reading and writing training run data."""

# Import consolidated schemas
# Import consolidated readers
from .consolidated_readers import (
    read_checkpoint,
    read_csv_metrics,
    read_directory_metrics,
    read_events_jsonl,
    read_json_report,
    read_jsonl,
    read_markdown_report,
    read_metrics_jsonl,
    read_reward_head,
    read_tensorboard_export,
    read_wandb_export,
)
from .consolidated_schemas import (
    ARTIFACT_SCHEMAS,
    CkptDiffReportV1,
    # JSON Schemas
    DeterminismCardV1,
    DeterminismCardV2,
    DriftCardV1,
    Event,
    MetricsSchema,
    PPOScanReportV1,
    RewardCardV1,
    RewardDriftReportV1,
    TrainingMetrics,
    create_event_from_row,
    dataframe_to_events,
    events_to_dataframe,
    get_schema_for_artifact,
    list_artifact_types,
    validate,
    validate_artifact,
)

# Import consolidated writers
from .consolidated_writers import (
    generate_reward_health_report,
    mkdir_reports,
    write_ckpt_diff_report,
    write_determinism_card,
    write_drift_analysis_csv,
    write_drift_card,
    write_environment_audit,
    write_eval_summary,
    write_events_jsonl,
    write_json,
    write_metrics_jsonl,
    write_png,
    write_ppo_scan_report,
    write_replay_comparison,
    write_reward_drift_report,
    write_reward_health_card,
    write_reward_health_summary,
    write_run_comparison,
    write_tracking_data,
)

# Import unified writer
from .unified_writer import (
    FileWriteError,
    RLDebugKitIOError,
    SchemaValidationError,
    UnifiedWriter,
)

# Import validator functions
from .validator import (
    create_jsonl_validator,
    validate_event_schema_compatibility,
    validate_jsonl_consistency,
    validate_jsonl_directory,
    validate_jsonl_file,
    validate_jsonl_schema,
)

__all__ = [
    # Schemas
    "TrainingMetrics",
    "MetricsSchema",
    "Event",
    "create_event_from_row",
    "events_to_dataframe",
    "dataframe_to_events",
    "validate",
    "get_schema_for_artifact",
    "validate_artifact",
    "list_artifact_types",
    "DeterminismCardV1",
    "DeterminismCardV2",
    "DriftCardV1",
    "RewardCardV1",
    "PPOScanReportV1",
    "CkptDiffReportV1",
    "RewardDriftReportV1",
    "ARTIFACT_SCHEMAS",

    # Readers
    "read_metrics_jsonl",
    "read_jsonl",
    "read_tensorboard_export",
    "read_wandb_export",
    "read_checkpoint",
    "read_reward_head",
    "read_events_jsonl",
    "read_csv_metrics",
    "read_json_report",
    "read_markdown_report",
    "read_directory_metrics",

    # Writers
    "write_drift_card",
    "write_reward_health_card",
    "write_drift_analysis_csv",
    "write_reward_health_summary",
    "generate_reward_health_report",
    "write_events_jsonl",
    "write_metrics_jsonl",
    "write_determinism_card",
    "write_ppo_scan_report",
    "write_ckpt_diff_report",
    "write_reward_drift_report",
    "write_run_comparison",
    "write_eval_summary",
    "write_environment_audit",
    "write_replay_comparison",
    "write_tracking_data",
    "mkdir_reports",
    "write_json",
    "write_png",

    # Unified Writer
    "UnifiedWriter",
    "RLDebugKitIOError",
    "FileWriteError",
    "SchemaValidationError",

    # Validators
    "validate_jsonl_schema",
    "validate_jsonl_file",
    "validate_jsonl_directory",
    "create_jsonl_validator",
    "validate_event_schema_compatibility",
    "validate_jsonl_consistency",
]
