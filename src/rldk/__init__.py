"""RL Debug Kit - Library and CLI for debugging reinforcement learning training runs."""

__version__ = "0.1.0"

# Core functionality (existing)
from .ingest import ingest_runs, ingest_runs_to_events
from .diff import first_divergence, DivergenceReport
from .determinism import check, DeterminismReport
from .bisect import bisect_commits, BisectResult
from .reward import health, RewardHealthReport, compare_models
from .evals import run, EvalResult
from .replay import replay, ReplayReport

# Forensics functionality (existing)
from .forensics import (
    scan_logs, diff_checkpoints, audit_environment,
    ComprehensivePPOForensics, ComprehensivePPOMetrics,
    ComprehensiveGRPOForensics, ComprehensiveGRPOMetrics
)

# Tracking functionality (existing)
from .tracking import (
    ExperimentTracker, TrackingConfig,
    DatasetTracker, ModelTracker, EnvironmentTracker,
    SeedTracker, GitTracker
)

# Card generation (existing)
from .cards import (
    generate_determinism_card, generate_drift_card, generate_reward_card
)

# Adapters (existing)
from .adapters import (
    BaseAdapter, TRLAdapter, OpenRLHFAdapter, WandBAdapter, CustomJSONLAdapter, DemoJSONLAdapter
)

# Configuration (existing)
from .config import settings, RLDKSettings, ConfigSchema

# Utility functions (existing)
from .io import (
    write_json, write_png, mkdir_reports, validate,
    read_jsonl, read_reward_head
)

# Seeding utilities (new)
from .utils.seed import (
    set_global_seed, get_global_seed, ensure_seeded,
    seeded_random_state, restore_random_state, DEFAULT_SEED
)

__all__ = [
    # Core functionality
    "ingest_runs",
    "ingest_runs_to_events", 
    "first_divergence",
    "DivergenceReport",
    "check",
    "DeterminismReport",
    "bisect_commits",
    "BisectResult",
    "health",
    "RewardHealthReport",
    "compare_models",
    "run",
    "EvalResult",
    "replay",
    "ReplayReport",
    
    # Forensics functionality
    "scan_logs",
    "diff_checkpoints", 
    "audit_environment",
    "ComprehensivePPOForensics",
    "ComprehensivePPOMetrics",
    "ComprehensiveGRPOForensics",
    "ComprehensiveGRPOMetrics",
    
    # Tracking functionality
    "ExperimentTracker",
    "TrackingConfig",
    "DatasetTracker",
    "ModelTracker", 
    "EnvironmentTracker",
    "SeedTracker",
    "GitTracker",
    
    # Card generation
    "generate_determinism_card",
    "generate_drift_card", 
    "generate_reward_card",
    
    # Adapters
    "BaseAdapter",
    "TRLAdapter",
    "OpenRLHFAdapter",
    "WandBAdapter",
    "CustomJSONLAdapter",
    "DemoJSONLAdapter",
    
    # Configuration
    "settings",
    "RLDKSettings",
    "ConfigSchema",
    
    # Utility functions
    "write_json",
    "write_png",
    "mkdir_reports",
    "validate",
    "read_jsonl",
    "read_reward_head",
    
    # Seeding utilities
    "set_global_seed",
    "get_global_seed",
    "ensure_seeded",
    "seeded_random_state",
    "restore_random_state",
    "DEFAULT_SEED",
]
