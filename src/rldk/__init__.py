"""RL Debug Kit - Library and CLI for debugging reinforcement learning training runs."""

__version__ = "0.1.0"

def _lazy_import_ingest():
    from .ingest import ingest_runs, ingest_runs_to_events
    return ingest_runs, ingest_runs_to_events

def _lazy_import_diff():
    from .diff import first_divergence, DivergenceReport
    return first_divergence, DivergenceReport

def _lazy_import_determinism():
    from .determinism import check, DeterminismReport
    return check, DeterminismReport

def _lazy_import_bisect():
    from .bisect import bisect_commits, BisectResult
    return bisect_commits, BisectResult

def _lazy_import_reward():
    from .reward import health, RewardHealthReport, compare_models
    return health, RewardHealthReport, compare_models

def _lazy_import_evals():
    from .evals import run, EvalResult
    return run, EvalResult

def _lazy_import_replay():
    from .replay import replay, ReplayReport
    return replay, ReplayReport

def _lazy_import_forensics():
    from .forensics import (
        scan_logs, diff_checkpoints, audit_environment,
        ComprehensivePPOForensics, ComprehensivePPOMetrics
    )
    return scan_logs, diff_checkpoints, audit_environment, ComprehensivePPOForensics, ComprehensivePPOMetrics

def _lazy_import_tracking():
    from .tracking import (
        ExperimentTracker, TrackingConfig,
        DatasetTracker, ModelTracker, EnvironmentTracker,
        SeedTracker, GitTracker
    )
    return ExperimentTracker, TrackingConfig, DatasetTracker, ModelTracker, EnvironmentTracker, SeedTracker, GitTracker

def _lazy_import_cards():
    from .cards import (
        generate_determinism_card, generate_drift_card, generate_reward_card
    )
    return generate_determinism_card, generate_drift_card, generate_reward_card

def _lazy_import_adapters():
    from .adapters import (
        BaseAdapter, TRLAdapter, OpenRLHFAdapter, WandBAdapter, CustomJSONLAdapter
    )
    return BaseAdapter, TRLAdapter, OpenRLHFAdapter, WandBAdapter, CustomJSONLAdapter

def _lazy_import_config():
    from .config import settings, RLDKSettings, ConfigSchema
    return settings, RLDKSettings, ConfigSchema

def _lazy_import_io():
    from .io import (
        write_json, write_png, mkdir_reports, validate,
        read_jsonl, read_reward_head
    )
    return write_json, write_png, mkdir_reports, validate, read_jsonl, read_reward_head

def _lazy_import_seed():
    from .utils.seed import (
        set_global_seed, get_current_seed, restore_seed_state,
        set_reproducible_environment, validate_seed_consistency
    )
    return set_global_seed, get_current_seed, restore_seed_state, set_reproducible_environment, validate_seed_consistency

def ingest_runs(*args, **kwargs):
    ingest_runs_func, _ = _lazy_import_ingest()
    return ingest_runs_func(*args, **kwargs)

def ingest_runs_to_events(*args, **kwargs):
    _, ingest_runs_to_events_func = _lazy_import_ingest()
    return ingest_runs_to_events_func(*args, **kwargs)

def first_divergence(*args, **kwargs):
    first_divergence_func, _ = _lazy_import_diff()
    return first_divergence_func(*args, **kwargs)

def check(*args, **kwargs):
    check_func, _ = _lazy_import_determinism()
    return check_func(*args, **kwargs)

def bisect_commits(*args, **kwargs):
    bisect_commits_func, _ = _lazy_import_bisect()
    return bisect_commits_func(*args, **kwargs)

def health(*args, **kwargs):
    health_func, _, _ = _lazy_import_reward()
    return health_func(*args, **kwargs)

def compare_models(*args, **kwargs):
    _, _, compare_models_func = _lazy_import_reward()
    return compare_models_func(*args, **kwargs)

def run(*args, **kwargs):
    run_func, _ = _lazy_import_evals()
    return run_func(*args, **kwargs)

def replay(*args, **kwargs):
    replay_func, _ = _lazy_import_replay()
    return replay_func(*args, **kwargs)

def scan_logs(*args, **kwargs):
    scan_logs_func, _, _, _, _ = _lazy_import_forensics()
    return scan_logs_func(*args, **kwargs)

def diff_checkpoints(*args, **kwargs):
    _, diff_checkpoints_func, _, _, _ = _lazy_import_forensics()
    return diff_checkpoints_func(*args, **kwargs)

def audit_environment(*args, **kwargs):
    _, _, audit_environment_func, _, _ = _lazy_import_forensics()
    return audit_environment_func(*args, **kwargs)

def generate_determinism_card(*args, **kwargs):
    generate_determinism_card_func, _, _ = _lazy_import_cards()
    return generate_determinism_card_func(*args, **kwargs)

def generate_drift_card(*args, **kwargs):
    _, generate_drift_card_func, _ = _lazy_import_cards()
    return generate_drift_card_func(*args, **kwargs)

def generate_reward_card(*args, **kwargs):
    _, _, generate_reward_card_func = _lazy_import_cards()
    return generate_reward_card_func(*args, **kwargs)

def write_json(*args, **kwargs):
    write_json_func, _, _, _, _, _ = _lazy_import_io()
    return write_json_func(*args, **kwargs)

def write_png(*args, **kwargs):
    _, write_png_func, _, _, _, _ = _lazy_import_io()
    return write_png_func(*args, **kwargs)

def mkdir_reports(*args, **kwargs):
    _, _, mkdir_reports_func, _, _, _ = _lazy_import_io()
    return mkdir_reports_func(*args, **kwargs)

def validate(*args, **kwargs):
    _, _, _, validate_func, _, _ = _lazy_import_io()
    return validate_func(*args, **kwargs)

def read_jsonl(*args, **kwargs):
    _, _, _, _, read_jsonl_func, _ = _lazy_import_io()
    return read_jsonl_func(*args, **kwargs)

def read_reward_head(*args, **kwargs):
    _, _, _, _, _, read_reward_head_func = _lazy_import_io()
    return read_reward_head_func(*args, **kwargs)

def set_global_seed(*args, **kwargs):
    set_global_seed_func, _, _, _, _ = _lazy_import_seed()
    return set_global_seed_func(*args, **kwargs)

def get_current_seed(*args, **kwargs):
    _, get_current_seed_func, _, _, _ = _lazy_import_seed()
    return get_current_seed_func(*args, **kwargs)

def restore_seed_state(*args, **kwargs):
    _, _, restore_seed_state_func, _, _ = _lazy_import_seed()
    return restore_seed_state_func(*args, **kwargs)

def set_reproducible_environment(*args, **kwargs):
    _, _, _, set_reproducible_environment_func, _ = _lazy_import_seed()
    return set_reproducible_environment_func(*args, **kwargs)

def validate_seed_consistency(*args, **kwargs):
    _, _, _, _, validate_seed_consistency_func = _lazy_import_seed()
    return validate_seed_consistency_func(*args, **kwargs)

class _LazyClass:
    def __init__(self, import_func, class_index):
        self._import_func = import_func
        self._class_index = class_index
        self._class = None
    
    def __call__(self, *args, **kwargs):
        if self._class is None:
            imports = self._import_func()
            self._class = imports[self._class_index]
        return self._class(*args, **kwargs)
    
    def __getattr__(self, name):
        if self._class is None:
            imports = self._import_func()
            self._class = imports[self._class_index]
        return getattr(self._class, name)

DivergenceReport = _LazyClass(_lazy_import_diff, 1)
DeterminismReport = _LazyClass(_lazy_import_determinism, 1)
BisectResult = _LazyClass(_lazy_import_bisect, 1)
RewardHealthReport = _LazyClass(_lazy_import_reward, 1)
EvalResult = _LazyClass(_lazy_import_evals, 1)
ReplayReport = _LazyClass(_lazy_import_replay, 1)
ComprehensivePPOForensics = _LazyClass(_lazy_import_forensics, 3)
ComprehensivePPOMetrics = _LazyClass(_lazy_import_forensics, 4)
ExperimentTracker = _LazyClass(_lazy_import_tracking, 0)
TrackingConfig = _LazyClass(_lazy_import_tracking, 1)
DatasetTracker = _LazyClass(_lazy_import_tracking, 2)
ModelTracker = _LazyClass(_lazy_import_tracking, 3)
EnvironmentTracker = _LazyClass(_lazy_import_tracking, 4)
SeedTracker = _LazyClass(_lazy_import_tracking, 5)
GitTracker = _LazyClass(_lazy_import_tracking, 6)
BaseAdapter = _LazyClass(_lazy_import_adapters, 0)
TRLAdapter = _LazyClass(_lazy_import_adapters, 1)
OpenRLHFAdapter = _LazyClass(_lazy_import_adapters, 2)
WandBAdapter = _LazyClass(_lazy_import_adapters, 3)
CustomJSONLAdapter = _LazyClass(_lazy_import_adapters, 4)

settings = _LazyClass(lambda: (_lazy_import_config()[0],), 0)
RLDKSettings = _LazyClass(lambda: (_lazy_import_config()[1],), 0)
ConfigSchema = _LazyClass(lambda: (_lazy_import_config()[2],), 0)

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
    
    # Seed management
    "set_global_seed",
    "get_current_seed",
    "restore_seed_state",
    "set_reproducible_environment",
    "validate_seed_consistency",
]
