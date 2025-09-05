"""RL Debug Kit - Library and CLI for debugging reinforcement learning training runs."""

__version__ = "0.1.0"

# Main public API - Core tracking functionality
from .tracking import ExperimentTracker, TrackingConfig

# Core functions from other modules
from .ingest import ingest_runs, ingest_runs_to_events
from .diff import first_divergence
from .determinism import check
from .reward import health, RewardHealthReport
from .evals import run, EvalResult

# Additional functionality
from .diff import bisect_commits  # Moved from bisect module

__all__ = [
    # Main public API
    "ExperimentTracker",
    "TrackingConfig",
    # Core functions
    "ingest_runs",
    "ingest_runs_to_events",
    "first_divergence", 
    "check",
    "bisect_commits",
    "health",
    "RewardHealthReport",
    "run",
    "EvalResult",
]
