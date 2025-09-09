"""RL Debug Kit - Library and CLI for debugging reinforcement learning training runs."""

__version__ = "0.1.0"

# Main public API - Core tracking functionality
from .tracking import ExperimentTracker, TrackingConfig

# Core analysis functions
from .ingest import ingest_runs
from .diff import first_divergence
from .determinism import check
from .bisect import bisect_commits
from .reward import health, RewardHealthReport
from .evals import run, EvalResult

__all__ = [
    # Main public API
    "ExperimentTracker",
    "TrackingConfig",
    
    # Core analysis functions
    "ingest_runs",
    "first_divergence", 
    "check",
    "bisect_commits",
    "health",
    "RewardHealthReport",
    "run",
    "EvalResult",
]
