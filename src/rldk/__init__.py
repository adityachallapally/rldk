"""RL Debug Kit - The Ultimate Post-Training Tool for RLHF, DPO, and All Reinforcement Learning."""

__version__ = "0.1.0"

# Core functionality
from .ingest import ingest_runs
from .diff import first_divergence
from .determinism import check
from .bisect import bisect_commits
from .reward import health, RewardHealthReport
from .evals import run, EvalResult

# Ultimate Post-Training Features
from .universal_monitor import UniversalMonitor, start_monitoring, quick_debug as monitor_quick_debug
from .anomaly_detector import AnomalyDetector, detect_anomalies, detect_training_anomalies
from .debug_training import TrainingDebugger, debug_training, quick_debug

__all__ = [
    # Core functionality
    "ingest_runs",
    "first_divergence",
    "check",
    "bisect_commits",
    "health",
    "RewardHealthReport",
    "run",
    "EvalResult",
    
    # Ultimate Post-Training Features
    "UniversalMonitor",
    "start_monitoring",
    "monitor_quick_debug",
    "AnomalyDetector",
    "detect_anomalies",
    "detect_training_anomalies",
    "TrainingDebugger",
    "debug_training",
    "quick_debug",
]
