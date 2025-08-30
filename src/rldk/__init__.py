"""RL Debug Kit - Library and CLI for debugging reinforcement learning training runs."""

__version__ = "0.1.0"

from .ingest import ingest_runs
from .diff import first_divergence
from .determinism import check_determinism
from .bisect import bisect_commits

__all__ = [
    "ingest_runs",
    "first_divergence", 
    "check_determinism",
    "bisect_commits",
]
