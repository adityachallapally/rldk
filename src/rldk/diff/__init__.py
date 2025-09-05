"""Run comparison and divergence detection."""

from .diff import first_divergence
from .bisect import bisect_commits, BisectResult
from .drift import generate_drift_card  # Moved from cards module

__all__ = ["first_divergence", "bisect_commits", "BisectResult", "generate_drift_card"]
