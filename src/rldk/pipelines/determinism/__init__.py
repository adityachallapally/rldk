"""Determinism checking for training runs."""

from .check import DeterminismReport, check

# Export both names for API consistency
check_determinism = check

__all__ = ["check", "check_determinism", "DeterminismReport"]
