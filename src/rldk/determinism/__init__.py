"""Determinism checking for training runs."""

from .check import check, DeterminismReport

# Export both names for API consistency
check_determinism = check

__all__ = ["check", "check_determinism", "DeterminismReport"]
