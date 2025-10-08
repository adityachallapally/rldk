"""Diff analysis for training runs."""

from .diff import DivergenceReport, first_divergence
from .training_metrics import compare_training_metrics_tables

__all__ = ["first_divergence", "DivergenceReport", "compare_training_metrics_tables"]
