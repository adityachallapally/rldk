"""Ingest training runs from various sources."""

from .ingest import ingest_runs, ingest_runs_to_events

__all__ = ["ingest_runs", "ingest_runs_to_events"]
