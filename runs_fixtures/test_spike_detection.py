#!/usr/bin/env python3
"""Test script to demonstrate KL spike detection."""

from rldk.ingest import ingest_runs
from rldk.diff import first_divergence

# Load the fixtures
df_clean = ingest_runs("runs_fixtures/clean_ppo.jsonl")
df_spike = ingest_runs("runs_fixtures/kl_spike.jsonl")

# Find divergence
report = first_divergence(df_clean, df_spike, ["kl_mean"], k_consecutive=3, window=20)

print(f"Divergence detected: {report.diverged}")
if report.diverged:
    print(f"First divergence at step: {report.first_step}")
    print(f"Tripped signals: {report.tripped_signals}")
