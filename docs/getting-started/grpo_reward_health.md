# GRPO Reward Health & Determinism

## Overview

Generalized Reinforcement Policy Optimization (GRPO) runs now integrate with RLDK's reward health, determinism, and drift tooling without manual wiring. This guide covers the new ingestion safeguards, automatic gold metric discovery, and reproducibility defaults so you can adopt the updated workflow immediately.

## Reward Health Ingestion

* **Canonical mapping.** The GRPO adapter resolves common reward aliases (`reward`, `group_reward_mean`, `normalized_reward_mean`, etc.) into the standard `reward_mean` / `reward_std` columns. Acceptance metrics such as `accept_rate` are normalized to `acceptance_rate`, and KL/entropy aliases are lifted into the canonical schema that downstream analyses expect.
* **Leakage protection.** Prompt, response, label, and other text-heavy fields are filtered during ingestion to keep sensitive labels out of reward health and doctor diagnostics.
* **Consistent phases and metadata.** Phase, run identifiers, and seeds are inferred from GRPO run directories so cross-run comparisons work even when logs omit explicit fields.

The result is a clean metrics table that plugs directly into `rldk reward-health` and the reward doctor without additional field maps.

## Automatic Gold Metric Discovery

Gold metrics are essential for over-optimization detection, but wiring them manually is easy to forget. Both reward health commands now accept an `--auto-gold` flag:

```bash
rldk reward-health run --scores path/to/run --out reports --auto-gold
rldk reward-health --run path/to/run --auto-gold
```

When enabled, RLDK will:

1. Reuse any trusted gold columns already embedded in the run (e.g., `gold_score`).
2. Search the run directory for GRPO-style artifacts such as `gold_scores.jsonl`, `gold_metrics.jsonl`, or other `*gold*.jsonl` files under `eval/`, `metrics/`, or `reward_model_eval/` folders.
3. Normalize discovered gold metrics automatically so over-optimization safeguards stay on by default.

Explicit `--gold` / `--gold-scores` paths still win and disable auto-discovery for full control.

## Deterministic Defaults

Calling `rldk.set_reproducible_environment()` now enforces a deterministic stack across Python, PyTorch, and TensorFlow:

* Environment variables such as `TOKENIZERS_PARALLELISM=false`, `CUDNN_DETERMINISTIC=true`, `CUDNN_BENCHMARK=false`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `TF_DETERMINISTIC_OPS=1`, and `TF_ENABLE_ONEDNN_OPTS=0` are set for you.
* PyTorch deterministic algorithms are enabled, TF32 acceleration is disabled, and the float32 matmul precision is pinned to a stable mode when available.

These defaults keep CI determinism checks stable and remove the need for bespoke setup scripts.

## CLI Determinism Alias

The determinism checker is now reachable via either command name:

```bash
rldk check-determinism --cmd "python train.py" --compare reward_mean,kl_mean
rldk determinism --cmd "python train.py" --compare reward_mean,kl_mean
```

The new alias matches the CLI documentation and reviewer expectations while retaining the original entry point.

## KL Drift Cards

KL drift analyses now write timestamped filenames (for example, `run123_20240212-153045_kl_drift_card.png`) so repeated investigations no longer overwrite previous visualizations.

---

With these improvements you can ingest GRPO runs, monitor reward health, and validate determinism with a single command, confident that gold metrics and reproducibility safeguards remain active.
