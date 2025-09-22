# Length Bias Trust Card

The length bias trust card surfaces how reward models behave when responses get longer or shorter. It combines correlation checks, quartile analysis, and ODIN-inspired efficiency metrics so you can quickly spot length-driven reward hacking.

## Concept Overview

Length bias occurs when the learned reward function systematically favors longer (or shorter) responses regardless of quality. The trust card visualizes three complementary signals:

- **Correlation scatter**: Pearson and Spearman correlation between response length and reward score, annotated with overall severity.
- **Quartile comparison**: Average reward per length quartile and the share of variance explained by length alone. Large gaps between Q1 and Q4 highlight suspicious optimization patterns.
- **ODIN heuristics**: Inspired by OpenAI's ODIN work, we report reward-per-token efficiency and flag when long responses dominate the reward budget.
- **Timeline**: Rolling reward trends coloured by length buckets reveal when bias emerges during training.

These panels complement the raw metrics already emitted by the [`LengthBiasDetector`](../reference/api.md) and the reward health pipeline. For background on other evaluation primitives, see the [evaluation metrics guide](../evaluation.md#3-bias-evaluation).

## Configuration & Threshold Guidance

The command defaults to a severity threshold of `0.35`, matching the built-in reward health presets. Tune it based on:

- **Tighter guardrails (`0.20`–`0.30`)** for high-stakes deployments or reinforcement learning from human feedback (RLHF) systems with short-form prompts.
- **Looser guardrails (`0.40`–`0.50`)** when prompts intentionally elicit long-form answers (technical reports, essays) and you already audit completion length elsewhere.

Additional configuration tips:

- Provide `--length-col` when metrics already contain token counts (e.g., `response_tokens`). Otherwise supply `--tokenizer-name` so the detector can measure tokens consistently.
- Use `--sample-size` to downsample large logs deterministically (`--seed`) for reproducible comparisons across runs.
- Align thresholds with your monitoring presets—[`health_thresholds.md`](../health_thresholds.md) includes reference values for CI gates.

## CLI Usage

```bash
# Run the detector on a JSONL run and emit a trust card
rldk reward length-bias \
  --run-path runs/my-run/events.jsonl \
  --response-col response_text \
  --reward-col reward_mean \
  --length-col response_tokens \
  --generate-card

# Leverage tokenizer-based length estimation
rldk reward length-bias \
  --run-path wandb://entity/project/run \
  --reward-col score \
  --tokenizer-name gpt2 \
  --threshold 0.30 \
  --sample-size 1000 \
  --generate-card
```

When `--generate-card` is set, artifacts are written to:

```
runs/<run_id>/rldk_cards/length_bias/
  ├── length_bias_card.json
  └── length_bias_card.png
```

If `--output-dir` is also provided, the CLI copies both files into that directory alongside `length_bias_report.json`.

## Monitoring & Presets

The live monitor presets now include dedicated length-bias guards. Pair the card with streaming alerts for continuous coverage:

```bash
# Enable the length bias preset together with PPO safeguards
rldk monitor --stream artifacts/run.jsonl \
  --rules ppo_safe,length_bias \
  --preset trl
```

The preset surfaces severity, correlation, and rank-correlation alerts so you can trigger retraining before the severity score crosses your configured gate. See the updated [monitor rules cookbook](../getting-started/monitor_rules_cookbook.md#new-length-bias-preset) for rule IDs and tuning tips.

## Interpreting ODIN Metrics

The ODIN panel tracks reward-per-token efficiency (`reward_per_token`) and normalized efficiency (`efficiency`). A rising optimization flag indicates long responses collect disproportionate reward relative to shorter completions—an early sign of exploitative behavior. Use it in tandem with quartile deltas to decide whether to:

- Tighten KL or entropy schedules.
- Resample prompts that invite rambling outputs.
- Introduce explicit penalties for excessive verbosity.

## Next Steps

- Integrate the JSON payload into dashboards or CI gates by reading `runs/<run_id>/rldk_cards/length_bias/length_bias_card.json`.
- Combine with the reward health report (`rldk reward reward-health run`) to correlate length bias with saturation, drift, and calibration findings.
- Browse the `examples/length_bias_card_demo.py` script for a programmatic workflow that generates cards from synthetic data.

