# Catastrophic Forgetting – Benchmark/Task Regression

The catastrophic forgetting evaluation component measures whether a model
regresses on previously mastered benchmarks after additional fine-tuning or
training updates. It compares current evaluation results against historical
baselines and reports normalized regression scores together with detailed
diagnostics that highlight problematic tasks.

## Expected Inputs

- **Tabular evaluation data** that follows the standard RLDK schema.
  - Required columns: `step`, `output`.
  - Recommended columns:
    - Task identifiers (any of `task`, `task_id`, `benchmark`, `dataset`,
      `evaluation_name`).
    - Numeric evaluation score (`score`, `reward`, `reward_mean`, `metric`,
      `metric_value`).
  - Additional metadata such as `events`, `reward`, or custom diagnostics is
    preserved and can be leveraged by downstream tooling.
- **Historical baselines** for each benchmark/task. Baselines may be provided
  as:
  - A mapping of task identifiers to summary statistics (`mean`, `std`,
    `count`, optional `timestamp`).
  - A DataFrame containing the same information with recognizable column
    names.

Tasks without baselines are reported in the output but are excluded from the
regression score so that missing references never cause the evaluation to fail.

## Output Structure

The component returns a dictionary with the following keys:

- `score`: Weighted mean regression score between 0.0 and 1.0 (1.0 indicates no
  regression).
- `regressed_tasks`: List of tasks that exceeded regression thresholds with
  per-task deltas, z-scores, and confidence intervals.
- `stable_tasks`: Tasks that remained within tolerance or lacked sufficient
  samples.
- `missing_baselines`: Sorted list of task identifiers without historical
  references.
- `recommendations`: Suggested mitigation steps (e.g., rehearsal or targeted
  fine-tuning).
- `warnings`: Non-fatal issues encountered while analysing the data.
- `metadata`: Additional context such as thresholds, weighting strategy, and
  sample counts.

These diagnostics are also attached to `EvalResult.raw_results` when run through
an evaluation suite.

## Configuration Hooks

Configuration values are sourced from `EvaluationConfig` and can be overridden
via environment variables or keyword arguments:

- `CATASTROPHIC_REGRESSION_THRESHOLD`: Maximum tolerated drop in mean score
  before flagging catastrophic regression (default: `-0.05`).
- `CATASTROPHIC_REGRESSION_Z_THRESHOLD`: Minimum z-score relative to the
  baseline distribution (default: `-2.0`).
- `CATASTROPHIC_MIN_SAMPLES`: Minimum number of samples required before a task
  contributes to the aggregate score (default: `5`).
- `CATASTROPHIC_WEIGHTING_STRATEGY`: Weighting scheme for aggregating task level
  scores (`baseline_count`, `sample_count`, or `custom`).

## Recommendations and Mitigation Guidance

When regression is detected the component highlights the largest drops and
suggests mitigation steps such as targeted fine-tuning or rehearsal sampling for
the affected tasks. Missing baselines trigger documentation-friendly
recommendations instead of hard failures, ensuring the evaluation can run during
early experimentation phases.
