"""Main evaluation runner for RL Debug Kit."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..utils.error_handling import EvaluationError, ValidationError
from ..utils.progress import progress_bar
from .metrics import calculate_confidence_intervals, calculate_effect_sizes
from .schema import (
    _create_enhanced_validation_error,
    get_schema_for_suite,
    validate_eval_input,
)
from .suites import get_eval_suite


@dataclass
class EvalResult:
    """Result of an evaluation suite run."""

    suite_name: str
    scores: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    effect_sizes: Dict[str, float]
    sample_size: int
    seed: int
    metadata: Dict[str, Any]
    raw_results: List[Dict[str, Any]]
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    @property
    def overall_score(self) -> Optional[float]:
        """
        Calculate overall score as unweighted mean of available numeric metrics.

        Returns:
            Overall score or None if no metrics available
        """
        from .schema import safe_mean

        valid_scores = []
        for metric, score in self.scores.items():
            if score is not None:
                try:
                    # Convert to float and check for NaN
                    score_float = float(score)
                    if not np.isnan(score_float):
                        valid_scores.append(score_float)
                except (ValueError, TypeError):
                    # Skip non-numeric scores
                    continue

        if not valid_scores:
            return None

        return safe_mean(valid_scores)

    @property
    def available_fraction(self) -> float:
        """
        Fraction of metrics that produced valid values.

        Returns:
            Float between 0 and 1 indicating fraction of available metrics
        """
        if not self.scores:
            return 0.0

        total_metrics = len(self.scores)
        valid_metrics = 0

        for score in self.scores.values():
            if score is not None:
                try:
                    # Convert to float and check for NaN
                    score_float = float(score)
                    if not np.isnan(score_float):
                        valid_metrics += 1
                except (ValueError, TypeError):
                    # Skip non-numeric scores
                    continue

        return valid_metrics / total_metrics if total_metrics > 0 else 0.0


def run(
    run_data: pd.DataFrame,
    suite: str = "quick",
    seed: int = 42,
    sample_size: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    column_mapping: Optional[Dict[str, str]] = None,
) -> EvalResult:
    """
    Run evaluation suite on training run data.

    Args:
        run_data: Training run data to evaluate
        suite: Name of evaluation suite to run
        seed: Random seed for reproducibility
        sample_size: Number of samples to evaluate (None for suite default)
        output_dir: Directory to save results (optional)
        column_mapping: Optional mapping from user column names to RLDK standard names

    Returns:
        EvalResult with comprehensive evaluation metrics

    Raises:
        ValidationError: If input validation fails
        EvaluationError: If evaluation fails
    """
    logger = logging.getLogger(__name__)

    # Validate inputs
    if run_data is None or not isinstance(run_data, pd.DataFrame):
        raise ValidationError(
            "run_data must be a pandas DataFrame",
            suggestion="Ensure you're passing a valid DataFrame",
            error_code="INVALID_RUN_DATA"
        )

    if run_data.empty:
        raise ValidationError(
            "run_data is empty",
            suggestion="Ensure the DataFrame contains data to evaluate",
            error_code="EMPTY_RUN_DATA"
        )

    # Get evaluation suite
    eval_suite = get_eval_suite(suite)
    if eval_suite is None:
        raise ValidationError(
            f"Unknown evaluation suite: {suite}",
            suggestion="Choose from: quick, comprehensive, safety, integrity, performance, trust, training_metrics",
            error_code="UNKNOWN_SUITE"
        )

    # Validate and normalize input data using schema
    schema = get_schema_for_suite(suite)

    if column_mapping:
        from .schema import normalize_columns
        run_data, effective_mapping = normalize_columns(run_data, column_mapping)
        logger.info(f"Applied column mapping: {effective_mapping}")
    else:
        from ..adapters.field_resolver import FieldResolver
        field_resolver = FieldResolver()
        auto_mapping = {}

        for col_spec in schema.required_columns:
            if col_spec.name not in run_data.columns:
                resolved = field_resolver.resolve_field(col_spec.name, run_data.columns.tolist())
                if resolved:
                    auto_mapping[resolved] = col_spec.name

        if auto_mapping:
            from .schema import normalize_columns
            run_data, effective_mapping = normalize_columns(run_data, auto_mapping)
            logger.info(f"Applied automatic column mapping: {effective_mapping}")
        else:
            effective_mapping = {}

    try:
        validated_data = validate_eval_input(run_data, schema, suite)
        logger.info(f"Data validation completed with {len(validated_data.warnings)} warnings")

        # Log warnings
        for warning in validated_data.warnings:
            logger.warning(f"Data validation warning: {warning}")

    except ValueError as e:
        enhanced_error = _create_enhanced_validation_error(
            str(e), run_data, schema, suite, effective_mapping
        )
        raise enhanced_error from e

    # Set random seed for reproducibility
    np.random.seed(seed)

    logger.info(f"Running {suite} evaluation suite on {len(validated_data.data)} records")

    # Determine sample size - use actual data size for empty or small datasets
    if sample_size is None:
        if len(validated_data.data) == 0:
            sample_size = 0
        elif len(validated_data.data) <= eval_suite.get("default_sample_size", 100):
            sample_size = len(validated_data.data)
        else:
            sample_size = eval_suite.get("default_sample_size", 100)

    # Sample data if needed
    if len(validated_data.data) > sample_size:
        sampled_data = validated_data.data.sample(n=sample_size, random_state=seed).reset_index(
            drop=True
        )
        logger.info(f"Sampled {sample_size} records from {len(validated_data.data)} total")
    else:
        sampled_data = validated_data.data.copy()

    evaluations_to_run = eval_suite["evaluations"].copy()
    skipped_evaluations = []

    if suite == "training_metrics":
        from .suites import EVAL_REQUIREMENTS
        available_columns = set(validated_data.data.columns)

        filtered_evaluations = {}
        for eval_name, eval_func in evaluations_to_run.items():
            if eval_name in EVAL_REQUIREMENTS:
                requirements = EVAL_REQUIREMENTS[eval_name]
                # Check if required columns are available
                required_cols = []
                for req in requirements:
                    if "|" in req:
                        alternatives = req.split("|")
                        if not any(alt in available_columns for alt in alternatives):
                            required_cols.append(req)
                    else:
                        if req not in available_columns:
                            required_cols.append(req)

                if required_cols:
                    logger.warning(f"Skipping {eval_name}: missing columns {required_cols}")
                    skipped_evaluations.append(eval_name)
                else:
                    filtered_evaluations[eval_name] = eval_func
            else:
                # Include evaluations not in EVAL_REQUIREMENTS
                filtered_evaluations[eval_name] = eval_func

        evaluations_to_run = filtered_evaluations

        if skipped_evaluations:
            validated_data.warnings.append(f"Skipped evaluations due to missing columns: {', '.join(skipped_evaluations)}")

    # Run evaluations with progress indication
    raw_results = []
    scores = {}
    failed_evaluations = []

    # If no data, return empty results
    if sample_size == 0:
        logger.warning("No data available for evaluation")
        for eval_name in evaluations_to_run.keys():
            raw_results.append(
                {
                    "evaluation": eval_name,
                    "result": {
                        "score": np.nan,
                        "details": f"{eval_name} evaluation based on 0 metrics",
                    },
                    "timestamp": pd.Timestamp.now().isoformat(),
                }
            )
            scores[eval_name] = np.nan
    else:
        # Run evaluations with progress tracking
        evaluations = list(evaluations_to_run.items())

        with progress_bar(len(evaluations), f"Running {suite} evaluations") as bar:
            for eval_name, eval_func in evaluations:
                try:
                    logger.debug(f"Running evaluation: {eval_name}")
                    result = eval_func(sampled_data, seed=seed)

                    raw_results.append(
                        {
                            "evaluation": eval_name,
                            "result": result,
                            "timestamp": pd.Timestamp.now().isoformat(),
                        }
                    )

                    # Extract score if available
                    if isinstance(result, dict) and "score" in result:
                        scores[eval_name] = result["score"]
                    elif isinstance(result, (int, float)):
                        scores[eval_name] = float(result)
                    else:
                        scores[eval_name] = np.nan
                        logger.warning(f"Could not extract score from {eval_name} result")

                except Exception as e:
                    logger.error(f"Evaluation {eval_name} failed: {e}")
                    failed_evaluations.append(eval_name)

                    raw_results.append(
                        {
                            "evaluation": eval_name,
                            "error": str(e),
                            "timestamp": pd.Timestamp.now().isoformat(),
                        }
                    )
                    scores[eval_name] = np.nan

                bar.update(1)

    # Check if too many evaluations failed
    if len(failed_evaluations) > len(evaluations) * 0.5:  # More than 50% failed
        raise EvaluationError(
            f"Too many evaluations failed: {len(failed_evaluations)}/{len(evaluations)}",
            suggestion="Check your data format and evaluation requirements",
            error_code="TOO_MANY_FAILURES",
            details={"failed_evaluations": failed_evaluations}
        )

    # Calculate confidence intervals with error handling
    try:
        confidence_intervals = calculate_confidence_intervals(scores, sample_size)
    except Exception as e:
        logger.warning(f"Failed to calculate confidence intervals: {e}")
        confidence_intervals = {}

    # Calculate effect sizes with error handling
    try:
        effect_sizes = calculate_effect_sizes(scores, eval_suite.get("baseline_scores", {}))
    except Exception as e:
        logger.warning(f"Failed to calculate effect sizes: {e}")
        effect_sizes = {}

    # Collect all warnings
    all_warnings = list(validated_data.warnings)

    # Add warnings for failed evaluations
    if failed_evaluations:
        all_warnings.append(f"Failed evaluations: {', '.join(failed_evaluations)}")

    # Add warning if no metrics are available
    if not scores or all(np.isnan(score) if isinstance(score, (int, float)) else score is None for score in scores.values()):
        all_warnings.append("No valid metrics computed - check data quality and evaluation requirements")

    # Create result object
    result = EvalResult(
        suite_name=suite,
        scores=scores,
        confidence_intervals=confidence_intervals,
        effect_sizes=effect_sizes,
        sample_size=sample_size,
        seed=seed,
        metadata={
            "suite_config": eval_suite,
            "run_data_shape": run_data.shape,
            "sampled_data_shape": sampled_data.shape,
            "evaluation_count": len(eval_suite["evaluations"]),
            "failed_evaluations": failed_evaluations,
            "normalized_columns": validated_data.normalized_columns,
            "effective_column_mapping": effective_mapping,
            "skipped_evaluations": skipped_evaluations if suite == "training_metrics" else [],
        },
        raw_results=raw_results,
        warnings=all_warnings,
    )

    # Save results if output directory specified
    if output_dir:
        try:
            save_eval_results(result, output_dir)
            logger.info(f"Results saved to {output_dir}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            # Don't fail the entire operation if saving fails

    logger.info(f"Evaluation completed: {len(scores)} metrics calculated")
    return result


def save_eval_results(result: EvalResult, output_dir: Union[str, Path]) -> None:
    """Save evaluation results to files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results as JSONL
    results_path = output_dir / "eval_results.jsonl"
    with open(results_path, "w") as f:
        for raw_result in result.raw_results:
            f.write(json.dumps(raw_result) + "\n")

    # Save summary as JSON
    summary_path = output_dir / "eval_summary.json"
    summary = {
        "suite_name": result.suite_name,
        "scores": result.scores,
        "confidence_intervals": result.confidence_intervals,
        "effect_sizes": result.effect_sizes,
        "sample_size": result.sample_size,
        "seed": result.seed,
        "metadata": result.metadata,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Generate evaluation card
    generate_eval_card(result, output_dir)


def generate_eval_card(result: EvalResult, output_dir: Path) -> None:
    """Generate human-readable evaluation card."""

    card_path = output_dir / "eval_card.md"

    with open(card_path, "w") as f:
        f.write("# Evaluation Results Card\n\n")
        f.write(f"**Suite:** {result.suite_name}\n")
        f.write(f"**Sample Size:** {result.sample_size}\n")
        f.write(f"**Seed:** {result.seed}\n")
        f.write(
            f"**Timestamp:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        f.write("## 📊 Overall Scores\n\n")
        f.write(f"**Overall Score:** {result.overall_score:.3f}" if result.overall_score is not None else "**Overall Score:** Not available")
        f.write(f"\n**Available Metrics:** {result.available_fraction:.1%}\n\n")

        f.write("| Metric | Score | Confidence Interval | Effect Size |\n")
        f.write("|--------|-------|-------------------|-------------|\n")

        for metric, score in result.scores.items():
            if not np.isnan(score) if isinstance(score, (int, float)) else score is not None:
                ci = result.confidence_intervals.get(metric, (np.nan, np.nan))
                effect_size = result.effect_sizes.get(metric, np.nan)

                ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if not np.isnan(ci[0]) else "N/A"
                effect_str = (
                    f"{effect_size:.3f}" if not np.isnan(effect_size) else "N/A"
                )

                f.write(f"| {metric} | {score:.3f} | {ci_str} | {effect_str} |\n")
            else:
                f.write(f"| {metric} | N/A | N/A | N/A |\n")

        # Add warnings section if there are any
        if result.warnings:
            f.write("\n## ⚠️ Warnings\n\n")
            for warning in result.warnings:
                f.write(f"- {warning}\n")
            f.write("\n")

        f.write("\n## 🔍 Detailed Results\n\n")

        for raw_result in result.raw_results:
            f.write(f"### {raw_result['evaluation']}\n\n")

            if "error" in raw_result:
                f.write(f"❌ **Error:** {raw_result['error']}\n\n")
            else:
                result_data = raw_result["result"]
                if isinstance(result_data, dict):
                    for key, value in result_data.items():
                        if key != "score":  # Already shown in summary
                            f.write(f"- **{key}:** {value}\n")
                else:
                    f.write(f"**Result:** {result_data}\n")
                f.write("\n")

        f.write("## 📁 Files Generated\n\n")
        f.write(f"- **Evaluation Card:** `{card_path.name}`\n")
        f.write("- **Detailed Results:** `eval_results.jsonl`\n")
        f.write("- **Summary:** `eval_summary.json`\n")

        if result.metadata.get("suite_config", {}).get("generates_plots", False):
            f.write("- **Plots:** `tradeoff_plots.png`\n")


def compare_evaluations(
    results: List[EvalResult], output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Compare multiple evaluation results.

    Args:
        results: List of EvalResult objects to compare
        output_dir: Directory to save comparison results

    Returns:
        Dictionary with comparison metrics
    """

    if len(results) < 2:
        raise ValueError("Need at least 2 evaluation results to compare")

    comparison = {
        "runs_compared": len(results),
        "run_names": [f"run_{i}" for i in range(len(results))],
        "comparisons": {},
    }

    # Get common metrics
    all_metrics = set()
    for result in results:
        all_metrics.update(result.scores.keys())

    # Compare each metric across runs
    for metric in all_metrics:
        metric_scores = []
        metric_cis = []

        for result in results:
            if metric in result.scores and not np.isnan(result.scores[metric]):
                metric_scores.append(result.scores[metric])
                if metric in result.confidence_intervals:
                    ci = result.confidence_intervals[metric]
                    if not np.isnan(ci[0]):
                        metric_cis.append(ci)

        if len(metric_scores) >= 2:
            # Calculate statistics
            comparison["comparisons"][metric] = {
                "scores": metric_scores,
                "mean": np.mean(metric_scores),
                "std": np.std(metric_scores),
                "min": np.min(metric_scores),
                "max": np.max(metric_scores),
                "range": np.max(metric_scores) - np.min(metric_scores),
                "cv": (
                    np.std(metric_scores) / np.mean(metric_scores)
                    if np.mean(metric_scores) != 0
                    else np.nan
                ),
            }

            # Add confidence intervals if available
            if len(metric_cis) >= 2:
                comparison["comparisons"][metric]["confidence_intervals"] = metric_cis

    # Save comparison if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        comparison_path = output_dir / "eval_comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2, default=str)

    return comparison
