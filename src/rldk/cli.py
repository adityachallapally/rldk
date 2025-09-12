"""Command-line interface for RL Debug Kit."""

import typer
from pathlib import Path
from typing import List, Optional
import numpy as np
import json
import logging
import sys
import pandas as pd

from rldk.ingest import ingest_runs, ingest_runs_to_events
from rldk.diff import first_divergence
from rldk.determinism.check import check
from rldk.bisect import bisect_commits
from rldk.io import write_json, generate_reward_health_report
from rldk.reward import health
from rldk.evals import run
from rldk.replay import replay
from rldk.tracking import ExperimentTracker, TrackingConfig
from rldk.config import settings
from rldk.utils.error_handling import (
    RLDKError, ValidationError, AdapterError, EvaluationError, TimeoutError,
    format_error_message, log_error_with_context, validate_file_path,
    validate_data_format, validate_required_fields, validate_adapter_source,
    print_usage_examples, print_troubleshooting_tips, check_dependencies,
    with_retry, with_timeout, handle_graceful_degradation, safe_operation
)
from rldk.utils.progress import (
    progress_bar, spinner, timed_operation, timed_operation_context, print_operation_status
)

# Import card generation modules
from rldk.cards import (
    generate_determinism_card,
    generate_drift_card,
    generate_reward_card,
)

# Import forensics modules
from rldk.forensics.ckpt_diff import diff_checkpoints
from rldk.forensics.env_audit import audit_environment
from rldk.forensics.log_scan import scan_logs
from rldk.io import write_json as write_json_report, write_png, mkdir_reports, validate
from rldk.io import DeterminismCardV1, PPOScanReportV1, CkptDiffReportV1, RewardDriftReportV1

# Import reward modules
from rldk.reward.drift import compare_models
from rldk.reward.health_analysis import health as reward_health_analysis
from rldk.reward.health_config.exit_codes import raise_on_failure
from rldk.reward.health_config.config import load_config, get_legacy_thresholds
from rldk.io import read_jsonl, read_reward_head

# Import evaluation modules
from rldk.evals.suites import QUICK_SUITE, COMPREHENSIVE_SUITE, SAFETY_SUITE
from rldk.evals.metrics import evaluate_throughput, evaluate_toxicity, evaluate_bias

def ensure_config_initialized():
    """Ensure configuration is initialized for CLI operations."""
    try:
        settings.initialize()
    except PermissionError as e:
        # If we can't create directories, that's okay for read-only operations
        # Just log a warning and continue
        logging.warning(f"Could not create RLDK directories: {e}")
    except Exception as e:
        # For other errors, log but don't fail
        logging.warning(f"Configuration initialization warning: {e}")

app = typer.Typer(
    name="rldk",
    help="RL Debug Kit - Library and CLI for debugging reinforcement learning training runs",
    add_completion=False,
)

# Create sub-apps
forensics_app = typer.Typer(name="forensics", help="Forensics commands for RL training analysis")
reward_app = typer.Typer(name="reward", help="Reward model analysis commands")
evals_app = typer.Typer(name="evals", help="Evaluation suite commands")

# Add sub-apps to main app
app.add_typer(forensics_app, name="forensics")
app.add_typer(reward_app, name="reward")
app.add_typer(evals_app, name="evals")


# ============================================================================
# FORENSICS COMMANDS
# ============================================================================

@forensics_app.command(name="compare-runs")
def forensics_compare_runs(
    run_a: str = typer.Argument(..., help="Path to first run directory"),
    run_b: str = typer.Argument(..., help="Path to second run directory"),
):
    """Compare two training runs and identify divergences."""
    try:
        ensure_config_initialized()
        typer.echo("Comparing runs:")
        typer.echo(f"  Run A: {run_a}")
        typer.echo(f"  Run B: {run_b}")

        # Scan both runs
        scan_a = scan_logs(run_a)
        scan_b = scan_logs(run_b)

        # Create comparison report
        comparison = {
            "version": "1",
            "run_a": {"path": run_a, "anomalies": scan_a.get("rules_fired", [])},
            "run_b": {"path": run_b, "anomalies": scan_b.get("rules_fired", [])},
            "earliest_divergent_step": None,
        }

        # Find earliest divergent step if both have step data
        if scan_a.get("earliest_step") and scan_b.get("earliest_step"):
            comparison["earliest_divergent_step"] = min(
                scan_a["earliest_step"], scan_b["earliest_step"]
            )

        # Write report
        mkdir_reports()
        write_json_report(comparison, "rldk_reports/run_comparison.json")

        typer.echo(
            "\nComparison complete. Report saved to rldk_reports/run_comparison.json"
        )

        # Print summary
        anomalies_a = len(scan_a.get("rules_fired", []))
        anomalies_b = len(scan_b.get("rules_fired", []))
        typer.echo(f"Run A anomalies: {anomalies_a}")
        typer.echo(f"Run B anomalies: {anomalies_b}")

        if comparison["earliest_divergent_step"]:
            typer.echo(
                f"Earliest divergent step: {comparison['earliest_divergent_step']}"
            )

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@forensics_app.command(name="diff-ckpt")
def forensics_diff_ckpt(
    ckpt_a: str = typer.Argument(..., help="Path to first checkpoint"),
    ckpt_b: str = typer.Argument(..., help="Path to second checkpoint"),
):
    """Compare two model checkpoints and identify parameter differences."""
    try:
        typer.echo("Comparing checkpoints:")
        typer.echo(f"  Checkpoint A: {ckpt_a}")
        typer.echo(f"  Checkpoint B: {ckpt_b}")

        # Diff checkpoints
        report = diff_checkpoints(ckpt_a, ckpt_b)

        # Validate report
        validate(CkptDiffReportV1, report)

        # Write report and plot
        mkdir_reports()
        write_json_report(report, "rldk_reports/ckpt_diff.json")

        # Create bar plot of top movers
        if report["top_movers"]:
            import matplotlib.pyplot as plt

            names = [m["name"] for m in report["top_movers"][:10]]
            l2_norms = [m["l2"] for m in report["top_movers"][:10]]

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(range(len(names)), l2_norms)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.set_xlabel("L2 Norm of Parameter Difference")
            ax.set_title("Top Parameter Changes")
            ax.invert_yaxis()

            write_png(fig, "rldk_reports/ckpt_diff.png")
            plt.close()

        typer.echo(
            "\nCheckpoint diff complete. Report saved to rldk_reports/ckpt_diff.json"
        )

        # Print summary
        summary = report["summary"]
        typer.echo(f"Total parameters: {summary['num_params']}")
        typer.echo(f"Common parameters: {summary['num_common_params']}")
        if summary['num_only_in_a'] > 0:
            typer.echo(f"Only in checkpoint A: {summary['num_only_in_a']}")
        if summary['num_only_in_b'] > 0:
            typer.echo(f"Only in checkpoint B: {summary['num_only_in_b']}")
        typer.echo(f"Average cosine similarity: {summary['avg_cosine']:.4f}")
        typer.echo(
            f"L2 norm percentiles - 5%: {summary['l2_p05']:.6f}, 50%: {summary['l2_p50']:.6f}, 95%: {summary['l2_p95']:.6f}"
        )

        if report["top_movers"]:
            typer.echo("\nTop parameter changes:")
            for i, mover in enumerate(report["top_movers"][:5]):
                note = mover.get('note', '')
                if note:
                    typer.echo(
                        f"  {i+1}. {mover['name']}: L2={mover['l2']:.6f}, cosine={mover['cosine']:.4f} ({note})"
                    )
                else:
                    typer.echo(
                        f"  {i+1}. {mover['name']}: L2={mover['l2']:.6f}, cosine={mover['cosine']:.4f}"
                    )

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@forensics_app.command(name="env-audit")
def forensics_env_audit(
    repo_or_run: str = typer.Argument(..., help="Path to repository or run directory"),
):
    """Audit environment for determinism and reproducibility."""
    try:
        typer.echo(f"Auditing environment for: {repo_or_run}")

        # Run audit
        determinism_card, lock_content = audit_environment(repo_or_run)

        # Validate determinism card
        validate(DeterminismCardV1, determinism_card)

        # Write outputs
        mkdir_reports()
        write_json_report(determinism_card, "rldk_reports/determinism_card.json")

        with open("rldk.lock", "w") as f:
            f.write(lock_content)

        typer.echo("\nEnvironment audit complete.")
        typer.echo("  Determinism card: rldk_reports/determinism_card.json")
        typer.echo("  Lock file: rldk.lock")

        # Print summary
        flags = determinism_card["flags"]
        typer.echo("\nKey findings:")
        typer.echo(f"  Deterministic: {determinism_card['pass']}")
        typer.echo(f"  CUDNN deterministic: {flags['cudnn_deterministic']}")
        typer.echo(f"  Tokenizers parallelism: {flags['tokenizers_parallelism']}")

        if determinism_card["nondeterminism_hints"]:
            typer.echo(
                f"  Nondeterminism hints: {len(determinism_card['nondeterminism_hints'])}"
            )
            for hint in determinism_card["nondeterminism_hints"][:3]:
                typer.echo(f"    - {hint}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@forensics_app.command(name="log-scan")
def forensics_log_scan(
    run_or_export: str = typer.Argument(..., help="Path to run or export directory"),
):
    """Scan training logs for PPO anomalies and issues."""
    try:
        typer.echo(f"Scanning logs: {run_or_export}")

        # Scan logs
        report = scan_logs(run_or_export)

        # Validate report
        validate(PPOScanReportV1, report)

        # Write report
        mkdir_reports()
        write_json_report(report, "rldk_reports/ppo_scan.json")

        typer.echo("\nLog scan complete. Report saved to rldk_reports/ppo_scan.json")

        # Print summary
        rules_fired = report.get("rules_fired", [])
        typer.echo(f"Rules fired: {len(rules_fired)}")

        if rules_fired:
            typer.echo("Anomalies detected:")
            for rule in rules_fired:
                typer.echo(f"  - {rule['rule']}: {rule['description']}")
                if rule.get("step_range"):
                    typer.echo(
                        f"    Steps: {rule['step_range'][0]} to {rule['step_range'][1]}"
                    )
        else:
            typer.echo("No anomalies detected.")

        if report.get("earliest_step"):
            typer.echo(f"Earliest step with data: {report['earliest_step']}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@forensics_app.command(name="doctor")
def forensics_doctor(
    run_or_repo: str = typer.Argument(..., help="Path to run or repository directory"),
):
    """Run comprehensive diagnostics on a training run or repository."""
    try:
        typer.echo(f"Running diagnostics on: {run_or_repo}")

        # Run env audit
        typer.echo("\n1. Environment audit...")
        determinism_card, lock_content = audit_environment(run_or_repo)

        # Run log scan
        typer.echo("2. Log scan...")
        scan_report = scan_logs(run_or_repo)

        # Write outputs
        mkdir_reports()
        write_json_report(determinism_card, "rldk_reports/determinism_card.json")
        write_json_report(scan_report, "rldk_reports/ppo_scan.json")

        with open("rldk.lock", "w") as f:
            f.write(lock_content)

        # Print summary
        typer.echo("\nDiagnostics complete!")
        typer.echo("Files generated:")
        typer.echo("  - rldk_reports/determinism_card.json")
        typer.echo("  - rldk_reports/ppo_scan.json")
        typer.echo("  - rldk.lock")

        # Health summary
        env_healthy = determinism_card["pass"]
        log_healthy = len(scan_report.get("rules_fired", [])) == 0

        if env_healthy and log_healthy:
            typer.echo("\n✅ All systems healthy!")
        else:
            typer.echo("\n⚠️  Issues detected:")
            if not env_healthy:
                typer.echo("  - Environment has nondeterminism issues")
            if not log_healthy:
                typer.echo(
                    f"  - Training logs show {len(scan_report.get('rules_fired', []))} anomalies"
                )

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


# ============================================================================
# REWARD COMMANDS
# ============================================================================

@reward_app.command(name="reward-drift")
def reward_drift(
    model_a: str = typer.Argument(..., help="Path to first reward model directory"),
    model_b: str = typer.Argument(..., help="Path to second reward model directory"),
    prompts: str = typer.Option(
        ..., "--prompts", "-p", help="Path to prompts JSONL file"
    ),
):
    """Compare two reward models and detect drift."""
    try:
        typer.echo("Comparing reward models:")
        typer.echo(f"  Model A: {model_a}")
        typer.echo(f"  Model B: {model_b}")
        typer.echo(f"  Prompts: {prompts}")

        # Read prompts
        prompt_list = list(read_jsonl(prompts))
        prompt_texts = [p.get("text", p.get("prompt", "")) for p in prompt_list]

        if not prompt_texts:
            raise ValueError("No valid prompts found in file")

        typer.echo(f"Loaded {len(prompt_texts)} prompts")

        # Compare models
        report = compare_models(model_a, model_b, prompt_texts)

        # Validate report
        validate(RewardDriftReportV1, report)

        # Write report and plot
        mkdir_reports()
        write_json_report(report, "rldk_reports/reward_drift.json")

        # Create scatter plot
        import matplotlib.pyplot as plt

        # Load model outputs for plotting
        model_a_fn = read_reward_head(model_a)
        model_b_fn = read_reward_head(model_b)

        scores_a = model_a_fn(prompt_texts)
        scores_b = model_b_fn(prompt_texts)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(scores_a, scores_b, alpha=0.6)

        # Add diagonal line
        min_val = min(min(scores_a), min(scores_b))
        max_val = max(max(scores_a), max(scores_b))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5)

        ax.set_xlabel("Model A Scores")
        ax.set_ylabel("Model B Scores")
        ax.set_title("Reward Model Comparison")

        # Add correlation info
        ax.text(
            0.05,
            0.95,
            f"Pearson: {report['pearson']:.3f}\nSpearman: {report['spearman']:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        write_png(fig, "rldk_reports/reward_drift.png")
        plt.close()

        typer.echo(
            "\nReward drift analysis complete. Report saved to rldk_reports/reward_drift.json"
        )

        # Print summary
        typer.echo("\nCorrelation metrics:")
        typer.echo(f"  Pearson correlation: {report['pearson']:.4f}")
        typer.echo(f"  Spearman correlation: {report['spearman']:.4f}")
        typer.echo(f"  MAE (z-scored): {report['mae_z']:.4f}")
        typer.echo(f"  L2 distance (z-scored): {report['l2_z']:.4f}")
        typer.echo(f"  Sign flip rate: {report['sign_flip_rate']:.4f}")

        if report["slice_deltas"]:
            typer.echo("\nSlice analysis:")
            for slice_name, slice_data in report["slice_deltas"].items():
                typer.echo(
                    f"  {slice_name}: delta_mean={slice_data['delta_mean']:.4f}, n={slice_data['n']}"
                )

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


# Create a sub-app for reward-health commands
reward_health_app = typer.Typer(name="reward-health", help="Reward health analysis commands")
reward_app.add_typer(reward_health_app, name="reward-health")


@reward_health_app.command(name="run")
def reward_health_run(
    scores: str = typer.Option(..., "--scores", help="Path to scores JSONL file"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to health configuration YAML file"),
    out: str = typer.Option(..., "--out", help="Output directory for reports"),
    adapter: Optional[str] = typer.Option(None, "--adapter", help="Adapter type for data ingestion (custom_jsonl, trl, openrlhf, wandb)"),
    gate: bool = typer.Option(False, "--gate", help="Enable CI gate mode with exit codes (0=pass, 1=warn, 2=fail). Use 'gate' subcommand for health.json-based gating."),
):
    """Run reward health analysis on scores data."""
    try:
        typer.echo(f"Running reward health analysis on scores: {scores}")
        
        # Ingest scores data
        typer.echo("Ingesting scores data...")
        scores_data = ingest_runs(scores, adapter_hint=adapter)
        
        # Load configuration (default or user-provided)
        if config:
            typer.echo(f"Using user configuration: {config}")
        else:
            typer.echo("Using default configuration (recipes/health_default.yaml)")
        config_data = load_config(config)
        
        # Extract thresholds from config
        legacy_thresholds = get_legacy_thresholds(config_data)
        threshold_drift = legacy_thresholds['threshold_drift']
        threshold_saturation = legacy_thresholds['threshold_saturation']
        threshold_calibration = legacy_thresholds['threshold_calibration']
        threshold_shortcut = legacy_thresholds['threshold_shortcut']
        threshold_leakage = legacy_thresholds['threshold_leakage']
        
        # Run reward health analysis
        typer.echo("Running reward health analysis...")
        health_report = reward_health_analysis(
            run_data=scores_data,
            reference_data=None,  # No reference data for now
            reward_col="reward_mean",
            step_col="step",
            threshold_drift=threshold_drift,
            threshold_saturation=threshold_saturation,
            threshold_calibration=threshold_calibration,
            threshold_shortcut=threshold_shortcut,
            threshold_leakage=threshold_leakage,
        )
        
        # Generate reports
        typer.echo("Generating reports...")
        generate_reward_health_report(health_report, out)
        
        # Display results
        if health_report.passed:
            typer.echo("\n✅ Reward health check passed")
            exit_code = 0
        else:
            typer.echo("\n🚨 Reward health issues detected")
            
            if health_report.drift_detected:
                typer.echo("  - Reward drift detected")
            if health_report.saturation_issues:
                typer.echo(f"  - {len(health_report.saturation_issues)} saturation issues")
            if health_report.calibration_score < threshold_calibration:
                typer.echo(f"  - Poor calibration (score: {health_report.calibration_score:.3f})")
            if health_report.shortcut_signals:
                typer.echo(f"  - {len(health_report.shortcut_signals)} shortcut signals")
            if health_report.label_leakage_risk > threshold_leakage:
                typer.echo(f"  - Label leakage risk: {health_report.label_leakage_risk:.3f}")
            
            # Determine exit code based on severity
            critical_issues = 0
            if health_report.drift_detected:
                critical_issues += 1
            if health_report.label_leakage_risk > threshold_leakage:
                critical_issues += 1
            
            if critical_issues > 0:
                exit_code = 2  # Fail - critical issues
            else:
                exit_code = 1  # Warn - non-critical issues
        
        typer.echo(f"\nReports saved to: {out}")
        typer.echo("  - reward_health_card.md")
        typer.echo("  - reward_health_summary.json")
        if health_report.drift_metrics is not None and not health_report.drift_metrics.empty:
            typer.echo("  - drift_analysis.csv")
        typer.echo("  - calibration_plots.png")
        
        # Handle gate mode
        if gate:
            if exit_code == 0:
                typer.echo("GATE: PASS")
            elif exit_code == 1:
                typer.echo("GATE: WARN")
            else:
                typer.echo("GATE: FAIL")
            raise typer.Exit(exit_code)
    
    except typer.Exit:
        # Re-raise typer.Exit to preserve intended exit codes
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        if gate:
            typer.echo("GATE: FAIL")
            raise typer.Exit(2)
        else:
            raise typer.Exit(1)


@reward_health_app.command(name="gate")
def reward_health_gate(
    from_path: str = typer.Option(..., "--from", help="Path to health.json file"),
):
    """Gate CI based on health.json results (exit codes: 0=pass, 3=fail)."""
    try:
        typer.echo(f"Reading health data from: {from_path}")
        raise_on_failure(from_path)
    except SystemExit:
        # Re-raise SystemExit to preserve exit codes from raise_on_failure
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


# ============================================================================
# EVALUATION COMMANDS
# ============================================================================

def load_jsonl_data(file_path: Path) -> pd.DataFrame:
    """
    Load data from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        DataFrame with loaded data
    """
    try:
        data = []
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue
        
        if not data:
            raise ValueError("No valid JSON records found in file")
        
        return pd.DataFrame(data)
    
    except Exception as e:
        logging.error(f"Failed to load JSONL file: {e}")
        raise


def run_evaluation_suite(
    data: pd.DataFrame,
    suite_name: str,
    output_column: str = "output",
    events_column: str = "events",
    **kwargs
) -> dict:
    """
    Run evaluation suite on data.
    
    Args:
        data: Input data DataFrame
        suite_name: Name of evaluation suite
        output_column: Column containing model outputs
        events_column: Column containing event logs
        **kwargs: Additional evaluation parameters
        
    Returns:
        Dictionary with evaluation results
    """
    if suite_name == "quick":
        suite = QUICK_SUITE
    elif suite_name == "comprehensive":
        suite = COMPREHENSIVE_SUITE
    elif suite_name == "safety":
        suite = SAFETY_SUITE
    else:
        raise ValueError(f"Unknown suite: {suite_name}")
    
    results = {
        "suite_name": suite_name,
        "suite_description": suite["description"],
        "evaluations": {},
        "summary": {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "errors": []
        }
    }
    
    # Add output column to data if not present
    if output_column not in data.columns:
        data[output_column] = "No output data available"
    
    # Add events column to data if not present
    if events_column not in data.columns:
        data[events_column] = "[]"
    
    for eval_name, eval_func in suite["evaluations"].items():
        try:
            logging.info(f"Running evaluation: {eval_name}")
            
            # Handle different evaluation types
            if eval_name == "throughput":
                result = evaluate_throughput(data, log_column=events_column, **kwargs)
            elif eval_name == "toxicity":
                result = evaluate_toxicity(data, output_column=output_column, **kwargs)
            elif eval_name == "bias":
                result = evaluate_bias(data, output_column=output_column, **kwargs)
            else:
                # For other evaluations, try with default parameters
                result = eval_func(data, **kwargs)
            
            results["evaluations"][eval_name] = result
            
            # Check if the evaluation actually succeeded (no error in result)
            if "error" in result and result["error"]:
                logging.warning(f"Evaluation {eval_name} completed but with errors: {result['error']}")
                results["summary"]["failed_evaluations"] += 1
            else:
                results["summary"]["successful_evaluations"] += 1
            
        except Exception as e:
            logging.error(f"Evaluation {eval_name} failed: {e}")
            results["evaluations"][eval_name] = {
                "score": 0.0,
                "details": f"Evaluation failed: {str(e)}",
                "error": str(e)
            }
            results["summary"]["errors"].append({
                "evaluation": eval_name,
                "error": str(e)
            })
            results["summary"]["failed_evaluations"] += 1
        
        results["summary"]["total_evaluations"] += 1
    
    # Calculate overall score
    successful_scores = [
        eval_result["score"] 
        for eval_result in results["evaluations"].values()
        if "score" in eval_result and "error" not in eval_result
    ]
    
    if successful_scores:
        results["summary"]["overall_score"] = sum(successful_scores) / len(successful_scores)
    else:
        results["summary"]["overall_score"] = 0.0
    
    return results


@evals_app.command()
def evaluate(
    input_file: Path = typer.Argument(..., help="Path to JSONL input file"),
    suite: str = typer.Option("quick", "--suite", "-s", help="Evaluation suite to run (quick/comprehensive/safety)"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to output JSON file"),
    output_column: str = typer.Option("output", "--output-column", help="Column name containing model outputs"),
    events_column: str = typer.Option("events", "--events-column", help="Column name containing event logs"),
    min_samples: int = typer.Option(10, "--min-samples", help="Minimum samples required for evaluation"),
    timeout: int = typer.Option(300, "--timeout", help="Timeout in seconds for evaluation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """
    Run evaluation suite on JSONL data.
    
    Examples:
        rldk evals evaluate data.jsonl --suite comprehensive --output results.json
        rldk evals evaluate data.jsonl --suite quick --min-samples 50 --verbose
        rldk evals evaluate data.jsonl --suite safety --timeout 600
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        # Validate input
        print_operation_status("Validating input", "start")
        
        # Validate input file
        input_path = validate_file_path(input_file, must_exist=True, file_extensions=[".jsonl"])
        
        # Validate suite
        valid_suites = ["quick", "comprehensive", "safety"]
        if suite not in valid_suites:
            raise ValidationError(
                f"Invalid evaluation suite: {suite}",
                suggestion=f"Use one of: {', '.join(valid_suites)}",
                error_code="INVALID_SUITE"
            )
        
        # Validate min_samples
        if min_samples < 1:
            raise ValidationError(
                f"Minimum samples must be at least 1, got: {min_samples}",
                suggestion="Use a positive integer for minimum samples",
                error_code="INVALID_MIN_SAMPLES"
            )
        
        print_operation_status("Input validation", "success")
        
        # Load data with progress indication
        with timed_operation_context("Data loading"):
            logging.info(f"Loading data from {input_file}")
            data = load_jsonl_data(input_file)
            
            if data.empty:
                raise ValidationError(
                    "No data found in input file",
                    suggestion="Ensure the JSONL file contains valid data",
                    error_code="NO_DATA_FOUND"
                )
            
            logging.info(f"Loaded {len(data)} records")
        
        # Validate data has required columns
        required_columns = [output_column, events_column]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print_operation_status("Data validation", "warning", f"Missing columns: {missing_columns}")
            # Add missing columns with default values
            for col in missing_columns:
                if col == output_column:
                    data[col] = "No output data available"
                elif col == events_column:
                    data[col] = "[]"
        
        # Check if we have enough samples
        if len(data) < min_samples:
            print_operation_status("Sample validation", "warning", 
                                 f"Only {len(data)} samples available, minimum is {min_samples}")
        
        # Run evaluation with timeout and progress indication
        @with_timeout(timeout)
        def run_evaluation():
            return run_evaluation_suite(
                data=data,
                suite_name=suite,
                output_column=output_column,
                events_column=events_column,
                min_samples=min_samples
            )
        
        with timed_operation_context(f"{suite} evaluation suite"):
            logging.info(f"Running {suite} evaluation suite")
            results = run_evaluation()
        
        # Output results
        if output_file:
            print_operation_status("Saving results", "start")
            try:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print_operation_status("Saving results", "success", f"Saved to {output_file}")
            except Exception as e:
                raise ValidationError(
                    f"Failed to save output file: {e}",
                    suggestion="Check write permissions and disk space",
                    error_code="SAVE_FAILED"
                ) from e
        else:
            # Print to stdout
            print(json.dumps(results, indent=2))
        
        # Print summary
        summary = results["summary"]
        print_operation_status("Evaluation", "success", 
                             f"{summary['successful_evaluations']}/{summary['total_evaluations']} successful")
        
        typer.echo(f"\n📊 Evaluation Results:")
        typer.echo(f"  Suite: {suite}")
        typer.echo(f"  Samples: {len(data)}")
        typer.echo(f"  Successful: {summary['successful_evaluations']}")
        typer.echo(f"  Failed: {summary['failed_evaluations']}")
        typer.echo(f"  Overall Score: {summary['overall_score']:.3f}")
        
        if summary["errors"]:
            typer.echo(f"\n⚠️  Failed Evaluations:")
            for error in summary["errors"]:
                typer.echo(f"  - {error['evaluation']}: {error['error']}")
        
        # Exit with error code if any evaluations failed
        if summary["failed_evaluations"] > 0:
            raise typer.Exit(1)
    
    except ValidationError as e:
        typer.echo(format_error_message(e), err=True)
        print_usage_examples("evaluate", [
            "rldk evals evaluate data.jsonl --suite comprehensive --output results.json",
            "rldk evals evaluate data.jsonl --suite quick --min-samples 50 --verbose",
            "rldk evals evaluate data.jsonl --suite safety --timeout 600"
        ])
        print_troubleshooting_tips([
            "Ensure the input file is a valid JSONL file",
            "Check that the specified columns exist in your data",
            "Use --verbose flag for detailed output",
            "Try reducing --min-samples if you have limited data"
        ])
        raise typer.Exit(1)
    except TimeoutError as e:
        typer.echo(format_error_message(e), err=True)
        print_troubleshooting_tips([
            "Try increasing the --timeout value",
            "Use a smaller dataset or --min-samples",
            "Check system resources and performance"
        ])
        raise typer.Exit(1)
    except EvaluationError as e:
        typer.echo(format_error_message(e), err=True)
        print_troubleshooting_tips([
            "Check that your data contains the required columns",
            "Ensure the evaluation suite is appropriate for your data",
            "Use --verbose flag to see detailed error information"
        ])
        raise typer.Exit(1)
    except Exception as e:
        log_error_with_context(e, "evaluate command")
        typer.echo(format_error_message(e, "Evaluation failed"), err=True)
        raise typer.Exit(1)


@evals_app.command()
def list_suites():
    """List available evaluation suites."""
    suites = {
        "quick": QUICK_SUITE,
        "comprehensive": COMPREHENSIVE_SUITE,
        "safety": SAFETY_SUITE
    }
    
    print("Available evaluation suites:")
    print()
    
    for name, suite in suites.items():
        print(f"  {name}:")
        print(f"    Description: {suite['description']}")
        print(f"    Default sample size: {suite['default_sample_size']}")
        print(f"    Estimated runtime: {suite['estimated_runtime']}")
        print(f"    Evaluations: {', '.join(suite['evaluations'].keys())}")
        print()


@evals_app.command(name="validate-data")
def validate_data(
    input_file: Path = typer.Argument(..., help="Path to JSONL input file to validate"),
    output_column: str = typer.Option("output", "--output-column", help="Column name containing model outputs"),
    events_column: str = typer.Option("events", "--events-column", help="Column name containing event logs")
):
    """
    Validate JSONL file structure and data.
    
    Example:
        rldk evals validate-data data.jsonl
    """
    try:
        logging.info(f"Validating {input_file}")
        data = load_jsonl_data(input_file)
        
        print(f"File validation results:")
        print(f"  Total records: {len(data)}")
        print(f"  Columns: {list(data.columns)}")
        
        # Check required columns
        if output_column in data.columns:
            output_count = data[output_column].notna().sum()
            print(f"  Output column '{output_column}': {output_count} non-null values")
        else:
            print(f"  Output column '{output_column}': NOT FOUND")
        
        if events_column in data.columns:
            events_count = data[events_column].notna().sum()
            print(f"  Events column '{events_column}': {events_count} non-null values")
        else:
            print(f"  Events column '{events_column}': NOT FOUND")
        
        # Check data quality
        print(f"  Missing values: {data.isnull().sum().sum()}")
        
        logging.info("Validation complete")
    
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        sys.exit(1)


# Backward compatibility alias
@evals_app.command(name="validate")
def validate_alias(
    input_file: Path = typer.Argument(..., help="Path to JSONL input file to validate"),
    output_column: str = typer.Option("output", "--output-column", help="Column name containing model outputs"),
    events_column: str = typer.Option("events", "--events-column", help="Column name containing event logs")
):
    """
    Validate JSONL file structure and data (alias for validate-data).
    
    Example:
        rldk evals validate data.jsonl
    """
    validate_data(input_file, output_column, events_column)


# ============================================================================
# MAIN CLI COMMANDS
# ============================================================================

@app.command(name="ingest")
def ingest(
    runs: str = typer.Argument(
        ..., help="Path to runs directory, file, or wandb:// URI"
    ),
    adapter: Optional[str] = typer.Option(
        None, "--adapter", "-a", help="Adapter type (trl, openrlhf, wandb, custom_jsonl)"
    ),
    output: Optional[str] = typer.Option(
        "metrics.jsonl", "--output", "-o", help="Output file path"
    ),
    validate: bool = typer.Option(
        True, "--validate/--no-validate", help="Validate input data before processing"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """Ingest training runs from various sources.
    
    Examples:
        rldk ingest /path/to/logs --adapter trl
        rldk ingest wandb://entity/project/run_id --adapter wandb
        rldk ingest data.jsonl --adapter custom_jsonl --output results.jsonl
    """
    try:
        # Setup logging
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        
        ensure_config_initialized()
        
        # Validate input
        if validate:
            print_operation_status("Validating input", "start")
            
            # Check if source exists and is accessible
            if runs.startswith("wandb://"):
                validate_adapter_source(runs, ["wandb:// URI"])
            else:
                source_path = validate_file_path(runs, must_exist=True)
                if source_path.is_file():
                    validate_file_path(runs, file_extensions=[".jsonl", ".log"])
                elif source_path.is_dir():
                    # Check if directory contains valid log files
                    log_files = list(source_path.glob("*.jsonl")) + list(source_path.glob("*.log"))
                    if not log_files:
                        raise ValidationError(
                            f"No log files found in directory: {source_path}",
                            suggestion="Ensure the directory contains .jsonl or .log files",
                            error_code="NO_LOG_FILES_FOUND"
                        )
            
            # Validate adapter if specified
            if adapter:
                valid_adapters = ["trl", "openrlhf", "wandb", "custom_jsonl"]
                if adapter not in valid_adapters:
                    raise ValidationError(
                        f"Invalid adapter: {adapter}",
                        suggestion=f"Use one of: {', '.join(valid_adapters)}",
                        error_code="INVALID_ADAPTER"
                    )
            
            print_operation_status("Input validation", "success")
        
        # Ingest the runs with progress indication
        with timed_operation_context("Data ingestion"):
            typer.echo(f"Ingesting runs from: {runs}")
            
            if adapter:
                typer.echo(f"Using adapter: {adapter}")
            
            df = ingest_runs(runs, adapter)
            
            if df.empty:
                raise ValidationError(
                    "No data found in source",
                    suggestion="Check that the source contains valid training data",
                    error_code="NO_DATA_FOUND"
                )

        # Save to output file
        if output:
            print_operation_status("Saving results", "start")
            try:
                df.to_json(output, orient="records", lines=True)
                print_operation_status("Saving results", "success", f"Saved to {output}")
            except Exception as e:
                raise ValidationError(
                    f"Failed to save output file: {e}",
                    suggestion="Check write permissions and disk space",
                    error_code="SAVE_FAILED"
                ) from e

        # Display summary
        typer.echo(f"\n📊 Ingestion Summary:")
        typer.echo(f"  Records: {len(df)}")
        typer.echo(f"  Columns: {', '.join(df.columns)}")
        if not df.empty and 'step' in df.columns:
            typer.echo(f"  Steps range: {df['step'].min()} to {df['step'].max()}")
        
        # Show sample data
        if not df.empty and verbose:
            typer.echo("\n📋 Sample data:")
            typer.echo(df.head().to_string())

    except ValidationError as e:
        typer.echo(format_error_message(e), err=True)
        print_usage_examples("ingest", [
            "rldk ingest /path/to/logs --adapter trl",
            "rldk ingest wandb://entity/project/run_id --adapter wandb",
            "rldk ingest data.jsonl --adapter custom_jsonl --output results.jsonl"
        ])
        print_troubleshooting_tips([
            "Ensure the source path exists and is accessible",
            "Check that the adapter matches your data format",
            "Use --verbose flag for detailed output",
            "Try auto-detection by omitting --adapter"
        ])
        raise typer.Exit(1)
    except AdapterError as e:
        typer.echo(format_error_message(e), err=True)
        print_troubleshooting_tips([
            "Check that the data format matches the specified adapter",
            "Try different adapter types: trl, openrlhf, wandb, custom_jsonl",
            "Use --verbose flag to see detailed error information"
        ])
        raise typer.Exit(1)
    except Exception as e:
        log_error_with_context(e, "ingest command")
        typer.echo(format_error_message(e, "Ingestion failed"), err=True)
        raise typer.Exit(1)


@app.command(name="diff")
def diff(
    a: str = typer.Option(..., "--a", "-a", help="Path or wandb:// URI for run A"),
    b: str = typer.Option(..., "--b", "-b", help="Path or wandb:// URI for run B"),
    signals: List[str] = typer.Option(
        ..., "--signals", "-s", help="Metrics to monitor for divergence"
    ),
    tolerance: float = typer.Option(
        2.0, "--tolerance", "-t", help="Z-score threshold for violation detection"
    ),
    k: int = typer.Option(
        3, "--k", "-k", help="Number of consecutive violations required"
    ),
    window: int = typer.Option(
        50, "--window", "-w", help="Rolling window size for z-score calculation"
    ),
    output_dir: str = typer.Option(
        "diff_analysis", "--output-dir", "-o", help="Output directory for reports"
    ),
):
    """Find first divergence between two training runs."""
    try:
        typer.echo("Comparing runs:")
        typer.echo(f"  Run A: {a}")
        typer.echo(f"  Run B: {b}")
        typer.echo(f"  Signals: {', '.join(signals)}")
        typer.echo(f"  Tolerance: {tolerance}")
        typer.echo(f"  K-consecutive: {k}")
        typer.echo(f"  Window size: {window}")

        # Ingest both runs
        typer.echo("\nIngesting run A...")
        df_a = ingest_runs(a)

        typer.echo("Ingesting run B...")
        df_b = ingest_runs(b)

        # Find divergence
        typer.echo("\nAnalyzing divergence...")
        # Handle case where signals might be a comma-separated string
        if isinstance(signals, str):
            signals = [s.strip() for s in signals.split(",")]
        elif len(signals) == 1 and "," in signals[0]:
            # Handle case where signals is a list with one comma-separated string
            signals = [s.strip() for s in signals[0].split(",")]
        report = first_divergence(df_a, df_b, signals, k, window, tolerance, output_dir)

        # Write reports
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create diff report JSON
        diff_report = {
            "version": "1",
            "diverged": report.diverged,
            "first_step": report.first_step,
            "tripped_signals": report.tripped_signals,
            "notes": report.notes,
            "suspected_causes": report.suspected_causes,
        }
        write_json(diff_report, output_path / "diff_report.json")

        # Create drift card JSON
        drift_card = {
            "version": "1",
            "diverged": report.diverged,
            "first_step": report.first_step,
            "signals_monitored": signals,
            "k_consecutive": k,
            "window_size": window,
            "tolerance": tolerance,
        }
        write_json(drift_card, output_path / "drift_card.json")

        # Save events CSV if details exist
        if not report.details.empty:
            report.details.to_csv(output_path / "diff_events.csv", index=False)

        # Display results
        if report.diverged:
            typer.echo(f"\n🚨 Divergence detected at step {report.first_step}")
            # Format tripped signals (now a list of dicts with 'signal' key)
            signal_names = [signal_info["signal"] for signal_info in report.tripped_signals]
            typer.echo(f"Tripped signals: {', '.join(signal_names)}")
        else:
            typer.echo("\n✅ No significant divergence detected")

        typer.echo(f"\nReports saved to: {output_dir}")
        typer.echo("  - diff_report.json")
        typer.echo("  - drift_card.json")
        if not report.details.empty:
            typer.echo("  - diff_events.csv")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="check-determinism")
def check_determinism_cmd(
    cmd: Optional[str] = typer.Option(None, "--cmd", "-c", help="Command to run for testing"),
    compare: Optional[str] = typer.Option(
        None, "--compare", "-m", help="Metrics to compare (comma-separated)"
    ),
    steps: Optional[str] = typer.Option(
        None, "--steps", "-s", help="Specific steps to compare (comma-separated)"
    ),
    stride: int = typer.Option(
        50, "--stride", help="Step interval for comparison if steps not specified"
    ),
    replicas: int = typer.Option(
        5, "--replicas", "-r", help="Number of replicas to run"
    ),
    runs: Optional[int] = typer.Option(
        None, "--runs", help="Number of runs for determinism check (alias for replicas)"
    ),
    tolerance: float = typer.Option(
        0.01, "--tolerance", "-t", help="Tolerance for metric differences"
    ),
    device: Optional[str] = typer.Option(
        None, "--device", "-d", help="Device to use (auto-detected if None)"
    ),
    output_dir: str = typer.Option(
        "determinism_analysis",
        "--output-dir",
        "-o",
        help="Output directory for reports",
    ),
    gate: bool = typer.Option(
        False, "--gate", help="Enable CI gate mode with exit codes (0=pass, 1=warn, 2=fail)"
    ),
):
    """Check if a training command is deterministic."""
    try:
        # Use runs parameter if provided, otherwise use replicas
        actual_replicas = runs if runs is not None else replicas
        
        # Handle simplified interface for gate mode
        if gate and not cmd and not compare:
            # For gate mode, use default values
            cmd = "python -c 'import torch; print(torch.randn(1).item())'"
            compare_list = ["loss"]  # Default metric
            typer.echo("Gate mode: Using default command and metrics")
        else:
            # Parse comma-separated values
            if not compare:
                raise ValueError("--compare parameter is required when not in gate mode")
            compare_list = [c.strip() for c in compare.split(",")]
        
        steps_list = None
        if steps:
            steps_list = [int(s.strip()) for s in steps.split(",")]

        typer.echo(f"Checking determinism for command: {cmd}")
        typer.echo(f"Metrics to compare: {', '.join(compare_list)}")
        if steps_list:
            typer.echo(f"Steps to compare: {steps_list}")
        else:
            typer.echo(f"Stride: {stride}")
        typer.echo(f"Runs: {actual_replicas}")
        typer.echo(f"Tolerance: {tolerance}")
        typer.echo(f"Device: {device or 'auto-detected'}")

        # Check determinism
        typer.echo("\nRunning determinism check...")
        report = check(cmd, compare_list, steps_list, actual_replicas, device)

        # Write report
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create determinism card JSON
        determinism_card = {
            "version": "1",
            "passed": report.passed,
            "culprit": report.culprit,
            "fixes": report.fixes,
            "replica_variance": report.replica_variance,
            "rng_map": report.rng_map,
            "mismatches": report.mismatches,
            "dataloader_notes": report.dataloader_notes,
        }
        write_json(determinism_card, output_path / "determinism_card.json")

        # Display results
        if report.passed:
            typer.echo("\n✅ Determinism check passed")
            exit_code = 0
        else:
            typer.echo("\n🚨 Determinism issues found")
            if report.culprit:
                typer.echo(f"Culprit operation: {report.culprit}")
            if report.fixes:
                typer.echo("\nRecommended fixes:")
                for fix in report.fixes[:3]:  # Show first 3 fixes
                    typer.echo(f"  - {fix}")
            
            # Determine exit code based on severity and tolerance
            if len(report.mismatches) > 0:
                # Check if mismatches exceed tolerance
                max_diff = max([m.get('difference', 0) for m in report.mismatches], default=0)
                if max_diff > tolerance:
                    exit_code = 2  # Fail - mismatches exceed tolerance
                else:
                    exit_code = 1  # Warn - mismatches within tolerance
            else:
                exit_code = 1  # Warn - potential issues but no hard failures

        typer.echo(f"\nReport saved to: {output_dir}/determinism_card.json")
        
        # Handle gate mode
        if gate:
            if exit_code == 0:
                typer.echo("GATE: PASS")
            elif exit_code == 1:
                typer.echo("GATE: WARN")
            else:
                typer.echo("GATE: FAIL")
            raise typer.Exit(exit_code)

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        if gate:
            typer.echo("GATE: FAIL")
            raise typer.Exit(2)
        else:
            raise typer.Exit(1)


@app.command(name="bisect")
def bisect(
    good: str = typer.Option(..., "--good", "-g", help="Known good commit SHA"),
    bad: str = typer.Option(
        "HEAD", "--bad", "-b", help="Known bad commit SHA (default: HEAD)"
    ),
    cmd: Optional[str] = typer.Option(
        None, "--cmd", "-c", help="Command to run for testing"
    ),
    metric: Optional[str] = typer.Option(
        None, "--metric", "-m", help="Metric name to monitor"
    ),
    cmp: Optional[str] = typer.Option(
        None, "--cmp", help="Comparison operator (e.g., '> 0.2')"
    ),
    window: int = typer.Option(
        100, "--window", "-w", help="Window size for metric statistics"
    ),
    shell_predicate: Optional[str] = typer.Option(
        None, "--shell-predicate", help="Shell command that returns non-zero on failure"
    ),
):
    """Find regression using git bisect."""
    try:
        typer.echo("Starting git bisect:")
        typer.echo(f"  Good commit: {good}")
        typer.echo(f"  Bad commit: {bad}")

        if cmd and metric and cmp:
            typer.echo(f"  Command: {cmd}")
            typer.echo(f"  Metric: {metric}")
            typer.echo(f"  Comparison: {cmp}")
            typer.echo(f"  Window: {window}")
        elif shell_predicate:
            typer.echo(f"  Shell predicate: {shell_predicate}")
        else:
            raise ValueError(
                "Must provide either (cmd, metric, cmp) or shell_predicate"
            )

        # Run bisect
        typer.echo("\nRunning git bisect...")
        result = bisect_commits(
            good_sha=good,
            bad_sha=bad,
            cmd=cmd,
            metric=metric,
            cmp=cmp,
            window=window,
            shell_predicate=shell_predicate,
        )

        # Display results
        typer.echo("\n🎯 Regression found!")
        typer.echo(f"Culprit commit: {result.culprit_sha}")
        typer.echo(f"Iterations: {result.iterations}")
        typer.echo(f"Logs: {result.logs_path}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="reward-health")
def reward_health(
    run_path: str = typer.Option(..., "--run", "-r", help="Path to training run data"),
    reference_path: Optional[str] = typer.Option(
        None, "--reference", "-ref", help="Path to reference run data"
    ),
    output_dir: str = typer.Option(
        "reward_analysis", "--output-dir", "-o", help="Output directory for reports"
    ),
    reward_col: str = typer.Option(
        "reward_mean", "--reward-col", help="Column name for reward values"
    ),
    step_col: str = typer.Option(
        "step", "--step-col", help="Column name for training steps"
    ),
    threshold_drift: float = typer.Option(
        0.1, "--threshold-drift", help="P-value threshold for drift detection"
    ),
    threshold_saturation: float = typer.Option(
        0.8, "--threshold-saturation", help="Threshold for saturation detection"
    ),
    threshold_calibration: float = typer.Option(
        0.7, "--threshold-calibration", help="Threshold for calibration quality"
    ),
    threshold_shortcut: float = typer.Option(
        0.6, "--threshold-shortcut", help="Threshold for shortcut signal detection"
    ),
    threshold_leakage: float = typer.Option(
        0.3, "--threshold-leakage", help="Threshold for label leakage risk"
    ),
    gate: bool = typer.Option(
        False, "--gate", help="Enable CI gate mode with exit codes (0=pass, 1=warn, 2=fail)"
    ),
):
    """Analyze reward model health and detect pathologies."""
    try:
        typer.echo(f"Analyzing reward health for run: {run_path}")

        # Ingest run data
        typer.echo("Ingesting run data...")
        run_data = ingest_runs(run_path)

        # Ingest reference data if provided
        reference_data = None
        if reference_path:
            typer.echo("Ingesting reference data...")
            reference_data = ingest_runs(reference_path)

        # Run reward health analysis
        typer.echo("Running reward health analysis...")
        health_report = health(
            run_data=run_data,
            reference_data=reference_data,
            reward_col=reward_col,
            step_col=step_col,
            threshold_drift=threshold_drift,
            threshold_saturation=threshold_saturation,
            threshold_calibration=threshold_calibration,
            threshold_shortcut=threshold_shortcut,
            threshold_leakage=threshold_leakage,
        )

        # Generate reports
        typer.echo("Generating reports...")
        generate_reward_health_report(health_report, output_dir)

        # Display results
        if health_report.passed:
            typer.echo("\n✅ Reward health check passed")
            exit_code = 0
        else:
            typer.echo("\n🚨 Reward health issues detected")

            if health_report.drift_detected:
                typer.echo("  - Reward drift detected")
            if health_report.saturation_issues:
                typer.echo(
                    f"  - {len(health_report.saturation_issues)} saturation issues"
                )
            if health_report.calibration_score < threshold_calibration:
                typer.echo(
                    f"  - Poor calibration (score: {health_report.calibration_score:.3f})"
                )
            if health_report.shortcut_signals:
                typer.echo(
                    f"  - {len(health_report.shortcut_signals)} shortcut signals"
                )
            if health_report.label_leakage_risk > threshold_leakage:
                typer.echo(
                    f"  - Label leakage risk: {health_report.label_leakage_risk:.3f}"
                )
            
            # Determine exit code based on severity
            critical_issues = 0
            if health_report.drift_detected:
                critical_issues += 1
            if health_report.label_leakage_risk > threshold_leakage:
                critical_issues += 1
            
            if critical_issues > 0:
                exit_code = 2  # Fail - critical issues
            else:
                exit_code = 1  # Warn - non-critical issues

        typer.echo(f"\nReports saved to: {output_dir}")
        typer.echo("  - reward_health_card.md")
        typer.echo("  - reward_health_summary.json")
        if (
            health_report.drift_metrics is not None
            and not health_report.drift_metrics.empty
        ):
            typer.echo("  - drift_analysis.csv")
        typer.echo("  - calibration_plots.png")
        
        # Handle gate mode
        if gate:
            if exit_code == 0:
                typer.echo("GATE: PASS")
            elif exit_code == 1:
                typer.echo("GATE: WARN")
            else:
                typer.echo("GATE: FAIL")
            raise typer.Exit(exit_code)

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        if gate:
            typer.echo("GATE: FAIL")
            raise typer.Exit(2)
        else:
            raise typer.Exit(1)


@app.command(name="replay")
def replay_cmd(
    run_path: str = typer.Option(
        ..., "--run", "-r", help="Path to original training run data"
    ),
    command: str = typer.Option(
        ..., "--command", "-c", help="Training command to replay (should accept --seed)"
    ),
    metrics: List[str] = typer.Option(
        ..., "--metrics", "-m", help="Metrics to compare (comma-separated)"
    ),
    tolerance: float = typer.Option(
        0.01, "--tolerance", "-t", help="Tolerance for metric differences (relative)"
    ),
    max_steps: Optional[int] = typer.Option(
        None, "--max-steps", "-s", help="Maximum steps to replay"
    ),
    output_dir: str = typer.Option(
        "replay_results", "--output-dir", "-o", help="Output directory for results"
    ),
    device: Optional[str] = typer.Option(
        None, "--device", "-d", help="Device to use (auto-detected if None)"
    ),
    no_wandb: bool = typer.Option(
        False, "--no-wandb", help="Disable W&B logging and use file logging only"
    ),
):
    """Replay a training run with the original seed and verify reproducibility."""
    try:
        # Parse comma-separated metrics
        metrics_list = [m.strip() for m in metrics.split(",")]

        typer.echo(f"Replaying training run: {run_path}")
        typer.echo(f"Training command: {command}")
        typer.echo(f"Metrics to compare: {', '.join(metrics_list)}")
        typer.echo(f"Tolerance: {tolerance}")
        if max_steps:
            typer.echo(f"Max steps: {max_steps}")
        typer.echo(f"Device: {device or 'auto-detected'}")

        # Run replay
        typer.echo("\nStarting seeded replay...")
        replay_report = replay(
            run_path=run_path,
            training_command=command,
            metrics_to_compare=metrics_list,
            tolerance=tolerance,
            max_steps=max_steps,
            output_dir=output_dir,
            device=device,
        )

        # Display results
        if replay_report.passed:
            typer.echo("\n✅ Seeded replay passed - metrics match within tolerance")
        else:
            typer.echo(
                f"\n🚨 Seeded replay failed - {len(replay_report.mismatches)} tolerance violations"
            )

            # Show summary of violations
            for metric in replay_report.metrics_compared:
                stats = replay_report.comparison_stats.get(metric, {})
                violations = stats.get("tolerance_violations", 0)
                max_diff = stats.get("max_diff", 0.0)
                if violations > 0:
                    typer.echo(
                        f"  {metric}: {violations} violations, max diff: {max_diff:.6f}"
                    )

        typer.echo(f"\nReplay completed in {replay_report.replay_duration:.2f} seconds")
        typer.echo(f"Original seed: {replay_report.original_seed}")
        typer.echo(f"Replay seed: {replay_report.replay_seed}")
        typer.echo(f"\nResults saved to: {output_dir}")
        typer.echo("  - replay_metrics.jsonl")
        typer.echo("  - replay_comparison.json")
        if replay_report.mismatches:
            typer.echo("  - replay_mismatches.json")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="eval")
def eval_cmd(
    run_path: str = typer.Option(..., "--run", "-r", help="Path to training run data"),
    suite: str = typer.Option("quick", "--suite", "-s", help="Evaluation suite to run"),
    output_dir: str = typer.Option(
        "eval_results", "--output-dir", "-o", help="Output directory for results"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    sample_size: Optional[int] = typer.Option(
        None, "--sample-size", help="Number of samples to evaluate"
    ),
    no_wandb: bool = typer.Option(
        False, "--no-wandb", help="Disable W&B logging and use file logging only"
    ),
):
    """Run evaluation suite with statistical analysis."""
    try:
        typer.echo(f"Running evaluation suite '{suite}' on run: {run_path}")

        # Ingest run data
        typer.echo("Ingesting run data...")
        run_data = ingest_runs(run_path)

        # Run evaluation
        typer.echo(f"Running {suite} evaluation suite...")
        eval_result = run(
            run_data=run_data,
            suite=suite,
            seed=seed,
            sample_size=sample_size,
            output_dir=output_dir,
        )

        # Display results
        typer.echo(f"\n📊 Evaluation Results for {suite} suite")
        typer.echo(f"Sample size: {eval_result.sample_size}")
        typer.echo(f"Seed: {eval_result.seed}")

        typer.echo("\nScores:")
        for metric, score in eval_result.scores.items():
            if not np.isnan(score):
                ci = eval_result.confidence_intervals.get(metric, (np.nan, np.nan))
                effect_size = eval_result.effect_sizes.get(metric, np.nan)

                ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if not np.isnan(ci[0]) else "N/A"
                effect_str = (
                    f"{effect_size:.3f}" if not np.isnan(effect_size) else "N/A"
                )

                typer.echo(
                    f"  {metric}: {score:.3f} (CI: {ci_str}, Effect: {effect_str})"
                )

        typer.echo(f"\nResults saved to: {output_dir}")
        typer.echo("  - eval_card.md")
        typer.echo("  - eval_results.jsonl")
        typer.echo("  - eval_summary.json")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


# Legacy command aliases for backward compatibility
@app.command(name="compare-runs")
def compare_runs(
    run_a: str = typer.Argument(..., help="Path to first run directory"),
    run_b: str = typer.Argument(..., help="Path to second run directory"),
):
    """Compare two training runs and identify divergences."""
    forensics_compare_runs(run_a, run_b)


@app.command(name="diff-ckpt")
def diff_ckpt(
    ckpt_a: str = typer.Argument(..., help="Path to first checkpoint"),
    ckpt_b: str = typer.Argument(..., help="Path to second checkpoint"),
):
    """Compare two model checkpoints and identify parameter differences."""
    forensics_diff_ckpt(ckpt_a, ckpt_b)


@app.command(name="env-audit")
def env_audit(
    repo_or_run: str = typer.Argument(..., help="Path to repository or run directory"),
):
    """Audit environment for determinism and reproducibility."""
    forensics_env_audit(repo_or_run)


@app.command(name="log-scan")
def log_scan(
    run_or_export: str = typer.Argument(..., help="Path to run or export directory"),
):
    """Scan training logs for PPO anomalies and issues."""
    forensics_log_scan(run_or_export)


@app.command(name="track")
def track(
    experiment_name: str = typer.Argument(..., help="Name of the experiment to track"),
    output_dir: str = typer.Option(
        "./runs", "--output-dir", "-o", help="Output directory for tracking data"
    ),
    no_wandb: bool = typer.Option(
        False, "--no-wandb", help="Disable W&B logging and use file logging only"
    ),
    wandb_project: Optional[str] = typer.Option(
        None, "--wandb-project", help="W&B project name (default: rldk-experiments)"
    ),
    tags: Optional[str] = typer.Option(
        None, "--tags", help="Comma-separated list of tags"
    ),
    notes: Optional[str] = typer.Option(
        None, "--notes", help="Additional notes for the experiment"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Keep tracker running in interactive mode"
    ),
):
    """Start tracking an experiment with W&B (default) or file logging."""
    try:
        typer.echo(f"Starting experiment tracking: {experiment_name}")
        
        # Parse tags if provided
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
        
        # Create tracking configuration
        config = TrackingConfig(
            experiment_name=experiment_name,
            output_dir=Path(output_dir),
            save_to_wandb=not no_wandb,  # Disable W&B if --no-wandb flag is used
            wandb_project=wandb_project,
            tags=tag_list,
            notes=notes,
        )
        
        # Create tracker
        tracker = ExperimentTracker(config)
        
        # Actually start the experiment tracking
        tracking_data = tracker.start_experiment()
        
        typer.echo(f"✅ Experiment tracking started successfully")
        typer.echo(f"  Experiment: {experiment_name}")
        typer.echo(f"  Experiment ID: {tracking_data['experiment_id']}")
        typer.echo(f"  Output directory: {output_dir}")
        typer.echo(f"  W&B enabled: {not no_wandb}")
        if not no_wandb:
            typer.echo(f"  W&B project: {config.wandb_project}")
        if tag_list:
            typer.echo(f"  Tags: {', '.join(tag_list)}")
        
        if interactive:
            typer.echo("\n🔄 Interactive mode enabled. Tracker is ready for use.")
            typer.echo("Available commands:")
            typer.echo("  tracker.log_metric('loss', 0.5)")
            typer.echo("  tracker.log_metric('accuracy', 0.8)")
            typer.echo("  tracker.track_dataset(dataset, 'my_dataset')")
            typer.echo("  tracker.track_model(model, 'my_model')")
            typer.echo("  tracker.finish_experiment()")
            typer.echo("\nPress Ctrl+C to finish the experiment and exit.")
            
            try:
                # Keep the process alive for interactive use
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                typer.echo("\n\nFinishing experiment...")
                summary = tracker.finish_experiment()
                typer.echo("✅ Experiment completed successfully!")
        else:
            # Non-interactive mode - finish immediately
            typer.echo("\n📊 Experiment tracking completed.")
            typer.echo("Environment, Git, and seed state have been captured.")
            typer.echo("Use --interactive flag to keep tracker running for manual logging.")
            
            summary = tracker.finish_experiment()
            typer.echo("✅ Experiment completed successfully!")
        
        return tracker
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="reward-drift")
def reward_drift_legacy(
    model_a: str = typer.Argument(..., help="Path to first reward model directory"),
    model_b: str = typer.Argument(..., help="Path to second reward model directory"),
    prompts: str = typer.Option(
        ..., "--prompts", "-p", help="Path to prompts JSONL file"
    ),
):
    """Compare two reward models and detect drift."""
    reward_drift(model_a, model_b, prompts)


@app.command(name="doctor")
def doctor(
    run_or_repo: str = typer.Argument(..., help="Path to run or repository directory"),
):
    """Run comprehensive diagnostics on a training run or repository."""
    forensics_doctor(run_or_repo)


@app.command(name="format-info")
def format_info(
    adapter: Optional[str] = typer.Option(None, "--adapter", "-a", help="Show format info for specific adapter"),
    examples: bool = typer.Option(False, "--examples", help="Show example data"),
):
    """Show data format information for adapters."""
    from .ingest.ingest import _get_adapter_format_requirements
    
    if adapter:
        # Show info for specific adapter
        requirements = _get_adapter_format_requirements(adapter)
        
        typer.echo(f"📋 Format requirements for '{adapter}' adapter:")
        typer.echo(f"  Description: {requirements['description']}")
        typer.echo(f"  File extensions: {', '.join(requirements['file_extensions'])}")
        typer.echo(f"  Required fields: {', '.join(requirements['required_fields'])}")
        typer.echo(f"  Optional fields: {', '.join(requirements['optional_fields'])}")
        typer.echo(f"  Suggestions: {requirements['suggestions']}")
        
        if examples and requirements['examples']:
            typer.echo("\n📝 Examples:")
            for i, example in enumerate(requirements['examples'], 1):
                typer.echo(f"  {i}. {example}")
    else:
        # Show info for all adapters
        adapters = ["trl", "openrlhf", "custom_jsonl", "wandb"]
        
        typer.echo("📋 Available adapters and their format requirements:")
        typer.echo()
        
        for adapter_name in adapters:
            requirements = _get_adapter_format_requirements(adapter_name)
            typer.echo(f"🔧 {adapter_name.upper()}:")
            typer.echo(f"  Description: {requirements['description']}")
            typer.echo(f"  File extensions: {', '.join(requirements['file_extensions'])}")
            typer.echo(f"  Required fields: {', '.join(requirements['required_fields'])}")
            typer.echo()
        
        typer.echo("💡 Use --adapter <name> to see detailed info for a specific adapter")
        typer.echo("💡 Use --examples to see example data formats")


@app.command(name="validate-format")
def validate_format(
    source: str = typer.Argument(..., help="Path to data source to validate"),
    adapter: Optional[str] = typer.Option(None, "--adapter", "-a", help="Adapter type to test"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed analysis"),
):
    """Validate data format and suggest appropriate adapter."""
    from .ingest.ingest import _analyze_source_format, _get_adapter_format_requirements
    
    typer.echo(f"🔍 Analyzing data format: {source}")
    
    # Analyze the source
    analysis = _analyze_source_format(source)
    
    typer.echo(f"📊 Analysis results:")
    typer.echo(f"  Type: {analysis['description']}")
    
    if analysis.get('files'):
        typer.echo(f"  Files found: {len(analysis['files'])}")
        if verbose:
            for file in analysis['files'][:5]:  # Show first 5 files
                typer.echo(f"    - {file}")
    
    if analysis.get('fields_found'):
        typer.echo(f"  Fields found: {', '.join(analysis['fields_found'])}")
    
    if analysis.get('issues'):
        typer.echo(f"  Issues detected: {len(analysis['issues'])}")
        if verbose:
            for issue in analysis['issues']:
                typer.echo(f"    - {issue}")
    
    # Test specific adapter if provided
    if adapter:
        typer.echo(f"\n🧪 Testing with '{adapter}' adapter...")
        try:
            from .adapters import TRLAdapter, OpenRLHFAdapter, CustomJSONLAdapter, WandBAdapter
            
            if adapter == "trl":
                adapter_instance = TRLAdapter(source)
            elif adapter == "openrlhf":
                adapter_instance = OpenRLHFAdapter(source)
            elif adapter == "custom_jsonl":
                adapter_instance = CustomJSONLAdapter(source)
            elif adapter == "wandb":
                adapter_instance = WandBAdapter(source)
            else:
                typer.echo(f"❌ Unknown adapter: {adapter}")
                raise typer.Exit(1)
            
            if adapter_instance.can_handle():
                typer.echo(f"✅ '{adapter}' adapter can handle this source")
            else:
                typer.echo(f"❌ '{adapter}' adapter cannot handle this source")
                requirements = _get_adapter_format_requirements(adapter)
                typer.echo(f"   Expected: {requirements['description']}")
        except Exception as e:
            typer.echo(f"❌ Error testing adapter: {e}")
    else:
        # Suggest adapters
        typer.echo(f"\n💡 Adapter suggestions:")
        
        if analysis['type'] == 'jsonl':
            fields = analysis.get('fields_found', [])
            if any(f in fields for f in ['global_step', 'reward_scalar', 'kl_to_ref']):
                typer.echo("  - custom_jsonl (has custom field names)")
            else:
                typer.echo("  - trl (standard format)")
                typer.echo("  - openrlhf (standard format)")
        elif analysis['type'] == 'log':
            typer.echo("  - trl (for .log files)")
            typer.echo("  - openrlhf (for .log files)")
        elif analysis['type'] == 'directory':
            typer.echo("  - trl (for directories with log files)")
            typer.echo("  - openrlhf (for directories with log files)")
            typer.echo("  - custom_jsonl (for directories with custom JSONL files)")
        
        typer.echo(f"\n💡 Use 'rldk format-info --adapter <name>' for detailed format requirements")
        typer.echo(f"💡 Use 'rldk validate-format {source} --adapter <name>' to test specific adapter")


@app.command(name="version")
def version():
    """Show version information."""
    from rldk import __version__

    typer.echo(f"RL Debug Kit version {__version__}")


@app.command(name="seed")
def seed_cmd(
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Seed value to set"),
    show: bool = typer.Option(False, "--show", help="Show current seed state"),
    deterministic: bool = typer.Option(True, "--deterministic/--non-deterministic", help="Enable deterministic behavior"),
    env: bool = typer.Option(False, "--env", help="Set environment variables for reproducibility"),
    validate: bool = typer.Option(False, "--validate", help="Validate seed consistency"),
):
    """Manage global seed for reproducible experiments.
    
    Examples:
        rldk seed --seed 42                    # Set seed to 42
        rldk seed --show                       # Show current seed state
        rldk seed --seed 1337 --env            # Set seed and environment variables
        rldk seed --validate                   # Validate current seed consistency
    """
    try:
        from rldk.utils.seed import (
            set_global_seed, get_current_seed, get_seed_state_summary,
            set_reproducible_environment, validate_seed_consistency
        )
        
        if show:
            # Show current seed state
            summary = get_seed_state_summary()
            typer.echo("🌱 Current seed state:")
            typer.echo(f"  Seed: {summary['seed']}")
            typer.echo(f"  Deterministic: {summary['deterministic']}")
            typer.echo(f"  Libraries: {', '.join(summary['libraries'])}")
            typer.echo(f"  PyTorch available: {summary['torch_available']}")
            typer.echo(f"  CUDA available: {summary['cuda_available']}")
            
            if summary['torch_available'] and summary['deterministic']:
                typer.echo(f"  CUDNN deterministic: {summary.get('cudnn_deterministic', False)}")
                typer.echo(f"  CUDNN benchmark: {summary.get('cudnn_benchmark', True)}")
        
        elif validate:
            # Validate seed consistency
            current_seed = get_current_seed()
            if current_seed is None:
                typer.echo("❌ No seed has been set")
                raise typer.Exit(1)
            
            typer.echo(f"🔍 Validating seed consistency for seed: {current_seed}")
            is_consistent = validate_seed_consistency(current_seed)
            
            if is_consistent:
                typer.echo("✅ Seed consistency validated successfully")
            else:
                typer.echo("❌ Seed consistency validation failed")
                raise typer.Exit(1)
        
        else:
            # Set seed
            if env:
                # Set up reproducible environment
                actual_seed = set_reproducible_environment(seed)
                typer.echo(f"🌱 Reproducible environment set with seed: {actual_seed}")
                typer.echo("  Environment variables configured for maximum reproducibility")
            else:
                # Just set the seed
                actual_seed = set_global_seed(seed, deterministic)
                typer.echo(f"🌱 Global seed set to: {actual_seed}")
            
            if deterministic:
                typer.echo("  Deterministic behavior enabled")
            else:
                typer.echo("  Non-deterministic behavior enabled")
    
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


# Card generation commands
@app.command(name="card")
def card(
    card_type: str = typer.Argument(
        ..., help="Type of card to generate (determinism, drift, reward)"
    ),
    run_a: str = typer.Argument(..., help="Path to first run directory"),
    run_b: Optional[str] = typer.Argument(
        None, help="Path to second run directory (for drift cards)"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Output directory for cards"
    ),
):
    """Generate trust cards for RL training runs."""
    try:
        if card_type == "determinism":
            typer.echo(f"Generating determinism card for run: {run_a}")

            # Ingest events
            events = ingest_runs_to_events(run_a)

            # Generate card
            card_data = generate_determinism_card(events, run_a, output_dir)

            typer.echo("✅ Determinism card generated")
            typer.echo(f"  Status: {'PASS' if card_data['passed'] else 'FAIL'}")
            typer.echo(f"  Replicas: {card_data['replicas']}")
            typer.echo(f"  Issues: {len(card_data['nondeterminism_hints'])}")

        elif card_type == "drift":
            if not run_b:
                typer.echo("Error: drift cards require two runs", err=True)
                raise typer.Exit(1)

            typer.echo("Generating drift card comparing runs:")
            typer.echo(f"  Run A: {run_a}")
            typer.echo(f"  Run B: {run_b}")

            # Ingest events
            events_a = ingest_runs_to_events(run_a)
            events_b = ingest_runs_to_events(run_b)

            # Generate card
            card_data = generate_drift_card(
                events_a, events_b, run_a, run_b, output_dir
            )

            typer.echo("✅ Drift card generated")
            typer.echo(f"  Diverged: {'Yes' if card_data['diverged'] else 'No'}")
            if card_data["first_step"]:
                typer.echo(f"  First divergence: Step {card_data['first_step']}")
            typer.echo(f"  Signals tripped: {len(card_data['tripped_signals'])}")

        elif card_type == "reward":
            typer.echo(f"Generating reward card for run: {run_a}")

            # Ingest events
            events = ingest_runs_to_events(run_a)

            # Generate card
            card_data = generate_reward_card(events, run_a, output_dir)

            typer.echo("✅ Reward card generated")
            typer.echo(f"  Status: {'HEALTHY' if card_data['passed'] else 'ISSUES'}")
            typer.echo(f"  Calibration: {card_data['calibration_score']:.2f}")
            typer.echo(
                f"  Drift detected: {'Yes' if card_data['drift_detected'] else 'No'}"
            )

        else:
            typer.echo(f"Error: Unknown card type '{card_type}'", err=True)
            typer.echo("Available types: determinism, drift, reward", err=True)
            raise typer.Exit(1)

        # Show output location
        if output_dir:
            typer.echo(f"\nCards saved to: {output_dir}")
        else:
            typer.echo("\nCards saved to runs/run_id/rldk_cards/")

        typer.echo("  - card_name.json")
        typer.echo("  - card_name.png")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
