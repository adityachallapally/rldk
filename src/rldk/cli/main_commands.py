"""Core CLI commands (ingest, diff, check-determinism, bisect, replay, eval, version, card)."""

import typer
import numpy as np
from pathlib import Path
from typing import List, Optional

from rldk.ingest import ingest_runs, ingest_runs_to_events
from rldk.diff import first_divergence
from rldk.determinism.check import check
from rldk.bisect import bisect_commits
from rldk.io import write_json, generate_reward_health_report
from rldk.reward import health
from rldk.evals import run
from rldk.replay import replay
from rldk.config import settings

# Import card generation modules
from rldk.cards import (
    generate_determinism_card,
    generate_drift_card,
    generate_reward_card,
)


def ensure_config_initialized():
    """Ensure configuration is initialized for CLI operations."""
    try:
        settings.initialize()
    except PermissionError as e:
        # If we can't create directories, that's okay for read-only operations
        # Just log a warning and continue
        import logging
        logging.warning(f"Could not create RLDK directories: {e}")
    except Exception as e:
        # For other errors, log but don't fail
        import logging
        logging.warning(f"Configuration initialization warning: {e}")


# Create main commands app
main_commands_app = typer.Typer(name="main", help="Core RL Debug Kit commands")


@main_commands_app.command(name="ingest")
def ingest(
    runs: str = typer.Argument(
        ..., help="Path to runs directory, file, or wandb:// URI"
    ),
    adapter: Optional[str] = typer.Option(
        None, "--adapter", "-a", help="Adapter type (trl, openrlhf, wandb)"
    ),
    output: Optional[str] = typer.Option(
        "metrics.jsonl", "--output", "-o", help="Output file path"
    ),
):
    """Ingest training runs from various sources."""
    try:
        ensure_config_initialized()
        typer.echo(f"Ingesting runs from: {runs}")

        # Ingest the runs
        df = ingest_runs(runs, adapter)

        # Save to output file
        if output:
            df.to_json(output, orient="records", lines=True)
            typer.echo(f"Saved {len(df)} records to {output}")

        # Display summary
        typer.echo(f"\nIngested {len(df)} training steps")
        typer.echo(f"Columns: {', '.join(df.columns)}")
        typer.echo(f"Steps range: {df['step'].min()} to {df['step'].max()}")

        # Show sample data
        if not df.empty:
            typer.echo("\nSample data:")
            typer.echo(df.head().to_string())

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@main_commands_app.command(name="diff")
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


@main_commands_app.command(name="check-determinism")
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


@main_commands_app.command(name="bisect")
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


@main_commands_app.command(name="reward-health")
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


@main_commands_app.command(name="replay")
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


@main_commands_app.command(name="eval")
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


@main_commands_app.command(name="version")
def version():
    """Show version information."""
    from rldk import __version__

    typer.echo(f"RL Debug Kit version {__version__}")


# Card generation commands
@main_commands_app.command(name="card")
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