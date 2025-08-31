"""Command-line interface for RL Debug Kit."""

import typer
from pathlib import Path
from typing import List, Optional
import pandas as pd

from rldk.ingest import ingest_runs
from rldk.diff import first_divergence
from rldk.determinism.check import check
from rldk.bisect import bisect_commits
from rldk.io import write_drift_card, write_determinism_card, write_diff_report, write_diff_events_csv
from rldk.reward import health
from rldk.evals import run
from rldk.io.reward_writers import generate_reward_health_report

app = typer.Typer(
    name="rldk",
    help="RL Debug Kit - Library and CLI for debugging reinforcement learning training runs",
    add_completion=False,
)


@app.command(name="ingest")
def ingest(
    runs: str = typer.Argument(..., help="Path to runs directory, file, or wandb:// URI"),
    adapter: Optional[str] = typer.Option(None, "--adapter", "-a", help="Adapter type (trl, openrlhf, wandb)"),
    output: Optional[str] = typer.Option("metrics.jsonl", "--output", "-o", help="Output file path"),
):
    """Ingest training runs from various sources."""
    try:
        typer.echo(f"Ingesting runs from: {runs}")
        
        # Ingest the runs
        df = ingest_runs(runs, adapter)
        
        # Save to output file
        if output:
            df.to_json(output, orient='records', lines=True)
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


@app.command(name="diff")
def diff(
    a: str = typer.Option(..., "--a", "-a", help="Path or wandb:// URI for run A"),
    b: str = typer.Option(..., "--b", "-b", help="Path or wandb:// URI for run B"),
    signals: List[str] = typer.Option(..., "--signals", "-s", help="Metrics to monitor for divergence"),
    tolerance: float = typer.Option(2.0, "--tolerance", "-t", help="Z-score threshold for violation detection"),
    k: int = typer.Option(3, "--k", "-k", help="Number of consecutive violations required"),
    window: int = typer.Option(50, "--window", "-w", help="Rolling window size for z-score calculation"),
    output_dir: str = typer.Option("diff_analysis", "--output-dir", "-o", help="Output directory for reports"),
):
    """Find first divergence between two training runs."""
    try:
        typer.echo(f"Comparing runs:")
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
            signals = [s.strip() for s in signals.split(',')]
        elif len(signals) == 1 and ',' in signals[0]:
            # Handle case where signals is a list with one comma-separated string
            signals = [s.strip() for s in signals[0].split(',')]
        report = first_divergence(df_a, df_b, signals, k, window, tolerance, output_dir)
        
        # Write reports
        output_path = Path(output_dir)
        write_diff_report(report, output_path)
        write_drift_card(report, output_path)
        write_diff_events_csv(report, output_path)
        
        # Display results
        if report.diverged:
            typer.echo(f"\n🚨 Divergence detected at step {report.first_step}")
            typer.echo(f"Tripped signals: {', '.join(report.tripped_signals)}")
        else:
            typer.echo("\n✅ No significant divergence detected")
        
        typer.echo(f"\nReports saved to: {output_dir}")
        typer.echo(f"  - diff_report.md")
        typer.echo(f"  - drift_card.md")
        if not report.details.empty:
            typer.echo(f"  - diff_events.csv")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="check-determinism")
def check_determinism_cmd(
    cmd: str = typer.Option(..., "--cmd", "-c", help="Command to run for testing"),
    compare: str = typer.Option(..., "--compare", "-m", help="Metrics to compare (comma-separated)"),
    steps: Optional[str] = typer.Option(None, "--steps", "-s", help="Specific steps to compare (comma-separated)"),
    stride: int = typer.Option(50, "--stride", help="Step interval for comparison if steps not specified"),
    replicas: int = typer.Option(5, "--replicas", "-r", help="Number of replicas to run"),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Device to use (auto-detected if None)"),
    output_dir: str = typer.Option("determinism_analysis", "--output-dir", "-o", help="Output directory for reports"),
):
    """Check if a training command is deterministic."""
    try:
        # Parse comma-separated values
        compare_list = [c.strip() for c in compare.split(',')]
        steps_list = None
        if steps:
            steps_list = [int(s.strip()) for s in steps.split(',')]
        
        typer.echo(f"Checking determinism for command: {cmd}")
        typer.echo(f"Metrics to compare: {', '.join(compare_list)}")
        if steps_list:
            typer.echo(f"Steps to compare: {steps_list}")
        else:
            typer.echo(f"Stride: {stride}")
        typer.echo(f"Device: {device or 'auto-detected'}")
        
        # Check determinism
        typer.echo("\nRunning determinism check...")
        report = check(cmd, compare_list, steps_list, stride, device)
        
        # Write report
        output_path = Path(output_dir)
        write_determinism_card(report, output_path)
        
        # Display results
        if report.passed:
            typer.echo("\n✅ Determinism check passed")
        else:
            typer.echo("\n🚨 Determinism issues found")
            if report.culprit:
                typer.echo(f"Culprit operation: {report.culprit}")
            if report.fixes:
                typer.echo("\nRecommended fixes:")
                for fix in report.fixes[:3]:  # Show first 3 fixes
                    typer.echo(f"  - {fix}")
        
        typer.echo(f"\nReport saved to: {output_dir}/determinism_card.md")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="bisect")
def bisect(
    good: str = typer.Option(..., "--good", "-g", help="Known good commit SHA"),
    bad: str = typer.Option("HEAD", "--bad", "-b", help="Known bad commit SHA (default: HEAD)"),
    cmd: Optional[str] = typer.Option(None, "--cmd", "-c", help="Command to run for testing"),
    metric: Optional[str] = typer.Option(None, "--metric", "-m", help="Metric name to monitor"),
    cmp: Optional[str] = typer.Option(None, "--cmp", help="Comparison operator (e.g., '> 0.2')"),
    window: int = typer.Option(100, "--window", "-w", help="Window size for metric statistics"),
    shell_predicate: Optional[str] = typer.Option(None, "--shell-predicate", help="Shell command that returns non-zero on failure"),
):
    """Find regression using git bisect."""
    try:
        typer.echo(f"Starting git bisect:")
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
            raise ValueError("Must provide either (cmd, metric, cmp) or shell_predicate")
        
        # Run bisect
        typer.echo("\nRunning git bisect...")
        result = bisect_commits(
            good_sha=good,
            bad_sha=bad,
            cmd=cmd,
            metric=metric,
            cmp=cmp,
            window=window,
            shell_predicate=shell_predicate
        )
        
        # Display results
        typer.echo(f"\n🎯 Regression found!")
        typer.echo(f"Culprit commit: {result.culprit_sha}")
        typer.echo(f"Iterations: {result.iterations}")
        typer.echo(f"Logs: {result.logs_path}")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="reward-health")
def reward_health(
    run_path: str = typer.Option(..., "--run", "-r", help="Path to training run data"),
    reference_path: Optional[str] = typer.Option(None, "--reference", "-ref", help="Path to reference run data"),
    output_dir: str = typer.Option("reward_analysis", "--output-dir", "-o", help="Output directory for reports"),
    reward_col: str = typer.Option("reward_mean", "--reward-col", help="Column name for reward values"),
    step_col: str = typer.Option("step", "--step-col", help="Column name for training steps"),
    threshold_drift: float = typer.Option(0.1, "--threshold-drift", help="P-value threshold for drift detection"),
    threshold_saturation: float = typer.Option(0.8, "--threshold-saturation", help="Threshold for saturation detection"),
    threshold_calibration: float = typer.Option(0.7, "--threshold-calibration", help="Threshold for calibration quality"),
    threshold_shortcut: float = typer.Option(0.6, "--threshold-shortcut", help="Threshold for shortcut signal detection"),
    threshold_leakage: float = typer.Option(0.3, "--threshold-leakage", help="Threshold for label leakage risk"),
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
            threshold_leakage=threshold_leakage
        )
        
        # Generate reports
        typer.echo("Generating reports...")
        generate_reward_health_report(health_report, output_dir)
        
        # Display results
        if health_report.passed:
            typer.echo("\n✅ Reward health check passed")
        else:
            typer.echo("\n🚨 Reward health issues detected")
            
            if health_report.drift_detected:
                typer.echo(f"  - Reward drift detected")
            if health_report.saturation_issues:
                typer.echo(f"  - {len(health_report.saturation_issues)} saturation issues")
            if health_report.calibration_score < threshold_calibration:
                typer.echo(f"  - Poor calibration (score: {health_report.calibration_score:.3f})")
            if health_report.shortcut_signals:
                typer.echo(f"  - {len(health_report.shortcut_signals)} shortcut signals")
            if health_report.label_leakage_risk > threshold_leakage:
                typer.echo(f"  - Label leakage risk: {health_report.label_leakage_risk:.3f}")
        
        typer.echo(f"\nReports saved to: {output_dir}")
        typer.echo(f"  - reward_health_card.md")
        typer.echo(f"  - reward_health_summary.json")
        if health_report.drift_metrics is not None and not health_report.drift_metrics.empty:
            typer.echo(f"  - drift_analysis.csv")
        typer.echo(f"  - calibration_plots.png")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="eval")
def eval_cmd(
    run_path: str = typer.Option(..., "--run", "-r", help="Path to training run data"),
    suite: str = typer.Option("quick", "--suite", "-s", help="Evaluation suite to run"),
    output_dir: str = typer.Option("eval_results", "--output-dir", "-o", help="Output directory for results"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    sample_size: Optional[int] = typer.Option(None, "--sample-size", help="Number of samples to evaluate"),
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
            output_dir=output_dir
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
                effect_str = f"{effect_size:.3f}" if not np.isnan(effect_size) else "N/A"
                
                typer.echo(f"  {metric}: {score:.3f} (CI: {ci_str}, Effect: {effect_str})")
        
        typer.echo(f"\nResults saved to: {output_dir}")
        typer.echo(f"  - eval_card.md")
        typer.echo(f"  - eval_results.jsonl")
        typer.echo(f"  - eval_summary.json")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="version")
def version():
    """Show version information."""
    from rldk import __version__
    typer.echo(f"RL Debug Kit version {__version__}")


if __name__ == "__main__":
    app()
