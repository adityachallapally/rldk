"""Command-line interface for RL Debug Kit."""

import typer
from pathlib import Path
from typing import List, Optional
import numpy as np

from rldk.ingest import ingest_runs
from rldk.diff import first_divergence
from rldk.determinism.check import check
from rldk.diff import bisect_commits
from rldk.io import write_json
from rldk.reward import health
from rldk.evals import run
from rldk.io.reward_writers import generate_reward_health_report
from rldk.determinism import replay
from rldk.tracking import ExperimentTracker, TrackingConfig

# Import forensics functions directly
from rldk.forensics.ckpt_diff import diff_checkpoints
from rldk.forensics.env_audit import audit_environment
from rldk.forensics.log_scan import scan_logs

# Import reward functions directly
from rldk.reward.drift import compare_models
from rldk.reward.health_config.exit_codes import raise_on_failure
from rldk.reward.health_config.config import load_config, get_legacy_thresholds

# Import IO utilities
from rldk.io.writers import write_json, write_png, mkdir_reports
from rldk.io.schemas import (
    validate,
    DeterminismCardV1,
    PPOScanReportV1,
    CkptDiffReportV1,
    RewardDriftReportV1,
)

# Import card generation modules
from rldk.determinism import generate_determinism_card
from rldk.diff import generate_drift_card
from rldk.reward import generate_reward_card
from rldk.ingest import ingest_runs, ingest_runs_to_events

app = typer.Typer(
    name="rldk",
    help="RL Debug Kit - Library and CLI for debugging reinforcement learning training runs",
    add_completion=False,
)


@app.command(name="ingest")
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
        typer.echo(f"Comparing runs:")
        typer.echo(f"  Run A: {a}")
        typer.echo(f"  Run B: {b}")
        typer.echo(f"  Signals: {', '.join(signals)}")

        # Find first divergence
        result = first_divergence(
            a, b, signals, tolerance=tolerance, k=k, window=window
        )

        if result is None:
            typer.echo("No divergence detected")
            return

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save results
        write_json(result, output_path / "divergence.json")

        typer.echo(f"\nDivergence detected at step {result['step']}")
        typer.echo(f"Signal: {result['signal']}")
        typer.echo(f"Z-score: {result['z_score']:.2f}")
        typer.echo(f"Report saved to: {output_path / 'divergence.json'}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="check-determinism")
def check_determinism(
    cmd: str = typer.Option(..., "--cmd", help="Command to run for determinism check"),
    compare: List[str] = typer.Option(
        ..., "--compare", help="Metrics to compare between runs"
    ),
    runs: int = typer.Option(2, "--runs", help="Number of runs to compare"),
    output_dir: str = typer.Option(
        "determinism_check", "--output-dir", "-o", help="Output directory for results"
    ),
):
    """Check if a command produces deterministic results."""
    try:
        typer.echo(f"Checking determinism for: {cmd}")
        typer.echo(f"Comparing metrics: {', '.join(compare)}")
        typer.echo(f"Running {runs} times...")

        # Run determinism check
        result = check(cmd, compare, runs=runs)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save results
        write_json(result, output_path / "determinism_check.json")

        if result["deterministic"]:
            typer.echo("✅ Command is deterministic")
        else:
            typer.echo("❌ Command is not deterministic")
            typer.echo(f"Divergence detected in: {', '.join(result['divergent_metrics'])}")

        typer.echo(f"Results saved to: {output_path / 'determinism_check.json'}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="replay")
def replay_experiment(
    run_path: str = typer.Argument(..., help="Path to run directory to replay"),
    command: str = typer.Option(
        ..., "--command", help="Command to run for replay"
    ),
    metrics: List[str] = typer.Option(
        ..., "--metrics", help="Metrics to compare during replay"
    ),
    output_dir: str = typer.Option(
        "replay_results", "--output-dir", "-o", help="Output directory for results"
    ),
):
    """Replay an experiment with the same seeds and compare results."""
    try:
        typer.echo(f"Replaying experiment: {run_path}")
        typer.echo(f"Command: {command}")
        typer.echo(f"Metrics: {', '.join(metrics)}")

        # Run replay
        result = replay(run_path, command, metrics)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save results
        write_json(result, output_path / "replay_results.json")

        if result["success"]:
            typer.echo("✅ Replay successful")
            typer.echo(f"Original run: {result['original_run']}")
            typer.echo(f"Replay run: {result['replay_run']}")
        else:
            typer.echo("❌ Replay failed")
            typer.echo(f"Error: {result['error']}")

        typer.echo(f"Results saved to: {output_path / 'replay_results.json'}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="eval")
def eval_experiment(
    run_path: str = typer.Argument(..., help="Path to run directory to evaluate"),
    suite: str = typer.Option(
        "quick", "--suite", help="Evaluation suite to run"
    ),
    output_dir: str = typer.Option(
        "eval_results", "--output-dir", "-o", help="Output directory for results"
    ),
):
    """Run evaluation suite on an experiment."""
    try:
        typer.echo(f"Evaluating experiment: {run_path}")
        typer.echo(f"Suite: {suite}")

        # Run evaluation
        result = run(run_path, suite)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save results
        write_json(result, output_path / "eval_results.json")

        typer.echo("✅ Evaluation complete")
        typer.echo(f"Results saved to: {output_path / 'eval_results.json'}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="track")
def track(
    experiment_name: str = typer.Argument(..., help="Name of the experiment to track"),
    output_dir: str = typer.Option(
        "./runs", "--output-dir", "-o", help="Output directory for tracking data"
    ),
    no_wandb: bool = typer.Option(
        False, "--no-wandb", help="Disable Weights & Biases integration"
    ),
    no_git: bool = typer.Option(
        False, "--no-git", help="Disable Git tracking"
    ),
    tags: List[str] = typer.Option(
        [], "--tag", help="Tags to add to the experiment"
    ),
    notes: str = typer.Option(
        "", "--notes", help="Notes to add to the experiment"
    ),
):
    """Start tracking an experiment."""
    try:
        typer.echo(f"Starting experiment tracking: {experiment_name}")

        # Create tracking config
        config = TrackingConfig(
            experiment_name=experiment_name,
            output_dir=output_dir,
            enable_wandb=not no_wandb,
            enable_git_tracking=not no_git,
            tags=tags,
            notes=notes,
        )

        # Create tracker
        tracker = ExperimentTracker(config)

        # Start experiment
        tracker.start_experiment()

        typer.echo("✅ Experiment tracking started")
        typer.echo(f"Experiment ID: {tracker.experiment_id}")
        typer.echo(f"Output directory: {output_dir}")

        # Save config for later use
        config_path = Path(output_dir) / "tracking_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(config.dict(), config_path)

        typer.echo(f"Config saved to: {config_path}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


# Forensics commands
@app.command(name="compare-runs")
def compare_runs(
    run_a: str = typer.Argument(..., help="Path to first run directory"),
    run_b: str = typer.Argument(..., help="Path to second run directory"),
):
    """Compare two training runs and identify divergences."""
    try:
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
        write_json(comparison, "rldk_reports/run_comparison.json")

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


@app.command(name="diff-ckpt")
def diff_ckpt(
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
        write_json(report, "rldk_reports/ckpt_diff.json")

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


@app.command(name="env-audit")
def env_audit(
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
        write_json(determinism_card, "rldk_reports/determinism_card.json")

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


@app.command(name="log-scan")
def log_scan(
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
        write_json(report, "rldk_reports/ppo_scan.json")

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


@app.command(name="doctor")
def doctor(
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
        write_json(determinism_card, "rldk_reports/determinism_card.json")
        write_json(scan_report, "rldk_reports/ppo_scan.json")

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


# Reward commands
@app.command(name="reward-drift")
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
        from rldk.io.readers import read_jsonl

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
        write_json(report, "rldk_reports/reward_drift.json")

        # Create scatter plot
        import matplotlib.pyplot as plt

        # Load model outputs for plotting
        from rldk.io.readers import read_reward_head

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


@app.command(name="reward-health")
def reward_health(
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
        health_report = health(
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


@app.command(name="reward-health-gate")
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

            typer.echo(f"Generating drift card for runs: {run_a} vs {run_b}")

            # Ingest events for both runs
            events_a = ingest_runs_to_events(run_a)
            events_b = ingest_runs_to_events(run_b)

            # Generate card
            card_data = generate_drift_card(events_a, events_b, run_a, run_b, output_dir)

            typer.echo("✅ Drift card generated")
            typer.echo(f"  Status: {'PASS' if card_data['passed'] else 'FAIL'}")
            typer.echo(f"  Divergence step: {card_data.get('divergence_step', 'N/A')}")
            typer.echo(f"  Issues: {len(card_data.get('issues', []))}")

        elif card_type == "reward":
            if not run_b:
                typer.echo("Error: reward cards require two runs", err=True)
                raise typer.Exit(1)

            typer.echo(f"Generating reward card for runs: {run_a} vs {run_b}")

            # Ingest events for both runs
            events_a = ingest_runs_to_events(run_a)
            events_b = ingest_runs_to_events(run_b)

            # Generate card
            card_data = generate_reward_card(events_a, events_b, run_a, run_b, output_dir)

            typer.echo("✅ Reward card generated")
            typer.echo(f"  Status: {'PASS' if card_data['passed'] else 'FAIL'}")
            typer.echo(f"  Correlation: {card_data.get('correlation', 'N/A')}")
            typer.echo(f"  Issues: {len(card_data.get('issues', []))}")

        else:
            typer.echo(f"Error: Unknown card type '{card_type}'", err=True)
            typer.echo("Valid types: determinism, drift, reward")
            raise typer.Exit(1)

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