"""Reward CLI commands for RL Debug Kit."""

import typer
from pathlib import Path
from typing import Optional

from rldk.reward.drift import compare_models
from rldk.reward.health import health
from rldk.reward.health.exit_codes import raise_on_failure
from rldk.io.writers import write_json, write_png, mkdir_reports
from rldk.io.schemas import validate, RewardDriftReportV1
from rldk.io.reward_writers import generate_reward_health_report
from rldk.ingest import ingest_runs

app = typer.Typer(name="reward", help="Reward model analysis commands")


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


# Create a sub-app for reward-health commands
reward_health_app = typer.Typer(name="reward-health", help="Reward health analysis commands")
app.add_typer(reward_health_app, name="reward-health")


@reward_health_app.command(name="run")
def reward_health_run(
    scores: str = typer.Option(..., "--scores", help="Path to scores JSONL file"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to health configuration YAML file"),
    out: str = typer.Option(..., "--out", help="Output directory for reports"),
    gate: bool = typer.Option(False, "--gate", help="Enable CI gate mode with exit codes (0=pass, 1=warn, 2=fail)"),
):
    """Run reward health analysis on scores data."""
    try:
        typer.echo(f"Running reward health analysis on scores: {scores}")
        
        # Ingest scores data
        typer.echo("Ingesting scores data...")
        scores_data = ingest_runs(scores)
        
        # Load configuration if provided
        config_data = {}
        if config and Path(config).exists():
            import yaml
            with open(config, 'r') as f:
                config_data = yaml.safe_load(f)
        
        # Extract thresholds from config
        threshold_drift = config_data.get('threshold_drift', 0.1)
        threshold_saturation = config_data.get('threshold_saturation', 0.8)
        threshold_calibration = config_data.get('threshold_calibration', 0.7)
        threshold_shortcut = config_data.get('threshold_shortcut', 0.6)
        threshold_leakage = config_data.get('threshold_leakage', 0.3)
        
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


@reward_health_app.command(name="gate")
def reward_health_gate(
    from_path: str = typer.Option(..., "--from", help="Path to health.json file"),
):
    """Gate CI based on health.json results."""
    try:
        typer.echo(f"Reading health data from: {from_path}")
        raise_on_failure(from_path)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
