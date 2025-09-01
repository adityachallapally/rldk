"""Reward CLI commands for RL Debug Kit."""

import typer

from rldk.reward.drift import compare_models
from rldk.io.writers import write_json, write_png, mkdir_reports
from rldk.io.schemas import validate, RewardDriftReportV1

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
