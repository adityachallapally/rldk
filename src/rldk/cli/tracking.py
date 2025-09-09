"""Tracking commands for experiment management."""

import typer
from pathlib import Path
from typing import Optional

from rldk.tracking import ExperimentTracker, TrackingConfig


# Create tracking sub-app
tracking_app = typer.Typer(name="tracking", help="Experiment tracking commands")


@tracking_app.command(name="track")
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