"""Main CLI app with sub-app registration and legacy aliases."""

import typer

# Import sub-apps
from .forensics import forensics_app
from .reward import reward_app
from .evals import evals_app
from .tracking import tracking_app
from .main_commands import main_commands_app

# Import individual command functions for legacy aliases
from .forensics import (
    compare_runs as forensics_compare_runs,
    diff_ckpt as forensics_diff_ckpt,
    env_audit as forensics_env_audit,
    log_scan as forensics_log_scan,
    doctor as forensics_doctor,
)
from .reward import reward_drift
from .main_commands import (
    ingest,
    diff,
    check_determinism_cmd,
    bisect,
    replay_cmd,
    eval_cmd,
    version,
    card,
    reward_health,
)
from .tracking import track


# Create main app
app = typer.Typer(
    name="rldk",
    help="RL Debug Kit - Library and CLI for debugging reinforcement learning training runs",
    add_completion=False,
)

# Add sub-apps to main app
app.add_typer(forensics_app, name="forensics")
app.add_typer(reward_app, name="reward")
app.add_typer(evals_app, name="evals")
app.add_typer(tracking_app, name="tracking")

# Add main commands directly to the main app
app.add_typer(main_commands_app, name="main")

# Register individual main commands directly on the main app
app.command(name="ingest")(ingest)
app.command(name="diff")(diff)
app.command(name="check-determinism")(check_determinism_cmd)
app.command(name="bisect")(bisect)
app.command(name="replay")(replay_cmd)
app.command(name="eval")(eval_cmd)
app.command(name="version")(version)
app.command(name="card")(card)
app.command(name="reward-health")(reward_health)
app.command(name="track")(track)

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


@app.command(name="doctor")
def doctor(
    run_or_repo: str = typer.Argument(..., help="Path to run or repository directory"),
):
    """Run comprehensive diagnostics on a training run or repository."""
    forensics_doctor(run_or_repo)


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


if __name__ == "__main__":
    app()