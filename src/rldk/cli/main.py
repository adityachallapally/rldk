"""Main CLI application and sub-app registration."""

import typer
import logging

from rldk.config import settings

# Import sub-apps
from .forensics import forensics_app
from .reward import reward_app
from .evals import evals_app
from .tracking import tracking_app


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


# Main CLI app
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

# Import main commands that should be available at the top level
from .main_commands import (
    ingest,
    diff,
    check_determinism_cmd,
    bisect,
    reward_health,
    replay_cmd,
    eval_cmd,
    track,
    version,
    card,
    # Legacy aliases
    compare_runs,
    diff_ckpt,
    env_audit,
    log_scan,
    doctor,
    reward_drift_legacy,
)

# Register main commands
app.command(name="ingest")(ingest)
app.command(name="diff")(diff)
app.command(name="check-determinism")(check_determinism_cmd)
app.command(name="bisect")(bisect)
app.command(name="reward-health")(reward_health)
app.command(name="replay")(replay_cmd)
app.command(name="eval")(eval_cmd)
app.command(name="track")(track)
app.command(name="version")(version)
app.command(name="card")(card)

# Legacy command aliases for backward compatibility
app.command(name="compare-runs")(compare_runs)
app.command(name="diff-ckpt")(diff_ckpt)
app.command(name="env-audit")(env_audit)
app.command(name="log-scan")(log_scan)
app.command(name="doctor")(doctor)
app.command(name="reward-drift")(reward_drift_legacy)


if __name__ == "__main__":
    app()