"""Command-line interface for RL Debug Kit."""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import typer
import click

from rldk.bisect import bisect_commits

# Import card generation modules
from rldk.cards import (
    generate_determinism_card,
    generate_drift_card,
    generate_kl_drift_card,
    generate_reward_card,
)
from rldk.config import settings
from rldk.determinism.check import check
from rldk.determinism.runner import run_deterministic_command
from rldk.diff import compare_training_metrics_tables
from rldk.evals import run
from rldk.evals.metrics import (
    evaluate_bias,
    evaluate_length_bias,
    evaluate_throughput,
    evaluate_toxicity,
)

# Import evaluation modules
from rldk.evals.suites import COMPREHENSIVE_SUITE, QUICK_SUITE, SAFETY_SUITE

# Import forensics modules
from rldk.forensics import ComprehensivePPOForensics
from rldk.forensics.ckpt_diff import diff_checkpoints
from rldk.forensics.env_audit import audit_environment
from rldk.forensics.log_scan import scan_logs
from rldk.ingest import ingest_runs, normalize_training_metrics_source
from rldk.ingest.training_metrics_normalizer import normalize_training_metrics
from rldk.io import (
    CkptDiffReportV1,
    DeterminismCardV1,
    PPOScanReportV1,
    RewardDriftReportV1,
    generate_reward_health_report,
    mkdir_reports,
    read_jsonl,
    read_reward_head,
    validate,
    write_json,
    write_png,
)
from rldk.io.event_schema import dataframe_to_events
from rldk.io import write_json as write_json_report
from rldk.monitor import (
    ActionDispatcher,
    AlertWriter,
    MonitorEngine,
    load_rules,
    read_events_once,
    read_stream,
    stream_from_mlflow,
    stream_from_wandb,
)
from rldk.monitor.presets import FIELD_MAP_PRESETS, RULE_PRESETS, get_field_map_preset
from rldk.emit import EventWriter
from rldk.replay import replay
from rldk.reward import health

__all__ = [
    "run_deterministic_command",
]

# Import reward modules
from rldk.reward.drift import compare_models, compare_score_lists
from rldk.reward.health_analysis import health as reward_health_analysis
from rldk.reward.health_config.config import get_legacy_thresholds, load_config
from rldk.reward.health_config.exit_codes import raise_on_failure
from rldk.tracking import ExperimentTracker, TrackingConfig
from rldk.utils.error_handling import (
    AdapterError,
    EvaluationError,
    RLDKTimeoutError,
    ValidationError,
    format_error_message,
    format_structured_error_message,
    log_error_with_context,
    print_troubleshooting_tips,
    print_usage_examples,
    validate_adapter_source,
    validate_file_path,
    validate_training_run_directory,
)
from rldk.utils.progress import print_operation_status, timed_operation_context
from rldk.utils.runtime import with_timeout


# Typer 0.9 expects click.Parameter.make_metavar to accept an optional context.
# Click 8.1 requires a context argument, so we provide a shim to keep help text working
# across versions.
if click.Parameter.make_metavar.__code__.co_argcount == 2:  # pragma: no cover - version guard
    _original_make_metavar = click.Parameter.make_metavar

    def _safe_make_metavar(self, ctx=None):
        return _original_make_metavar(self, ctx)

    click.Parameter.make_metavar = _safe_make_metavar


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


def _parse_field_map_option(field_map: Optional[str]) -> Optional[Dict[str, str]]:
    if not field_map:
        return None
    try:
        data = json.loads(field_map)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON for field map: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("field map must be a JSON object mapping source keys to canonical keys")
    return {str(key): str(value) for key, value in data.items()}


def _combine_field_maps(
    preset: Optional[str],
    raw_field_map: Optional[str],
) -> Optional[Dict[str, str]]:
    mapping: Dict[str, str] = {}
    if preset:
        preset_mapping = get_field_map_preset(preset)
        if preset_mapping is None:
            available = ", ".join(sorted(FIELD_MAP_PRESETS))
            raise ValueError(
                f"unknown field map preset '{preset}'. Available presets: {available}"
            )
        mapping.update(preset_mapping)
    custom_mapping = _parse_field_map_option(raw_field_map)
    if custom_mapping:
        mapping.update(custom_mapping)
    return mapping or None


def _fallback_directory_load(
    directory: Path, field_map: Optional[Dict[str, str]]
) -> pd.DataFrame:
    typer.echo("Fallback: loading metrics table directly...")
    candidates = sorted(directory.glob("*"))
    table: Optional[pd.DataFrame] = None
    for candidate in candidates:
        suffix = candidate.suffix.lower()
        try:
            if suffix == ".csv":
                table = pd.read_csv(candidate)
            elif suffix == ".tsv":
                table = pd.read_csv(candidate, sep="\t")
            elif suffix == ".parquet":
                table = pd.read_parquet(candidate)
        except Exception:
            continue
        if table is not None:
            break
    if table is None:
        raise TypeError(
            "Unable to load metrics table from directory; no supported files found"
        )
    return normalize_training_metrics(table, field_map=field_map)


_GOLD_FILE_CANDIDATES: Tuple[str, ...] = (
    "gold_scores.jsonl",
    "gold_metrics.jsonl",
    "gold.jsonl",
    "trusted_scores.jsonl",
    "trusted_metrics.jsonl",
    "eval_scores.jsonl",
)

_GOLD_DIR_HINTS: Tuple[str, ...] = (
    "eval",
    "evals",
    "evaluation",
    "evaluations",
    "metrics",
    "reports",
    "analysis",
    "analyzers",
    "reward_model_eval",
)

_GOLD_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "gold_metric",
    "gold_score",
    "trusted_score",
    "eval_score",
    "benchmark_score",
)


def _auto_detect_gold_artifact(source: str) -> Optional[Path]:
    base_path = Path(source)
    if not base_path.exists():
        return None

    search_queue: List[Path] = []
    if base_path.is_file():
        search_queue.append(base_path.parent)
    else:
        search_queue.append(base_path)

    visited: Set[Path] = set()

    while search_queue:
        current = search_queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        index_path = current / "index.json"
        if index_path.exists():
            try:
                index_data = json.loads(index_path.read_text())
                file_entries = index_data.get("files", [])
                for entry in file_entries:
                    if not isinstance(entry, str):
                        continue
                    if "gold" not in entry.lower():
                        continue
                    candidate = current / entry
                    if candidate.exists() and candidate.is_file():
                        return candidate
            except json.JSONDecodeError:
                pass

        for filename in _GOLD_FILE_CANDIDATES:
            candidate = current / filename
            if candidate.exists() and candidate.is_file():
                return candidate

        for match in sorted(current.glob("*gold*.jsonl")):
            if match.is_file():
                return match

        for subdir in _GOLD_DIR_HINTS:
            candidate_dir = current / subdir
            if candidate_dir.exists() and candidate_dir.is_dir() and candidate_dir not in visited:
                search_queue.append(candidate_dir)

    return None


def _dataframe_has_gold_metrics(df: pd.DataFrame) -> bool:
    return any(column in df.columns for column in _GOLD_COLUMN_CANDIDATES)


def _normalize_signals_option(signals: Sequence[str]) -> List[str]:
    """Expand comma-delimited signal entries into a unique list."""

    normalized: List[str] = []
    seen = set()
    for entry in signals:
        if not isinstance(entry, str):
            continue
        for part in (segment.strip() for segment in entry.split(",")):
            if not part or part in seen:
                continue
            normalized.append(part)
            seen.add(part)
    return normalized

def _parse_json_mapping(raw: Optional[str], field: str) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} must be valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{field} must be a JSON object")
    return data


def _load_tokenizer_by_name(tokenizer_name: Optional[str]) -> Optional[Any]:
    """Load a Hugging Face tokenizer if the optional dependency is available."""

    if not tokenizer_name:
        return None

    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "transformers is required for --tokenizer-name but is not installed"
        ) from exc

    try:
        return AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as exc:  # pragma: no cover - network/config errors
        raise RuntimeError(f"Failed to load tokenizer '{tokenizer_name}': {exc}") from exc


def _derive_alert_text_path(alerts_path: Optional[Path], override: Optional[Path]) -> Optional[Path]:
    if override is not None:
        return override
    if alerts_path is None:
        return None
    if alerts_path.suffix:
        return alerts_path.with_suffix(".txt")
    return Path(f"{alerts_path}.txt")

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

_MONITOR_RULES_HELP = "Path to a YAML rules file or preset name"
if RULE_PRESETS:
    _MONITOR_RULES_HELP += f" ({', '.join(sorted(RULE_PRESETS))})"
_MONITOR_RULES_HELP += (
    ". Use '--rules length-bias-gate' to enable the upcoming preset for reward "
    "length bias monitoring."
)

@app.command(name="monitor")
def monitor(
    stream: Optional[str] = typer.Option(
        None,
        "--stream",
        help="Path to a JSONL metrics file or '-' to read from stdin.",
    ),
    once: Optional[Path] = typer.Option(
        None,
        "--once",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Analyze an existing JSONL metrics file once and exit.",
    ),
    from_wandb: Optional[str] = typer.Option(
        None,
        "--from-wandb",
        help="Stream metrics directly from a W&B run path (entity/project[/run_id]).",
    ),
    from_mlflow: Optional[str] = typer.Option(
        None,
        "--from-mlflow",
        help="Stream metrics from an MLflow run ID using the active tracking URI.",
    ),
    rules: str = typer.Option(..., "--rules", help=_MONITOR_RULES_HELP),
    report: Optional[Path] = typer.Option(
        None,
        "--report",
        help="Optional path to write the monitoring summary report as JSON.",
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help=(
            "Field map preset to normalize common trainer keys"
            f" ({', '.join(sorted(FIELD_MAP_PRESETS))})"
            if FIELD_MAP_PRESETS
            else "Field map preset name (e.g. trl, grpo)."
        ),
    ),
    field_map: Optional[str] = typer.Option(
        None,
        "--field-map",
        help="JSON object mapping input keys to canonical event fields.",
    ),
    regex: Optional[str] = typer.Option(
        None,
        "--regex",
        help="Regex preset or pattern to parse text logs (e.g. 'trl').",
    ),
    pid: Optional[int] = typer.Option(
        None,
        "--pid",
        help="PID of the training process to terminate when stop actions fire.",
    ),
    alerts: Path = typer.Option(
        Path("artifacts/alerts.jsonl"),
        "--alerts",
        help="Path to write alert activations as JSONL.",
        dir_okay=False,
    ),
    alerts_txt: Optional[Path] = typer.Option(
        None,
        "--alerts-txt",
        help="Optional path for human-readable alert summaries.",
        dir_okay=False,
    ),
    reward_health_window: Optional[int] = typer.Option(
        None,
        "--reward-health-window",
        help=(
            "Number of recent steps to batch for reward health checks. "
            "Provide a positive value to enable synthetic reward health alerts."
        ),
    ),
    kill_timeout_sec: float = typer.Option(
        5.0,
        "--kill-timeout-sec",
        help="Seconds to wait between SIGTERM and SIGKILL for stop actions.",
    ),
    http_timeout_sec: float = typer.Option(
        5.0,
        "--http-timeout-sec",
        help="Timeout in seconds for shell and HTTP actions.",
    ),
    retries: int = typer.Option(
        0,
        "--retries",
        help="Number of retries for shell and HTTP actions.",
    ),
) -> None:
    """Monitor JSONL metrics with streaming or batch analysis."""
    ensure_config_initialized()
    stream_source = stream
    once_source = once
    wandb_source = from_wandb
    mlflow_source = from_mlflow
    env_stream = os.environ.get("RLDK_METRICS_PATH")
    if (
        stream_source is None
        and once_source is None
        and wandb_source is None
        and mlflow_source is None
    ):
        if env_stream:
            stream_source = env_stream
        elif not sys.stdin.isatty():
            stream_source = "-"
    selections = [
        stream_source is not None,
        once_source is not None,
        wandb_source is not None,
        mlflow_source is not None,
    ]
    if sum(1 for selected in selections if selected) != 1:
        typer.echo(
            "Provide exactly one source via --stream, --once, --from-wandb, or --from-mlflow (stdin also supported).",
            err=True,
        )
        raise typer.Exit(1)
    try:
        mapping = _combine_field_maps(preset, field_map)
    except ValueError as exc:
        typer.echo(f"Invalid field map: {exc}", err=True)
        raise typer.Exit(1)
    try:
        rule_defs = load_rules(rules)
    except Exception as exc:
        typer.echo(f"Failed to load rules: {exc}", err=True)
        raise typer.Exit(1)
    dispatcher = ActionDispatcher(
        pid=pid,
        kill_timeout_sec=kill_timeout_sec,
        http_timeout_sec=http_timeout_sec,
        retries=retries,
    )
    if reward_health_window is not None and reward_health_window <= 0:
        reward_health_window = None

    engine = MonitorEngine(
        rule_defs,
        action_executor=dispatcher,
        reward_health_window=reward_health_window,
    )
    alerts_text_path = _derive_alert_text_path(alerts, alerts_txt)
    alert_writer = AlertWriter(alerts, alerts_text_path)

    def emit_alerts(alerts):
        for alert in alerts:
            alert_writer.write(alert)
            message = alert.summary()
            if alert.status == "error":
                typer.echo(message, err=True)
            else:
                typer.echo(message)

    mode: str
    if once_source is not None:
        mode = "once"
    elif wandb_source is not None:
        mode = "wandb"
    elif mlflow_source is not None:
        mode = "mlflow"
    else:
        mode = "stream"

    try:
        if mode == "stream":
            for event in read_stream(stream_source, field_map=mapping, regex=regex):
                emit_alerts(engine.process_event(event))
        elif mode == "once":
            events = read_events_once(once_source, field_map=mapping, regex=regex)
            for event in events:
                emit_alerts(engine.process_event(event))
        elif mode == "wandb":
            for event in stream_from_wandb(wandb_source):
                emit_alerts(engine.process_event(event))
        elif mode == "mlflow":
            for event in stream_from_mlflow(mlflow_source):
                emit_alerts(engine.process_event(event))
    except KeyboardInterrupt:
        typer.echo("Monitoring interrupted by user", err=True)
    except EOFError:
        pass
    except typer.Exit:
        raise
    except ValueError as exc:
        typer.echo(f"Failed to parse metrics: {exc}", err=True)
        raise typer.Exit(1)
    except RuntimeError as exc:
        message = str(exc) or exc.__class__.__name__
        typer.echo(f"Monitoring failed: {message}", err=True)
        raise typer.Exit(1)
    except Exception as exc:
        message = str(exc) or exc.__class__.__name__
        typer.echo(f"Monitoring failed: {message}", err=True)
        raise typer.Exit(1)

    report_payload = engine.generate_report().to_dict()
    if report is not None:
        try:
            if report.parent and not report.parent.exists():
                report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(report_payload, indent=2, sort_keys=True))
        except Exception as exc:
            typer.echo(f"Failed to write report: {exc}", err=True)
            raise typer.Exit(1)
    elif once_source is not None:
        typer.echo(json.dumps(report_payload, indent=2, sort_keys=True))


@app.command(name="emit")
def emit_event(
    to: Path = typer.Option(
        ...,
        "--to",
        help="Path to the JSONL file that should receive the event.",
    ),
    name: str = typer.Option(..., "--name", help="Metric name."),
    value: float = typer.Option(..., "--value", help="Metric value."),
    step: int = typer.Option(..., "--step", help="Training step associated with the metric."),
    time: Optional[str] = typer.Option(None, "--time", help="ISO8601 timestamp for the event."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Optional identifier for the training run."),
    tags: Optional[str] = typer.Option(None, "--tags", help="JSON object of tag key/value pairs."),
    meta: Optional[str] = typer.Option(None, "--meta", help="JSON object of metadata values."),
) -> None:
    """Append a canonical monitoring event to a JSONL file."""
    ensure_config_initialized()
    try:
        tags_payload = _parse_json_mapping(tags, "tags")
        meta_payload = _parse_json_mapping(meta, "meta")
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    writer = EventWriter(to)
    try:
        event = writer.log(
            step=step,
            name=name,
            value=value,
            time=time,
            run_id=run_id,
            tags=tags_payload,
            meta=meta_payload,
        )
    except ValueError as exc:
        typer.echo(f"Failed to emit event: {exc}", err=True)
        raise typer.Exit(1)
    finally:
        writer.close()
    typer.echo(json.dumps(event, indent=2, sort_keys=True))



# ============================================================================
# FORENSICS COMMANDS
# ============================================================================

@forensics_app.command(name="compare-runs")
def forensics_compare_runs(
    run_a: str = typer.Argument(..., help="Path to first run directory or file"),
    run_b: str = typer.Argument(..., help="Path to second run directory or file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed format detection info"),
):
    """Compare two training runs and identify divergences.

    Supports multiple formats: JSONL, CSV, JSON, Parquet files or directories.
    Automatically detects format and handles field mapping for common RL frameworks.
    """
    try:
        ensure_config_initialized()
        typer.echo("Comparing runs:")
        typer.echo(f"  Run A: {run_a}")
        typer.echo(f"  Run B: {run_b}")

        if verbose:
            typer.echo("\nDetecting formats...")

        # Validate input paths
        try:
            validated_run_a = validate_training_run_directory(run_a)
            validated_run_b = validate_training_run_directory(run_b)
        except ValidationError as e:
            typer.echo(format_error_message(e), err=True)
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(
                format_structured_error_message(
                    "Path validation failed",
                    f"{run_a}, {run_b}",
                    "Valid training run directories or files",
                    f"Error accessing paths: {e}",
                    "Check that both paths exist and you have read permissions"
                ),
                err=True
            )
            raise typer.Exit(1)

        # Scan both runs with enhanced error handling
        try:
            scan_a = scan_logs(validated_run_a)
            if verbose:
                typer.echo(f"  Run A: Successfully loaded {len(scan_a.get('rules_fired', []))} anomaly rules")
        except (ValidationError, AdapterError) as e:
            typer.echo(
                format_structured_error_message(
                    "Run A loading failed",
                    str(validated_run_a),
                    "Training logs with RL fields (step, reward, etc.)",
                    f"Format/validation issue: {e}",
                    "Check that Run A contains valid RL training data"
                ),
                err=True
            )
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(
                format_structured_error_message(
                    "Unexpected error loading Run A",
                    str(validated_run_a),
                    "Successful log loading",
                    f"Error: {e}",
                    "Check log file format and try again, or use --verbose for more details"
                ),
                err=True
            )
            raise typer.Exit(1)

        try:
            scan_b = scan_logs(validated_run_b)
            if verbose:
                typer.echo(f"  Run B: Successfully loaded {len(scan_b.get('rules_fired', []))} anomaly rules")
        except (ValidationError, AdapterError) as e:
            typer.echo(
                format_structured_error_message(
                    "Run B loading failed",
                    str(validated_run_b),
                    "Training logs with RL fields (step, reward, etc.)",
                    f"Format/validation issue: {e}",
                    "Check that Run B contains valid RL training data"
                ),
                err=True
            )
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(
                format_structured_error_message(
                    "Unexpected error loading Run B",
                    str(validated_run_b),
                    "Successful log loading",
                    f"Error: {e}",
                    "Check log file format and try again, or use --verbose for more details"
                ),
                err=True
            )
            raise typer.Exit(1)

        # Create comparison report
        comparison = {
            "version": "1",
            "run_a": {"path": run_a, "anomalies": scan_a.get("rules_fired", [])},
            "run_b": {"path": run_b, "anomalies": scan_b.get("rules_fired", [])},
            "earliest_divergent_step": None,
            "format_info": {
                "run_a_stats": scan_a.get("stats", {}),
                "run_b_stats": scan_b.get("stats", {})
            }
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

        if verbose and comparison["format_info"]:
            typer.echo("\nFormat Statistics:")
            for run_name, stats in [("Run A", scan_a.get("stats", {})), ("Run B", scan_b.get("stats", {}))]:
                if stats:
                    typer.echo(f"  {run_name}:")
                    for key, value in stats.items():
                        typer.echo(f"    {key}: {value}")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(
            format_structured_error_message(
                "Unexpected error",
                f"{run_a}, {run_b}",
                "Successful run comparison",
                f"Unexpected error: {e}",
                "Please report this issue with the full error message"
            ),
            err=True
        )
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
        
        # Validate checkpoint files
        try:
            ckpt_a_path = validate_file_path(ckpt_a, must_exist=True)
            ckpt_b_path = validate_file_path(ckpt_b, must_exist=True)
            
            # Check if files are likely checkpoint files
            checkpoint_extensions = ['.pt', '.pth', '.bin', '.safetensors', '.ckpt']
            if ckpt_a_path.suffix not in checkpoint_extensions:
                typer.echo(f"Warning: {ckpt_a} may not be a checkpoint file (extension: {ckpt_a_path.suffix})")
            if ckpt_b_path.suffix not in checkpoint_extensions:
                typer.echo(f"Warning: {ckpt_b} may not be a checkpoint file (extension: {ckpt_b_path.suffix})")
                
        except ValidationError as e:
            typer.echo(format_error_message(e), err=True)
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(
                format_structured_error_message(
                    "Checkpoint validation failed",
                    f"{ckpt_a}, {ckpt_b}",
                    "Valid checkpoint files (.pt, .pth, .bin, .safetensors, .ckpt)",
                    f"Error accessing files: {e}",
                    "Check that both checkpoint files exist and you have read permissions"
                ),
                err=True
            )
            raise typer.Exit(1)

        # Diff checkpoints
        try:
            report = diff_checkpoints(ckpt_a_path, ckpt_b_path)
        except Exception as e:
            typer.echo(
                format_structured_error_message(
                    "Checkpoint comparison failed",
                    f"{ckpt_a_path}, {ckpt_b_path}",
                    "Successful checkpoint comparison",
                    f"Error during comparison: {e}",
                    "Ensure both files are valid PyTorch checkpoint files"
                ),
                err=True
            )
            raise typer.Exit(1)

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
            ax.barh(range(len(names)), l2_norms)
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
        
        # Validate input path
        try:
            validated_path = validate_file_path(repo_or_run, must_exist=True)
            if not validated_path.is_dir():
                raise ValidationError(
                    format_structured_error_message(
                        "Invalid input type",
                        str(validated_path),
                        "Directory (repository or run directory)",
                        "File",
                        "Provide a directory path for environment audit"
                    ),
                    error_code="EXPECTED_DIRECTORY"
                )
        except ValidationError as e:
            typer.echo(format_error_message(e), err=True)
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(
                format_structured_error_message(
                    "Path validation failed",
                    repo_or_run,
                    "Valid directory path",
                    f"Error accessing path: {e}",
                    "Check that the path exists and you have read permissions"
                ),
                err=True
            )
            raise typer.Exit(1)

        # Run audit
        try:
            determinism_card, lock_content = audit_environment(validated_path)
        except Exception as e:
            typer.echo(
                format_structured_error_message(
                    "Environment audit failed",
                    str(validated_path),
                    "Successful environment analysis",
                    f"Error during audit: {e}",
                    "Ensure the directory is accessible and contains valid project files"
                ),
                err=True
            )
            raise typer.Exit(1)

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

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(
            format_structured_error_message(
                "Unexpected error",
                repo_or_run,
                "Successful environment audit",
                f"Unexpected error: {e}",
                "Please report this issue with the full error message"
            ),
            err=True
        )
        raise typer.Exit(1)


@forensics_app.command(name="log-scan")
def forensics_log_scan(
    run_or_export: str = typer.Argument(..., help="Path to run or export directory"),
):
    """Scan training logs for PPO anomalies and issues."""
    try:
        typer.echo(f"Scanning logs: {run_or_export}")
        
        # Validate input path and directory contents
        try:
            validated_path = validate_training_run_directory(run_or_export)
        except ValidationError as e:
            typer.echo(format_error_message(e), err=True)
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(
                format_structured_error_message(
                    "Path validation failed",
                    run_or_export,
                    "Valid training run directory or file",
                    f"Error accessing path: {e}",
                    "Check that the path exists and you have read permissions"
                ),
                err=True
            )
            raise typer.Exit(1)

        # Scan logs
        try:
            report = scan_logs(validated_path)
        except (ValidationError, AdapterError) as e:
            typer.echo(
                format_structured_error_message(
                    "Log scanning failed",
                    str(validated_path),
                    "Training logs with standard RL fields (step, reward, etc.)",
                    f"Data format issue: {e}",
                    "Ensure logs contain RL training metrics or use 'rldk validate-format' to check data"
                ),
                err=True
            )
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(
                format_structured_error_message(
                    "Unexpected error during log scanning",
                    str(validated_path),
                    "Successful log analysis",
                    f"Error: {e}",
                    "Check log file format and try again, or contact support if issue persists"
                ),
                err=True
            )
            raise typer.Exit(1)

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

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(
            format_structured_error_message(
                "Unexpected error",
                run_or_export,
                "Successful log scan operation",
                f"Unexpected error: {e}",
                "Please report this issue with the full error message"
            ),
            err=True
        )
        raise typer.Exit(1)


@forensics_app.command(name="kl-drift")
def forensics_kl_drift(
    run_path: str = typer.Argument(..., help="Path to training run metrics (file or directory)"),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Directory where the KL drift card should be saved",
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help="Field map preset for the metrics source (e.g. trl, grpo, accelerate)",
    ),
    field_map: Optional[str] = typer.Option(
        None,
        "--field-map",
        help="JSON mapping from source column names to canonical names",
    ),
    step_col: str = typer.Option("step", "--step-col", help="Canonical step column"),
    kl_col: str = typer.Option("kl", "--kl-col", help="Canonical KL column"),
    kl_coef_col: str = typer.Option(
        "kl_coef",
        "--kl-coef-col",
        help="Canonical KL coefficient column",
    ),
    drift_threshold: float = typer.Option(
        0.15, "--drift-threshold", help="Threshold for declaring KL drift"
    ),
    drift_window_size: int = typer.Option(
        100, "--drift-window-size", help="Rolling window size for drift analysis"
    ),
    reference_period: int = typer.Option(
        500,
        "--reference-period",
        help="Number of initial steps to treat as the drift reference",
    ),
):
    """Analyze KL drift for a single training run."""

    def _resolve_column(df: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        raise ValidationError(
            f"Could not find a {label} column in the normalized data",
            suggestion=(
                "Use --field-map/--preset or provide the column via --kl-col/--kl-coef-col"
            ),
            error_code=f"MISSING_{label.upper()}_COLUMN",
        )

    try:
        ensure_config_initialized()

        try:
            mapping = _combine_field_maps(preset, field_map)
        except ValueError as exc:
            typer.echo(f"Invalid field map: {exc}", err=True)
            raise typer.Exit(1)

        typer.echo(f"Loading training metrics from: {run_path}")
        metrics_df = normalize_training_metrics_source(run_path, field_map=mapping)

        if metrics_df.empty:
            raise ValidationError(
                "Normalized metrics data is empty",
                suggestion="Ensure the run contains KL metrics and try again",
                error_code="EMPTY_KL_DATA",
            )

        step_column = _resolve_column(metrics_df, [step_col, "global_step"], "step")
        kl_column = _resolve_column(
            metrics_df,
            [kl_col, "kl", "kl_mean", "ppo/policy/kl_mean", "train/kl"],
            "kl",
        )
        kl_coef_column = _resolve_column(
            metrics_df,
            [kl_coef_col, "kl_coef", "kl_coefficient", "kl_coeff", "ppo/policy/kl_coef"],
            "kl coefficient",
        )

        metrics_df = metrics_df.sort_values(step_column)

        median_kl = metrics_df[kl_column].dropna().median()
        kl_target = float(median_kl) if not np.isnan(median_kl) else 0.1

        forensics = ComprehensivePPOForensics(
            enable_kl_drift_tracking=True,
            kl_target=kl_target,
            kl_target_tolerance=0.05,
            window_size=max(200, drift_window_size),
            enable_kl_schedule_tracking=True,
            enable_gradient_norms_analysis=False,
            enable_advantage_statistics=False,
            kl_drift_threshold=drift_threshold,
            kl_drift_window_size=drift_window_size,
            kl_drift_reference_period=reference_period,
            enable_length_bias_detection=False,
        )

        entropy_candidates = ["entropy", "entropy_mean", "ppo/policy/entropy"]
        reward_candidates = ["reward", "reward_mean", "train/reward"]
        reward_std_candidates = ["reward_std", "train/reward_std"]

        def _maybe_get(series: pd.Series, options: Sequence[str], default: float = 0.0) -> float:
            for option in options:
                if option in series and pd.notna(series.get(option)):
                    return float(series.get(option))
            return default

        for _, row in metrics_df.iterrows():
            forensics.update(
                step=int(row.get(step_column, 0)),
                kl=float(row.get(kl_column, 0.0)),
                kl_coef=float(row.get(kl_coef_column, 1.0)),
                entropy=_maybe_get(row, entropy_candidates, 0.0),
                reward_mean=_maybe_get(row, reward_candidates, 0.0),
                reward_std=_maybe_get(row, reward_std_candidates, 0.0),
            )

        drift_analysis = forensics.get_kl_drift_analysis()
        if not drift_analysis:
            raise RuntimeError("KL drift analysis could not be computed from the provided data")

        run_id = Path(run_path).stem
        metrics_payload = [metric.to_dict() for metric in forensics.comprehensive_metrics_history]
        card = generate_kl_drift_card(
            metrics_payload,
            output_dir=output_dir,
            run_id=run_id,
            reference_period=reference_period,
        )

        typer.echo("\nKL drift analysis summary:")
        typer.echo(f"  Detected: {drift_analysis['detected']}")
        typer.echo(f"  Drift score: {drift_analysis['score']:.3f}")
        typer.echo(f"  Divergence: {drift_analysis['divergence']:.4f}")
        typer.echo(f"  Trend: {drift_analysis['trend']}")

        png_path = card.get("visualizations", {}).get("timeline")
        if png_path:
            typer.echo(f"\nKL drift card saved to: {png_path}")
            json_path = Path(png_path).with_suffix(".json")
            if json_path.exists():
                typer.echo(f"JSON card: {json_path}")

        if drift_analysis["detected"]:
            typer.echo(
                "\n⚠️  KL drift detected. Review the card for mitigation recommendations.",
                err=True,
            )

    except ValidationError as exc:
        typer.echo(format_error_message(exc), err=True)
        raise typer.Exit(1)
    except Exception as exc:
        typer.echo(
            format_structured_error_message(
                "KL drift analysis failed",
                run_path,
                "Successful KL drift analysis",
                str(exc),
                "Verify the metrics source contains KL and KL coefficient columns",
            ),
            err=True,
        )
        raise typer.Exit(1)


@forensics_app.command(name="doctor")
def forensics_doctor(
    run_or_repo: str = typer.Argument(..., help="Path to run or repository directory"),
):
    """Run comprehensive diagnostics on a training run or repository."""
    try:
        typer.echo(f"Running diagnostics on: {run_or_repo}")
        
        # Validate input path
        try:
            validated_path = validate_file_path(run_or_repo, must_exist=True)
            if not validated_path.is_dir():
                raise ValidationError(
                    format_structured_error_message(
                        "Invalid input type",
                        str(validated_path),
                        "Directory (repository or training run)",
                        "File",
                        "Provide a directory path for comprehensive diagnostics"
                    ),
                    error_code="EXPECTED_DIRECTORY"
                )
        except ValidationError as e:
            typer.echo(format_error_message(e), err=True)
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(
                format_structured_error_message(
                    "Path validation failed",
                    run_or_repo,
                    "Valid directory path",
                    f"Error accessing path: {e}",
                    "Check that the path exists and you have read permissions"
                ),
                err=True
            )
            raise typer.Exit(1)

        # Run env audit
        typer.echo("\n1. Environment audit...")
        try:
            determinism_card, lock_content = audit_environment(validated_path)
        except Exception as e:
            typer.echo(
                format_structured_error_message(
                    "Environment audit failed",
                    str(validated_path),
                    "Successful environment analysis",
                    f"Error during audit: {e}",
                    "Ensure the directory is accessible and contains valid project files"
                ),
                err=True
            )
            raise typer.Exit(1)

        # Run log scan
        typer.echo("2. Log scan...")
        try:
            scan_report = scan_logs(validated_path)
        except (ValidationError, AdapterError) as e:
            typer.echo(
                format_structured_error_message(
                    "Log scan failed",
                    str(validated_path),
                    "Directory with training logs",
                    f"No valid training logs found: {e}",
                    "Ensure directory contains training log files (.jsonl, .log, .csv) or skip log analysis"
                ),
                err=True
            )
            scan_report = {"rules_fired": [], "note": "Log scan skipped due to missing training files"}

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

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(
            format_structured_error_message(
                "Unexpected error",
                run_or_repo,
                "Successful diagnostic operation",
                f"Unexpected error: {e}",
                "Please report this issue with the full error message"
            ),
            err=True
        )
        raise typer.Exit(1)


# ============================================================================
# REWARD COMMANDS
# ============================================================================

def _load_score_file(path: str, label: str) -> Tuple[List[str], List[float]]:
    """Load prompts and scores from a JSONL score file."""

    prompts: List[str] = []
    scores: List[float] = []

    for line_number, record in enumerate(read_jsonl(path), start=1):
        if not isinstance(record, dict):
            raise ValueError(
                f"{label} score file contains a non-object record on line {line_number}."
            )

        if "score" not in record:
            raise ValueError(
                f"Missing 'score' field in {label} score file on line {line_number}."
            )

        score_value = record.get("score")
        if score_value is None:
            raise ValueError(
                f"Score value is missing for line {line_number} in {label} score file."
            )

        try:
            score_float = float(score_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Score value on line {line_number} in {label} score file is not numeric: {score_value!r}."
            ) from exc

        prompt_value = record.get("prompt") or record.get("text") or ""
        prompts.append(str(prompt_value))
        scores.append(score_float)

    if not scores:
        raise ValueError(f"No scores found in {label} score file at {path}.")

    return prompts, scores


def _has_prompt_text(prompts: Sequence[str]) -> bool:
    return any(str(prompt).strip() for prompt in prompts)


def _validate_score_alignment(prompts_a: Sequence[str], prompts_b: Sequence[str]) -> None:
    if len(prompts_a) != len(prompts_b):
        raise ValueError(
            "Score files must have the same number of records for drift comparison."
        )

    if _has_prompt_text(prompts_a) and _has_prompt_text(prompts_b):
        mismatched_rows = [
            str(index + 1)
            for index, (prompt_a, prompt_b) in enumerate(zip(prompts_a, prompts_b))
            if str(prompt_a) != str(prompt_b)
        ]
        if mismatched_rows:
            joined_rows = ", ".join(mismatched_rows[:5])
            suffix = "" if len(mismatched_rows) <= 5 else ", ..."
            raise ValueError(
                "Score files contain mismatched prompts at rows: "
                f"{joined_rows}{suffix}."
            )


def _select_prompts(prompts_a: Sequence[str], prompts_b: Sequence[str]) -> List[str]:
    if _has_prompt_text(prompts_a):
        return [str(prompt) for prompt in prompts_a]
    if _has_prompt_text(prompts_b):
        return [str(prompt) for prompt in prompts_b]
    return [""] * len(prompts_a)


@reward_app.command(name="reward-drift")
def reward_drift(
    model_a: Optional[str] = typer.Argument(
        None, help="Path to first reward model directory"
    ),
    model_b: Optional[str] = typer.Argument(
        None, help="Path to second reward model directory"
    ),
    prompts: Optional[str] = typer.Option(
        None, "--prompts", "-p", help="Path to prompts JSONL file"
    ),
    scores_a: Optional[str] = typer.Option(
        None, "--scores-a", help="Path to JSONL score file for the first model"
    ),
    scores_b: Optional[str] = typer.Option(
        None, "--scores-b", help="Path to JSONL score file for the second model"
    ),
):
    """Compare two reward models and detect drift."""
    try:
        using_score_files = scores_a is not None or scores_b is not None
        using_model_dirs = model_a is not None or model_b is not None

        if using_score_files and using_model_dirs:
            raise ValueError(
                "Provide either model directories with --prompts or two score files, not both."
            )

        if using_score_files:
            if scores_a is None or scores_b is None:
                raise ValueError(
                    "Both --scores-a and --scores-b must be provided when comparing score files."
                )

            typer.echo("Comparing reward scores from files:")
            typer.echo(f"  Scores A: {scores_a}")
            typer.echo(f"  Scores B: {scores_b}")

            prompts_a, scores_list_a = _load_score_file(scores_a, "First")
            prompts_b, scores_list_b = _load_score_file(scores_b, "Second")
            _validate_score_alignment(prompts_a, prompts_b)
            prompt_texts = _select_prompts(prompts_a, prompts_b)

            report = compare_score_lists(prompt_texts, scores_list_a, scores_list_b)
            scores_a_values = scores_list_a
            scores_b_values = scores_list_b
        else:
            if model_a is None or model_b is None:
                raise ValueError(
                    "Model directory paths are required when score files are not provided."
                )
            if prompts is None:
                raise ValueError(
                    "--prompts is required when comparing model directories."
                )

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

            model_a_fn = read_reward_head(model_a)
            model_b_fn = read_reward_head(model_b)

            scores_a_values = model_a_fn(prompt_texts)
            scores_b_values = model_b_fn(prompt_texts)

            report = compare_score_lists(prompt_texts, scores_a_values, scores_b_values)

        # Validate report
        validate(RewardDriftReportV1, report)

        # Write report and plot
        mkdir_reports()
        write_json_report(report, "rldk_reports/reward_drift.json")

        # Create scatter plot
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(scores_a_values, scores_b_values, alpha=0.6)

        # Add diagonal line
        min_val = min(min(scores_a_values), min(scores_b_values))
        max_val = max(max(scores_a_values), max(scores_b_values))
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
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
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
        typer.echo(f"  Drift magnitude: {report['drift_magnitude']:.4f}")
        typer.echo(f"  Effect size (Cohen's d): {report['effect_size']:.4f}")
        typer.echo(f"  Confidence: {report['confidence_summary']}")

        if report["slice_deltas"]:
            typer.echo("\nSlice analysis:")
            for slice_name, slice_data in report["slice_deltas"].items():
                typer.echo(
                    f"  {slice_name}: delta_mean={slice_data['delta_mean']:.4f}, n={slice_data['n']}"
                )

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@reward_app.command(name="length-bias")
def reward_length_bias(
    run_path: str = typer.Option(
        ..., "--run-path", "-r", help="Path to run metrics (file, directory, or wandb:// URI)"
    ),
    response_col: Optional[str] = typer.Option(
        None,
        "--response-col",
        help="Column containing response text for length analysis",
    ),
    reward_col: Optional[str] = typer.Option(
        None,
        "--reward-col",
        help="Column containing reward scores",
    ),
    length_col: Optional[str] = typer.Option(
        None,
        "--length-col",
        help="Column containing token counts or response lengths",
    ),
    threshold: float = typer.Option(
        0.35, "--threshold", help="Severity threshold for length bias detection"
    ),
    sample_size: Optional[int] = typer.Option(
        None, "--sample-size", help="Optional sample size for evaluation"
    ),
    adapter: Optional[str] = typer.Option(
        None, "--adapter", help="Adapter hint to use during ingestion"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Directory for JSON outputs"
    ),
    generate_card: bool = typer.Option(
        False,
        "--generate-card",
        help="Write a summary card JSON alongside the detailed report",
    ),
    tokenizer_name: Optional[str] = typer.Option(
        None,
        "--tokenizer-name",
        help="Optional Hugging Face tokenizer name for token counting",
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Random seed used when sampling rows"
    ),
) -> None:
    """Run reward length bias analysis using evaluation helpers."""

    ensure_config_initialized()
    typer.echo(f"Loading run data from: {run_path}")

    try:
        run_data = ingest_runs(run_path, adapter_hint=adapter)
    except (AdapterError, ValidationError) as exc:
        typer.echo(f"Failed to ingest run data: {exc}", err=True)
        raise typer.Exit(1)
    except Exception as exc:  # pragma: no cover - defensive path
        typer.echo(f"Unexpected ingestion error: {exc}", err=True)
        raise typer.Exit(1)

    if run_data.empty:
        typer.echo("Ingested data is empty; cannot analyze length bias.", err=True)
        raise typer.Exit(1)

    try:
        tokenizer = _load_tokenizer_by_name(tokenizer_name)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)

    try:
        result = evaluate_length_bias(
            run_data,
            response_col=response_col,
            reward_col=reward_col,
            length_col=length_col,
            threshold=threshold,
            sample_size=sample_size,
            seed=seed,
            tokenizer=tokenizer,
        )
    except Exception as exc:
        typer.echo(f"Length bias evaluation failed: {exc}", err=True)
        raise typer.Exit(1)

    severity = result.get("severity")
    typer.echo("\nLength bias evaluation summary:")
    typer.echo(f"  Responses analyzed: {result.get('response_count', 0)}")
    typer.echo(f"  Valid samples: {result.get('num_samples', 0)}")
    if severity is not None:
        typer.echo(f"  Severity: {severity:.3f}")
    typer.echo(f"  Score: {result.get('score', float('nan')):.3f}")
    typer.echo(
        f"  Passed threshold ({threshold}): {'yes' if result.get('passed') else 'no'}"
    )

    recommendations = result.get("recommendations") or []
    if recommendations:
        typer.echo("  Recommendations:")
        for line in recommendations:
            typer.echo(f"    - {line}")

    json_summary = json.dumps(result, indent=2)
    typer.echo("\nJSON report:")
    typer.echo(json_summary)

    report_path: Optional[Path] = None
    card_path: Optional[Path] = None
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report_path = output_path / "length_bias_report.json"
        write_json(result, report_path)
        typer.echo(f"\nReport saved to: {report_path}")

        if generate_card:
            card = {
                "version": "1.0",
                "generated_at": datetime.utcnow().isoformat(),
                "source": run_path,
                "passed": result.get("passed"),
                "score": result.get("score"),
                "severity": severity,
                "threshold": result.get("threshold"),
                "recommendations": recommendations,
                "metrics": result.get("metrics"),
            }
            card_path = output_path / "length_bias_card.json"
            write_json(card, card_path)
            typer.echo(f"Card saved to: {card_path}")
    elif generate_card:
        card = {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat(),
            "source": run_path,
            "passed": result.get("passed"),
            "score": result.get("score"),
            "severity": severity,
            "threshold": result.get("threshold"),
            "recommendations": recommendations,
            "metrics": result.get("metrics"),
        }
        card_path = Path("length_bias_card.json")
        write_json(card, card_path)
        typer.echo(f"\nCard saved to: {card_path}")

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
    gold_scores: Optional[str] = typer.Option(None, "--gold-scores", help="Optional path to trusted gold metrics for overoptimization detection"),
    gold_metric_col: Optional[str] = typer.Option(None, "--gold-metric-col", help="Column name containing gold metrics (in scores or gold dataset)"),
    auto_gold: bool = typer.Option(
        False,
        "--auto-gold",
        help="Automatically discover trusted gold metrics from the run directory when available",
    ),
    overopt_window: int = typer.Option(100, "--overopt-window", help="Window size (steps) for early/late delta comparison"),
    overopt_delta_threshold: float = typer.Option(0.2, "--overopt-delta-threshold", help="Minimum proxy-minus-gold delta to trigger overoptimization warning"),
    overopt_min_samples: int = typer.Option(100, "--overopt-min-samples", help="Minimum paired samples required for overoptimization detector"),
):
    """Run reward health analysis on scores data."""
    try:
        typer.echo(f"Running reward health analysis on scores: {scores}")

        # Ingest scores data
        typer.echo("Ingesting scores data...")
        scores_data = ingest_runs(scores, adapter_hint=adapter)

        gold_data: Optional[pd.DataFrame] = None
        if gold_scores and auto_gold:
            typer.echo(
                "⚠️  --gold-scores provided; ignoring --auto-gold discovery.", err=True
            )
            auto_gold = False

        if gold_scores:
            typer.echo(f"Ingesting gold metrics from: {gold_scores}")
            gold_data = ingest_runs(gold_scores, adapter_hint=adapter)
        elif auto_gold:
            if _dataframe_has_gold_metrics(scores_data):
                typer.echo(
                    "Auto-gold: detected trusted metric column in scores data; reusing embedded gold metrics."
                )
            else:
                auto_gold_path = _auto_detect_gold_artifact(scores)
                if auto_gold_path is not None:
                    typer.echo(f"Auto-detected gold metrics at: {auto_gold_path}")
                    gold_data = ingest_runs(str(auto_gold_path), adapter_hint=adapter)
                else:
                    typer.echo(
                        "⚠️  Auto gold enabled but no gold metrics were discovered alongside the run; continuing without gold metrics.",
                        err=True,
                    )

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
            gold_metrics=gold_data,
            gold_metric_col=gold_metric_col,
            overoptimization_window=overopt_window,
            overoptimization_delta_threshold=overopt_delta_threshold,
            overoptimization_min_samples=overopt_min_samples,
        )

        # Generate reports
        typer.echo("Generating reports...")
        generate_reward_health_report(health_report, out)

        # Display results
        severity = health_report.length_bias_metrics.bias_severity
        overopt = getattr(health_report, "overoptimization", None)
        if health_report.passed:
            typer.echo("\n✅ Reward health check passed")
            if severity is not None:
                typer.echo(f"  Length bias severity: {severity:.3f}")
            if overopt and getattr(overopt, "gold_metrics_available", False):
                typer.echo(
                    f"  Overoptimization delta: {getattr(overopt, 'delta', 0.0):.3f}"
                )
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
            if overopt and getattr(overopt, "flagged", False):
                typer.echo(
                    f"  - Reward overoptimization suspected (delta {getattr(overopt, 'delta', 0.0):.3f})"
                )
            elif overopt and overopt.gold_metrics_available:
                typer.echo(
                    f"  - Overoptimization delta {getattr(overopt, 'delta', 0.0):.3f} (below threshold)"
                )
            elif overopt and overopt.warning:
                typer.echo(f"  - Overoptimization check: {overopt.warning}")

            # Determine exit code based on severity
            critical_issues = 0
            if health_report.drift_detected:
                critical_issues += 1
            if health_report.label_leakage_risk > threshold_leakage:
                critical_issues += 1
            if overopt and getattr(overopt, "flagged", False):
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
        with open(file_path) as f:
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
    min_samples: int = 10,
    column_mapping: Optional[Dict[str, str]] = None,
    **kwargs
) -> dict:
    """
    Run evaluation suite on data.

    Args:
        data: Input data DataFrame
        suite_name: Name of evaluation suite
        output_column: Column containing model outputs
        events_column: Column containing event logs
        min_samples: Minimum samples required for evaluation
        column_mapping: Optional mapping from user column names to RLDK standard names
        **kwargs: Additional evaluation parameters

    Returns:
        Dictionary with evaluation results
    """
    from .evals.runner import run
    
    try:
        result = run(
            data,
            suite=suite_name,
            seed=42,
            sample_size=min_samples if min_samples > 0 else None,
            column_mapping=column_mapping
        )

        raw_scores = result.scores
        failed_from_metadata = set(result.metadata.get("failed_evaluations", []))
        skipped_from_metadata = set(result.metadata.get("skipped_evaluations", []))
        skipped_from_scores = {
            name
            for name, score in raw_scores.items()
            if pd.isna(score) and name not in failed_from_metadata
        }

        skipped_all = skipped_from_metadata.union(skipped_from_scores)
        failed_evaluations = len(failed_from_metadata)
        skipped_evaluations = len(skipped_all)

        serializable_scores = {}
        for key, value in raw_scores.items():
            if pd.isna(value):
                serializable_scores[key] = None
            elif callable(value):
                serializable_scores[key] = str(value)
            else:
                serializable_scores[key] = float(value) if isinstance(value, (int, float)) else value
        
        serializable_metadata = {}
        for key, value in result.metadata.items():
            if callable(value):
                serializable_metadata[key] = str(value)
            elif isinstance(value, dict):
                serializable_metadata[key] = {k: (str(v) if callable(v) else v) for k, v in value.items()}
            else:
                serializable_metadata[key] = value

        if skipped_all and "skipped_evaluations" not in serializable_metadata:
            serializable_metadata["skipped_evaluations"] = sorted(skipped_all)

        successful_evaluations = len(
            [
                name
                for name, score in raw_scores.items()
                if not pd.isna(score) and name not in failed_from_metadata
            ]
        )

        return {
            "suite_name": suite_name,
            "suite_description": f"Evaluation suite: {suite_name}",
            "evaluations": serializable_scores,
            "summary": {
                "total_evaluations": len(raw_scores),
                "successful_evaluations": successful_evaluations,
                "failed_evaluations": failed_evaluations,
                "skipped_evaluations": skipped_evaluations,
                "overall_score": float(result.overall_score) if not pd.isna(result.overall_score) else None,
                "errors": []
            },
            "metadata": serializable_metadata,
            "warnings": list(result.warnings)
        }
        
    except Exception as e:
        logging.error(f"Evaluation suite {suite_name} failed: {e}")
        return {
            "suite_name": suite_name,
            "suite_description": f"Evaluation suite: {suite_name}",
            "evaluations": {},
            "summary": {
                "total_evaluations": 0,
                "successful_evaluations": 0,
                "failed_evaluations": 1,
                "overall_score": 0.0,
                "errors": [{"evaluation": "suite_execution", "error": str(e)}]
            },
            "metadata": {},
            "warnings": [f"Suite execution failed: {str(e)}"]
        }


@evals_app.command()
def evaluate(
    input_path: Path = typer.Argument(
        ..., help="Path to run directory, metrics table, or evaluation dataset"
    ),
    suite: str = typer.Option(
        "quick",
        "--suite",
        "-s",
        help="Evaluation suite to run (quick/comprehensive/safety/training_metrics)",
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Path to output JSON file"
    ),
    output_column: str = typer.Option(
        "output", "--output-column", help="Column name containing model outputs"
    ),
    events_column: str = typer.Option(
        "events", "--events-column", help="Column name containing event logs"
    ),
    min_samples: int = typer.Option(
        10, "--min-samples", help="Minimum samples required for evaluation"
    ),
    timeout: int = typer.Option(
        300, "--timeout", help="Timeout in seconds for evaluation"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    column_mapping: Optional[str] = typer.Option(
        None,
        "--column-mapping",
        help="Column mapping as JSON or key=value pairs (e.g., 'global_step=step,kl=kl_mean' or '{\"global_step\":\"step\"}')",
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help=(
            "Field map preset to normalize training metrics "
            f"({', '.join(sorted(FIELD_MAP_PRESETS))})"
            if FIELD_MAP_PRESETS
            else "Field map preset name (e.g. trl, grpo)."
        ),
    ),
    field_map: Optional[str] = typer.Option(
        None,
        "--field-map",
        help="JSON object mapping source columns to canonical training metrics.",
    ),
):
    """
    Run evaluation suite on normalized runs or prepared datasets.

    Examples:
        rldk evals evaluate /path/to/run --suite quick --preset trl
        rldk evals evaluate metrics.csv --suite training_metrics --field-map '{"progress":"step"}'
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

        # Validate input path
        resolved_path = validate_file_path(input_path, must_exist=True)

        # Validate suite
        valid_suites = ["quick", "comprehensive", "safety", "training_metrics"]
        if suite not in valid_suites:
            raise ValidationError(
                f"Invalid evaluation suite: {suite}",
                suggestion=f"Use one of: {', '.join(valid_suites)}",
                error_code="INVALID_SUITE"
            )

        # Validate min_samples
        if min_samples < 0:
            raise ValidationError(
                f"Minimum samples must be non-negative, got: {min_samples}",
                suggestion="Use zero to auto-detect or a positive integer to sample",
                error_code="INVALID_MIN_SAMPLES"
            )

        print_operation_status("Input validation", "success")

        # Parse column mapping
        parsed_column_mapping = None
        if column_mapping:
            try:
                if column_mapping.startswith('{'):
                    parsed_column_mapping = json.loads(column_mapping)
                else:
                    parsed_column_mapping = {}
                    for pair in column_mapping.split(','):
                        if '=' in pair:
                            key, value = pair.split('=', 1)
                            parsed_column_mapping[key.strip()] = value.strip()
                        else:
                            raise ValueError(f"Invalid mapping format: {pair}")
                
                logging.info(f"Parsed column mapping: {parsed_column_mapping}")
            except (json.JSONDecodeError, ValueError) as e:
                raise ValidationError(
                    f"Invalid column mapping format: {e}",
                    suggestion="Use JSON format like '{\"old\":\"new\"}' or key=value pairs like 'old=new,old2=new2'",
                    error_code="INVALID_COLUMN_MAPPING"
                )

        try:
            combined_field_map = _combine_field_maps(preset, field_map)
        except ValueError as exc:
            raise ValidationError(
                f"Invalid field map: {exc}",
                suggestion="Provide JSON like '{\"source\":\"canonical\"}' or use --preset",
                error_code="INVALID_FIELD_MAP",
            ) from exc

        use_normalizer = False
        first_record: Optional[Dict[str, Any]] = None
        suffix = resolved_path.suffix.lower()
        if resolved_path.is_dir() or suffix in {".csv", ".tsv", ".parquet"}:
            use_normalizer = True
        if suite == "training_metrics" or combined_field_map:
            use_normalizer = True
        if suffix in {".jsonl", ".ndjson"}:
            try:
                with resolved_path.open("r", encoding="utf-8") as handle:
                    for raw_line in handle:
                        raw_line = raw_line.strip()
                        if not raw_line:
                            continue
                        first_record = json.loads(raw_line)
                        break
            except (json.JSONDecodeError, OSError):
                first_record = None

            if isinstance(first_record, dict):
                keys = {str(key) for key in first_record.keys()}
                output_like = {"output", "response", "completion", "text", "generation"}
                if not keys.intersection(output_like):
                    metric_keys = {
                        "reward",
                        "reward_mean",
                        "kl",
                        "kl_mean",
                        "tokens_in",
                        "tokens_out",
                    }
                    if {"name", "value"}.issubset(keys) or keys.intersection(metric_keys):
                        use_normalizer = True

        effective_field_map = combined_field_map
        if use_normalizer and parsed_column_mapping:
            effective_field_map = {**(effective_field_map or {}), **parsed_column_mapping}

        if use_normalizer and effective_field_map:
            logging.info(f"Using field map: {effective_field_map}")

        # Load data with progress indication
        with timed_operation_context("Data loading"):
            logging.info(f"Loading data from {resolved_path}")
            if use_normalizer:
                jsonl_like_table = (
                    suffix in {".jsonl", ".ndjson"}
                    and isinstance(first_record, dict)
                    and not {"name", "value"}.issubset({str(k) for k in first_record.keys()})
                )

                if jsonl_like_table:
                    raw_df = load_jsonl_data(resolved_path)
                    data = normalize_training_metrics(
                        raw_df, field_map=effective_field_map
                    )
                else:
                    data = normalize_training_metrics_source(
                        resolved_path, field_map=effective_field_map
                    )
            else:
                data = load_jsonl_data(resolved_path)

            if data.empty:
                raise ValidationError(
                    "No data found in source",
                    suggestion="Ensure the path contains valid training metrics or evaluation data",
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
                min_samples=min_samples,
                column_mapping=parsed_column_mapping
            )

        with timed_operation_context(f"{suite} evaluation suite"):
            logging.info(f"Running {suite} evaluation suite")
            results = run_evaluation()

        # Output results
        if output_file:
            print_operation_status("Saving results", "start")
            try:
                def make_serializable(obj):
                    if isinstance(obj, dict):
                        return {k: make_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [make_serializable(item) for item in obj]
                    elif callable(obj):
                        return str(obj)
                    elif hasattr(obj, '__dict__'):
                        return str(obj)
                    else:
                        return obj
                
                serializable_results = make_serializable(results)
                with open(output_file, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
                print_operation_status("Saving results", "success", f"Saved to {output_file}")
            except Exception as e:
                raise ValidationError(
                    f"Failed to save output file: {e}",
                    suggestion="Check write permissions and disk space",
                    error_code="SAVE_FAILED"
                ) from e
        else:
            # Print to stdout
            def make_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                elif callable(obj):
                    return str(obj)
                elif hasattr(obj, '__dict__'):
                    return str(obj)
                else:
                    return obj
            
            serializable_results = make_serializable(results)
            print(json.dumps(serializable_results, indent=2))

        # Print summary
        summary = results["summary"]
        print_operation_status("Evaluation", "success",
                             f"{summary['successful_evaluations']}/{summary['total_evaluations']} successful")

        typer.echo("\n📊 Evaluation Results:")
        typer.echo(f"  Suite: {suite}")
        typer.echo(f"  Samples: {len(data)}")
        typer.echo(f"  Successful: {summary['successful_evaluations']}")
        typer.echo(f"  Failed: {summary['failed_evaluations']}")
        if summary.get('skipped_evaluations', 0) > 0:
            typer.echo(f"  Skipped: {summary['skipped_evaluations']}")
        if summary['overall_score'] is not None:
            typer.echo(f"  Overall Score: {summary['overall_score']:.3f}")
        else:
            typer.echo("  Overall Score: N/A")

        if summary["errors"]:
            typer.echo("\n⚠️  Failed Evaluations:")
            for error in summary["errors"]:
                typer.echo(f"  - {error['evaluation']}: {error['error']}")

        # Exit with error code if any evaluations failed (but not if they were just skipped)
        if summary["failed_evaluations"] > 0:
            raise typer.Exit(1)

    except ValidationError as e:
        typer.echo(format_error_message(e), err=True)
        print_usage_examples("evaluate", [
            "rldk evals evaluate /path/to/run --suite quick --preset trl",
            "rldk evals evaluate metrics.csv --suite training_metrics --field-map '{\"progress\":\"step\"}'",
            "rldk evals evaluate data.jsonl --suite comprehensive --output results.json"
        ])
        print_troubleshooting_tips([
            "Ensure the input path exists and points to a run, table, or dataset",
            "Use --preset or --field-map to align custom metric column names",
            "Check that the specified columns exist in your data",
            "Use --verbose flag for detailed output",
            "Try reducing --min-samples if you have limited data"
        ])
        raise typer.Exit(1)
    except RLDKTimeoutError as e:
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

        print("File validation results:")
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
        None, "--adapter", "-a", help="Adapter type (trl, openrlhf, wandb, custom_jsonl, flexible)"
    ),
    output: Optional[str] = typer.Option(
        "metrics.jsonl", "--output", "-o", help="Output file path"
    ),
    field_map: Optional[str] = typer.Option(
        None, "--field-map", help="JSON string or file path with field mapping (e.g., '{\"step\": \"global_step\"}')"
    ),
    config_file: Optional[str] = typer.Option(
        None, "--config-file", "-c", help="YAML/JSON config file with field mapping"
    ),
    validation_mode: str = typer.Option(
        "flexible", "--validation-mode", help="Validation mode: strict, flexible, or lenient"
    ),
    required_fields: Optional[str] = typer.Option(
        None, "--required-fields", help="Comma-separated list of required fields (default: step,reward)"
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
        rldk ingest data.jsonl --adapter flexible --field-map '{"step": "global_step"}'
        rldk ingest data.jsonl --config-file field_mapping.yaml --validation-mode strict
        rldk ingest data.jsonl --required-fields step,reward,kl --validation-mode lenient
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
                valid_adapters = ["trl", "openrlhf", "wandb", "custom_jsonl", "flexible"]
                if adapter not in valid_adapters:
                    raise ValidationError(
                        f"Invalid adapter: {adapter}",
                        suggestion=f"Use one of: {', '.join(valid_adapters)}",
                        error_code="INVALID_ADAPTER"
                    )

            print_operation_status("Input validation", "success")

        parsed_field_map = None
        if field_map:
            if field_map.startswith('{'):
                import json
                try:
                    parsed_field_map = json.loads(field_map)
                except json.JSONDecodeError as e:
                    raise ValidationError(
                        f"Invalid JSON in field_map: {e}",
                        suggestion="Use valid JSON format like '{\"step\": \"global_step\"}'",
                        error_code="INVALID_FIELD_MAP_JSON"
                    )
            else:
                import json
                from pathlib import Path

                import yaml
                field_map_path = Path(field_map)
                if not field_map_path.exists():
                    raise ValidationError(
                        f"Field map file not found: {field_map}",
                        suggestion="Ensure the file path exists and is accessible",
                        error_code="FIELD_MAP_FILE_NOT_FOUND"
                    )
                try:
                    if field_map_path.suffix.lower() in ['.yaml', '.yml']:
                        with open(field_map_path) as f:
                            config = yaml.safe_load(f)
                            parsed_field_map = config.get('field_map', {})
                    else:
                        with open(field_map_path) as f:
                            parsed_field_map = json.load(f)
                except Exception as e:
                    raise ValidationError(
                        f"Failed to parse field map file: {e}",
                        suggestion="Ensure the file contains valid JSON or YAML",
                        error_code="FIELD_MAP_PARSE_ERROR"
                    )

        parsed_required_fields = None
        if required_fields:
            parsed_required_fields = [f.strip() for f in required_fields.split(',')]

        # Validate validation mode
        valid_validation_modes = ["strict", "flexible", "lenient"]
        if validation_mode not in valid_validation_modes:
            raise ValidationError(
                f"Invalid validation mode: {validation_mode}",
                suggestion=f"Use one of: {', '.join(valid_validation_modes)}",
                error_code="INVALID_VALIDATION_MODE"
            )

        if parsed_field_map or config_file:
            if not adapter:
                adapter = "flexible"
                if verbose:
                    typer.echo("🔧 Auto-selecting flexible adapter for field mapping options")
            elif adapter != "flexible":
                typer.echo("⚠️  Field mapping options require flexible adapter. Switching to flexible adapter.", err=True)
                adapter = "flexible"

        # Ingest the runs with progress indication
        with timed_operation_context("Data ingestion"):
            typer.echo(f"Ingesting runs from: {runs}")

            if adapter:
                typer.echo(f"Using adapter: {adapter}")

            if parsed_field_map and verbose:
                typer.echo(f"Field mapping: {parsed_field_map}")

            if validation_mode != "flexible" and verbose:
                typer.echo(f"Validation mode: {validation_mode}")

            df = ingest_runs(
                runs,
                adapter,
                field_map=parsed_field_map,
                config_file=config_file,
                validation_mode=validation_mode,
                required_fields=parsed_required_fields
            )

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
        typer.echo("\n📊 Ingestion Summary:")
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
            "Try auto-detection by omitting --adapter",
            "Use --adapter flexible with --field-map for custom schemas",
            "Try --validation-mode lenient for partial data"
        ])
        raise typer.Exit(1)
    except AdapterError as e:
        typer.echo(format_error_message(e), err=True)
        print_troubleshooting_tips([
            "Check that the data format matches the specified adapter",
            "Try different adapter types: trl, openrlhf, wandb, custom_jsonl, flexible",
            "Use --verbose flag to see detailed error information",
            "Try --adapter flexible with --field-map for custom field names"
        ])
        raise typer.Exit(1)
    except Exception as e:
        log_error_with_context(e, "ingest command")
        typer.echo(format_error_message(e, "Ingestion failed"), err=True)
        raise typer.Exit(1)


@app.command(name="diff")
def diff(
    a: str = typer.Option(
        ..., "--a", "-a", help="Path to training metrics for side A (file or directory)"
    ),
    b: str = typer.Option(
        ..., "--b", "-b", help="Path to training metrics for side B (file or directory)"
    ),
    signals: List[str] = typer.Option(
        ..., "--signals", "-s", help="Training metric columns to compare"
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help=(
            "Field map preset to normalize training metrics"
            f" ({', '.join(sorted(FIELD_MAP_PRESETS))})"
            if FIELD_MAP_PRESETS
            else "Field map preset name (e.g. trl, grpo)."
        ),
    ),
    field_map: Optional[str] = typer.Option(
        None,
        "--field-map",
        help="JSON object mapping source columns to canonical training metrics.",
    ),
    output_dir: str = typer.Option(
        "diff_analysis", "--output-dir", "-o", help="Output directory for the diff report"
    ),
):
    """Compare normalized training metrics between two runs."""

    try:
        typer.echo("Comparing training runs:")
        typer.echo(f"  Run A: {a}")
        typer.echo(f"  Run B: {b}")

        try:
            combined_map = _combine_field_maps(preset, field_map)
        except ValueError as exc:
            typer.echo(f"Invalid field map: {exc}", err=True)
            raise typer.Exit(1)

        mapping_dict = combined_map or None
        signal_list = _normalize_signals_option(signals)
        typer.echo(f"  Signals: {', '.join(signal_list)}")

        typer.echo("\nNormalizing run A...")
        df_a = normalize_training_metrics_source(a, field_map=mapping_dict)

        typer.echo("Normalizing run B...")
        df_b = normalize_training_metrics_source(b, field_map=mapping_dict)

        typer.echo("\nComputing diff statistics...")
        report = compare_training_metrics_tables(df_a, df_b, signal_list)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report_path = output_path / "diff_report.json"
        write_json(report, report_path)

        summary = report.get("summary", {})
        typer.echo("\nSummary:")
        typer.echo(f"  Signals compared: {summary.get('signals_compared', 0)}")
        typer.echo(f"  Verdict: {summary.get('verdict', 'unknown')}")
        max_abs_delta = summary.get("max_abs_delta")
        if max_abs_delta is not None:
            typer.echo(f"  Max abs delta: {max_abs_delta}")
        typer.echo(f"\nReport saved to: {report_path}")

    except ValidationError as exc:
        typer.echo(format_error_message(exc), err=True)
        raise typer.Exit(1)
    except Exception as exc:  # pragma: no cover - unexpected failure path
        typer.echo(f"Error: {exc}", err=True)
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
        report = check(cmd or "", compare_list, steps_list, actual_replicas, device)

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


app.command(name="determinism")(check_determinism_cmd)


@app.command(name="reward-health")
def reward_health(
    run_path: str = typer.Option(..., "--run", "-r", help="Path to training run data"),
    reference_path: Optional[str] = typer.Option(
        None, "--reference", "-ref", help="Path to reference run data"
    ),
    output_dir: str = typer.Option(
        "reward_analysis", "--output-dir", "-o", help="Output directory for reports"
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help=(
            "Field map preset to normalize training metrics"
            f" ({', '.join(sorted(FIELD_MAP_PRESETS))})"
            if FIELD_MAP_PRESETS
            else "Field map preset name (e.g. trl, grpo)."
        ),
    ),
    field_map: Optional[str] = typer.Option(
        None,
        "--field-map",
        help="JSON object mapping source columns to canonical training metrics.",
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
    response_col: Optional[str] = typer.Option(
        None,
        "--response-col",
        help="Column containing response text for length bias detection",
    ),
    length_col: Optional[str] = typer.Option(
        None,
        "--length-col",
        help="Column containing response lengths or token counts",
    ),
    threshold_length_bias: float = typer.Option(
        0.4,
        "--threshold-length-bias",
        help="Severity threshold for length bias detection",
    ),
    enable_length_bias_detection: bool = typer.Option(
        True,
        "--enable-length-bias-detection/--disable-length-bias-detection",
        help="Toggle dedicated length bias detection",
    ),
    gold_path: Optional[str] = typer.Option(
        None,
        "--gold",
        help="Optional path to trusted gold metrics for overoptimization analysis",
    ),
    gold_metric_col: Optional[str] = typer.Option(
        None,
        "--gold-col",
        help="Column containing gold metrics (in run or gold dataset)",
    ),
    auto_gold: bool = typer.Option(
        False,
        "--auto-gold",
        help="Automatically discover trusted gold metrics alongside the run artifacts",
    ),
    overopt_window: int = typer.Option(
        100,
        "--overopt-window",
        help="Window size (steps) used for early/late proxy vs gold comparison",
    ),
    overopt_delta_threshold: float = typer.Option(
        0.2,
        "--overopt-delta-threshold",
        help="Minimum proxy-minus-gold delta to raise overoptimization flag",
    ),
    overopt_min_samples: int = typer.Option(
        100,
        "--overopt-min-samples",
        help="Minimum paired samples required to evaluate overoptimization",
    ),
    gate: bool = typer.Option(
        False, "--gate", help="Enable CI gate mode with exit codes (0=pass, 1=warn, 2=fail)"
    ),
):
    """Analyze reward model health and detect pathologies."""

    def _ensure_column_present(df: pd.DataFrame, column: str, kind: str) -> None:
        if column not in df.columns:
            raise ValidationError(
                f"{kind.capitalize()} column '{column}' not found after normalization",
                suggestion=(
                    "Use --preset or --field-map to map your metrics to canonical names, "
                    "for example --field-map '{\"reward\": \"reward_mean\"}'"
                ),
                error_code=f"MISSING_{kind.upper()}_COLUMN",
            )
        if df[column].dropna().empty:
            raise ValidationError(
                f"Normalized {kind} column '{column}' is empty",
                suggestion=(
                    "Use --preset or --field-map to map the correct column or verify the source data"
                ),
                error_code=f"EMPTY_{kind.upper()}_COLUMN",
            )

    try:
        typer.echo(f"Analyzing reward health for run: {run_path}")

        try:
            combined_map = _combine_field_maps(preset, field_map)
        except ValueError as exc:
            typer.echo(f"Invalid field map: {exc}", err=True)
            raise typer.Exit(1)

        mapping_dict = combined_map or None

        # Normalize run data
        typer.echo("Normalizing run data...")
        try:
            run_data = normalize_training_metrics_source(run_path, field_map=mapping_dict)
        except TypeError as exc:
            if not Path(run_path).is_dir():
                raise
            try:
                run_data = _fallback_directory_load(Path(run_path), mapping_dict)
            except Exception as load_exc:
                raise exc from load_exc

        if run_data.empty:
            raise ValidationError(
                "Normalized run data is empty",
                suggestion="Ensure the source contains reward metrics",
                error_code="EMPTY_RUN_DATA",
            )

        _ensure_column_present(run_data, step_col, "step")
        _ensure_column_present(run_data, reward_col, "reward")

        reference_data = None
        if reference_path:
            typer.echo("Normalizing reference data...")
            try:
                reference_data = normalize_training_metrics_source(
                    reference_path, field_map=mapping_dict
                )
            except TypeError as exc:
                if not Path(reference_path).is_dir():
                    raise
                try:
                    reference_data = _fallback_directory_load(
                        Path(reference_path), mapping_dict
                    )
                except Exception as load_exc:
                    raise exc from load_exc
            if reference_data.empty:
                raise ValidationError(
                    "Normalized reference data is empty",
                    suggestion="Verify the reference source contains reward metrics",
                    error_code="EMPTY_REFERENCE_DATA",
                )
            _ensure_column_present(reference_data, step_col, "step")
            _ensure_column_present(reference_data, reward_col, "reward")

        gold_data = None
        if gold_path and auto_gold:
            typer.echo("⚠️  --gold provided; ignoring --auto-gold discovery.", err=True)
            auto_gold = False

        if gold_path:
            typer.echo("Normalizing gold metrics...")
            try:
                gold_data = normalize_training_metrics_source(
                    gold_path, field_map=mapping_dict
                )
            except TypeError as exc:
                if not Path(gold_path).is_dir():
                    raise
                try:
                    gold_data = _fallback_directory_load(Path(gold_path), mapping_dict)
                except Exception as load_exc:
                    raise exc from load_exc
            if gold_data.empty:
                raise ValidationError(
                    "Normalized gold metrics are empty",
                    suggestion="Ensure the gold source contains the trusted metric column",
                    error_code="EMPTY_GOLD_DATA",
                )
        elif auto_gold:
            if _dataframe_has_gold_metrics(run_data):
                typer.echo(
                    "Auto-gold: detected trusted metric column in normalized run data; using embedded gold metrics."
                )
            else:
                auto_gold_source = _auto_detect_gold_artifact(run_path)
                if auto_gold_source is not None:
                    typer.echo(f"Auto-detected gold metrics at: {auto_gold_source}")
                    try:
                        gold_data = normalize_training_metrics_source(
                            auto_gold_source, field_map=mapping_dict
                        )
                    except TypeError as exc:
                        if not Path(auto_gold_source).is_dir():
                            raise
                        try:
                            gold_data = _fallback_directory_load(
                                Path(auto_gold_source), mapping_dict
                            )
                        except Exception as load_exc:
                            raise exc from load_exc
                    if gold_data is not None and gold_data.empty:
                        typer.echo(
                            "⚠️  Auto-detected gold metrics are empty; continuing without gold metrics.",
                            err=True,
                        )
                        gold_data = None
                else:
                    typer.echo(
                        "⚠️  Auto gold enabled but no gold metrics were discovered alongside the run; continuing without gold metrics.",
                        err=True,
                    )

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
            response_col=response_col,
            length_col=length_col,
            threshold_length_bias=threshold_length_bias,
            enable_length_bias_detection=enable_length_bias_detection,
            gold_metrics=gold_data,
            gold_metric_col=gold_metric_col,
            overoptimization_window=overopt_window,
            overoptimization_delta_threshold=overopt_delta_threshold,
            overoptimization_min_samples=overopt_min_samples,
        )

        # Generate reports
        typer.echo("Generating reports...")
        generate_reward_health_report(health_report, output_dir)

        # Display results
        severity = health_report.length_bias_metrics.bias_severity
        overopt = getattr(health_report, "overoptimization", None)
        if health_report.passed:
            typer.echo("\n✅ Reward health check passed")
            if severity is not None:
                typer.echo(f"  Length bias severity: {severity:.3f}")
            if overopt and getattr(overopt, "gold_metrics_available", False):
                typer.echo(
                    f"  Overoptimization delta: {getattr(overopt, 'delta', 0.0):.3f}"
                )
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
            if overopt and getattr(overopt, "flagged", False):
                typer.echo(
                    f"  - Reward overoptimization suspected (delta {getattr(overopt, 'delta', 0.0):.3f})"
                )
            elif overopt and overopt.gold_metrics_available:
                typer.echo(
                    f"  - Overoptimization delta {getattr(overopt, 'delta', 0.0):.3f} (below threshold)"
                )
            elif overopt and overopt.warning:
                typer.echo(f"  - Overoptimization check: {overopt.warning}")
            if health_report.length_bias_detected:
                typer.echo(
                    f"  - Length bias severity {severity:.3f} exceeds threshold"
                    if severity is not None
                    else "  - Length bias detected"
                )
            elif severity is not None:
                typer.echo(
                    f"  - Length bias severity: {severity:.3f} (below threshold)"
                )

            # Determine exit code based on severity
            critical_issues = 0
            if health_report.drift_detected:
                critical_issues += 1
            if health_report.label_leakage_risk > threshold_leakage:
                critical_issues += 1
            if overopt and getattr(overopt, "flagged", False):
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

    except (ValidationError, AdapterError) as exc:
        typer.echo(format_error_message(exc), err=True)
        if gate:
            typer.echo("GATE: FAIL")
            raise typer.Exit(2)
        raise typer.Exit(1)
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
        metrics_list = [m.strip() for m in (metrics.split(",") if isinstance(metrics, str) else metrics)]

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

        typer.echo("✅ Experiment tracking started successfully")
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
                tracker.finish_experiment()
                typer.echo("✅ Experiment completed successfully!")
        else:
            # Non-interactive mode - finish immediately
            typer.echo("\n📊 Experiment tracking completed.")
            typer.echo("Environment, Git, and seed state have been captured.")
            typer.echo("Use --interactive flag to keep tracker running for manual logging.")

            tracker.finish_experiment()
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

    typer.echo("📊 Analysis results:")
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
            from .adapters import (
                CustomJSONLAdapter,
                OpenRLHFAdapter,
                TRLAdapter,
                WandBAdapter,
            )

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
        typer.echo("\n💡 Adapter suggestions:")

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

        typer.echo("\n💡 Use 'rldk format-info --adapter <name>' for detailed format requirements")
        typer.echo(f"💡 Use 'rldk validate-format {source} --adapter <name>' to test specific adapter")


@app.command(name="version")
def version():
    """Show version information."""
    from rldk import __version__

    typer.echo(f"RL Debug Kit version {__version__}")


@app.command(name="seed")
def seed_cmd(
    seed_value: Optional[int] = typer.Option(None, "--seed", "-s", help="Seed value to set"),
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
            get_current_seed,
            get_seed_state_summary,
            set_global_seed,
            set_reproducible_environment,
            validate_seed_consistency,
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
                actual_seed = set_reproducible_environment(seed_value)
                typer.echo(f"🌱 Reproducible environment set with seed: {actual_seed}")
                typer.echo("  Environment variables configured for maximum reproducibility")
            else:
                # Just set the seed
                actual_seed = set_global_seed(seed_value, deterministic)
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
    run_a: str = typer.Argument(
        ..., help="Path to first run directory or metrics file"
    ),
    run_b: Optional[str] = typer.Argument(
        None, help="Path to second run directory (for drift cards)"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Output directory for cards"
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help=(
            "Field map preset to normalize training metrics"
            f" ({', '.join(sorted(FIELD_MAP_PRESETS))})"
            if FIELD_MAP_PRESETS
            else "Field map preset name (e.g. trl, grpo)."
        ),
    ),
    field_map: Optional[str] = typer.Option(
        None,
        "--field-map",
        help="JSON object mapping source columns to canonical training metrics.",
    ),
):
    """Generate trust cards for RL training runs."""
    try:
        try:
            combined_field_map = _combine_field_maps(preset, field_map)
        except ValueError as exc:
            typer.echo(f"Invalid field map: {exc}", err=True)
            raise typer.Exit(1)

        def _normalize_to_events(path: str) -> Tuple[pd.DataFrame, List[Any]]:
            try:
                df = normalize_training_metrics_source(
                    path, field_map=combined_field_map
                )
            except ValidationError as exc:
                typer.echo(format_structured_error_message(exc), err=True)
                raise typer.Exit(1)

            run_id_value: Optional[str] = None
            if "run_id" in df.columns:
                non_null = df["run_id"].dropna()
                if not non_null.empty:
                    run_id_value = str(non_null.iloc[0])

            git_sha_value: Optional[str] = None
            if "git_sha" in df.columns:
                git_non_null = df["git_sha"].dropna()
                if not git_non_null.empty:
                    git_sha_value = str(git_non_null.iloc[0])

            try:
                events = dataframe_to_events(
                    df, run_id=run_id_value, git_sha=git_sha_value
                )
            except ValidationError as exc:
                typer.echo(format_structured_error_message(exc), err=True)
                raise typer.Exit(1)

            return df, events

        if card_type == "determinism":
            typer.echo(f"Generating determinism card for run: {run_a}")

            # Normalize run to events
            _, events = _normalize_to_events(run_a)

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

            # Normalize runs to events
            _, events_a = _normalize_to_events(run_a)
            _, events_b = _normalize_to_events(run_b)

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

            # Normalize run to events
            metrics_df, events = _normalize_to_events(run_a)

            if metrics_df.empty or metrics_df["reward_mean"].dropna().empty:
                typer.echo(
                    "Warning: No reward_mean values found after normalization. "
                    "Use --preset or --field-map to map your reward metric.",
                    err=True,
                )

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
