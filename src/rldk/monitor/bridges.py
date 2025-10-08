from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from numbers import Number
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Tuple

from .engine import Event, _now_iso, canonicalize_event

logger = logging.getLogger(__name__)

_WANDB_TRANSIENT_TOKENS = ("rate limit", "temporarily unavailable", "timeout", "timed out", "connection", "429", "502", "503")
_MLFLOW_TRANSIENT_TOKENS = ("temporarily unavailable", "timeout", "connection", "429", "502", "503")
_WANDB_SKIP_KEYS = {
    "_step",
    "_timestamp",
    "_runtime",
    "_session_id",
    "_wandb",
    "_runtime_seconds",
    "global_step",
    "epoch",
    "_metric_system",
    "_metric_variant",
    "_runtime_ms",
    "_runtime_min",
    "_runtime_epoch",
    "_runtime_hr",
    "_runtime_day",
    "_timestamp_seconds",
    "_timestamp_ms",
    "_timestamp_min",
    "_timestamp_hour",
    "_timestamp_day",
    "_index",
    "_batch",
    "_runtime_total",
    "_runtime_iter",
    "_runtime_step",
}


def _timestamp_to_iso(timestamp: Any) -> str:
    if timestamp is None:
        return _now_iso()
    try:
        value = float(timestamp)
    except (TypeError, ValueError):
        return _now_iso()
    if value > 1e12:
        value /= 1000.0
    try:
        parsed = datetime.fromtimestamp(value, tz=timezone.utc)
    except (OSError, OverflowError):
        return _now_iso()
    return parsed.isoformat().replace("+00:00", "Z")


def _sanitize_tags(tags: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in tags.items() if value not in (None, "")}


def _is_numeric(value: Any) -> bool:
    if not isinstance(value, Number):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def _should_retry(message: str, tokens: Sequence[str]) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in tokens)


def _parse_wandb_target(target: str) -> Tuple[str, str, Optional[str]]:
    raw = target.strip()
    if raw.startswith("wandb://"):
        raw = raw[len("wandb://") :]
    parts = [part for part in raw.split("/") if part]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], None
    raise ValueError(
        "W&B target must be formatted as 'entity/project/run_id' or 'entity/project'"
    )


def stream_from_wandb(
    target: str,
    poll_interval: float = 5.0,
    idle_sleep: float = 5.0,
) -> Iterator[Event]:
    """Stream metrics from a W&B run or project."""

    try:
        import wandb  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "wandb package is required for --from-wandb. Install with 'pip install wandb'."
        ) from exc

    try:
        entity, project, run_id = _parse_wandb_target(target)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    api = wandb.Api()
    run = None
    resolved_target = None
    try:
        if run_id:
            resolved_target = f"{entity}/{project}/{run_id}"
            run = api.run(resolved_target)
        else:
            candidates = list(
                api.runs(f"{entity}/{project}", order="-updated_at", per_page=1)
            )
            if not candidates:
                raise RuntimeError(
                    f"No runs found for W&B project '{entity}/{project}'."
                )
            run = candidates[0]
            resolved_target = f"{entity}/{project}/{run.id}"
    except Exception as exc:  # pragma: no cover - network interaction
        raise RuntimeError(f"Failed to resolve W&B run '{target}': {exc}") from exc

    logger.info(
        "Streaming metrics from W&B run %s (%s/%s)",
        getattr(run, "id", resolved_target),
        getattr(run, "entity", entity),
        getattr(run, "project", project),
    )

    seen: set[Tuple[str, int, Optional[float], Optional[float]]] = set()
    last_step = -1
    backoff = poll_interval
    while True:
        try:
            run.reload()
            history: Iterable[Dict[str, Any]] = run.scan_history(
                page_size=500, min_step=max(last_step + 1, 0)
            )
        except Exception as exc:  # pragma: no cover - network interaction
            wait = min(max(backoff, poll_interval), 60.0)
            message = str(exc) or exc.__class__.__name__
            if _should_retry(message, _WANDB_TRANSIENT_TOKENS):
                logger.warning("W&B API unavailable (%s). Retrying in %.1fs", message, wait)
                time.sleep(wait)
                backoff = min(wait * 2, 60.0)
                continue
            raise RuntimeError(f"W&B streaming failed: {message}") from exc
        emitted = False
        backoff = poll_interval
        for row in history:
            step_raw = row.get("_step") or row.get("step") or row.get("global_step")
            try:
                step = int(step_raw) if step_raw is not None else None
            except (TypeError, ValueError):
                step = None
            if step is None:
                continue
            last_step = max(last_step, step)
            timestamp = _timestamp_to_iso(row.get("_timestamp"))
            tags = _sanitize_tags(
                {
                    "source": "wandb",
                    "wandb_entity": getattr(run, "entity", entity),
                    "wandb_project": getattr(run, "project", project),
                    "wandb_run": getattr(run, "id", None),
                }
            )
            meta: Dict[str, Any] = {}
            if getattr(run, "url", None):
                meta["wandb_url"] = run.url
            for key, value in row.items():
                if key in _WANDB_SKIP_KEYS or key.startswith("_"):
                    continue
                if not _is_numeric(value):
                    continue
                signature = (key, step, row.get("_timestamp"), row.get("_runtime"))
                if signature in seen:
                    continue
                seen.add(signature)
                payload = {
                    "time": timestamp,
                    "step": step,
                    "name": key,
                    "value": float(value),
                    "run_id": getattr(run, "id", resolved_target),
                    "tags": tags,
                }
                if meta:
                    payload["meta"] = meta
                try:
                    event = canonicalize_event(payload, None)
                except ValueError as exc:
                    logger.debug(
                        "Skipping W&B metric %s at step %s: %s", key, step, exc
                    )
                    continue
                emitted = True
                yield event
        if not emitted:
            time.sleep(idle_sleep)


def stream_from_mlflow(
    run_id: str,
    poll_interval: float = 5.0,
) -> Iterator[Event]:
    """Stream metrics from an MLflow run."""

    try:
        from mlflow.tracking import MlflowClient  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "mlflow package is required for --from-mlflow. Install with 'pip install mlflow'."
        ) from exc

    client = MlflowClient()
    try:
        run = client.get_run(run_id)
    except Exception as exc:  # pragma: no cover - network interaction
        raise RuntimeError(f"Failed to load MLflow run '{run_id}': {exc}") from exc

    known_metrics = set(run.data.metrics.keys())
    metric_offsets: Dict[str, int] = {name: 0 for name in known_metrics}
    backoff = poll_interval

    while True:
        try:
            run = client.get_run(run_id)
        except Exception as exc:  # pragma: no cover - network interaction
            wait = min(max(backoff, poll_interval), 60.0)
            message = str(exc) or exc.__class__.__name__
            if _should_retry(message, _MLFLOW_TRANSIENT_TOKENS):
                logger.warning(
                    "MLflow tracking API unavailable (%s). Retrying in %.1fs", message, wait
                )
                time.sleep(wait)
                backoff = min(wait * 2, 60.0)
                continue
            raise RuntimeError(f"MLflow streaming failed: {message}") from exc

        backoff = poll_interval
        tags = dict(run.data.tags)
        experiment_id = getattr(run.info, "experiment_id", None)
        run_name = getattr(run.info, "run_name", None)
        source_tags = _sanitize_tags(
            {
                "source": "mlflow",
                "mlflow_experiment_id": experiment_id,
                "mlflow_run_name": run_name,
            }
        )
        metric_names = set(run.data.metrics.keys()) | known_metrics
        known_metrics = metric_names
        emitted = False
        for name in sorted(metric_names):
            try:
                history = client.get_metric_history(run_id, name)
            except Exception as exc:  # pragma: no cover - network interaction
                message = str(exc) or exc.__class__.__name__
                if _should_retry(message, _MLFLOW_TRANSIENT_TOKENS):
                    wait = min(max(backoff, poll_interval), 60.0)
                    logger.warning(
                        "MLflow metric fetch failed for %s (%s). Retrying in %.1fs",
                        name,
                        message,
                        wait,
                    )
                    time.sleep(wait)
                    backoff = min(wait * 2, 60.0)
                    continue
                logger.error("Failed to fetch MLflow history for %s: %s", name, message)
                continue
            offset = metric_offsets.get(name, 0)
            if offset >= len(history):
                continue
            for metric in history[offset:]:
                if not _is_numeric(metric.value):
                    continue
                value = float(metric.value)
                timestamp_value = getattr(metric, "timestamp", None)
                timestamp = _timestamp_to_iso(
                    (timestamp_value / 1000.0) if timestamp_value is not None else None
                )
                payload = {
                    "time": timestamp,
                    "step": int(metric.step),
                    "name": name,
                    "value": value,
                    "run_id": run_id,
                    "tags": source_tags,
                }
                if tags:
                    payload["meta"] = {"mlflow_tags": dict(tags)}
                try:
                    event = canonicalize_event(payload, None)
                except ValueError as exc:
                    logger.debug(
                        "Skipping MLflow metric %s at step %s: %s", name, metric.step, exc
                    )
                    continue
                emitted = True
                yield event
            metric_offsets[name] = len(history)
        if not emitted:
            time.sleep(poll_interval)


__all__ = ["stream_from_mlflow", "stream_from_wandb"]
