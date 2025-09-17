"""Adapter for TRL (Transformer Reinforcement Learning) logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from ..utils.error_handling import AdapterError, ValidationError
from .base import BaseAdapter
from .flexible import FlexibleDataAdapter


class TRLAdapter(BaseAdapter):
    """Adapter that normalizes TRL logs using the shared TrainingMetrics schema."""

    PRESET = "trl"
    _SUPPORTED_EXTENSIONS = {".jsonl", ".ndjson"}
    _COLUMN_ALIASES: Dict[str, str] = {
        "reward": "reward_mean",
        "kl": "kl_mean",
        "entropy": "entropy_mean",
        "clipfrac": "clip_frac",
        "learning_rate": "lr",
        "gradient_norm": "grad_norm",
        "total_loss": "loss",
        "policy_loss": "loss",
        "input_tokens": "tokens_in",
        "output_tokens": "tokens_out",
        "experiment_id": "run_id",
        "commit_hash": "git_sha",
    }

    def can_handle(self) -> bool:
        """Return ``True`` if the source looks like a TRL metrics export."""

        if not self.source.exists():
            return False

        if self.source.is_file():
            return self.source.suffix.lower() in self._SUPPORTED_EXTENSIONS

        if self.source.is_dir():
            return any(
                path.suffix.lower() in self._SUPPORTED_EXTENSIONS
                for path in self._iter_candidate_files(self.source)
            )

        return False

    def load(self) -> pd.DataFrame:
        """Load TRL logs and normalize them into the TrainingMetrics table."""

        if not self.can_handle():
            raise AdapterError(
                f"Cannot handle source: {self.source}",
                suggestion=(
                    "Provide a TRL JSONL/NDJSON file or directory containing TRL events"
                ),
                error_code="CANNOT_HANDLE_SOURCE",
            )

        frames: List[pd.DataFrame] = []

        for path in self._iter_candidate_files(self.source):
            try:
                frame = self._load_jsonl_file(path)
            except ValidationError as exc:
                self.logger.warning(
                    "TRL adapter could not normalize %s directly (%s); falling back to the flexible adapter. "
                    "Provide --field-map if your metrics use custom names.",
                    path,
                    exc,
                )
                frame = self._fallback_with_flexible(path)
            frames.append(self._apply_legacy_aliases(frame))

        if not frames:
            raise AdapterError(
                f"No TRL metrics found in {self.source}",
                suggestion="Ensure the path contains TRL JSONL event files",
                error_code="NO_VALID_METRICS_FOUND",
            )

        combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

        try:
            from ..ingest.training_metrics_normalizer import standardize_training_metrics

            return standardize_training_metrics(combined)
        except ValidationError as exc:
            raise AdapterError(
                f"Failed to standardize TRL metrics from {self.source}: {exc}",
                suggestion="Provide --field-map or use the flexible adapter for custom schemas",
                error_code="SCHEMA_STANDARDIZATION_FAILED",
            ) from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _iter_candidate_files(self, source: Path) -> Iterable[Path]:
        if source.is_file():
            yield source
            return

        for pattern in ("*.jsonl", "*.ndjson"):
            for path in sorted(source.glob(pattern)):
                yield path

    def _load_jsonl_file(self, path: Path) -> pd.DataFrame:
        format_hint, first_record = self._detect_jsonl_format(path)

        if format_hint == "event_stream":
            from ..ingest.stream_normalizer import stream_jsonl_to_dataframe

            return stream_jsonl_to_dataframe(path, preset=self.PRESET)

        if format_hint == "event_dict":
            from ..io.event_schema import events_to_dataframe

            records = self._read_jsonl_records(path)
            return events_to_dataframe(records)

        if format_hint == "flat_table":
            records = self._read_jsonl_records(path)
            if not records:
                raise ValidationError(
                    f"No valid TRL records found in {path}",
                    suggestion="Ensure the file contains JSON objects",
                    error_code="NO_VALID_EVENTS",
                )
            return pd.DataFrame(records)

        raise ValidationError(
            f"Unrecognized TRL log structure in {path}",
            suggestion="Use --field-map with the flexible adapter to describe custom layouts",
            error_code="UNKNOWN_TRL_FORMAT",
            details={"first_record": first_record},
        )

    def _detect_jsonl_format(self, path: Path) -> Tuple[str | None, Dict[str, object] | None]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        continue
                    metrics = record.get("metrics")
                    if isinstance(metrics, dict):
                        return "event_dict", record
                    if metrics is not None and not isinstance(metrics, dict):
                        return None, record
                    if any(key in record for key in ("metric", "metric_name", "name")):
                        return "event_stream", record
                    return "flat_table", record
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"Invalid JSON in {path}: {exc}",
                suggestion="Ensure each line is a valid JSON object",
                error_code="INVALID_JSONL",
            ) from exc

        return None, None

    def _read_jsonl_records(self, path: Path) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        skipped = 0

        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
                else:
                    skipped += 1

        if skipped:
            self.logger.warning(
                "Skipped %d invalid JSONL line%s while reading %s",
                skipped,
                "" if skipped == 1 else "s",
                path,
            )

        return records

    def _fallback_with_flexible(self, path: Path) -> pd.DataFrame:
        adapter = FlexibleDataAdapter(
            path,
            validation_mode="flexible",
            required_fields=["step"],
        )

        try:
            df = adapter.load()
        except (AdapterError, ValidationError) as exc:
            raise AdapterError(
                f"Flexible adapter could not read {path}",
                suggestion="Provide --field-map to map your custom metric names",
                error_code="TRL_FALLBACK_FAILED",
            ) from exc

        return df

    def _apply_legacy_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        normalized = df.copy()

        if "wall_time" not in normalized and "wall_time_ms" in normalized:
            normalized["wall_time"] = pd.to_numeric(
                normalized["wall_time_ms"], errors="coerce"
            ) / 1000.0
        if "wall_time_ms" in normalized.columns:
            normalized = normalized.drop(columns=["wall_time_ms"])

        for alias, canonical in self._COLUMN_ALIASES.items():
            if canonical in normalized.columns:
                continue
            if alias in normalized.columns:
                normalized[canonical] = normalized[alias]

        return normalized


__all__ = ["TRLAdapter"]

