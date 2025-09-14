"""Adapter for TRL (Transformer Reinforcement Learning) logs."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ..utils.error_handling import AdapterError
from .base import BaseAdapter


class TRLAdapter(BaseAdapter):
    """Adapter for TRL training logs."""

    def can_handle(self) -> bool:
        """Check if source contains TRL logs."""
        try:
            if not self.source.exists():
                return False

            # Look for TRL-specific files or patterns
            if self.source.is_file():
                return self._is_trl_file(self.source)
            elif self.source.is_dir():
                # Check for common TRL log files including new RLDK JSONL events
                trl_files = (list(self.source.glob("*.log")) +
                            list(self.source.glob("trainer_log.jsonl")) +
                            list(self.source.glob("*_events.jsonl")) +
                            list(self.source.glob("*.jsonl")))  # Be more permissive
                return len(trl_files) > 0

            return False
        except Exception as e:
            self.logger.warning(f"Error checking if source can be handled: {e}")
            return False

    def _is_trl_file(self, file_path: Path) -> bool:
        """Check if a file contains TRL logs."""
        try:
            self._handle_file_error(file_path, "read")

            if file_path.suffix == ".jsonl":
                with open(file_path) as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        # Check for TRL-specific keywords or our test fixture format
                        # First check if data is a dict and has the required keys
                        if isinstance(data, dict):
                            # Check for new Event schema format
                            if "metrics" in data and "model_info" in data:
                                return True
                            # Check for old format - be more flexible
                            has_required_fields = all(
                                key in data
                                for key in ["step", "phase", "reward_mean", "kl_mean"]
                            )
                            has_trl_keywords = (
                                "trl" in str(data).lower()
                                or "trainer" in str(data).lower()
                            )
                            # Accept if it has required fields OR TRL keywords
                            return has_required_fields or has_trl_keywords
                        else:
                            # For non-dict data, only check string content
                            return (
                                "trl" in str(data).lower()
                                or "trainer" in str(data).lower()
                            )
            elif file_path.suffix == ".log":
                with open(file_path) as f:
                    content = f.read()
                    return "trl" in content.lower() or "trainer" in content.lower()
        except (OSError, UnicodeDecodeError, TypeError) as e:
            # Log the specific error for debugging but don't fail the check
            self.logger.warning(f"Error checking if file {file_path} is TRL format: {e}")
            return False
        except json.JSONDecodeError as e:
            # Log JSON decode error but don't fail the check
            self.logger.warning(f"JSON decode error in {file_path}: {e}")
            return False
        except AdapterError:
            # Re-raise adapter errors
            raise
        except Exception as e:
            self.logger.warning(f"Unexpected error checking TRL file {file_path}: {e}")
            return False
        return False

    def load(self) -> pd.DataFrame:
        """Load TRL logs and convert to standard format."""
        if not self.can_handle():
            raise AdapterError(
                f"Cannot handle source: {self.source}",
                suggestion="Check that the source contains TRL training logs",
                error_code="CANNOT_HANDLE_SOURCE"
            )

        self._log_operation("Loading TRL data", {"source": str(self.source)})

        metrics = []

        try:
            if self.source.is_file():
                metrics = self._parse_file(self.source)
            elif self.source.is_dir():
                # Find and parse all TRL log files
                log_files = list(self.source.glob("*.log")) + list(
                    self.source.glob("*.jsonl")
                )
                if not log_files:
                    raise AdapterError(
                        f"No log files found in directory: {self.source}",
                        suggestion="Ensure the directory contains .log or .jsonl files",
                        error_code="NO_LOG_FILES_FOUND"
                    )

                for log_file in log_files:
                    try:
                        file_metrics = self._parse_file(log_file)
                        metrics.extend(file_metrics)
                    except Exception as e:
                        self.logger.warning(f"Failed to parse {log_file}: {e}")
                        continue

            if not metrics:
                # Try fallback parsing with more lenient requirements
                metrics = self._fallback_parse()
                if not metrics:
                    raise AdapterError(
                        f"No valid TRL metrics found in {self.source}",
                        suggestion="Check that the source contains valid TRL training data. Required fields: step, phase, reward_mean, kl_mean",
                        error_code="NO_VALID_METRICS_FOUND"
                    )

            # Convert to DataFrame
            df = pd.DataFrame(metrics)
            df = self._validate_dataframe(df)

            # Ensure required columns exist
            required_cols = [
                "step",
                "phase",
                "reward_mean",
                "reward_std",
                "kl_mean",
                "entropy_mean",
                "clip_frac",
                "grad_norm",
                "lr",
                "loss",
                "tokens_in",
                "tokens_out",
                "wall_time",
                "seed",
                "run_id",
                "git_sha",
            ]

            for col in required_cols:
                if col not in df.columns:
                    df[col] = None

            self._log_operation("TRL data loaded successfully", {"records": len(df)})
            return df[required_cols]

        except AdapterError:
            raise
        except Exception as e:
            raise AdapterError(
                f"Failed to load TRL data: {e}",
                suggestion="Check that the source contains valid TRL training logs",
                error_code="LOAD_FAILED",
                details={"source": str(self.source)}
            ) from e

    def _parse_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse a single TRL log file."""
        metrics = []

        try:
            if file_path.suffix == ".jsonl":
                with open(file_path) as f:
                    for line_num, line in enumerate(f, 1):  # Start line numbering from 1
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            metric = self._extract_trl_metric(data, line_num)
                            if metric:
                                metrics.append(metric)
                        except json.JSONDecodeError as e:
                            print(f"Warning: JSON decode error in {file_path} at line {line_num}: {e}")
                            continue

            elif file_path.suffix == ".log":
                with open(file_path) as f:
                    for line_num, line in enumerate(f, 1):  # Start line numbering from 1
                        metric = self._parse_log_line(line, line_num)
                        if metric:
                            metrics.append(metric)

        except (OSError, UnicodeDecodeError) as e:
            print(f"Warning: Error parsing {file_path}: {e}")
            # Re-raise the exception with context
            raise RuntimeError(f"Failed to parse TRL file {file_path}: {e}") from e

        return metrics

    def _extract_trl_metric(
        self, data: Dict[str, Any], line_num: int
    ) -> Dict[str, Any]:
        """Extract metric from TRL JSON data."""
        # Handle both old format and new Event schema format
        if "metrics" in data and "model_info" in data:
            # New Event schema format
            metrics = data.get("metrics", {})
            model_info = data.get("model_info", {})
            rng = data.get("rng", {})
            data_slice = data.get("data_slice", {})

            metric = {
                "step": data.get("step", line_num),
                "phase": model_info.get("phase", "train"),
                "reward_mean": metrics.get("reward_mean"),
                "reward_std": metrics.get("reward_std"),
                "kl_mean": metrics.get("kl_mean"),
                "entropy_mean": metrics.get("entropy_mean"),
                "clip_frac": metrics.get("clip_frac"),
                "grad_norm": metrics.get("grad_norm"),
                "lr": metrics.get("lr"),
                "loss": metrics.get("loss"),
                "tokens_in": data_slice.get("tokens_in"),
                "tokens_out": data_slice.get("tokens_out"),
                "wall_time": data.get("wall_time"),
                "seed": rng.get("seed"),
                "run_id": model_info.get("run_id"),
                "git_sha": model_info.get("git_sha"),
            }
        else:
            # Old format (backward compatibility)
            metric = {
                "step": data.get("step", line_num),
                "phase": data.get("phase", "train"),
                "reward_mean": data.get("reward_mean")
                or data.get("reward", {}).get("mean"),
                "reward_std": data.get("reward_std") or data.get("reward", {}).get("std"),
                "kl_mean": data.get("kl_mean") or data.get("kl_div", {}).get("mean"),
                "entropy_mean": data.get("entropy_mean")
                or data.get("entropy", {}).get("mean"),
                "clip_frac": data.get("clip_frac") or data.get("clipped_ratio"),
                "grad_norm": data.get("grad_norm") or data.get("gradient_norm"),
                "lr": data.get("lr") or data.get("learning_rate"),
                "loss": data.get("loss") or data.get("total_loss"),
                "tokens_in": data.get("tokens_in") or data.get("input_tokens"),
                "tokens_out": data.get("tokens_out") or data.get("output_tokens"),
                "wall_time": data.get("wall_time")
                or (
                    data.get("wall_time_ms", 0) / 1000.0
                    if data.get("wall_time_ms") is not None
                    else None
                ),
                "seed": data.get("seed"),
                "run_id": data.get("run_id") or data.get("experiment_id"),
                "git_sha": data.get("git_sha") or data.get("commit_hash"),
            }

        return metric

    def _parse_log_line(self, line: str, line_num: int) -> Dict[str, Any]:
        """Parse a single log line for metrics."""
        # Simple regex patterns for common TRL log formats
        patterns = {
            "step": r"step[:\s]+(\d+)",
            "reward": r"reward[:\s]+([\d.-]+)",
            "kl": r"kl[:\s]+([\d.-]+)",
            "entropy": r"entropy[:\s]+([\d.-]+)",
            "loss": r"loss[:\s]+([\d.-]+)",
            "lr": r"lr[:\s]+([\d.-]+)",
        }

        metric = {"step": line_num, "phase": "train"}

        for key, pattern in patterns.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    if key == "reward":
                        metric["reward_mean"] = value
                    elif key == "kl":
                        metric["kl_mean"] = value
                    elif key == "entropy":
                        metric["entropy_mean"] = value
                    elif key == "loss":
                        metric["loss"] = value
                    elif key == "lr":
                        metric["lr"] = value
                except ValueError:
                    continue

        # Only return if we found some meaningful metrics
        if len(metric) > 2:  # More than just step and phase
            return metric

        return None

    def _fallback_parse(self) -> List[Dict[str, Any]]:
        """Fallback parsing with more lenient requirements."""
        metrics = []

        try:
            if self.source.is_file():
                metrics = self._fallback_parse_file(self.source)
            elif self.source.is_dir():
                # Try all JSONL files with fallback parsing
                jsonl_files = list(self.source.glob("*.jsonl"))
                for jsonl_file in jsonl_files:
                    try:
                        file_metrics = self._fallback_parse_file(jsonl_file)
                        metrics.extend(file_metrics)
                    except Exception as e:
                        self.logger.warning(f"Fallback parsing failed for {jsonl_file}: {e}")
                        continue
        except Exception as e:
            self.logger.warning(f"Fallback parsing failed: {e}")

        return metrics

    def _fallback_parse_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Fallback parsing for a single file with lenient requirements."""
        metrics = []

        if file_path.suffix != ".jsonl":
            return metrics

        try:
            with open(file_path) as f:
                for line_num, line in enumerate(f, 1):  # Start line numbering from 1
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        if isinstance(data, dict):
                            # More lenient metric extraction
                            metric = self._extract_metric_lenient(data, line_num)
                            if metric:
                                metrics.append(metric)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"JSON decode error in {file_path} at line {line_num}: {e}")
                        continue
        except Exception as e:
            self.logger.warning(f"Error in fallback parsing {file_path}: {e}")

        return metrics

    def _extract_metric_lenient(self, data: Dict[str, Any], line_num: int) -> Dict[str, Any]:
        """Extract metric with lenient requirements - only need step and some metrics."""
        # Check if we have at least a step and some meaningful data
        if "step" not in data:
            return None

        # Try to extract any available metrics
        metric = {
            "step": data.get("step", line_num),
            "phase": data.get("phase", "train"),
        }

        # Add any available metrics
        metric_fields = [
            "reward_mean", "reward_std", "kl_mean", "entropy_mean",
            "clip_frac", "grad_norm", "lr", "loss", "tokens_in",
            "tokens_out", "wall_time", "seed", "run_id", "git_sha"
        ]

        for field in metric_fields:
            if field in data:
                metric[field] = data[field]

        # Only return if we have at least step and one other metric
        if len(metric) > 2:
            return metric

        return None
