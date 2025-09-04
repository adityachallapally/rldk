"""Adapter for TRL (Transformer Reinforcement Learning) logs."""

import json
import re
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from .base import BaseAdapter


class TRLAdapter(BaseAdapter):
    """Adapter for TRL training logs."""

    def can_handle(self) -> bool:
        """Check if source contains TRL logs."""
        if not self.source.exists():
            return False

        # Look for TRL-specific files or patterns
        if self.source.is_file():
            return self._is_trl_file(self.source)
        elif self.source.is_dir():
            # Check for common TRL log files
            trl_files = list(self.source.glob("*.log")) + list(
                self.source.glob("trainer_log.jsonl")
            )
            return len(trl_files) > 0

        return False

    def _is_trl_file(self, file_path: Path) -> bool:
        """Check if a file contains TRL logs."""
        try:
            if file_path.suffix == ".jsonl":
                with open(file_path, "r") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        # Check for TRL-specific keywords or our test fixture format
                        # First check if data is a dict and has the required keys
                        if isinstance(data, dict):
                            # Check for new Event schema format
                            if "metrics" in data and "model_info" in data:
                                return True
                            # Check for old format
                            return (
                                "trl" in str(data).lower()
                                or "trainer" in str(data).lower()
                                or all(
                                    key in data
                                    for key in ["step", "phase", "reward_mean", "kl_mean"]
                                )
                            )
                        else:
                            # For non-dict data, only check string content
                            return (
                                "trl" in str(data).lower()
                                or "trainer" in str(data).lower()
                            )
            elif file_path.suffix == ".log":
                with open(file_path, "r") as f:
                    content = f.read()
                    return "trl" in content.lower() or "trainer" in content.lower()
        except (OSError, IOError, UnicodeDecodeError, TypeError) as e:
            # Log the specific error for debugging but don't fail the check
            print(f"Warning: Error checking if file {file_path} is TRL format: {e}")
            return False
        except json.JSONDecodeError as e:
            # Log JSON decode error but don't fail the check
            print(f"Warning: JSON decode error in {file_path}: {e}")
            return False
        return False

    def load(self) -> pd.DataFrame:
        """Load TRL logs and convert to standard format."""
        if not self.can_handle():
            raise ValueError(f"Cannot handle source: {self.source}")

        metrics = []

        if self.source.is_file():
            metrics = self._parse_file(self.source)
        elif self.source.is_dir():
            # Find and parse all TRL log files
            log_files = list(self.source.glob("*.log")) + list(
                self.source.glob("*.jsonl")
            )
            for log_file in log_files:
                metrics.extend(self._parse_file(log_file))

        if not metrics:
            raise ValueError(f"No valid TRL metrics found in {self.source}")

        # Convert to DataFrame
        df = pd.DataFrame(metrics)

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

        return df[required_cols]

    def _parse_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse a single TRL log file."""
        metrics = []

        try:
            if file_path.suffix == ".jsonl":
                with open(file_path, "r") as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            metric = self._extract_trl_metric(data, line_num)
                            if metric:
                                metrics.append(metric)
                        except json.JSONDecodeError as e:
                            print(f"Warning: JSON decode error in {file_path} at line {line_num + 1}: {e}")
                            continue

            elif file_path.suffix == ".log":
                with open(file_path, "r") as f:
                    for line_num, line in enumerate(f):
                        metric = self._parse_log_line(line, line_num)
                        if metric:
                            metrics.append(metric)

        except (OSError, IOError, UnicodeDecodeError) as e:
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
