"""Custom adapter for our JSONL training logs."""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from .base import BaseAdapter


class CustomJSONLAdapter(BaseAdapter):
    """Adapter for our custom JSONL training logs."""

    def can_handle(self) -> bool:
        """Check if source contains our custom JSONL logs."""
        if not self.source.exists():
            return False

        if self.source.is_file():
            return self._is_custom_jsonl_file(self.source)
        elif self.source.is_dir():
            # Check for our custom JSONL log files
            jsonl_files = list(self.source.glob("*.jsonl"))
            return len(jsonl_files) > 0

        return False

    def _is_custom_jsonl_file(self, file_path: Path) -> bool:
        """Check if a file contains our custom JSONL logs."""
        try:
            if file_path.suffix == ".jsonl":
                with open(file_path, "r") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        # Check for our custom schema - be more flexible
                        # Only check for required keys if data is a dict
                        if isinstance(data, dict):
                            # Check for custom format indicators (either old or new format)
                            custom_indicators = [
                                "global_step", "reward_scalar", "kl_to_ref",  # Old format
                                "step", "reward_mean", "kl_mean", "loss"      # Standard format
                            ]
                            # If it has any of these indicators, it's likely our format
                            return any(key in data for key in custom_indicators)
                        else:
                            # For non-dict data, it's not our custom format
                            return False
        except (OSError, IOError, json.JSONDecodeError, UnicodeDecodeError, TypeError) as e:
            # Log the specific error for debugging but don't fail the check
            print(f"Warning: Error checking if file {file_path} is custom JSONL format: {e}")
            return False
        return False

    def load(self) -> pd.DataFrame:
        """Load our custom JSONL logs and convert to standard format."""
        if not self.can_handle():
            raise ValueError(f"Cannot handle source: {self.source}")

        metrics = []

        if self.source.is_file():
            metrics = self._parse_file(self.source)
        elif self.source.is_dir():
            # Find and parse all JSONL log files
            jsonl_files = list(self.source.glob("*.jsonl"))
            for jsonl_file in jsonl_files:
                metrics.extend(self._parse_file(jsonl_file))

        if not metrics:
            raise ValueError(f"No valid custom JSONL metrics found in {self.source}")

        # Convert to DataFrame
        df = pd.DataFrame(metrics)

        # Ensure required columns exist for the standard format
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
        """Parse a single custom JSONL log file."""
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
                            metric = self._extract_custom_metric(data, line_num)
                            if metric:
                                metrics.append(metric)
                        except json.JSONDecodeError as e:
                            print(f"Warning: JSON decode error in {file_path} at line {line_num + 1}: {e}")
                            continue
        except (OSError, IOError, UnicodeDecodeError) as e:
            print(f"Error parsing {file_path}: {e}")
            # Re-raise the exception with context
            raise RuntimeError(f"Failed to parse custom JSONL file {file_path}: {e}") from e

        return metrics

    def _extract_custom_metric(
        self, data: Dict[str, Any], line_num: int
    ) -> Dict[str, Any]:
        """Extract metric from our custom JSONL format."""
        try:
            # Map our custom schema to the expected format
            # Handle various possible field names for better compatibility
            step = data.get("global_step") or data.get("step", line_num)
            reward_scalar = data.get("reward_scalar") or data.get("reward_mean", 0.0)
            kl_value = data.get("kl_to_ref") or data.get("kl_mean", 0.0)
            loss_value = data.get("loss", 0.0)
            
            # Extract RNG seed from various possible locations
            seed = (data.get("rng.python") or 
                   data.get("seed") or 
                   data.get("rng", {}).get("python") if isinstance(data.get("rng"), dict) else None or 
                   42)
            
            # Extract additional metrics if available
            entropy_value = data.get("entropy") or data.get("entropy_mean", 0.0)
            clip_frac_value = data.get("clip_frac", 0.0)
            grad_norm_value = data.get("grad_norm", 0.0)
            lr_value = data.get("lr") or data.get("learning_rate", 0.0)
            reward_std_value = data.get("reward_std", 0.0)
            
            # Extract data slice information if available
            tokens_in = data.get("tokens_in", 0)
            tokens_out = data.get("tokens_out", 0)
            
            # Extract wall time if available
            wall_time = data.get("wall_time") or data.get("timestamp", 0.0)
            
            metric = {
                "step": int(step),
                "phase": data.get("phase", "train"),
                "reward_mean": float(reward_scalar),
                "reward_std": float(reward_std_value),
                "kl_mean": float(kl_value),
                "entropy_mean": float(entropy_value),
                "clip_frac": float(clip_frac_value),
                "grad_norm": float(grad_norm_value),
                "lr": float(lr_value),
                "loss": float(loss_value),
                "tokens_in": int(tokens_in),
                "tokens_out": int(tokens_out),
                "wall_time": float(wall_time),
                "seed": int(seed),
                "run_id": data.get("run_id", f"custom_{line_num}"),
                "git_sha": data.get("git_sha", "unknown"),
            }

            return metric
        except Exception as e:
            print(f"Error extracting metric from line {line_num}: {e}")
            return None
