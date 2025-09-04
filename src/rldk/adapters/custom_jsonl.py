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
                        # Check for our custom schema - be very specific to avoid misclassifying TRL/OpenRLHF
                        # Only check for required keys if data is a dict
                        if isinstance(data, dict):
                            # Check for explicit custom format indicators (not standard format)
                            # These are the unique identifiers of our custom format
                            custom_indicators = [
                                "global_step", "reward_scalar", "kl_to_ref"  # Unique to custom format
                            ]
                            
                            # Must have at least one of the unique custom indicators
                            has_custom_indicators = any(key in data for key in custom_indicators)
                            
                            # Also check that it doesn't look like standard TRL/OpenRLHF format
                            # Standard formats typically have nested metrics or different structure
                            is_standard_format = (
                                "reward" in data and isinstance(data["reward"], dict) or  # Nested metrics
                                "metrics" in data and isinstance(data["metrics"], dict) or  # Nested metrics
                                ("step" in data and "reward_mean" in data and "kl_mean" in data and 
                                 "entropy_mean" in data and "clip_frac" in data and "grad_norm" in data)  # Full standard schema
                            )
                            
                            # Only classify as custom if it has custom indicators AND doesn't look like standard format
                            return has_custom_indicators and not is_standard_format
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
            # Use 'in' checks to avoid skipping valid zeros
            step = data.get("global_step") if "global_step" in data else data.get("step", line_num)
            reward_scalar = data.get("reward_scalar") if "reward_scalar" in data else data.get("reward_mean", 0.0)
            kl_value = data.get("kl_to_ref") if "kl_to_ref" in data else data.get("kl_mean", 0.0)
            loss_value = data.get("loss", 0.0)
            
            # Extract RNG seed from various possible locations
            # Fix operator precedence by using proper conditional logic
            seed = 42  # Default fallback
            if "rng.python" in data:
                seed = data["rng.python"]
            elif "seed" in data:
                seed = data["seed"]
            elif "rng" in data and isinstance(data["rng"], dict) and "python" in data["rng"]:
                seed = data["rng"]["python"]
            
            # Extract additional metrics if available
            # Use 'in' checks to avoid skipping valid zeros
            entropy_value = data.get("entropy") if "entropy" in data else data.get("entropy_mean", 0.0)
            clip_frac_value = data.get("clip_frac", 0.0)
            grad_norm_value = data.get("grad_norm", 0.0)
            lr_value = data.get("lr") if "lr" in data else data.get("learning_rate", 0.0)
            reward_std_value = data.get("reward_std", 0.0)
            
            # Extract data slice information if available
            tokens_in = data.get("tokens_in", 0)
            tokens_out = data.get("tokens_out", 0)
            
            # Extract wall time if available
            # Use 'in' check to avoid skipping valid zeros
            wall_time = data.get("wall_time") if "wall_time" in data else data.get("timestamp", 0.0)
            
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
