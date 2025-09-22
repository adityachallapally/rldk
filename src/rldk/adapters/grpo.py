"""Adapter for GRPO training logs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .base import BaseAdapter


class GRPOAdapter(BaseAdapter):
    """Adapter for GRPO training logs produced by run.jsonl artifacts."""

    _RUN_FILENAME = "run.jsonl"
    _REQUIRED_KEYS = {"kl", "entropy", "advantage_mean", "grad_norm_policy", "reward_mean"}

    def can_handle(self) -> bool:
        """Return ``True`` when the source looks like a GRPO log directory."""
        if not self.source.exists():
            return False

        if self.source.is_file():
            return self._is_grpo_file(self.source)

        if self.source.is_dir():
            run_files = list(self._discover_run_files(self.source))
            return any(self._is_grpo_file(path) for path in run_files)

        return False

    def load(self) -> pd.DataFrame:
        """Load GRPO training metrics and map them to the canonical schema."""
        if not self.can_handle():
            raise ValueError(f"Cannot handle source: {self.source}")

        run_files = list(self._discover_run_files(self.source))
        if not run_files and self.source.is_file():
            run_files = [self.source]

        metrics: List[Dict[str, Any]] = []
        for run_file in run_files:
            metrics.extend(self._parse_run_file(run_file))

        if not metrics:
            raise ValueError(f"No GRPO metrics found in {self.source}")

        df = pd.DataFrame(metrics)

        if "phase" not in df.columns:
            df["phase"] = "train"
        else:
            df["phase"] = df["phase"].fillna("train")

        for column in ["reward_mean", "reward_std", "kl_mean", "entropy_mean"]:
            if column not in df.columns:
                df[column] = None

        if "step" not in df.columns:
            raise ValueError("GRPO logs must include a step field")

        return df

    def _discover_run_files(self, root: Path) -> Iterable[Path]:
        if root.is_file():
            if root.name == self._RUN_FILENAME:
                yield root
            return

        for candidate in sorted(root.rglob(self._RUN_FILENAME)):
            yield candidate

    def _is_grpo_file(self, file_path: Path) -> bool:
        if file_path.suffix != ".jsonl":
            return False

        try:
            with open(file_path) as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        continue
                    return self._REQUIRED_KEYS.issubset(data)
        except (OSError, json.JSONDecodeError):
            return False
        return False

    def _parse_run_file(self, file_path: Path) -> List[Dict[str, Any]]:
        metrics: List[Dict[str, Any]] = []
        seed_hint = self._extract_seed(file_path)
        run_id_hint = self._extract_run_id(file_path)

        try:
            with open(file_path) as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)
                    if not isinstance(data, dict):
                        continue

                    record: Dict[str, Any] = {
                        "step": data.get("step", line_number - 1),
                        "phase": data.get("phase") or "train",
                        "reward_mean": data.get("reward_mean"),
                        "reward_std": data.get("reward_std"),
                        "kl_mean": data.get("kl"),
                        "entropy_mean": data.get("entropy"),
                    }

                    seed_value = data.get("seed", seed_hint)
                    if seed_value is not None:
                        record["seed"] = seed_value

                    run_id_value = data.get("run_id", run_id_hint)
                    if run_id_value is not None:
                        record["run_id"] = run_id_value

                    for key, value in data.items():
                        if key in {"step", "phase", "reward_mean", "reward_std", "kl", "entropy", "seed", "run_id"}:
                            continue
                        record[key] = value

                    metrics.append(record)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Failed to parse GRPO log {file_path}: {exc}") from exc

        return metrics

    def _extract_seed(self, file_path: Path) -> Any:
        match = re.search(r"seed_(\d+)", str(file_path))
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return match.group(1)
        return None

    def _extract_run_id(self, file_path: Path) -> str | None:
        parent = file_path.parent
        if parent.name and parent.name != ".":
            return parent.name
        return None
