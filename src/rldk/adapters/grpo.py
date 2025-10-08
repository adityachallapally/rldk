"""Adapter for GRPO training logs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from .base import BaseAdapter


class GRPOAdapter(BaseAdapter):
    """Adapter for GRPO training logs produced by run.jsonl artifacts."""

    _RUN_FILENAME = "run.jsonl"

    # Canonical metrics that must be present (in any aliased form) for a file to
    # be considered a GRPO training artifact.
    _REQUIRED_KEY_ALIASES: Dict[str, Set[str]] = {
        "reward_mean": {
            "reward_mean",
            "reward",
            "reward_avg",
            "mean_reward",
            "group_reward_mean",
            "normalized_reward_mean",
        },
        "kl_mean": {"kl", "kl_mean", "policy_kl", "kl_avg", "kl_value", "kl_divergence"},
        "entropy_mean": {"entropy", "entropy_mean", "policy_entropy"},
    }

    # Full alias mapping for canonical columns that we expose downstream. Keys
    # are the normalized column names in the RLDK schema.
    _COLUMN_ALIASES: Dict[str, Set[str]] = {
        "reward_mean": _REQUIRED_KEY_ALIASES["reward_mean"],
        "reward_std": {"reward_std", "reward_stddev", "std_reward", "group_reward_std"},
        "kl_mean": _REQUIRED_KEY_ALIASES["kl_mean"],
        "kl_coef": {"kl_coef", "kl_coeff", "kl_coefficient", "kl_beta", "beta"},
        "entropy_mean": _REQUIRED_KEY_ALIASES["entropy_mean"],
        "acceptance_rate": {
            "acceptance_rate",
            "accept_rate",
            "acceptance",
            "policy_acceptance_rate",
        },
        "advantage_mean": {
            "advantage_mean",
            "adv_mean",
            "normalized_advantage_mean",
        },
        "advantage_std": {"advantage_std", "adv_std", "advantage_stddev"},
        "grad_norm_policy": {
            "grad_norm_policy",
            "policy_grad_norm",
            "pi_grad_norm",
            "actor_grad_norm",
        },
        "grad_norm_value": {
            "grad_norm_value",
            "value_grad_norm",
            "critic_grad_norm",
        },
        "phase": {"phase", "training_phase"},
        "seed": {"seed"},
        "run_id": {"run_id", "run"},
    }

    _SENSITIVE_KEYS: Set[str] = {
        "prompt",
        "prompts",
        "response",
        "responses",
        "completion",
        "completions",
        "chosen",
        "rejected",
        "question",
        "questions",
        "answer",
        "answers",
        "input",
        "inputs",
        "output",
        "outputs",
        "reference",
        "references",
        "gold",
        "gold_label",
        "gold_labels",
        "label",
        "labels",
        "target",
        "targets",
        "ground_truth",
        "ground_truths",
        "sample",
        "samples",
        "raw_text",
        "metadata_text",
    }

    _SENSITIVE_SUFFIXES: Tuple[str, ...] = (
        "_prompt",
        "_prompts",
        "_response",
        "_responses",
        "_completion",
        "_completions",
        "_text",
        "_answer",
        "_question",
    )

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

        for column in ["reward_mean", "reward_std", "kl_mean", "entropy_mean", "acceptance_rate"]:
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
                    if self._has_required_keys(data):
                        return True
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

                    step_value = data.get("step")
                    if step_value is None:
                        step_value = line_number - 1

                    record: Dict[str, Any] = {"step": step_value}

                    consumed_keys = {"step"}

                    # Map all known aliases into their canonical schema.
                    for canonical, aliases in self._COLUMN_ALIASES.items():
                        alias, value = self._resolve_alias(data, aliases)
                        if alias is None:
                            continue
                        consumed_keys.add(alias)
                        if canonical == "phase" and value is None:
                            continue
                        record[canonical] = value

                    if "phase" not in record:
                        record["phase"] = "train"

                    if "seed" not in record and seed_hint is not None:
                        record["seed"] = seed_hint

                    if "run_id" not in record and run_id_hint is not None:
                        record["run_id"] = run_id_hint

                    for key, value in data.items():
                        if key in consumed_keys:
                            continue
                        if key in {"kl", "entropy"}:
                            # These have been normalized into kl_mean/entropy_mean.
                            continue
                        if self._is_sensitive_key(key):
                            continue
                        if isinstance(value, (dict, list, tuple, set)):
                            continue
                        record[key] = value

                    metrics.append(record)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Failed to parse GRPO log {file_path}: {exc}") from exc

        return metrics

    def _has_required_keys(self, data: Dict[str, Any]) -> bool:
        for aliases in self._REQUIRED_KEY_ALIASES.values():
            if not any(key in data for key in aliases):
                return False
        return True

    def _resolve_alias(
        self, data: Dict[str, Any], aliases: Set[str]
    ) -> Tuple[Optional[str], Optional[Any]]:
        for alias in aliases:
            if alias in data:
                return alias, data.get(alias)
        return None, None

    def _is_sensitive_key(self, key: str) -> bool:
        key_lower = key.lower()
        if key_lower in self._SENSITIVE_KEYS:
            return True
        return any(key_lower.endswith(suffix) for suffix in self._SENSITIVE_SUFFIXES)

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
