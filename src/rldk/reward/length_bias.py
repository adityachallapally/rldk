"""Length bias analysis utilities for reward model auditing.

This module provides a reusable :class:`LengthBiasDetector` that inspects
response/reward pairs for length-driven optimization patterns.  It produces a
structured :class:`LengthBiasMetrics` report that callers can serialize to JSON
or feed into higher level health checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:  # pragma: no cover - exercised indirectly in tests when SciPy is present
    from scipy.stats import pearsonr, spearmanr
except Exception:  # pragma: no cover - SciPy is optional at runtime
    pearsonr = None  # type: ignore[assignment]
    spearmanr = None  # type: ignore[assignment]


@dataclass
class LengthBiasMetrics:
    """Summary statistics describing length correlation with reward signals."""

    response_count: int = 0
    valid_sample_count: int = 0
    mean_length: Optional[float] = None
    mean_reward: Optional[float] = None
    pearson_correlation: Optional[float] = None
    pearson_pvalue: Optional[float] = None
    spearman_correlation: Optional[float] = None
    spearman_pvalue: Optional[float] = None
    variance_explained: Optional[float] = None
    quartile_metrics: Dict[str, Dict[str, Optional[float]]] = field(
        default_factory=dict
    )
    odin_reward_per_token: Optional[float] = None
    odin_efficiency: Optional[float] = None
    odin_optimization_flag: bool = False
    optimization_patterns: List[str] = field(default_factory=list)
    bias_severity: Optional[float] = None
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of the metric values."""

        return asdict(self)


class LengthBiasDetector:
    """Detect systematic reward differences that correlate with response length."""

    def __init__(self, tokenizer: Optional[Any] = None) -> None:
        self._tokenizer = tokenizer

    def analyze_length_bias(
        self,
        responses: Sequence[Any],
        rewards: Sequence[float],
        lengths: Optional[Sequence[Optional[float]]] = None,
    ) -> LengthBiasMetrics:
        """Compute correlation, quartile, and ODIN metrics for length bias."""

        responses_list = list(responses)
        rewards_array = np.asarray(list(rewards), dtype=float)

        if len(responses_list) != len(rewards_array):
            raise ValueError("Responses and rewards must have equal length")

        raw_lengths = self._extract_lengths(responses_list, lengths)
        lengths_array = np.asarray(raw_lengths, dtype=float)

        valid_mask = (~np.isnan(rewards_array)) & (~np.isnan(lengths_array))

        metrics = LengthBiasMetrics(response_count=len(responses_list))
        metrics.valid_sample_count = int(np.sum(valid_mask))

        if metrics.valid_sample_count == 0:
            metrics.recommendations = [
                "Unable to compute length bias metrics due to empty inputs.",
            ]
            return metrics

        valid_lengths = lengths_array[valid_mask]
        valid_rewards = rewards_array[valid_mask]

        metrics.mean_length = float(np.mean(valid_lengths))
        metrics.mean_reward = float(np.mean(valid_rewards))

        correlation_metrics = self._calculate_correlation_metrics(
            valid_lengths, valid_rewards
        )
        metrics.pearson_correlation = correlation_metrics.get("pearson")
        metrics.pearson_pvalue = correlation_metrics.get("pearson_pvalue")
        metrics.spearman_correlation = correlation_metrics.get("spearman")
        metrics.spearman_pvalue = correlation_metrics.get("spearman_pvalue")

        quartile_report = self._analyze_length_quartiles(valid_lengths, valid_rewards)
        metrics.quartile_metrics = quartile_report["quartiles"]
        metrics.variance_explained = quartile_report["variance_explained"]

        odin_metrics = self.calculate_odin_metrics(valid_lengths, valid_rewards)
        metrics.odin_reward_per_token = odin_metrics.get("reward_per_token")
        metrics.odin_efficiency = odin_metrics.get("efficiency")
        metrics.odin_optimization_flag = bool(
            odin_metrics.get("optimization_flag", False)
        )

        metrics.optimization_patterns = self._detect_optimization_patterns(
            valid_lengths, valid_rewards, quartile_report["quartiles"], correlation_metrics
        )

        metrics.bias_severity = self._calculate_bias_severity(
            correlation_metrics, quartile_report["variance_explained"]
        )

        metrics.recommendations = self._generate_recommendations(metrics)

        return metrics

    def detect_reward_hacking(
        self,
        responses: Sequence[Any],
        rewards: Sequence[float],
        lengths: Optional[Sequence[Optional[float]]] = None,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable reward hacking report."""

        metrics = self.analyze_length_bias(responses, rewards, lengths)
        report = metrics.to_dict()
        report["recommendations"] = metrics.recommendations
        return report

    def calculate_odin_metrics(
        self, lengths: np.ndarray, rewards: np.ndarray
    ) -> Dict[str, Optional[float]]:
        """Compute ODIN-inspired efficiency metrics."""

        if lengths.size == 0:
            return {
                "reward_per_token": None,
                "efficiency": None,
                "optimization_flag": False,
            }

        total_tokens = float(np.sum(lengths))
        total_reward = float(np.sum(rewards))
        mean_length = float(np.mean(lengths)) if lengths.size else None
        mean_reward = float(np.mean(rewards)) if rewards.size else None

        reward_per_token = (
            total_reward / total_tokens if total_tokens > 0 else None
        )
        efficiency = (
            (mean_reward / mean_length)
            if mean_length is not None and mean_length > 0
            else None
        )

        if lengths.size == 0:
            optimization_flag = False
        else:
            median_length = float(np.median(lengths))
            long_mask = lengths >= median_length
            short_mask = lengths < median_length
            long_mean = float(np.mean(rewards[long_mask])) if np.any(long_mask) else None
            short_mean = (
                float(np.mean(rewards[short_mask])) if np.any(short_mask) else None
            )
            optimization_flag = False
            if long_mean is not None and short_mean is not None:
                diff = long_mean - short_mean
                baseline = np.mean(np.abs(rewards)) + 1e-8
                optimization_flag = bool(diff > 0.1 * baseline)

        return {
            "reward_per_token": reward_per_token,
            "efficiency": efficiency,
            "optimization_flag": optimization_flag,
        }

    def _extract_lengths(
        self,
        responses: Sequence[Any],
        lengths: Optional[Sequence[Optional[float]]],
    ) -> List[float]:
        extracted: List[float] = []

        if lengths is not None:
            for value in lengths:
                if value is None:
                    extracted.append(np.nan)
                else:
                    try:
                        extracted.append(float(value))
                    except (TypeError, ValueError):
                        extracted.append(np.nan)
            if len(extracted) != len(responses):
                raise ValueError("Length list must match number of responses")
            return extracted

        for response in responses:
            extracted.append(self._measure_length(response))

        return extracted

    def _measure_length(self, response: Any) -> float:
        if self._tokenizer is not None:
            try:
                if hasattr(self._tokenizer, "encode"):
                    tokens = self._tokenizer.encode(str(response))
                else:
                    tokens = self._tokenizer(str(response))
                    if isinstance(tokens, dict) and "input_ids" in tokens:
                        tokens = tokens["input_ids"]
                    elif hasattr(tokens, "input_ids"):
                        tokens = tokens.input_ids
                if isinstance(tokens, (list, tuple, np.ndarray)):
                    return float(len(tokens))
            except Exception:
                pass
        return float(len(str(response)))

    def _calculate_correlation_metrics(
        self, lengths: np.ndarray, rewards: np.ndarray
    ) -> Dict[str, Optional[float]]:
        result: Dict[str, Optional[float]] = {
            "pearson": None,
            "pearson_pvalue": None,
            "spearman": None,
            "spearman_pvalue": None,
        }

        if lengths.size < 2:
            return result

        if float(np.std(lengths)) == 0.0 or float(np.std(rewards)) == 0.0:
            return result

        if pearsonr is not None:
            try:
                corr, pvalue = pearsonr(lengths, rewards)
                result["pearson"] = float(corr)
                result["pearson_pvalue"] = float(pvalue)
            except Exception:
                pass
        else:
            corr = self._safe_corrcoef(lengths, rewards)
            if corr is not None:
                result["pearson"] = corr

        if spearmanr is not None:
            try:
                corr, pvalue = spearmanr(lengths, rewards)
                result["spearman"] = float(corr)
                result["spearman_pvalue"] = float(pvalue)
            except Exception:
                pass
        else:
            length_ranks = self._rankdata(lengths)
            reward_ranks = self._rankdata(rewards)
            corr = self._safe_corrcoef(length_ranks, reward_ranks)
            if corr is not None:
                result["spearman"] = corr

        return result

    def _analyze_length_quartiles(
        self, lengths: np.ndarray, rewards: np.ndarray
    ) -> Dict[str, Any]:
        quartile_report: Dict[str, Any] = {
            "quartiles": {},
            "variance_explained": None,
        }

        if lengths.size == 0:
            return quartile_report

        edges = np.quantile(lengths, [0.0, 0.25, 0.5, 0.75, 1.0])
        quartiles: Dict[str, Dict[str, Optional[float]]] = {}
        overall_mean = float(np.mean(rewards))
        total_var = float(np.var(rewards))
        between_var = 0.0

        for idx, label in enumerate(["q1", "q2", "q3", "q4"]):
            low = float(edges[idx])
            high = float(edges[idx + 1])
            if idx < 3:
                mask = (lengths >= low) & (lengths < high)
            else:
                mask = (lengths >= low) & (lengths <= high)

            subset_rewards = rewards[mask]
            subset_lengths = lengths[mask]
            if subset_rewards.size == 0:
                quartiles[label] = {
                    "length_min": low,
                    "length_max": high,
                    "mean_reward": None,
                    "mean_length": None,
                    "count": 0,
                }
                continue

            reward_mean = float(np.mean(subset_rewards))
            length_mean = float(np.mean(subset_lengths))
            quartiles[label] = {
                "length_min": low,
                "length_max": high,
                "mean_reward": reward_mean,
                "mean_length": length_mean,
                "count": int(subset_rewards.size),
            }

            if total_var > 0:
                between_var += subset_rewards.size * (reward_mean - overall_mean) ** 2

        variance_explained = None
        if total_var > 0 and lengths.size > 0:
            variance_explained = float(between_var / (total_var * lengths.size))

        quartile_report["quartiles"] = quartiles
        quartile_report["variance_explained"] = variance_explained
        return quartile_report

    def _detect_optimization_patterns(
        self,
        lengths: np.ndarray,
        rewards: np.ndarray,
        quartiles: Dict[str, Dict[str, Optional[float]]],
        correlations: Dict[str, Optional[float]],
    ) -> List[str]:
        patterns: List[str] = []

        pearson_corr = correlations.get("pearson") or 0.0
        spearman_corr = correlations.get("spearman") or 0.0

        dominant_corr = pearson_corr if abs(pearson_corr) >= abs(spearman_corr) else spearman_corr

        if dominant_corr > 0.2:
            patterns.append("Longer responses are rewarded")
        elif dominant_corr < -0.2:
            patterns.append("Shorter responses are rewarded")

        q1 = quartiles.get("q1", {})
        q4 = quartiles.get("q4", {})
        q1_mean = q1.get("mean_reward")
        q4_mean = q4.get("mean_reward")

        if q1_mean is not None and q4_mean is not None:
            delta = q4_mean - q1_mean
            baseline = np.mean(np.abs(rewards)) + 1e-8
            if delta > 0.15 * baseline:
                patterns.append("Top length quartile outperforms short responses")
            elif delta < -0.15 * baseline:
                patterns.append("Short responses outperform long responses")

        reward_diff = np.max(rewards) - np.min(rewards)
        if reward_diff > 0 and len(set(lengths.tolist())) <= 3:
            patterns.append("Reward concentrated on narrow length range")

        return patterns

    def _calculate_bias_severity(
        self,
        correlations: Dict[str, Optional[float]],
        variance_explained: Optional[float],
    ) -> Optional[float]:
        pearson_corr = abs(correlations.get("pearson") or 0.0)
        spearman_corr = abs(correlations.get("spearman") or 0.0)
        dominant_corr = max(pearson_corr, spearman_corr)
        variance_component = variance_explained or 0.0
        severity = 0.6 * dominant_corr + 0.4 * variance_component
        return float(min(max(severity, 0.0), 1.0))

    def _generate_recommendations(self, metrics: LengthBiasMetrics) -> List[str]:
        recommendations: List[str] = []

        if metrics.valid_sample_count == 0:
            return ["Collect reward data before running length bias analysis."]

        severity = metrics.bias_severity or 0.0
        if severity < 0.2:
            recommendations.append("No significant length bias detected.")
        elif severity < 0.5:
            recommendations.append(
                "Monitor response length during training; mild bias observed."
            )
        else:
            recommendations.append(
                "Consider penalizing overly long responses or diversifying training prompts."
            )

        if metrics.optimization_patterns:
            recommendations.append(
                "Investigate prompts where the listed patterns appear strongest."
            )

        if metrics.odin_optimization_flag:
            recommendations.append(
                "ODIN heuristics suggest reward hacking; audit tokenizer configuration."
            )

        return recommendations

    @staticmethod
    def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> Optional[float]:
        if x.size < 2 or y.size < 2:
            return None
        x_std = float(np.std(x))
        y_std = float(np.std(y))
        if x_std == 0.0 or y_std == 0.0:
            return None
        corr = float(np.corrcoef(x, y)[0, 1])
        if np.isnan(corr):
            return None
        return corr

    @staticmethod
    def _rankdata(values: np.ndarray) -> np.ndarray:
        sorter = np.argsort(values)
        ranks = np.empty_like(sorter, dtype=float)
        ranks[sorter] = np.arange(1, len(values) + 1, dtype=float)
        return ranks

