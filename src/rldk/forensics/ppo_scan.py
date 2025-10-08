"""PPO forensics analysis."""

from typing import Any, Dict, Iterator, List

import numpy as np


def scan_ppo_events(events: Iterator[Dict[str, Any]]) -> Dict[str, Any]:
    """Scan PPO training events for anomalies."""

    # Convert iterator to list for analysis
    events_list = list(events)

    if not events_list:
        return {"version": "1", "rules_fired": [], "earliest_step": None, "stats": {}}

    # Extract required fields
    steps = []
    kl_values = []
    kl_coefs = []
    entropies = []
    advantage_means = []
    advantage_stds = []
    grad_norm_policies = []
    grad_norm_values = []

    for event in events_list:
        steps.append(event.get("step", event.get("global_step", 0)))
        kl_values.append(event.get("kl", event.get("kl_div", 0.0)))
        kl_coefs.append(event.get("kl_coef", event.get("kl_coefficient", 1.0)))
        entropies.append(event.get("entropy", 0.0))
        advantage_means.append(event.get("advantage_mean", event.get("adv_mean", 0.0)))
        advantage_stds.append(event.get("advantage_std", event.get("adv_std", 1.0)))
        grad_norm_policies.append(
            event.get("grad_norm_policy", event.get("policy_grad_norm", 0.0))
        )
        grad_norm_values.append(
            event.get("grad_norm_value", event.get("value_grad_norm", 0.0))
        )

    # Run anomaly detection rules
    rules_fired = []

    # Rule 1: KL spike detection
    kl_spike_rules = detect_kl_spikes(steps, kl_values)
    rules_fired.extend(kl_spike_rules)

    # Rule 2: KL controller stuck
    controller_rules = detect_kl_controller_stuck(steps, kl_values, kl_coefs)
    rules_fired.extend(controller_rules)

    # Rule 3: Gradient ratio anomalies
    grad_rules = detect_gradient_ratio_anomalies(
        steps, grad_norm_policies, grad_norm_values
    )
    rules_fired.extend(grad_rules)

    # Rule 4: Advantage sanity checks
    advantage_rules = detect_advantage_anomalies(
        steps, advantage_means, advantage_stds, entropies, kl_values
    )
    rules_fired.extend(advantage_rules)

    # Compute statistics
    stats = compute_stats(kl_values, grad_norm_policies, grad_norm_values, entropies)

    # Find earliest step
    earliest_step = min(steps) if steps else None

    return {
        "version": "1",
        "rules_fired": rules_fired,
        "earliest_step": earliest_step,
        "stats": stats,
    }


def detect_kl_spikes(steps: List[int], kl_values: List[float]) -> List[Dict[str, Any]]:
    """Detect KL spikes (4x running median for 5+ consecutive updates)."""
    rules = []

    if len(kl_values) < 10:
        return rules

    # Compute running median
    window_size = 10
    running_medians = []

    for i in range(len(kl_values)):
        start_idx = max(0, i - window_size + 1)
        window = kl_values[start_idx : i + 1]
        running_medians.append(np.median(window))

    # Detect spikes
    spike_threshold = 4.0
    consecutive_threshold = 3  # Reduced from 5 to catch more spikes

    consecutive_spikes = 0
    spike_start = None

    for i, (step, kl, median) in enumerate(zip(steps, kl_values, running_medians)):
        if median > 0 and kl > spike_threshold * median:
            if consecutive_spikes == 0:
                spike_start = step
            consecutive_spikes += 1
        else:
            if consecutive_spikes >= consecutive_threshold:
                rules.append(
                    {
                        "rule": "kl_spike",
                        "description": f"KL spike detected: {consecutive_spikes} consecutive updates with KL > {spike_threshold}x median",
                        "step_range": [spike_start, step],
                    }
                )
            consecutive_spikes = 0
            spike_start = None

    # Check for spike at the end
    if consecutive_spikes >= consecutive_threshold:
        rules.append(
            {
                "rule": "kl_spike",
                "description": f"KL spike detected: {consecutive_spikes} consecutive updates with KL > {spike_threshold}x median",
                "step_range": [spike_start, steps[-1]],
            }
        )

    return rules


def detect_kl_controller_stuck(
    steps: List[int], kl_values: List[float], kl_coefs: List[float]
) -> List[Dict[str, Any]]:
    """Detect KL controller stuck (KL outside target range while coef changes < 5% for 10+ updates)."""
    rules = []

    if len(kl_values) < 10:
        return rules

    # Default KL target range
    target_low, target_high = 0.01, 0.15

    # Detect when KL is outside target range
    kl_outside_target = []
    for i, (step, kl) in enumerate(zip(steps, kl_values)):
        if kl < target_low or kl > target_high:
            kl_outside_target.append(i)

    if len(kl_outside_target) < 10:
        return rules

    # Check for controller stuck periods
    consecutive_threshold = 10
    coef_change_threshold = 0.05  # 5%

    consecutive_stuck = 0
    stuck_start = None

    for i in range(len(kl_outside_target) - 1):
        current_idx = kl_outside_target[i]
        next_idx = kl_outside_target[i + 1]

        # Check if consecutive and coef change is small
        if next_idx == current_idx + 1:
            coef_change = abs(kl_coefs[next_idx] - kl_coefs[current_idx]) / max(
                kl_coefs[current_idx], 1e-8
            )

            if coef_change < coef_change_threshold:
                if consecutive_stuck == 0:
                    stuck_start = steps[current_idx]
                consecutive_stuck += 1
            else:
                if consecutive_stuck >= consecutive_threshold:
                    rules.append(
                        {
                            "rule": "kl_controller_stuck",
                            "description": f"KL controller stuck: {consecutive_stuck} consecutive updates with KL outside [{target_low}, {target_high}] and coef change < {coef_change_threshold*100}%",
                            "step_range": [stuck_start, steps[current_idx]],
                        }
                    )
                consecutive_stuck = 0
                stuck_start = None
        else:
            # Not consecutive, reset
            if consecutive_stuck >= consecutive_threshold:
                rules.append(
                    {
                        "rule": "kl_controller_stuck",
                        "description": f"KL controller stuck: {consecutive_stuck} consecutive updates with KL outside [{target_low}, {target_high}] and coef change < {coef_change_threshold*100}%",
                        "step_range": [stuck_start, steps[current_idx]],
                    }
                )
            consecutive_stuck = 0
            stuck_start = None

    # Check for stuck period at the end
    if consecutive_stuck >= consecutive_threshold:
        rules.append(
            {
                "rule": "kl_controller_stuck",
                "description": f"KL controller stuck: {consecutive_stuck} consecutive updates with KL outside [{target_low}, {target_high}] and coef change < {coef_change_threshold*100}%",
                "step_range": [stuck_start, steps[kl_outside_target[-1]]],
            }
        )

    return rules


def detect_gradient_ratio_anomalies(
    steps: List[int], grad_norm_policies: List[float], grad_norm_values: List[float]
) -> List[Dict[str, Any]]:
    """Detect gradient ratio anomalies (policy/value ratio < 0.1 or > 10 for 5+ updates)."""
    rules = []

    if len(grad_norm_policies) < 5:
        return rules

    consecutive_threshold = 5
    ratio_low_threshold = 0.1
    ratio_high_threshold = 10.0

    consecutive_low = 0
    consecutive_high = 0
    low_start = None
    high_start = None

    for i, (step, policy_norm, value_norm) in enumerate(
        zip(steps, grad_norm_policies, grad_norm_values)
    ):
        if policy_norm > 0 and value_norm > 0:
            ratio = policy_norm / value_norm

            # Check for low ratio
            if ratio < ratio_low_threshold:
                if consecutive_low == 0:
                    low_start = step
                consecutive_low += 1
                consecutive_high = 0
                high_start = None
            # Check for high ratio
            elif ratio > ratio_high_threshold:
                if consecutive_high == 0:
                    high_start = step
                consecutive_high += 1
                consecutive_low = 0
                low_start = None
            else:
                # Reset both counters
                if consecutive_low >= consecutive_threshold:
                    rules.append(
                        {
                            "rule": "grad_ratio_low",
                            "description": f"Gradient ratio too low: {consecutive_low} consecutive updates with policy/value ratio < {ratio_low_threshold}",
                            "step_range": [low_start, step],
                        }
                    )
                if consecutive_high >= consecutive_threshold:
                    rules.append(
                        {
                            "rule": "grad_ratio_high",
                            "description": f"Gradient ratio too high: {consecutive_high} consecutive updates with policy/value ratio > {ratio_high_threshold}",
                            "step_range": [high_start, step],
                        }
                    )
                consecutive_low = 0
                consecutive_high = 0
                low_start = None
                high_start = None

    # Check for anomalies at the end
    if consecutive_low >= consecutive_threshold:
        rules.append(
            {
                "rule": "grad_ratio_low",
                "description": f"Gradient ratio too low: {consecutive_low} consecutive updates with policy/value ratio < {ratio_low_threshold}",
                "step_range": [low_start, steps[-1]],
            }
        )
    if consecutive_high >= consecutive_threshold:
        rules.append(
            {
                "rule": "grad_ratio_high",
                "description": f"Gradient ratio too high: {consecutive_high} consecutive updates with policy/value ratio > {ratio_high_threshold}",
                "step_range": [high_start, steps[-1]],
            }
        )

    return rules


def detect_advantage_anomalies(
    steps: List[int],
    advantage_means: List[float],
    advantage_stds: List[float],
    entropies: List[float],
    kl_values: List[float],
) -> List[Dict[str, Any]]:
    """Detect advantage anomalies (mean rising while entropy falling and KL rising)."""
    rules = []

    if len(advantage_means) < 10:
        return rules

    # Compute trends over sliding window
    window_size = 5

    for i in range(window_size, len(steps)):
        # Get window data
        adv_window = advantage_means[i - window_size : i + 1]
        entropy_window = entropies[i - window_size : i + 1]
        kl_window = kl_values[i - window_size : i + 1]

        # Compute trends (simple linear regression slope)
        adv_trend = np.polyfit(range(len(adv_window)), adv_window, 1)[0]
        entropy_trend = np.polyfit(range(len(entropy_window)), entropy_window, 1)[0]
        kl_trend = np.polyfit(range(len(kl_window)), kl_window, 1)[0]

        # Check for reward hacking pattern: adv rising, entropy falling, kl rising
        if adv_trend > 0 and entropy_trend < 0 and kl_trend > 0:
            rules.append(
                {
                    "rule": "advantage_reward_hacking",
                    "description": f"Potential reward hacking: advantage rising, entropy falling, KL rising at step {steps[i]}",
                    "step_range": [steps[i - window_size], steps[i]],
                }
            )

    return rules


def compute_stats(
    kl_values: List[float],
    grad_norm_policies: List[float],
    grad_norm_values: List[float],
    entropies: List[float],
) -> Dict[str, Any]:
    """Compute summary statistics."""
    stats = {}

    if kl_values:
        stats["kl_median"] = float(np.median(kl_values))

    # Compute gradient ratios
    grad_ratios = []
    for policy_norm, value_norm in zip(grad_norm_policies, grad_norm_values):
        if policy_norm > 0 and value_norm > 0:
            grad_ratios.append(policy_norm / value_norm)

    if grad_ratios:
        stats["grad_ratio_median"] = float(np.median(grad_ratios))

    # Compute entropy trend
    if len(entropies) >= 10:
        entropy_trend = np.polyfit(range(len(entropies)), entropies, 1)[0]
        if entropy_trend > 0.01:
            stats["entropy_trend"] = "increasing"
        elif entropy_trend < -0.01:
            stats["entropy_trend"] = "decreasing"
        else:
            stats["entropy_trend"] = "stable"

    return stats
