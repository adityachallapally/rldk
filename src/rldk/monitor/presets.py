"""Built-in rule and field map presets for the streaming monitor."""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional

RulePreset = Dict[str, object]
FieldMapPreset = Dict[str, str]


RULE_PRESETS: Dict[str, RulePreset] = {
    "ppo_safe": {
        "rules": [
            {
                "id": "ppo_high_kl_guard",  # Warn and stop when KL runs hot
                "where": "name in (\"kl\", \"kl_mean\", \"ppo/policy/kl_mean\", \"train/kl\")",
                "condition": "value > 0.35",
                "window": {"size": 5, "kind": "consecutive"},
                "grace_steps": 5,
                "cooldown_steps": 5,
                "actions": [
                    {"warn": {"msg": "KL spike {value:.3f} at step {step}"}},
                    {"stop": {"msg": "Requesting stop due to KL {value:.3f}"}},
                ],
            },
            {
                "id": "ppo_reward_freefall",  # Catch collapsing rewards
                "where": "name in (\"reward\", \"reward_mean\", \"ppo/rewards/mean\", \"train/reward\")",
                "condition": "mean(value) < -0.2",
                "window": {"size": 12, "kind": "rolling"},
                "grace_steps": 10,
                "cooldown_steps": 10,
                "actions": [
                    {
                        "warn": {
                            "msg": "Reward dropped to {value:.3f} (rolling mean)"
                        }
                    }
                ],
            },
            {
                "id": "ppo_grad_norm_spike",  # High gradient norms usually precede divergence
                "where": "name in (\"grad_norm\", \"policy_grad_norm\", \"ppo/policy/grad_norm\")",
                "condition": "value > 12.0",
                "window": {"size": 3, "kind": "consecutive"},
                "grace_steps": 6,
                "cooldown_steps": 6,
                "actions": [
                    {"warn": {"msg": "Gradient norm {value:.2f} at step {step}"}},
                ],
            },
            {
                "id": "ppo_kl_drift_detection",
                "where": "name in (\"kl\", \"kl_mean\", \"ppo/policy/kl_mean\", \"train/kl\")",
                "condition": "kl_drift_score > 0.15",
                "window": {"size": 20, "kind": "rolling"},
                "grace_steps": 10,
                "cooldown_steps": 15,
                "actions": [
                    {
                        "warn": {
                            "msg": "KL drift detected: score {kl_drift_score:.3f} at step {step}"
                        }
                    },
                    {
                        "stop": {
                            "msg": "Stopping due to KL drift (score={kl_drift_score:.3f})"
                        }
                    },
                ],
            },
            {
                "id": "ppo_length_bias_severity",
                "where": "name == \"length_bias_score\"",
                "condition": "value > meta.length_bias_threshold",
                "window": {"size": 3, "kind": "consecutive"},
                "grace_steps": 6,
                "cooldown_steps": 12,
                "actions": [
                    {
                        "warn": {
                            "msg": "Length bias severity {value:.3f} exceeds threshold"
                        }
                    }
                ],
            },
            {
                "id": "ppo_length_bias_corr",
                "where": "name == \"length_reward_correlation_abs\"",
                "condition": "value > meta.length_bias_corr_threshold",
                "window": {"size": 6, "kind": "rolling"},
                "grace_steps": 8,
                "cooldown_steps": 12,
                "actions": [
                    {
                        "warn": {
                            "msg": "Absolute Pearson corr {value:.3f} with length"
                        }
                    }
                ],
            },
            {
                "id": "ppo_length_bias_rank_corr",
                "where": "name == \"length_reward_spearman_abs\"",
                "condition": "value > meta.length_bias_corr_threshold",
                "window": {"size": 6, "kind": "rolling"},
                "grace_steps": 8,
                "cooldown_steps": 12,
                "actions": [
                    {
                        "warn": {
                            "msg": "Absolute Spearman corr {value:.3f} with length"
                        }
                    }
                ],
            },
        ]
    },
    "ppo_strict": {
        "rules": [
            {
                "id": "ppo_strict_kl_stop",
                # Strict preset keeps the 0.25 gate to mirror production overrides.
                "where": "name in (\"kl\", \"kl_mean\", \"ppo/policy/kl_mean\", \"train/kl\")",
                "condition": "value > 0.25",
                "window": {"size": 3, "kind": "consecutive"},
                "grace_steps": 3,
                "cooldown_steps": 8,
                "actions": [
                    {"warn": {"msg": "KL exceeded strict gate at {value:.3f}"}},
                    {"stop": {"msg": "Strict KL gate fired (value={value:.3f})"}},
                ],
            },
            {
                "id": "ppo_entropy_collapse",
                "where": "name in (\"entropy\", \"entropy_mean\", \"ppo/policy/entropy\")",
                "condition": "mean(value) < 0.1",
                "window": {"size": 8, "kind": "rolling"},
                "grace_steps": 8,
                "cooldown_steps": 12,
                "actions": [
                    {"warn": {"msg": "Entropy collapsing: {value:.3f} at {step}"}},
                ],
            },
            {
                "id": "ppo_clipfrac_breach",
                "where": "name in (\"clipfrac\", \"clip_frac\", \"ppo/policy/clipfrac\")",
                "condition": "mean(value) > 0.35",
                "window": {"size": 6, "kind": "rolling"},
                "grace_steps": 6,
                "cooldown_steps": 10,
                "actions": [
                    {
                        "warn": {
                            "msg": "Clip fraction averaging {value:.3f} (rolling window)"
                        }
                    }
                ],
            },
            {
                "id": "ppo_reward_stall",
                "where": "name in (\"reward\", \"reward_mean\", \"ppo/rewards/mean\", \"train/reward\")",
                "condition": "max(value) - min(value) < 0.05",
                "window": {"size": 12, "kind": "rolling"},
                "grace_steps": 12,
                "cooldown_steps": 12,
                "actions": [
                    {"warn": {"msg": "Reward plateau detected at {value:.3f}"}},
                ],
            },
            {
                "id": "ppo_strict_kl_drift",
                "where": "name in (\"kl\", \"kl_mean\", \"ppo/policy/kl_mean\", \"train/kl\")",
                "condition": "kl_drift_score > 0.10",
                "window": {"size": 15, "kind": "rolling"},
                "grace_steps": 5,
                "cooldown_steps": 10,
                "actions": [
                    {"warn": {"msg": "KL drift threshold exceeded: {kl_drift_score:.3f}"}},
                    {
                        "stop": {
                            "msg": "Strict KL drift gate fired (score={kl_drift_score:.3f})"
                        }
                    },
                ],
            },
            {
                "id": "ppo_strict_length_bias_severity",
                "where": "name == \"length_bias_score\"",
                "condition": "value > meta.length_bias_threshold",
                "window": {"size": 2, "kind": "consecutive"},
                "grace_steps": 4,
                "cooldown_steps": 10,
                "actions": [
                    {
                        "warn": {
                            "msg": "Strict length bias guard fired: {value:.3f}"
                        }
                    },
                    {
                        "stop": {
                            "msg": "Stopping run due to persistent length bias"
                        }
                    },
                ],
            },
            {
                "id": "ppo_strict_length_bias_corr",
                "where": "name == \"length_reward_correlation_abs\"",
                "condition": "value > meta.length_bias_corr_threshold",
                "window": {"size": 4, "kind": "rolling"},
                "grace_steps": 6,
                "cooldown_steps": 10,
                "actions": [
                    {
                        "warn": {
                            "msg": "Length bias correlation {value:.3f} exceeded strict guard"
                        }
                    }
                ],
            },
            {
                "id": "ppo_strict_length_bias_rank_corr",
                "where": "name == \"length_reward_spearman_abs\"",
                "condition": "value > meta.length_bias_corr_threshold",
                "window": {"size": 4, "kind": "rolling"},
                "grace_steps": 6,
                "cooldown_steps": 10,
                "actions": [
                    {
                        "warn": {
                            "msg": "Rank correlation {value:.3f} indicates reward-length coupling"
                        }
                    }
                ],
            },
        ]
    },
    "dpo_basic": {
        "rules": [
            {
                "id": "dpo_high_kl",
                # DPO preset retains the 0.2 limit tuned for its narrower KL envelope.
                "where": "name in (\"dpo/kl\", \"kl\", \"kl_divergence\")",
                "condition": "value > 0.2",
                "window": {"size": 4, "kind": "consecutive"},
                "grace_steps": 4,
                "cooldown_steps": 6,
                "actions": [
                    {"warn": {"msg": "DPO KL {value:.3f} beyond limit"}},
                    {"stop": {"msg": "Halting run due to DPO KL {value:.3f}"}},
                ],
            },
            {
                "id": "dpo_reward_regression",
                "where": "name in (\"dpo/reward\", \"reward\", \"reward_mean\")",
                "condition": "mean(value) < 0.0",
                "window": {"size": 10, "kind": "rolling"},
                "grace_steps": 6,
                "cooldown_steps": 10,
                "actions": [
                    {
                        "warn": {
                            "msg": "DPO reward trending negative ({value:.3f})"
                        }
                    }
                ],
            },
            {
                "id": "dpo_reference_drift",
                "where": "name in (\"dpo/ref_logprob\", \"ref_logprob\", \"reference_logprob\")",
                "condition": "mean(value) < -5.0",
                "window": {"size": 8, "kind": "rolling"},
                "grace_steps": 6,
                "cooldown_steps": 10,
                "actions": [
                    {
                        "warn": {
                            "msg": "Reference policy drift: logprob {value:.2f}"
                        }
                    }
                ],
            },
        ]
    },
    "kl_drift": {
        "rules": [
            {
                "id": "kl_drift_early_warning",
                "where": "name in (\"kl\", \"kl_mean\", \"ppo/policy/kl_mean\", \"train/kl\")",
                "condition": "kl_drift_score > 0.08",
                "window": {"size": 10, "kind": "rolling"},
                "grace_steps": 5,
                "cooldown_steps": 10,
                "actions": [
                    {
                        "warn": {
                            "msg": "Early KL drift warning: {kl_drift_score:.3f}"
                        }
                    }
                ],
            },
            {
                "id": "kl_drift_critical",
                "where": "name in (\"kl\", \"kl_mean\", \"ppo/policy/kl_mean\", \"train/kl\")",
                "condition": "kl_drift_score > 0.20",
                "window": {"size": 5, "kind": "consecutive"},
                "grace_steps": 3,
                "cooldown_steps": 8,
                "actions": [
                    {"warn": {"msg": "Critical KL drift: {kl_drift_score:.3f}"}},
                    {
                        "stop": {
                            "msg": "Emergency stop due to critical KL drift"
                        }
                    },
                ],
            },
        ]
    },
    "length_bias": {
        "rules": [
            {
                "id": "length_bias_severity_gate",
                "where": "name == \"length_bias_score\"",
                "condition": "value > meta.length_bias_threshold",
                "window": {"size": 3, "kind": "consecutive"},
                "grace_steps": 5,
                "cooldown_steps": 12,
                "actions": [
                    {
                        "warn": {
                            "msg": "Length bias severity {value:.3f} exceeds configured guard"
                        }
                    },
                    {
                        "stop": {
                            "msg": "Halting run: length bias severity over limit"
                        }
                    },
                ],
            },
            {
                "id": "length_bias_corr_guard",
                "where": "name == \"length_reward_correlation_abs\"",
                "condition": "value > meta.length_bias_corr_threshold",
                "window": {"size": 5, "kind": "rolling"},
                "grace_steps": 6,
                "cooldown_steps": 10,
                "actions": [
                    {
                        "warn": {
                            "msg": "Absolute Pearson correlation {value:.3f} with response length"
                        }
                    }
                ],
            },
            {
                "id": "length_bias_rank_corr_guard",
                "where": "name == \"length_reward_spearman_abs\"",
                "condition": "value > meta.length_bias_corr_threshold",
                "window": {"size": 5, "kind": "rolling"},
                "grace_steps": 6,
                "cooldown_steps": 10,
                "actions": [
                    {
                        "warn": {
                            "msg": "Absolute Spearman correlation {value:.3f} with response length"
                        }
                    }
                ],
            },
        ]
    },
}


FIELD_MAP_PRESETS: Dict[str, FieldMapPreset] = {
    "trl": {
        "timestamp": "time",
        "time": "time",
        "wall_time": "time",
        "global_step": "step",
        "step": "step",
        "s": "step",
        "metric": "name",
        "metric_name": "name",
        "key": "name",
        "name": "name",
        "metric_key": "name",
        "value": "value",
        "metric_value": "value",
        "scalar": "value",
        "v": "value",
        "run": "run_id",
        "run_id": "run_id",
        "tags": "tags",
        "metadata": "meta",
        "meta": "meta",
        "context": "meta",
    },
    "accelerate": {
        "timestamp": "time",
        "time": "time",
        "completed_steps": "step",
        "step": "step",
        "iteration": "step",
        "metric": "name",
        "metric_name": "name",
        "event": "name",
        "name": "name",
        "value": "value",
        "metric_value": "value",
        "scalar": "value",
        "logged_value": "value",
        "run": "run_id",
        "run_id": "run_id",
        "tags": "tags",
        "metadata": "meta",
        "meta": "meta",
        "extra": "meta",
    },
    "openrlhf": {
        "timestamp": "time",
        "time": "time",
        "global_step": "step",
        "trainer_step": "step",
        "step": "step",
        "metric": "name",
        "metric_name": "name",
        "key": "name",
        "name": "name",
        "value": "value",
        "metric_value": "value",
        "scalar": "value",
        "measurement": "value",
        "run": "run_id",
        "run_id": "run_id",
        "metadata": "meta",
        "meta": "meta",
        "context": "meta",
    },
}


def get_rule_preset(name: str) -> Optional[RulePreset]:
    """Return a deep copy of the requested rule preset, if it exists."""
    key = name.lower()
    preset = RULE_PRESETS.get(key)
    return deepcopy(preset) if preset is not None else None


def get_field_map_preset(name: str) -> Optional[FieldMapPreset]:
    """Return a deep copy of the requested field-map preset, if it exists."""
    key = name.lower()
    preset = FIELD_MAP_PRESETS.get(key)
    return deepcopy(preset) if preset is not None else None


__all__ = [
    "FIELD_MAP_PRESETS",
    "RULE_PRESETS",
    "get_field_map_preset",
    "get_rule_preset",
]
