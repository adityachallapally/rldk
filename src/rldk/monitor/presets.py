"""Built-in rule and field map presets for the streaming monitor."""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional

RulePreset = Dict[str, object]
FieldMapPreset = Dict[str, str]


RULE_PRESETS: Dict[str, RulePreset] = {
    "grpo_safe": {
        "rules": [
            {
                "id": "grpo_safe_kl_spike",
                "where": "name in (\"kl\", \"kl_mean\", \"policy_kl\", \"grpo/kl_mean\")",
                "condition": "value > 0.30",
                "window": {"size": 2, "kind": "consecutive"},
                "grace_steps": 12,
                "cooldown_steps": 12,
                "actions": [
                    {"warn": {"msg": "GRPO KL spike {value:.3f} at step {step}"}},
                    {"stop": {"msg": "Halting due to GRPO KL {value:.3f}"}},
                ],
            },
            {
                "id": "grpo_safe_kl_coef_stall",
                "where": "name in (\"kl_coef\", \"kl_coeff\", \"kl_coefficient\", \"beta\")",
                "condition": "max(value) - min(value) < 0.003",
                "window": {"size": 20, "kind": "rolling"},
                "grace_steps": 25,
                "cooldown_steps": 30,
                "actions": [
                    {
                        "warn": {
                            "msg": "KL coefficient stalled near {value:.4f}"
                        }
                    }
                ],
            },
            {
                "id": "grpo_safe_advantage_collapse",
                "where": "name in (\"advantage_std\", \"adv_std\", \"advantage_stddev\")",
                "condition": "mean(value) < 0.35",
                "window": {"size": 8, "kind": "rolling"},
                "grace_steps": 12,
                "cooldown_steps": 18,
                "actions": [
                    {
                        "warn": {
                            "msg": "Advantage variance collapsing: std {value:.3f}"
                        }
                    }
                ],
            },
            {
                "id": "grpo_safe_acceptance_swings",
                "where": "name in (\"acceptance_rate\", \"accept_rate\", \"acceptance\")",
                "condition": "max(value) - min(value) > 0.4",
                "window": {"size": 12, "kind": "rolling"},
                "grace_steps": 12,
                "cooldown_steps": 20,
                "actions": [
                    {
                        "warn": {
                            "msg": "Acceptance rate swing detected (latest {value:.2f})"
                        }
                    }
                ],
            },
            {
                "id": "grpo_safe_policy_grad_spike",
                "where": "name == \"grad_norm_policy\"",
                "condition": "value > 8.0",
                "window": {"size": 2, "kind": "consecutive"},
                "grace_steps": 6,
                "cooldown_steps": 18,
                "actions": [
                    {
                        "warn": {
                            "msg": "Policy gradient norm spiking: {value:.2f}"
                        }
                    }
                ],
            },
            {
                "id": "grpo_safe_grad_ratio_imbalance",
                "where": "name in (\"grad_norm_policy\", \"grad_norm_value\")",
                "condition": "('policy_over_value' in meta and meta['policy_over_value'] > 4.5) or ('value_over_policy' in meta and meta['value_over_policy'] > 4.5)",
                "window": {"size": 1, "kind": "consecutive"},
                "grace_steps": 6,
                "cooldown_steps": 24,
                "actions": [
                    {
                        "warn": {
                            "msg": "Policy/value gradient imbalance detected; guard ratio > 4.5 (latest {value:.2f})"
                        }
                    }
                ],
            },
            {
                "id": "grpo_safe_entropy_floor",
                "where": "name in (\"entropy\", \"entropy_mean\", \"policy_entropy\")",
                "condition": "mean(value) < 1.8",
                "window": {"size": 10, "kind": "rolling"},
                "grace_steps": 15,
                "cooldown_steps": 20,
                "actions": [
                    {"warn": {"msg": "Entropy below floor: {value:.3f}"}},
                ],
            },
            {
                "id": "grpo_safe_diversity_pass_floor",
                "where": "name == \"diversity_pass_at_1\"",
                "condition": "mean(value) < 0.24",
                "window": {"size": 10, "kind": "rolling"},
                "grace_steps": 12,
                "cooldown_steps": 18,
                "actions": [
                    {
                        "warn": {
                            "msg": "Pass@1 diversity collapsing: mean {value:.3f}"
                        }
                    }
                ],
            },
            {
                "id": "grpo_safe_diversity_distinct_collapse",
                "where": "name == \"diversity_distinct_4\"",
                "condition": "mean(value) < 0.10",
                "window": {"size": 8, "kind": "rolling"},
                "grace_steps": 10,
                "cooldown_steps": 15,
                "actions": [
                    {
                        "warn": {
                            "msg": "Distinct-4 diversity eroding: mean {value:.3f}"
                        }
                    }
                ],
            },
            {
                "id": "grpo_safe_self_bleu_spike",
                "where": "name == \"diversity_self_bleu\"",
                "condition": "mean(value) > 0.90",
                "window": {"size": 8, "kind": "rolling"},
                "grace_steps": 10,
                "cooldown_steps": 15,
                "actions": [
                    {
                        "warn": {
                            "msg": "Self-BLEU spike indicates mode collapse: mean {value:.3f}"
                        }
                    }
                ],
            },
            {
                "id": "grpo_safe_output_entropy_floor",
                "where": "name == \"diversity_output_entropy\"",
                "condition": "mean(value) < 1.20",
                "window": {"size": 8, "kind": "rolling"},
                "grace_steps": 10,
                "cooldown_steps": 15,
                "actions": [
                    {
                        "warn": {
                            "msg": "Output entropy collapsing: mean {value:.3f}"
                        }
                    }
                ],
            },
            {
                "id": "grpo_safe_reward_saturation",
                "where": "name in (\"reward_mean\", \"reward\", \"group_reward_mean\")",
                "condition": "max(value) - min(value) < 0.05",
                "window": {"size": 30, "kind": "rolling"},
                "grace_steps": 30,
                "cooldown_steps": 30,
                "actions": [
                    {
                        "warn": {
                            "msg": "Reward trend saturated around {value:.3f}"
                        }
                    }
                ],
            },
            {
                "id": "grpo_safe_reward_health_drift",
                "where": "name == \"reward_health.drift_flag\"",
                "condition": "value >= 1",
                "window": {"size": 1, "kind": "consecutive"},
                "cooldown_steps": 200,
                "actions": [
                    {
                        "warn": {
                            "msg": "Reward health detected drift in recent window"
                        }
                    }
                ],
            },
            {
                "id": "grpo_safe_reward_health_saturation",
                "where": "name == \"reward_health.saturation_flag\"",
                "condition": "value >= 1",
                "window": {"size": 1, "kind": "consecutive"},
                "cooldown_steps": 120,
                "actions": [
                    {
                        "warn": {
                            "msg": "Reward health saturation issues detected ({value:.0f} alerts)"
                        }
                    }
                ],
            },
            {
                "id": "grpo_safe_reward_health_shortcuts",
                "where": "name == \"reward_health.shortcut_flag\"",
                "condition": "value >= 1",
                "window": {"size": 1, "kind": "consecutive"},
                "cooldown_steps": 120,
                "actions": [
                    {
                        "warn": {
                            "msg": "Reward health shortcut signals detected ({value:.0f})"
                        }
                    }
                ],
            },
            {
                "id": "grpo_safe_reward_health_label_leakage",
                "where": "name == \"reward_health.label_leakage_risk\"",
                "condition": "value >= 0.3",
                "window": {"size": 1, "kind": "consecutive"},
                "cooldown_steps": 200,
                "actions": [
                    {
                        "warn": {
                            "msg": "Reward health label leakage risk {value:.2f}"
                        }
                    }
                ],
            },
            {
                "id": "grpo_safe_reward_health_overoptimization",
                "where": "name == \"reward_health.overoptimization_flag\"",
                "condition": "value >= 0",
                "window": {"size": 1, "kind": "consecutive"},
                "cooldown_steps": 200,
                "actions": [
                    {
                        "warn": {
                            "msg": "Reward health overoptimization suspected (Δ={value:.3f})"
                        }
                    }
                ],
            },
        ]
    },
    "grpo_strict": {
        "rules": [
            {
                "id": "grpo_strict_kl_spike",
                "where": "name in (\"kl\", \"kl_mean\", \"policy_kl\", \"grpo/kl_mean\")",
                "condition": "value > 0.22",
                "window": {"size": 2, "kind": "consecutive"},
                "grace_steps": 8,
                "cooldown_steps": 10,
                "actions": [
                    {"warn": {"msg": "Strict GRPO KL guard tripped at {value:.3f}"}},
                    {"stop": {"msg": "Stopping due to GRPO KL {value:.3f}"}},
                ],
            },
            {
                "id": "grpo_strict_kl_coef_stall",
                "where": "name in (\"kl_coef\", \"kl_coeff\", \"kl_coefficient\", \"beta\")",
                "condition": "max(value) - min(value) < 0.002",
                "window": {"size": 15, "kind": "rolling"},
                "grace_steps": 18,
                "cooldown_steps": 25,
                "actions": [
                    {
                        "warn": {
                            "msg": "KL coefficient flatlined near {value:.4f}"
                        }
                    },
                    {
                        "stop": {
                            "msg": "Strict KL coefficient stall detected"
                        }
                    },
                ],
            },
            {
                "id": "grpo_strict_advantage_collapse",
                "where": "name in (\"advantage_std\", \"adv_std\", \"advantage_stddev\")",
                "condition": "mean(value) < 0.6",
                "window": {"size": 6, "kind": "rolling"},
                "grace_steps": 8,
                "cooldown_steps": 14,
                "actions": [
                    {"warn": {"msg": "Advantage std collapsed to {value:.3f}"}},
                    {"stop": {"msg": "Stopping due to advantage variance collapse"}},
                ],
            },
            {
                "id": "grpo_strict_acceptance_swings",
                "where": "name in (\"acceptance_rate\", \"accept_rate\", \"acceptance\")",
                "condition": "max(value) - min(value) > 0.3",
                "window": {"size": 10, "kind": "rolling"},
                "grace_steps": 10,
                "cooldown_steps": 18,
                "actions": [
                    {"warn": {"msg": "Acceptance rate unstable (latest {value:.2f})"}},
                    {"stop": {"msg": "Stopping due to acceptance-rate instability"}},
                ],
            },
            {
                "id": "grpo_strict_policy_grad_spike",
                "where": "name == \"grad_norm_policy\"",
                "condition": "value > 6.5",
                "window": {"size": 2, "kind": "consecutive"},
                "grace_steps": 5,
                "cooldown_steps": 16,
                "actions": [
                    {
                        "warn": {
                            "msg": "Strict policy gradient ceiling breached: {value:.2f}"
                        }
                    },
                    {"stop": {"msg": "Stopping due to runaway policy gradients"}},
                ],
            },
            {
                "id": "grpo_strict_grad_ratio_imbalance",
                "where": "name in (\"grad_norm_policy\", \"grad_norm_value\")",
                "condition": "('policy_over_value' in meta and meta['policy_over_value'] > 3.5) or ('value_over_policy' in meta and meta['value_over_policy'] > 3.5)",
                "window": {"size": 1, "kind": "consecutive"},
                "grace_steps": 5,
                "cooldown_steps": 20,
                "actions": [
                    {
                        "warn": {
                            "msg": "Strict gradient ratio imbalance detected; guard ratio > 3.5 (latest {value:.2f})"
                        }
                    },
                    {"stop": {"msg": "Stopping due to gradient imbalance"}},
                ],
            },
            {
                "id": "grpo_strict_entropy_floor",
                "where": "name in (\"entropy\", \"entropy_mean\", \"policy_entropy\")",
                "condition": "mean(value) < 1.95",
                "window": {"size": 8, "kind": "rolling"},
                "grace_steps": 10,
                "cooldown_steps": 15,
                "actions": [
                    {"warn": {"msg": "Entropy breached strict floor at {value:.3f}"}},
                    {"stop": {"msg": "Stopping due to entropy collapse"}},
                ],
            },
            {
                "id": "grpo_strict_diversity_pass_floor",
                "where": "name == \"diversity_pass_at_1\"",
                "condition": "mean(value) < 0.32",
                "window": {"size": 8, "kind": "rolling"},
                "grace_steps": 8,
                "cooldown_steps": 14,
                "actions": [
                    {
                        "warn": {
                            "msg": "Strict pass@1 diversity floor breached: mean {value:.3f}"
                        }
                    },
                    {
                        "stop": {
                            "msg": "Stopping due to collapsing pass@1 diversity"
                        }
                    },
                ],
            },
            {
                "id": "grpo_strict_diversity_distinct_collapse",
                "where": "name == \"diversity_distinct_4\"",
                "condition": "mean(value) < 0.16",
                "window": {"size": 6, "kind": "rolling"},
                "grace_steps": 8,
                "cooldown_steps": 12,
                "actions": [
                    {
                        "warn": {
                            "msg": "Strict distinct-4 diversity collapse: mean {value:.3f}"
                        }
                    },
                    {
                        "stop": {
                            "msg": "Stopping due to distinct-4 collapse"
                        }
                    },
                ],
            },
            {
                "id": "grpo_strict_self_bleu_spike",
                "where": "name == \"diversity_self_bleu\"",
                "condition": "mean(value) > 0.82",
                "window": {"size": 8, "kind": "rolling"},
                "grace_steps": 8,
                "cooldown_steps": 12,
                "actions": [
                    {
                        "warn": {
                            "msg": "Strict self-BLEU spike detected: mean {value:.3f}"
                        }
                    },
                    {
                        "stop": {
                            "msg": "Stopping due to self-BLEU spike"
                        }
                    },
                ],
            },
            {
                "id": "grpo_strict_output_entropy_floor",
                "where": "name == \"diversity_output_entropy\"",
                "condition": "mean(value) < 1.35",
                "window": {"size": 6, "kind": "rolling"},
                "grace_steps": 8,
                "cooldown_steps": 12,
                "actions": [
                    {
                        "warn": {
                            "msg": "Strict output entropy floor breached: mean {value:.3f}"
                        }
                    },
                    {
                        "stop": {
                            "msg": "Stopping due to output entropy collapse"
                        }
                    },
                ],
            },
            {
                "id": "grpo_strict_reward_saturation",
                "where": "name in (\"reward_mean\", \"reward\", \"group_reward_mean\")",
                "condition": "max(value) - min(value) < 0.02",
                "window": {"size": 25, "kind": "rolling"},
                "grace_steps": 25,
                "cooldown_steps": 25,
                "actions": [
                    {"warn": {"msg": "Rewards saturated around {value:.3f}"}},
                    {"stop": {"msg": "Stopping due to saturated rewards"}},
                ],
            },
            {
                "id": "grpo_strict_reward_health_drift",
                "where": "name == \"reward_health.drift_flag\"",
                "condition": "value >= 1",
                "window": {"size": 1, "kind": "consecutive"},
                "cooldown_steps": 200,
                "actions": [
                    {
                        "warn": {
                            "msg": "Reward health detected drift in strict preset window"
                        }
                    }
                ],
            },
            {
                "id": "grpo_strict_reward_health_saturation",
                "where": "name == \"reward_health.saturation_flag\"",
                "condition": "value >= 1",
                "window": {"size": 1, "kind": "consecutive"},
                "cooldown_steps": 120,
                "actions": [
                    {
                        "warn": {
                            "msg": "Reward health saturation trip count {value:.0f}"
                        }
                    }
                ],
            },
            {
                "id": "grpo_strict_reward_health_shortcuts",
                "where": "name == \"reward_health.shortcut_flag\"",
                "condition": "value >= 1",
                "window": {"size": 1, "kind": "consecutive"},
                "cooldown_steps": 120,
                "actions": [
                    {
                        "warn": {
                            "msg": "Reward health shortcut signals detected ({value:.0f})"
                        }
                    }
                ],
            },
            {
                "id": "grpo_strict_reward_health_label_leakage",
                "where": "name == \"reward_health.label_leakage_risk\"",
                "condition": "value >= 0.3",
                "window": {"size": 1, "kind": "consecutive"},
                "cooldown_steps": 200,
                "actions": [
                    {
                        "warn": {
                            "msg": "Reward health label leakage risk {value:.2f}"
                        }
                    },
                    {
                        "stop": {
                            "msg": "Stopping due to reward health label leakage risk {value:.2f}"
                        }
                    },
                ],
            },
            {
                "id": "grpo_strict_reward_health_overoptimization",
                "where": "name == \"reward_health.overoptimization_flag\"",
                "condition": "value >= 0",
                "window": {"size": 1, "kind": "consecutive"},
                "cooldown_steps": 200,
                "actions": [
                    {
                        "warn": {
                            "msg": "Reward health overoptimization suspected (Δ={value:.3f})"
                        }
                    },
                    {
                        "stop": {
                            "msg": "Stopping due to reward health overoptimization signal"
                        }
                    },
                ],
            },
        ]
    },
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
    "grpo": {
        "timestamp": "time",
        "time": "time",
        "wall_time": "time",
        "global_step": "step",
        "trainer_step": "step",
        "step": "step",
        "iteration": "step",
        "epoch_step": "step",
        "reward": "reward_mean",
        "reward_mean": "reward_mean",
        "normalized_reward_mean": "reward_mean",
        "group_reward_mean": "reward_mean",
        "reward_avg": "reward_mean",
        "reward_std": "reward_std",
        "reward_stddev": "reward_std",
        "group_reward_std": "reward_std",
        "kl": "kl",
        "kl_mean": "kl",
        "policy_kl": "kl",
        "kl_avg": "kl",
        "kl_value": "kl",
        "kl_divergence": "kl",
        "kl_coef": "kl_coef",
        "kl_coeff": "kl_coef",
        "kl_coefficient": "kl_coef",
        "kl_beta": "kl_coef",
        "beta": "kl_coef",
        "entropy": "entropy",
        "entropy_mean": "entropy",
        "policy_entropy": "entropy",
        "advantage_mean": "advantage_mean",
        "adv_mean": "advantage_mean",
        "normalized_advantage_mean": "advantage_mean",
        "advantage_std": "advantage_std",
        "adv_std": "advantage_std",
        "advantage_stddev": "advantage_std",
        "policy_grad_norm": "grad_norm_policy",
        "pi_grad_norm": "grad_norm_policy",
        "actor_grad_norm": "grad_norm_policy",
        "grad_norm_policy": "grad_norm_policy",
        "value_grad_norm": "grad_norm_value",
        "critic_grad_norm": "grad_norm_value",
        "grad_norm_value": "grad_norm_value",
        "acceptance_rate": "acceptance_rate",
        "accept_rate": "acceptance_rate",
        "acceptance": "acceptance_rate",
        "policy_acceptance_rate": "acceptance_rate",
        "diversity/pass_at_1": "diversity_pass_at_1",
        "diversity_pass_at_1": "diversity_pass_at_1",
        "pass_at_1": "diversity_pass_at_1",
        "pass@1": "diversity_pass_at_1",
        "diversity/distinct_4": "diversity_distinct_4",
        "diversity_distinct_4": "diversity_distinct_4",
        "distinct_4": "diversity_distinct_4",
        "diversity/self_bleu": "diversity_self_bleu",
        "diversity_self_bleu": "diversity_self_bleu",
        "self_bleu": "diversity_self_bleu",
        "diversity/output_entropy": "diversity_output_entropy",
        "diversity_output_entropy": "diversity_output_entropy",
        "output_entropy": "diversity_output_entropy",
        "response_entropy": "diversity_output_entropy",
        "run": "run_id",
        "run_id": "run_id",
        "seed": "seed",
        "phase": "phase",
        "training_phase": "phase",
        "tags": "tags",
        "metadata": "meta",
        "meta": "meta",
        "context": "meta",
    },
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
