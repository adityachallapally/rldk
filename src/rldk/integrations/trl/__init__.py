"""TRL integration for RLDK."""

from .callbacks import RLDKCallback, RLDKMetrics, RLDKMonitor
from .dashboard import RLDKDashboard
from .monitors import CheckpointMetrics, CheckpointMonitor, PPOMetrics, PPOMonitor
from .utils import (
    check_trl_compatibility,
    create_dpo_trainer,
    fix_generation_config,
    prepare_models_for_ppo,
    simple_reward_function,
    validate_ppo_setup,
)

__all__ = [
    "RLDKCallback",
    "RLDKMonitor",
    "RLDKMetrics",
    "PPOMonitor",
    "PPOMetrics",
    "CheckpointMonitor",
    "CheckpointMetrics",
    "RLDKDashboard",
    "fix_generation_config",
    "prepare_models_for_ppo",
    "create_dpo_trainer",
    "simple_reward_function",
    "check_trl_compatibility",
    "validate_ppo_setup",
]
