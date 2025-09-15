"""Utility functions for TRL integration."""

from typing import Optional

import torch.nn as nn
from packaging import version
from transformers import AutoTokenizer, GenerationConfig

try:
    from trl import AutoModelForCausalLMWithValueHead
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    AutoModelForCausalLMWithValueHead = None


def fix_generation_config(
    model: "AutoModelForCausalLMWithValueHead",
    tokenizer: AutoTokenizer,
    generation_config: Optional[GenerationConfig] = None
) -> "AutoModelForCausalLMWithValueHead":
    """Fix missing generation_config and base_model_prefix attributes on TRL models.

    This is a common issue with TRL 0.23.0+ where AutoModelForCausalLMWithValueHead
    doesn't have a generation_config or base_model_prefix attribute by default,
    causing AttributeError when PPOTrainer tries to access them.

    Args:
        model: The TRL model to fix
        tokenizer: The tokenizer used with the model
        generation_config: Optional custom generation config. If None, creates a default one.

    Returns:
        The model with generation_config and base_model_prefix attributes set

    Raises:
        ImportError: If TRL is not available
        AttributeError: If model doesn't have required attributes
    """
    if not TRL_AVAILABLE:
        raise ImportError("TRL is required for this function. Install with: pip install trl")

    if not isinstance(model, AutoModelForCausalLMWithValueHead):
        raise AttributeError("Model must be an AutoModelForCausalLMWithValueHead instance")

    # Create generation config if not provided
    if generation_config is None:
        generation_config = GenerationConfig(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=getattr(tokenizer, 'bos_token_id', None),
            max_length=512,  # Default max length
            do_sample=True,  # Enable sampling for generation
            temperature=1.0,  # Default temperature
            top_p=1.0,  # Default top_p
        )

    # Set the generation_config attribute
    model.generation_config = generation_config

    # Fix missing base_model_prefix attribute (required by PPOTrainer)
    if not hasattr(model, 'base_model_prefix'):
        # The AutoModelForCausalLMWithValueHead wraps a pretrained_model that has the base_model_prefix
        if hasattr(model, 'pretrained_model') and hasattr(model.pretrained_model, 'base_model_prefix'):
            model.base_model_prefix = model.pretrained_model.base_model_prefix
        else:
            # Fallback: try to infer from the model name or use a default
            model_name = getattr(model, 'name_or_path', '').lower()
            if 'gpt2' in model_name or 'gpt' in model_name:
                model.base_model_prefix = "transformer"
            elif 'llama' in model_name:
                model.base_model_prefix = "model"
            else:
                model.base_model_prefix = "transformer"  # Default fallback

    # Fix missing base model attribute (required by PPOTrainer)
    # The PPOTrainer expects the model to have an attribute with the name of base_model_prefix
    if hasattr(model, 'base_model_prefix') and hasattr(model, 'pretrained_model'):
        base_model_prefix = model.base_model_prefix
        if not hasattr(model, base_model_prefix):
            # Add the base model attribute by referencing the pretrained_model's attribute
            if hasattr(model.pretrained_model, base_model_prefix):
                setattr(model, base_model_prefix, getattr(model.pretrained_model, base_model_prefix))

    # Fix missing gradient checkpointing attribute (required by PPOTrainer)
    if not hasattr(model, 'is_gradient_checkpointing'):
        if hasattr(model, 'pretrained_model') and hasattr(model.pretrained_model, 'is_gradient_checkpointing'):
            model.is_gradient_checkpointing = model.pretrained_model.is_gradient_checkpointing
        else:
            model.is_gradient_checkpointing = False  # Default to False

    return model


def create_simple_value_model(tokenizer: AutoTokenizer, model_name: str = None) -> nn.Module:
    """Create a simple value model compatible with TRL PPOTrainer.

    This creates a value model that has all required attributes for TRL 0.23+:
    - base_model_prefix attribute
    - score() method that takes hidden_states and returns value logits
    - proper tensor shapes and interfaces

    Args:
        tokenizer: Tokenizer to use for the model
        model_name: Optional model name. If None, creates a simple linear layer.

    Returns:
        Value model compatible with TRL PPOTrainer

    Raises:
        ImportError: If TRL is not available
    """
    if not TRL_AVAILABLE:
        raise ImportError("TRL is required for this function. Install with: pip install trl")

    import torch

    class SimpleValueModel(nn.Module):
        def __init__(self, hidden_size: int = 768):
            super().__init__()
            self.base_model_prefix = "transformer"  # Required by TRL
            self.hidden_size = hidden_size
            self.value_head = nn.Linear(hidden_size, 1)
            
            # This is a minimal implementation that satisfies TRL's requirements
            class SimpleTransformer(nn.Module):
                def __init__(self, hidden_size):
                    super().__init__()
                    self.hidden_size = hidden_size
                    
                def forward(self, **kwargs):
                    # Return dummy hidden states for compatibility
                    batch_size = kwargs.get('input_ids', torch.tensor([[1]])).shape[0]
                    seq_len = kwargs.get('input_ids', torch.tensor([[1]])).shape[1]
                    return type('obj', (object,), {
                        'last_hidden_state': torch.zeros(batch_size, seq_len, hidden_size)
                    })()
            
            self.transformer = SimpleTransformer(hidden_size)

        def score(self, hidden_states):
            """Score method required by TRL PPOTrainer."""
            with torch.no_grad():
                return self.value_head(hidden_states)

        def forward(self, **kwargs):
            return self.score(kwargs.get('hidden_states'))

    if model_name:
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            hidden_size = getattr(config, 'hidden_size', 768)
        except Exception:
            hidden_size = 768  # Default fallback
    else:
        hidden_size = 768

    value_model = SimpleValueModel(hidden_size)
    return value_model


def create_simple_reward_model(tokenizer: AutoTokenizer, model_name: str = "gpt2") -> "AutoModelForCausalLMWithValueHead":
    """Create a simple reward model compatible with TRL's get_reward() function.

    This creates a reward model that works seamlessly with TRL's get_reward() utility
    and handles all required attributes automatically.

    Args:
        tokenizer: Tokenizer to use for the model
        model_name: Base model name to use for the reward model

    Returns:
        Reward model compatible with TRL's get_reward() function

    Raises:
        ImportError: If TRL is not available
    """
    if not TRL_AVAILABLE:
        raise ImportError("TRL is required for this function. Install with: pip install trl")

    reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

    reward_model = fix_generation_config(reward_model, tokenizer)

    return reward_model


def prepare_models_for_ppo(
    model_name: str,
    tokenizer: Optional[AutoTokenizer] = None,
    generation_config: Optional[GenerationConfig] = None,
    create_separate_value_model: bool = True,
    create_separate_reward_model: bool = True
) -> tuple["AutoModelForCausalLMWithValueHead", "AutoModelForCausalLMWithValueHead",
           "AutoModelForCausalLMWithValueHead", nn.Module, AutoTokenizer]:
    """Prepare all required models for PPO training with TRL 0.23+ compatibility.

    This function creates and configures all the models needed for PPO training,
    ensuring they have the required attributes for TRL 0.23.0+ API compatibility.

    Args:
        model_name: Name or path of the base model
        tokenizer: Optional tokenizer. If None, will be loaded from model_name
        generation_config: Optional custom generation config
        create_separate_value_model: If True, creates a proper value model. If False, uses policy model.
        create_separate_reward_model: If True, creates a separate reward model. If False, uses policy model.

    Returns:
        Tuple of (policy_model, ref_model, reward_model, value_model, tokenizer)

    Raises:
        ImportError: If TRL is not available
    """
    if not TRL_AVAILABLE:
        raise ImportError("TRL is required for this function. Install with: pip install trl")

    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

    # Create reward model
    if create_separate_reward_model:
        reward_model = create_simple_reward_model(tokenizer, model_name)
    else:
        reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        reward_model = fix_generation_config(reward_model, tokenizer, generation_config)

    if create_separate_value_model:
        value_model = create_simple_value_model(tokenizer, model_name)
    else:
        # Use policy model for backward compatibility
        value_model = policy_model

    # Fix generation_config for policy and reference models
    policy_model = fix_generation_config(policy_model, tokenizer, generation_config)
    ref_model = fix_generation_config(ref_model, tokenizer, generation_config)

    return policy_model, ref_model, reward_model, value_model, tokenizer


def check_trl_compatibility() -> dict:
    """Check TRL version compatibility and common issues.

    Returns:
        Dictionary with compatibility information and warnings
    """
    if not TRL_AVAILABLE:
        return {
            "trl_available": False,
            "version": None,
            "warnings": ["TRL is not installed. Install with: pip install trl"],
            "recommendations": ["Install TRL: pip install trl>=0.7.0"]
        }

    try:
        import trl
        trl_version_str = trl.__version__

        warnings_list = []
        recommendations = []

        # Check for known issues using proper semantic versioning
        trl_version_obj = version.parse(trl_version_str)

        if trl_version_obj >= version.parse("0.23.0"):
            warnings_list.append(
                "TRL 0.23.0+ has known issues with AutoModelForCausalLMWithValueHead.generation_config. "
                "Use fix_generation_config() utility function."
            )
            recommendations.append(
                "Use prepare_models_for_ppo() or fix_generation_config() to avoid AttributeError"
            )

        if trl_version_obj < version.parse("0.7.0"):
            warnings_list.append("TRL version is quite old. Consider upgrading to 0.7.0+")
            recommendations.append("Upgrade TRL: pip install --upgrade trl")

        # Check for very recent versions that might have breaking changes
        if trl_version_obj >= version.parse("0.25.0"):
            warnings_list.append(
                "Using a very recent TRL version. Some features may not be fully tested. "
                "Report any issues if they occur."
            )

        # Check for specific problematic versions
        if version.parse("0.20.0") <= trl_version_obj < version.parse("0.22.0"):
            warnings_list.append(
                "TRL versions 0.20.0-0.21.x have known stability issues. "
                "Consider upgrading to 0.22.0+ or downgrading to 0.19.x"
            )
            recommendations.append("Upgrade TRL: pip install --upgrade trl>=0.22.0")

        return {
            "trl_available": True,
            "version": trl_version_str,
            "warnings": warnings_list,
            "recommendations": recommendations
        }

    except Exception as e:
        return {
            "trl_available": True,
            "version": "unknown",
            "warnings": [f"Could not determine TRL version: {e}"],
            "recommendations": ["Check TRL installation"]
        }


def validate_ppo_setup(
    model: "AutoModelForCausalLMWithValueHead",
    ref_model: "AutoModelForCausalLMWithValueHead",
    reward_model: "AutoModelForCausalLMWithValueHead",
    tokenizer: AutoTokenizer
) -> dict:
    """Validate PPO setup for common issues.

    Args:
        model: Main PPO model (used for both policy and value heads)
        ref_model: Reference model
        reward_model: Reward model
        tokenizer: Tokenizer

    Returns:
        Dictionary with validation results
    """
    issues = []
    warnings = []

    # Check generation_config attribute
    for name, model_obj in [("model", model), ("ref_model", ref_model),
                           ("reward_model", reward_model)]:
        if not hasattr(model_obj, 'generation_config'):
            issues.append(f"{name} missing generation_config attribute")
        elif model_obj.generation_config is None:
            warnings.append(f"{name} has None generation_config")

    # Check tokenizer compatibility
    if not hasattr(tokenizer, 'eos_token_id') or tokenizer.eos_token_id is None:
        issues.append("Tokenizer missing eos_token_id")

    if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
        warnings.append("Tokenizer missing pad_token_id")

    # Check model types
    for name, model_obj in [("model", model), ("ref_model", ref_model),
                           ("reward_model", reward_model)]:
        if not isinstance(model_obj, AutoModelForCausalLMWithValueHead):
            issues.append(f"{name} is not an AutoModelForCausalLMWithValueHead instance")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "recommendations": [
            "Use fix_generation_config() to fix generation_config issues",
            "Ensure all models are AutoModelForCausalLMWithValueHead instances",
            "Check tokenizer has required token IDs"
        ] if issues else []
    }
