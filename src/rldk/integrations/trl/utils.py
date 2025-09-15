"""Utility functions for TRL integration."""

from typing import Optional

from packaging import version
from transformers import AutoTokenizer, GenerationConfig

try:
    from datasets import Dataset
except ImportError:
    Dataset = None

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


def prepare_models_for_ppo(
    model_name: str,
    tokenizer: Optional[AutoTokenizer] = None,
    generation_config: Optional[GenerationConfig] = None
) -> tuple["AutoModelForCausalLMWithValueHead", "AutoModelForCausalLMWithValueHead",
           "AutoModelForCausalLMWithValueHead", "AutoModelForCausalLMWithValueHead", AutoTokenizer]:
    """Prepare all required models for PPO training with proper generation_config.

    This function creates and configures all the models needed for PPO training,
    ensuring they have the required generation_config attribute to avoid AttributeError.
    
    Note: The same model is used for both policy and value heads (standard approach).
    This avoids the base_model_prefix AttributeError in TRL 0.23.0+.

    Args:
        model_name: Name or path of the base model
        tokenizer: Optional tokenizer. If None, will be loaded from model_name
        generation_config: Optional custom generation config

    Returns:
        Tuple of (model, ref_model, reward_model, value_model, tokenizer)

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

    # Create models - use same model for policy and value heads
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    value_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

    # Fix generation_config for all models
    model = fix_generation_config(model, tokenizer, generation_config)
    ref_model = fix_generation_config(ref_model, tokenizer, generation_config)
    reward_model = fix_generation_config(reward_model, tokenizer, generation_config)
    value_model = fix_generation_config(value_model, tokenizer, generation_config)

    return model, ref_model, reward_model, value_model, tokenizer


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


def create_ppo_trainer(
    model_name: str,
    ppo_config,
    train_dataset,
    callbacks: Optional[list] = None,
    **kwargs
):
    """Create PPOTrainer with automatic parameter handling for different TRL versions.
    
    This function creates a PPOTrainer with all required parameters, automatically
    handling API differences between TRL versions. For TRL 0.23.0+, all parameters
    including reward_model, value_model, and train_dataset are required.
    
    Args:
        model_name: Name or path of the base model
        ppo_config: PPO configuration object
        train_dataset: Dataset for training
        callbacks: Optional list of callbacks
        **kwargs: Additional arguments passed to PPOTrainer
        
    Returns:
        Configured PPOTrainer instance
        
    Raises:
        ImportError: If TRL is not available
    """
    if not TRL_AVAILABLE:
        raise ImportError("TRL is required for this function. Install with: pip install trl")
    
    try:
        from trl import PPOTrainer
    except ImportError:
        raise ImportError("PPOTrainer not available. Install with: pip install trl")
    
    # Prepare all models
    model, ref_model, reward_model, value_model, tokenizer = prepare_models_for_ppo(model_name)
    
    compatibility = check_trl_compatibility()
    trl_version_str = compatibility.get("version", "0.7.0")
    
    try:
        if trl_version_str and trl_version_str not in ("unknown", "None", "null"):
            trl_version = version.parse(trl_version_str)
        else:
            trl_version = version.parse("0.23.0")  # Safe fallback to minimum supported version
    except (TypeError, ValueError, Exception):
        trl_version = version.parse("0.23.0")  # Fallback for any parsing errors
    
    if trl_version >= version.parse("0.23.0"):
        return PPOTrainer(
            args=ppo_config,
            model=model,
            ref_model=ref_model,
            reward_model=reward_model,
            value_model=value_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            callbacks=callbacks or [],
            **kwargs
        )
    else:
        return PPOTrainer(
            args=ppo_config,
            model=model,
            ref_model=ref_model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            callbacks=callbacks or [],
            **kwargs
        )
