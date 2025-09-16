"""Utility functions for TRL integration."""

from typing import List, Optional

from packaging import version
from transformers import AutoTokenizer, GenerationConfig

try:
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, DPOConfig, DPOTrainer
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    AutoModelForCausalLMWithValueHead = None
    PPOConfig = None
    PPOTrainer = None
    DPOConfig = None
    DPOTrainer = None


def fix_generation_config(
    model: "AutoModelForCausalLM",
    tokenizer: AutoTokenizer,
    generation_config: Optional[GenerationConfig] = None
) -> "AutoModelForCausalLM":
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

    from transformers import PreTrainedModel
    if not isinstance(model, PreTrainedModel):
        raise AttributeError("Model must be a PreTrainedModel instance")

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


# Simple reward function for basic RL training
def simple_reward_function(text: str) -> float:
    """Simple reward function for demonstration purposes.
    
    In a real implementation, this would be replaced with a proper reward model
    or human feedback system.
    
    Args:
        text: Generated text to evaluate
        
    Returns:
        Reward score (higher is better)
    """
    # Simple heuristics for demonstration
    reward = 0.0
    
    # Reward for length (not too short, not too long)
    length = len(text.split())
    if 5 <= length <= 50:
        reward += 0.1
    
    # Reward for common positive words
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
    for word in positive_words:
        if word in text.lower():
            reward += 0.2
    
    # Penalty for repetition
    words = text.lower().split()
    if len(words) > 1:
        unique_words = len(set(words))
        repetition_ratio = unique_words / len(words)
        reward += repetition_ratio * 0.1
    
    return min(reward, 1.0)  # Cap at 1.0


def prepare_models_for_ppo(
    model_name: str,
    tokenizer: Optional[AutoTokenizer] = None,
    generation_config: Optional[GenerationConfig] = None
) -> tuple["AutoModelForCausalLM", "AutoModelForCausalLM",
           "AutoModelForCausalLM", "AutoModelForCausalLM", AutoTokenizer]:
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

    # Create models with simplified architecture for TRL compatibility
    # Use regular AutoModelForCausalLM instead of AutoModelForCausalLMWithValueHead
    # This avoids the tuple output and missing .score() method issues
    
    from transformers import AutoModelForCausalLM
    
    # Create policy model (main model)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create reference model (can be same as base for simplicity)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # For reward and value models, we can share the same model instance
    # This is memory-efficient and commonly used in practice
    reward_model = model  # Share with policy model
    value_model = model   # Share with policy model

    # Fix generation_config for all models
    model = fix_generation_config(model, tokenizer, generation_config)
    ref_model = fix_generation_config(ref_model, tokenizer, generation_config)
    reward_model = fix_generation_config(reward_model, tokenizer, generation_config)
    value_model = fix_generation_config(value_model, tokenizer, generation_config)

    # No need for complex wrappers with simplified architecture
    # Regular AutoModelForCausalLM returns proper ModelOutput objects with .logits
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
    model: "AutoModelForCausalLM",
    ref_model: "AutoModelForCausalLM",
    reward_model: "AutoModelForCausalLM",
    value_model: "AutoModelForCausalLM",
    tokenizer: AutoTokenizer
) -> dict:
    """Validate PPO setup for common issues.

    Args:
        model: Main PPO model (used for both policy and value heads)
        ref_model: Reference model
        reward_model: Reward model
        value_model: Value model (required for TRL 0.23.0+)
        tokenizer: Tokenizer

    Returns:
        Dictionary with validation results
    """
    issues = []
    warnings = []

    # Check generation_config attribute
    for name, model_obj in [("model", model), ("ref_model", ref_model),
                           ("reward_model", reward_model), ("value_model", value_model)]:
        if not hasattr(model_obj, 'generation_config'):
            issues.append(f"{name} missing generation_config attribute")
        elif model_obj.generation_config is None:
            warnings.append(f"{name} has None generation_config")

    # Check tokenizer compatibility
    if not hasattr(tokenizer, 'eos_token_id') or tokenizer.eos_token_id is None:
        issues.append("Tokenizer missing eos_token_id")

    if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
        warnings.append("Tokenizer missing pad_token_id")

    # Check model types (accept any PreTrainedModel)
    from transformers import PreTrainedModel
    for name, model_obj in [("model", model), ("ref_model", ref_model),
                           ("reward_model", reward_model), ("value_model", value_model)]:
        if not isinstance(model_obj, PreTrainedModel):
            issues.append(f"{name} is not a PreTrainedModel instance")

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


def create_dpo_trainer(
    model_name: str,
    dpo_config: "DPOConfig",
    train_dataset: "Dataset",
    callbacks: Optional[List] = None,
    **kwargs
) -> "DPOTrainer":
    """Create DPOTrainer with simplified model architecture for better compatibility.
    
    This factory function creates a DPOTrainer using regular AutoModelForCausalLM models,
    which are much more compatible than the complex AutoModelForCausalLMWithValueHead
    required by PPOTrainer.
    
    Args:
        model_name: Name or path of the base model
        dpo_config: DPO configuration object
        train_dataset: Training dataset (should have 'prompt', 'chosen', 'rejected' columns)
        callbacks: Optional list of callbacks
        **kwargs: Additional arguments passed to DPOTrainer
        
    Returns:
        Configured DPOTrainer instance
        
    Raises:
        ImportError: If TRL is not available
        ValueError: If required parameters are missing
    """
    if not TRL_AVAILABLE:
        raise ImportError("TRL is required for this function. Install with: pip install trl")
    
    if DPOConfig is None or DPOTrainer is None:
        raise ImportError("TRL DPOTrainer components not available. Check TRL installation.")
    
    # Validate required parameters
    if not isinstance(dpo_config, DPOConfig):
        raise ValueError("dpo_config must be a DPOConfig instance")
    
    if train_dataset is None:
        raise ValueError("train_dataset is required")
    
    # Check dataset format
    required_columns = ['prompt', 'chosen', 'rejected']
    missing_columns = [col for col in required_columns if col not in train_dataset.column_names]
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {missing_columns}")
    
    # Prepare models using simplified architecture
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create models (much simpler than PPO)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Fix generation config
        model = fix_generation_config(model, tokenizer)
        ref_model = fix_generation_config(ref_model, tokenizer)
        
    except Exception as e:
        raise RuntimeError(f"Failed to prepare models for DPO: {e}") from e
    
    # Create DPOTrainer (much simpler than PPOTrainer)
    try:
        return DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            callbacks=callbacks or [],
            **kwargs
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create DPOTrainer: {e}") from e
