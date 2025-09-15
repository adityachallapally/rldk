"""Utility functions for TRL integration."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from packaging import version
from transformers import AutoTokenizer, GenerationConfig

try:
    from trl import AutoModelForCausalLMWithValueHead
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    AutoModelForCausalLMWithValueHead = None

if TYPE_CHECKING:  # pragma: no cover - imports for type checkers only
    from datasets import Dataset
    from trl import PPOConfig, PPOTrainer


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
) -> Tuple[
    "AutoModelForCausalLMWithValueHead",
    "AutoModelForCausalLMWithValueHead",
    "AutoModelForCausalLMWithValueHead",
    "AutoModelForCausalLMWithValueHead",
    AutoTokenizer,
]:
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


def _ensure_generation_config(
    candidate: Optional["AutoModelForCausalLMWithValueHead"],
    tokenizer: Optional[AutoTokenizer],
    generation_config: Optional[GenerationConfig],
) -> Optional["AutoModelForCausalLMWithValueHead"]:
    """Apply :func:`fix_generation_config` when possible."""

    if candidate is None or not isinstance(candidate, AutoModelForCausalLMWithValueHead):
        return candidate

    if tokenizer is None:
        raise ValueError(
            "Tokenizer is required when providing custom TRL models. "
            "Pass a tokenizer or allow prepare_models_for_ppo() to create one."
        )

    return fix_generation_config(candidate, tokenizer, generation_config)


def create_ppo_trainer(
    model_name: str,
    ppo_config: "PPOConfig",
    train_dataset: "Dataset",
    callbacks: Optional[List[Any]] = None,
    *,
    tokenizer: Optional[AutoTokenizer] = None,
    generation_config: Optional[GenerationConfig] = None,
    model: Optional["AutoModelForCausalLMWithValueHead"] = None,
    ref_model: Optional["AutoModelForCausalLMWithValueHead"] = None,
    reward_model: Optional[Any] = None,
    value_model: Optional["AutoModelForCausalLMWithValueHead"] = None,
    validate_setup: bool = True,
    **ppo_kwargs: Any,
) -> "PPOTrainer":
    """Create a :class:`trl.PPOTrainer` with version-aware defaults.

    This factory wraps TRL's ``PPOTrainer`` to ensure all required models are
    prepared consistently across TRL versions. When models or tokenizer are not
    provided they are instantiated via :func:`prepare_models_for_ppo`.

    Args:
        model_name: Base model identifier used for automatic loading.
        ppo_config: PPO configuration object.
        train_dataset: Training dataset for PPO.
        callbacks: Optional list of callbacks passed to the trainer.
        tokenizer: Optional tokenizer to reuse.
        generation_config: Optional generation configuration shared across
            prepared models.
        model: Optional policy model. When provided its ``generation_config`` is
            validated/fixed.
        ref_model: Optional reference model. Same handling as ``model``.
        reward_model: Optional reward model. When not provided a default value
            head model is created.
        value_model: Optional value model. When not provided a default value
            head model is created.
        validate_setup: Whether to validate the prepared components before
            creating the trainer.
        **ppo_kwargs: Additional keyword arguments forwarded to ``PPOTrainer``.

    Returns:
        Configured ``PPOTrainer`` instance.

    Raises:
        ImportError: If TRL is not installed.
        ValueError: If required components are missing or misconfigured.
    """

    if not TRL_AVAILABLE:
        raise ImportError("TRL is required for this function. Install with: pip install trl")

    # Local import to avoid import errors when TRL is missing at module import time
    from trl import PPOTrainer  # type: ignore

    callbacks_list: List[Any] = list(callbacks) if callbacks else []

    components_missing = any(
        component is None for component in (model, ref_model, reward_model, value_model)
    ) or tokenizer is None

    if components_missing:
        (
            default_model,
            default_ref_model,
            default_reward_model,
            default_value_model,
            default_tokenizer,
        ) = prepare_models_for_ppo(
            model_name,
            tokenizer=tokenizer,
            generation_config=generation_config,
        )

        model = model or default_model
        ref_model = ref_model or default_ref_model
        reward_model = reward_model or default_reward_model
        value_model = value_model or default_value_model
        tokenizer = tokenizer or default_tokenizer
    else:
        model = _ensure_generation_config(model, tokenizer, generation_config)
        ref_model = _ensure_generation_config(ref_model, tokenizer, generation_config)
        if isinstance(reward_model, AutoModelForCausalLMWithValueHead):
            reward_model = _ensure_generation_config(reward_model, tokenizer, generation_config)
        value_model = _ensure_generation_config(value_model, tokenizer, generation_config)

    if tokenizer is None:
        raise ValueError("Tokenizer could not be prepared for PPO training")

    if validate_setup:
        validation = validate_ppo_setup(model, ref_model, reward_model, value_model, tokenizer)
        if not validation["valid"]:
            issues = "; ".join(validation["issues"])
            raise ValueError(f"Invalid PPO setup detected: {issues}")

    compatibility = check_trl_compatibility()
    trl_version: Optional[version.Version]
    try:
        trl_version = (
            version.parse(compatibility["version"]) if compatibility.get("version") else None
        )
    except Exception:
        trl_version = None

    use_new_api = trl_version is None or trl_version >= version.parse("0.23.0")

    trainer_kwargs: Dict[str, Any] = {
        "args": ppo_config,
        "model": model,
        "ref_model": ref_model,
        "processing_class": tokenizer,
        "train_dataset": train_dataset,
        "callbacks": callbacks_list,
    }
    trainer_kwargs.update(ppo_kwargs)

    if use_new_api:
        trainer_kwargs.update({
            "reward_model": reward_model,
            "value_model": value_model,
        })

    trainer = PPOTrainer(**trainer_kwargs)

    # Ensure reward/value models remain accessible even for legacy TRL versions
    if not use_new_api:
        if reward_model is not None and not hasattr(trainer, "reward_model"):
            setattr(trainer, "reward_model", reward_model)
        if value_model is not None and not hasattr(trainer, "value_model"):
            setattr(trainer, "value_model", value_model)

    return trainer


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
            "recommendations": ["Install TRL: pip install trl>=0.23.0"],
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

        if trl_version_obj < version.parse("0.23.0"):
            warnings_list.append("TRL version is quite old. Consider upgrading to 0.23.0+")
            recommendations.append("Upgrade TRL: pip install --upgrade trl>=0.23.0")

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
    model: Optional["AutoModelForCausalLMWithValueHead"],
    ref_model: Optional["AutoModelForCausalLMWithValueHead"],
    reward_model: Optional[Any],
    value_model: Optional["AutoModelForCausalLMWithValueHead"],
    tokenizer: Optional[AutoTokenizer]
) -> Dict[str, Any]:
    """Validate PPO setup for common issues.

    Args:
        model: Main PPO model (policy model)
        ref_model: Reference model
        reward_model: Reward model
        value_model: Value function model
        tokenizer: Tokenizer used for the models

    Returns:
        Dictionary with validation results
    """
    issues: List[str] = []
    warnings: List[str] = []

    # Helper to validate TRL value-head models
    def _check_value_head(name: str, model_obj: Optional[Any]) -> None:
        if model_obj is None:
            issues.append(f"{name} is missing")
            return

        if not isinstance(model_obj, AutoModelForCausalLMWithValueHead):
            issues.append(f"{name} is not an AutoModelForCausalLMWithValueHead instance")
            return

        if not hasattr(model_obj, "generation_config"):
            issues.append(f"{name} missing generation_config attribute")
        elif model_obj.generation_config is None:
            warnings.append(f"{name} has None generation_config")

    _check_value_head("model", model)
    _check_value_head("ref_model", ref_model)
    _check_value_head("value_model", value_model)

    if reward_model is None:
        issues.append("reward_model is missing")
    elif isinstance(reward_model, AutoModelForCausalLMWithValueHead):
        if not hasattr(reward_model, "generation_config"):
            issues.append("reward_model missing generation_config attribute")
        elif reward_model.generation_config is None:
            warnings.append("reward_model has None generation_config")
    elif not hasattr(reward_model, "forward"):
        warnings.append("reward_model does not define a forward method")

    if tokenizer is None:
        issues.append("Tokenizer is missing")
    else:
        if not hasattr(tokenizer, "eos_token_id") or tokenizer.eos_token_id is None:
            issues.append("Tokenizer missing eos_token_id")
        if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
            warnings.append("Tokenizer missing pad_token_id")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "recommendations": [
            "Use fix_generation_config() to fix generation_config issues",
            "Ensure PPO components include policy, reference, reward, and value models",
            "Verify tokenizer has required token IDs",
        ] if issues else [],
    }
