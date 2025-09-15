"""Utility functions for TRL integration."""

from __future__ import annotations

import copy
from typing import Optional, Tuple, Union

import torch
from packaging import version
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

try:
    from trl import AutoModelForCausalLMWithValueHead
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    AutoModelForCausalLMWithValueHead = None


def fix_generation_config(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    generation_config: Optional[GenerationConfig] = None,
) -> PreTrainedModel:
    """Ensure a model exposes a usable :class:`~transformers.GenerationConfig`.

    Newer versions of TRL and Transformers expect policy models to always expose a
    ``generation_config`` with sensible defaults. Older checkpoints – especially
    those loaded through TRL wrappers – sometimes omit this attribute which causes
    the PPO trainer to crash when it prepares stopping criteria.  This helper
    attaches a generation configuration and mirrors the ``base_model_prefix`` so
    downstream utilities can locate the underlying backbone module.

    Args:
        model: Model to update. Any :class:`~transformers.PreTrainedModel` works.
        tokenizer: Tokenizer associated with the model.
        generation_config: Optional configuration to attach.  When ``None`` a
            conservative default is built from the tokenizer.

    Returns:
        The provided ``model`` for fluent-style usage.
    """

    if not hasattr(model, "config"):
        raise AttributeError("Model is missing a Transformers configuration and cannot expose generation settings.")

    if generation_config is None:
        generation_config = GenerationConfig(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=getattr(tokenizer, "bos_token_id", None),
            max_length=min(getattr(tokenizer, "model_max_length", 1024) or 1024, 1024),
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
        )

    model.generation_config = generation_config

    # Mirror the backbone attribute when possible so PPOTrainer can reach the
    # base transformer module regardless of wrapper structure.
    if not hasattr(model, "base_model_prefix"):
        if hasattr(model, "pretrained_model") and hasattr(model.pretrained_model, "base_model_prefix"):
            model.base_model_prefix = model.pretrained_model.base_model_prefix
        else:
            model.base_model_prefix = _infer_base_model_prefix(getattr(model, "name_or_path", ""))

    if hasattr(model, "pretrained_model"):
        base_model_prefix = model.base_model_prefix
        if not hasattr(model, base_model_prefix) and hasattr(model.pretrained_model, base_model_prefix):
            setattr(model, base_model_prefix, getattr(model.pretrained_model, base_model_prefix))

        if not hasattr(model, "is_gradient_checkpointing") and hasattr(
            model.pretrained_model, "is_gradient_checkpointing"
        ):
            model.is_gradient_checkpointing = model.pretrained_model.is_gradient_checkpointing
    elif not hasattr(model, "is_gradient_checkpointing"):
        model.is_gradient_checkpointing = False

    return model


def prepare_models_for_ppo(
    model_name: str,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    generation_config: Optional[GenerationConfig] = None,
    *,
    policy_model_kwargs: Optional[dict] = None,
    ref_model_kwargs: Optional[dict] = None,
    value_model_kwargs: Optional[dict] = None,
) -> Tuple[PreTrainedModel, PreTrainedModel, nn.Module, PreTrainedTokenizerBase]:
    """Instantiate a minimal set of models compatible with TRL's PPO trainer.

    The modern TRL API expects individual modules for the policy, reference,
    value, and reward networks.  This helper focuses on the policy, reference
    and value components.  The returned tokenizer can be passed directly to the
    trainer as the ``processing_class``.  See :func:`create_simple_reward_model`
    for a reward model that complements the output of this function.

    Args:
        model_name: Pretrained identifier or local path for the policy model.
        tokenizer: Optional tokenizer instance. When omitted a matching
            tokenizer is loaded and padded with ``eos_token`` if required.
        generation_config: Optional generation configuration applied to both the
            policy and reference models.
        policy_model_kwargs: Extra keyword arguments forwarded to
            :func:`AutoModelForCausalLM.from_pretrained` when instantiating the
            policy.
        ref_model_kwargs: Keyword arguments used for the reference model. When
            ``None`` they default to ``policy_model_kwargs``.
        value_model_kwargs: Additional options passed to
            :func:`create_simple_value_model`.

    Returns:
        ``(policy_model, ref_model, value_model, tokenizer)``

    Raises:
        ImportError: If TRL is not available in the runtime environment.
    """

    if not TRL_AVAILABLE:
        raise ImportError("TRL is required for this function. Install with: pip install trl")

    policy_kwargs = dict(policy_model_kwargs or {})
    reference_kwargs = dict(ref_model_kwargs or policy_kwargs)
    value_kwargs = dict(value_model_kwargs or {})

    tokenizer = _prepare_tokenizer(model_name, tokenizer)

    policy_model = AutoModelForCausalLM.from_pretrained(model_name, **policy_kwargs)
    policy_model = fix_generation_config(policy_model, tokenizer, generation_config)

    ref_model = AutoModelForCausalLM.from_pretrained(model_name, **reference_kwargs)
    ref_model = fix_generation_config(ref_model, tokenizer, generation_config)

    base_model_override = value_kwargs.pop("base_model", policy_model)
    value_model = create_simple_value_model(base_model_override, **value_kwargs)

    return policy_model, ref_model, value_model, tokenizer


def create_simple_value_model(
    base_model: Union[str, PreTrainedModel],
    *,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[Union[str, dict]] = None,
    head_init_std: float = 0.02,
    freeze_backbone: bool = False,
    **model_kwargs,
) -> nn.Module:
    """Create a lightweight value model compatible with TRL's PPO trainer.

    The value model exposes a ``score`` head that maps token hidden states to a
    single scalar per token – mirroring the behaviour expected by
    :func:`trl.trainer.utils.get_reward`.  By default the backbone shares the
    same architecture as the policy model.  Supplying an existing policy model
    instance avoids an additional download while still copying the weights to
    keep the modules independent during optimisation.

    Args:
        base_model: Either a model identifier understood by
            :func:`AutoModelForCausalLM.from_pretrained` or an instantiated
            :class:`~transformers.PreTrainedModel`.
        torch_dtype: Optional dtype override when loading from a model name.
        device_map: Device placement hint forwarded to ``from_pretrained`` when
            ``base_model`` is a string.
        head_init_std: Standard deviation used to initialise the linear value
            head.  Set to ``0`` to start from zero predictions.
        freeze_backbone: When ``True`` the backbone parameters are frozen.  This
            can be useful when the value network should only train the scoring
            head.
        **model_kwargs: Additional arguments forwarded to
            ``AutoModelForCausalLM.from_pretrained`` when ``base_model`` is a
            string identifier.

    Returns:
        ``nn.Module`` exposing ``base_model_prefix`` and ``score`` attributes.
    """

    backbone = _load_backbone(base_model, torch_dtype=torch_dtype, device_map=device_map, **model_kwargs)
    return _SimpleScoringModel(backbone, head_init_std=head_init_std, freeze_backbone=freeze_backbone)


def create_simple_reward_model(
    tokenizer: PreTrainedTokenizerBase,
    base_model: Optional[Union[str, PreTrainedModel]] = None,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[Union[str, dict]] = None,
    trainable: bool = False,
    head_init_std: float = 0.0,
    **model_kwargs,
) -> nn.Module:
    """Create a reward model compatible with :func:`trl.trainer.utils.get_reward`.

    The reward model mirrors the value model but defaults to a frozen backbone
    and zero-initialised head.  This produces deterministic rewards and avoids
    accidental training of the reward network when optimising the policy.

    Args:
        tokenizer: Tokenizer associated with the base model. Used to infer the
            default checkpoint when ``base_model`` is omitted.
        base_model: Optional identifier or instantiated model to base the reward
            network on.  When ``None`` the tokenizer's ``name_or_path`` is used.
        torch_dtype: Optional dtype override when loading from a model name.
        device_map: Device placement hint forwarded to ``from_pretrained`` when
            ``base_model`` is a string.
        trainable: Whether the returned module should require gradients.  The
            default (``False``) freezes all parameters.
        head_init_std: Standard deviation for the score head initialisation.
            The default ``0.0`` initialises the head to return zero rewards.
        **model_kwargs: Additional keyword arguments forwarded to
            ``AutoModelForCausalLM.from_pretrained`` when a model name is used.

    Returns:
        Frozen reward model suitable for PPO training.
    """

    model_identifier = base_model if base_model is not None else _resolve_model_name_from_tokenizer(tokenizer)
    backbone = _load_backbone(model_identifier, torch_dtype=torch_dtype, device_map=device_map, **model_kwargs)
    reward_model = _SimpleScoringModel(backbone, head_init_std=head_init_std, freeze_backbone=not trainable)
    if not trainable:
        reward_model.eval()
        for parameter in reward_model.parameters():
            parameter.requires_grad = False
    return reward_model


def _prepare_tokenizer(model_name: str, tokenizer: Optional[PreTrainedTokenizerBase]) -> PreTrainedTokenizerBase:
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError(
                "Tokenizer is missing a pad token and no suitable fallback (eos/unk) was found. "
                "Please provide a tokenizer with explicit padding support."
            )
    return tokenizer


def _resolve_model_name_from_tokenizer(tokenizer: PreTrainedTokenizerBase) -> str:
    for attribute in ("name_or_path", "_name_or_path"):
        value = getattr(tokenizer, attribute, None)
        if value:
            return value
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    value = init_kwargs.get("name_or_path")
    if value:
        return value
    raise ValueError(
        "Could not determine the base model name from the tokenizer. Please pass `base_model` explicitly."
    )


def _load_backbone(
    base_model: Union[str, PreTrainedModel],
    *,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[Union[str, dict]] = None,
    **model_kwargs,
) -> PreTrainedModel:
    if isinstance(base_model, PreTrainedModel):
        return copy.deepcopy(base_model)

    load_kwargs = dict(model_kwargs)
    if torch_dtype is not None:
        load_kwargs.setdefault("torch_dtype", torch_dtype)
    if device_map is not None:
        load_kwargs.setdefault("device_map", device_map)

    if not isinstance(base_model, str):
        raise TypeError("base_model must be a model name or a PreTrainedModel instance")

    return AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)


def _infer_hidden_size(config) -> int:
    for attribute in ("hidden_size", "n_embd", "d_model", "model_dim", "embed_dim", "word_embed_proj_dim"):
        value = getattr(config, attribute, None)
        if isinstance(value, int):
            return value
    hidden_sizes = getattr(config, "hidden_sizes", None)
    if isinstance(hidden_sizes, (list, tuple)) and hidden_sizes:
        return hidden_sizes[-1]
    raise ValueError("Could not infer hidden size from model configuration")


def _infer_base_model_prefix(model_name: str) -> str:
    lowered = (model_name or "").lower()
    if "llama" in lowered or "mistral" in lowered or "phi" in lowered:
        return "model"
    return "transformer"


def _init_linear_head(linear: nn.Linear, std: float) -> None:
    with torch.no_grad():
        if std > 0:
            linear.weight.normal_(mean=0.0, std=std)
        else:
            linear.weight.zero_()
        if linear.bias is not None:
            linear.bias.zero_()


class _SimpleScoringModel(nn.Module):
    """Minimal wrapper exposing ``score`` and ``base_model_prefix`` attributes."""

    def __init__(self, backbone: PreTrainedModel, *, head_init_std: float, freeze_backbone: bool) -> None:
        super().__init__()
        if not isinstance(backbone, PreTrainedModel):
            raise TypeError("backbone must be a Transformers PreTrainedModel instance")

        self.config = backbone.config
        if hasattr(self.config, "output_hidden_states"):
            self.config.output_hidden_states = True

        prefix = getattr(backbone, "base_model_prefix", None)
        module = None
        if prefix and hasattr(backbone, prefix):
            module = getattr(backbone, prefix)
        else:
            for candidate in ("model", "transformer", backbone.__class__.__name__.lower()):
                if hasattr(backbone, candidate):
                    prefix = candidate
                    module = getattr(backbone, candidate)
                    break
        if module is None:
            prefix = "model"
            module = backbone

        self.base_model_prefix = prefix
        setattr(self, prefix, module)

        if freeze_backbone:
            for parameter in module.parameters():
                parameter.requires_grad = False

        hidden_size = _infer_hidden_size(module.config if hasattr(module, "config") else self.config)
        self.score = nn.Linear(hidden_size, 1)
        _init_linear_head(self.score, std=head_init_std)

    def forward(self, **kwargs):
        backbone = getattr(self, self.base_model_prefix)
        outputs = backbone(
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            **kwargs,
        )
        return self.score(outputs.hidden_states[-1])


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
                "TRL 0.23.0+ expects explicit policy, value, and reward modules. Use the RLDK helpers "
                "to automatically attach generation configs and scoring heads."
            )
            recommendations.append(
                "Use prepare_models_for_ppo() and create_simple_reward_model() to avoid manual patches"
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
    policy_model: nn.Module,
    ref_model: Optional[nn.Module],
    reward_model: Optional[nn.Module],
    tokenizer: PreTrainedTokenizerBase,
    value_model: Optional[nn.Module] = None,
) -> dict:
    """Validate a PPO configuration for common pitfalls."""

    issues: list[str] = []
    warnings: list[str] = []

    for name, candidate in (("policy_model", policy_model), ("ref_model", ref_model)):
        if candidate is None:
            continue
        if not hasattr(candidate, "generation_config") or getattr(candidate, "generation_config") is None:
            issues.append(f"{name} is missing a valid generation_config")

    if not hasattr(tokenizer, "eos_token_id") or tokenizer.eos_token_id is None:
        issues.append("Tokenizer missing eos_token_id")
    if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
        warnings.append("Tokenizer missing pad_token_id")

    for name, candidate in (
        ("value_model", value_model),
        ("reward_model", reward_model),
    ):
        if candidate is None:
            continue
        if not hasattr(candidate, "base_model_prefix"):
            issues.append(f"{name} missing base_model_prefix attribute")
        if not hasattr(candidate, "score"):
            issues.append(f"{name} missing score head compatible with TRL")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "recommendations": [
            "Ensure policy/ref models expose generation_config",
            "Ensure value and reward models define base_model_prefix and score",
            "Check tokenizer has required token IDs",
        ] if issues else [],
    }
