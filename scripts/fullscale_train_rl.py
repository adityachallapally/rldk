#!/usr/bin/env python3
"""Fullscale on-policy RL training script for GPT-2 medium on CPU.

This script is intentionally heavyweight compared to the quick demos that ship with
RLDK.  It exercises a PPO-lite training loop that uses a frozen GPT-2 backbone with
an adapter head trained via REINFORCE-style updates.  Metrics are emitted through the
canonical :class:`rldk.emit.EventWriter` interface so downstream tooling can ingest the
run and normalize it into the ``TrainingMetrics`` table.  The defaults are tuned so the
script runs on CPU within a few hours while still producing a rich event stream.  Use
``--simulate-anomalies`` to opt back into the older scripted collapse/spike perturbations
if you need deterministic alert demonstrations.

The implementation focuses on debuggability rather than raw throughput.  Extensive
logging, deterministic seeding, and reward decomposition make it easier for RLDK
pipelines (monitoring, reward health, diffing, determinism checks, and trust cards)
to surface actionable insights.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from rldk.emit import EventWriter


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - optional GPU path
        torch.cuda.manual_seed_all(seed)


@dataclass
class SyntheticPrompt:
    """Container for a synthetic prompt specification."""

    text: str
    domain: str
    keywords: Tuple[str, ...]
    fragility: float


def build_prompt_corpus(size: int) -> List[SyntheticPrompt]:
    """Create a diverse synthetic corpus spanning several domains.

    The corpus mixes math proofs, coding interviews, literary vignettes, science
    explainers, and speculative scenarios.  Each prompt encodes domain-specific
    keywords so the reward function can evaluate task fidelity.
    """

    base_templates: Sequence[Tuple[str, Sequence[str]]] = (
        (
            "Explain the theorem of {topic} and provide an intuitive example.",
            [" theorem", "example", "intuition"],
        ),
        (
            "Write a Python function that computes {topic} and describe its complexity.",
            ["def", "return", "complexity"],
        ),
        (
            "Narrate a short story about {topic} in a cyberpunk city.",
            ["city", "neon", "shadow"],
        ),
        (
            "Provide a scientific report discussing {topic} with hypotheses and conclusions.",
            ["hypothesis", "experiment", "conclusion"],
        ),
        (
            "Compose an imaginative dialogue where two characters debate {topic} politely.",
            ["dialogue", "character", "debate"],
        ),
        (
            "Create pseudocode for {topic} and mention edge cases.",
            ["loop", "case", "handle"],
        ),
        (
            "Summarize research on {topic} and cite at least two findings.",
            ["study", "finding", "evidence"],
        ),
        (
            "Draft release notes for a software update that focuses on {topic}.",
            ["release", "update", "issue"],
        ),
    )

    topics: Sequence[Tuple[str, str]] = (
        ("modular arithmetic", "math"),
        ("quantum entanglement", "science"),
        ("graph traversal", "code"),
        ("recursive poetry", "literature"),
        ("ethical AI", "policy"),
        ("climate adaptation", "science"),
        ("distributed consensus", "systems"),
        ("game balancing", "design"),
        ("symbolic reasoning", "math"),
        ("dragon folklore", "myth"),
        ("topological art", "art"),
        ("ocean exploration", "science"),
        ("tournament scheduling", "operations"),
        ("probabilistic robotics", "robotics"),
        ("ancient astronomy", "history"),
    )

    prompts: List[SyntheticPrompt] = []
    rng = random.Random(17)
    for idx in range(size):
        template, keyword_seed = base_templates[idx % len(base_templates)]
        topic, domain = topics[idx % len(topics)]
        variation = rng.choice(
            [
                "with practical insights",
                "emphasizing trade-offs",
                "and outline testing strategy",
                "highlighting open questions",
                "alongside real-world analogies",
                "and cover failure cases",
                "giving historical context",
                "and explore counterfactuals",
            ]
        )
        prompt = template.format(topic=topic) + f" {variation}."
        keywords = tuple(keyword_seed)
        fragility = 0.15 + 0.7 * rng.random()
        prompts.append(SyntheticPrompt(prompt, domain, keywords, fragility))
    return prompts


class AdapterPolicy(nn.Module):
    """Policy head that learns a residual over a frozen GPT-2 backbone."""

    def __init__(self, backbone: GPT2LMHeadModel) -> None:
        super().__init__()
        hidden_size = backbone.config.n_embd
        vocab_size = backbone.config.vocab_size
        self.backbone = backbone
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, vocab_size, bias=False),
        )
        nn.init.zeros_(self.adapter[-1].weight)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone.transformer(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        hidden_states = outputs.last_hidden_state
        base_logits = self.backbone.lm_head(hidden_states)
        adapter_logits = self.adapter(hidden_states)
        return base_logits + adapter_logits


@dataclass
class EpisodeStats:
    text: str
    total_reward: float
    task_reward: float
    aux_reward: float
    kl_penalty: float
    kl_value: float
    entropy: float
    log_prob_sum: torch.Tensor
    policy_logits: torch.Tensor
    generated_ids: torch.Tensor
    prompt_len: int


def _sample_response(
    policy: AdapterPolicy,
    tokenizer: GPT2TokenizerFast,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> Tuple[str, torch.Tensor, int]:
    """Sample a continuation using the current policy (no gradients)."""

    policy.eval()
    device = next(policy.parameters()).device
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    generated: List[int] = []
    past_key_values = None
    current_input = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = policy.backbone.transformer(
                input_ids=current_input,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )
            hidden = outputs.last_hidden_state[:, -1:, :]
            past_key_values = outputs.past_key_values
            base_logits = policy.backbone.lm_head(hidden)
            adapter_logits = policy.adapter(hidden)
            logits = (base_logits + adapter_logits) / temperature
            probs = torch.softmax(logits, dim=-1)
            token = torch.distributions.Categorical(probs.squeeze(0)).sample()
            generated.append(token.item())
            current_input = token.view(1, 1)
            attention_mask = torch.ones_like(current_input)
            if token.item() == tokenizer.eos_token_id:
                break

    if generated:
        continuation_ids = torch.tensor(generated, dtype=torch.long, device=device).view(1, -1)
        full_ids = torch.cat([input_ids, continuation_ids], dim=-1)
    else:
        continuation_ids = torch.empty((1, 0), dtype=torch.long, device=device)
        full_ids = input_ids

    decoded = tokenizer.decode(full_ids[0], skip_special_tokens=True)
    policy.train()
    return decoded, full_ids, input_ids.shape[1]


def _compute_rewards(
    text: str,
    prompt: SyntheticPrompt,
    base_reward: float,
    aux_weight: float,
    noise_weight: float,
) -> Tuple[float, float, float]:
    lower = text.lower()
    keyword_hits = sum(1 for keyword in prompt.keywords if keyword in lower)
    task_reward = keyword_hits / max(len(prompt.keywords), 1)
    words = text.split()
    diversity = len(set(words)) / max(len(words), 1)
    length_bonus = min(len(words) / 80.0, 1.0)
    aux_reward = aux_weight * (0.5 * diversity + 0.5 * length_bonus)

    fragile_signal = 0.0
    if "???" in text or "[fragile]" in text:
        fragile_signal += 0.6
    if any(symbol in text for symbol in {";", "{", "}"}):
        fragile_signal += 0.25
    if random.random() < prompt.fragility:
        fragile_signal -= 0.2 + 0.3 * random.random()

    total_reward = base_reward * task_reward + aux_reward + noise_weight * fragile_signal
    return total_reward, task_reward, aux_reward + noise_weight * fragile_signal


def _compute_episode(
    policy: AdapterPolicy,
    reference: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    prompt: SyntheticPrompt,
    max_new_tokens: int,
    temperature: float,
    kl_coeff: float,
    aux_weight: float,
    noise_weight: float,
) -> EpisodeStats:
    text, full_ids, prompt_len = _sample_response(
        policy, tokenizer, prompt.text, max_new_tokens, temperature
    )
    device = full_ids.device
    attention_mask = torch.ones_like(full_ids)

    logits = policy(full_ids, attention_mask)
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    actions = full_ids[:, 1:]
    generated_slice = slice(prompt_len - 1, actions.shape[1])
    generated_ids = actions[:, generated_slice]
    selected_log_probs = log_probs[:, generated_slice, :].gather(2, generated_ids.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        reference_outputs = reference(
            input_ids=full_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        reference_log_probs = torch.log_softmax(reference_outputs.logits[:, :-1, :], dim=-1)

    policy_probs = torch.softmax(logits[:, :-1, :], dim=-1)
    kl_all = policy_probs * (log_probs - reference_log_probs)
    kl_values = kl_all[:, generated_slice, :].sum(dim=-1)
    kl_mean = kl_values.mean()

    entropy = -(policy_probs[:, generated_slice, :] * log_probs[:, generated_slice, :]).sum(dim=-1).mean()

    total_reward, task_reward, aux_reward = _compute_rewards(
        text, prompt, base_reward=1.0, aux_weight=aux_weight, noise_weight=noise_weight
    )
    total_reward = float(total_reward - kl_coeff * float(kl_mean))

    log_prob_sum = selected_log_probs.sum()
    return EpisodeStats(
        text=text,
        total_reward=total_reward,
        task_reward=task_reward,
        aux_reward=aux_reward,
        kl_penalty=float(kl_coeff * float(kl_mean)),
        kl_value=float(kl_mean),
        entropy=float(entropy),
        log_prob_sum=log_prob_sum,
        policy_logits=logits[:, generated_slice, :],
        generated_ids=generated_ids,
        prompt_len=prompt_len,
    )


@dataclass
class RunningMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> Tuple[float, float]:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        variance = self.m2 / (self.count - 1) if self.count > 1 else 0.0
        return self.mean, math.sqrt(max(variance, 0.0))


def determine_event_path(outdir: Path, run_id: str, override: Optional[str]) -> Path:
    if override:
        return Path(override)
    env_path = os.environ.get("RLDK_METRICS_PATH")
    if env_path:
        return Path(env_path)
    return outdir / f"{run_id}.jsonl"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fullscale RL acceptance training run")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--kl-coeff", type=float, default=0.08)
    parser.add_argument("--aux-weight", type=float, default=0.35)
    parser.add_argument("--noise-weight", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--dataset-size", type=int, default=4096)
    parser.add_argument("--model-name", type=str, default="gpt2-medium")
    parser.add_argument("--outdir", type=Path, default=Path("artifacts/fullscale"))
    parser.add_argument("--run-id", type=str, default="run")
    parser.add_argument("--ema-beta", type=float, default=0.04)
    parser.add_argument("--max-grad-norm", type=float, default=2.5)
    parser.add_argument("--disable-updates", action="store_true", help="Skip optimizer updates for baseline runs")
    parser.add_argument("--override-event-path", type=str, default=None)
    parser.add_argument("--print-interval", type=int, default=10)
    parser.add_argument(
        "--simulate-anomalies",
        action="store_true",
        help="Re-enable scripted collapse/spike behavior for alert demos",
    )

    args = parser.parse_args(argv)

    _set_seed(args.seed)

    args.outdir.mkdir(parents=True, exist_ok=True)
    config_path = args.outdir / f"{args.run_id}_config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "kl_coeff": args.kl_coeff,
                "aux_weight": args.aux_weight,
                "noise_weight": args.noise_weight,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "dataset_size": args.dataset_size,
                "model_name": args.model_name,
                "disable_updates": args.disable_updates,
                "simulate_anomalies": args.simulate_anomalies,
            },
            indent=2,
        )
    )

    print(f"[fullscale] loading tokenizer and models for {args.model_name}")
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    backbone = GPT2LMHeadModel.from_pretrained(args.model_name)
    reference = GPT2LMHeadModel.from_pretrained(args.model_name)
    device = torch.device("cpu")
    backbone.to(device)
    reference.to(device)
    reference.eval()

    for param in backbone.parameters():
        param.requires_grad_(False)
    for param in reference.parameters():
        param.requires_grad_(False)

    policy = AdapterPolicy(backbone).to(device)
    optimizer = torch.optim.Adam(policy.adapter.parameters(), lr=args.learning_rate)

    corpus = build_prompt_corpus(args.dataset_size)
    print(f"[fullscale] synthetic corpus contains {len(corpus)} prompts")

    event_path = determine_event_path(args.outdir, args.run_id, args.override_event_path)
    event_writer = EventWriter(event_path)
    print(f"[fullscale] writing metrics to {event_path}")

    baseline = 0.0
    ema_reward = 0.0
    ema_kl = 0.0
    reward_moments = RunningMoments()

    start_time = time.time()

    for step in range(1, args.max_steps + 1):
        step_start = time.time()
        optimizer.zero_grad()
        batch_prompts = random.sample(corpus, args.batch_size)
        episodes = [
            _compute_episode(
                policy,
                reference,
                tokenizer,
                prompt,
                args.max_new_tokens,
                args.temperature,
                args.kl_coeff,
                args.aux_weight,
                args.noise_weight,
            )
            for prompt in batch_prompts
        ]

        total_reward = float(np.mean([ep.total_reward for ep in episodes]))
        task_reward = float(np.mean([ep.task_reward for ep in episodes]))
        aux_reward = float(np.mean([ep.aux_reward for ep in episodes]))
        kl_value = float(np.mean([ep.kl_value for ep in episodes]))
        entropy = float(np.mean([ep.entropy for ep in episodes]))
        kl_penalty = float(np.mean([ep.kl_penalty for ep in episodes]))

        mean_reward, std_reward = reward_moments.update(total_reward)
        ema_reward = (1 - args.ema_beta) * ema_reward + args.ema_beta * total_reward
        ema_kl = (1 - args.ema_beta) * ema_kl + args.ema_beta * kl_value

        advantages = []
        for ep in episodes:
            baseline = 0.9 * baseline + 0.1 * ep.total_reward
            advantages.append(ep.total_reward - baseline)
        loss = -sum(adv * ep.log_prob_sum for adv, ep in zip(advantages, episodes))
        loss = loss / max(len(episodes), 1)

        adv_mean = float(np.mean(advantages)) if advantages else 0.0
        adv_std = float(np.std(advantages)) if len(advantages) > 1 else 0.0

        grad_norm = 0.0
        if not args.disable_updates:
            loss.backward()
            grad_norm = float(clip_grad_norm_(policy.adapter.parameters(), args.max_grad_norm))
            optimizer.step()
        else:
            grad_norm = float(
                math.sqrt(
                    sum(torch.sum(param.detach() ** 2).item() for param in policy.adapter.parameters())
                )
            )
        grad_norm_clipped = min(grad_norm, args.max_grad_norm)

        logged_reward = total_reward
        logged_kl = kl_value
        collapse_active = 0.0
        spike_active = 0.0
        if args.simulate_anomalies:
            collapse_active = 1.0 if step % 55 in range(28, 52) else 0.0
            spike_active = 1.0 if step % 60 == 0 else 0.0
            if collapse_active:
                logged_reward -= 0.35
            if spike_active:
                logged_kl += 0.45

        wall_time = time.time() - step_start
        ema_reward_gap = total_reward - ema_reward
        baseline_gap = total_reward - baseline
        ema_kl_gap = kl_value - ema_kl
        advantage_norm = adv_mean / (abs(adv_std) + 1e-6)
        ratio_reward_kl = logged_reward / (abs(logged_kl) + 1e-6)

        metrics = {
            "reward_total": logged_reward,
            "reward_total_raw": total_reward,
            "reward_mean": mean_reward,
            "reward_std": std_reward,
            "reward_task": task_reward,
            "reward_aux": aux_reward,
            "reward_ema": ema_reward,
            "kl": logged_kl,
            "kl_penalty": kl_penalty,
            "kl_ema": ema_kl,
            "loss": float(loss.detach().cpu()),
            "grad_norm": grad_norm,
            "entropy": entropy,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "baseline_value": baseline,
            "advantage_mean": adv_mean,
            "advantage_std": adv_std,
            "collapse_window_active": collapse_active,
            "kl_spike_window": spike_active,
            "reward_total_abs": abs(logged_reward),
            "reward_total_sqr": logged_reward * logged_reward,
            "kl_raw": kl_value,
            "entropy_bits": entropy / math.log(2.0),
            "ema_reward_gap": ema_reward_gap,
            "baseline_gap": baseline_gap,
            "advantage_norm": advantage_norm,
            "step_wall_time": wall_time,
            "ema_kl_gap": ema_kl_gap,
            "reward_pressure_flag": 1.0 if logged_reward < 0.15 else 0.0,
            "kl_deviation_flag": 1.0 if logged_kl > 0.3 else 0.0,
            "reward_to_kl_ratio": ratio_reward_kl,
            "grad_norm_clipped": grad_norm_clipped,
        }

        for name, value in metrics.items():
            event_writer.log(
                step=step,
                name=name,
                value=float(value),
                run_id=args.run_id,
                meta={
                    "model": args.model_name,
                    "disable_updates": args.disable_updates,
                    "batch_size": args.batch_size,
                },
            )

        if step % args.print_interval == 0 or step == 1:
            print(
                f"[fullscale] step {step:04d} | reward={logged_reward:.3f} mean={mean_reward:.3f} "
                f"kl={logged_kl:.3f} grad={grad_norm:.3f}"
            )
            sys.stdout.flush()

    duration = time.time() - start_time
    event_writer.close()
    print(
        f"[fullscale] completed {args.max_steps} steps in {duration/60:.2f} minutes; "
        f"metrics located at {event_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
