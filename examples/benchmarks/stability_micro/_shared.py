"""Shared utilities for the RLDK stability micro-benchmark."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List


GSM8K_TARGET_SAMPLES = 500
HUMAN_EVAL_DEFAULT_STEPS = 60

HUMAN_EVAL_TASKS: List[Dict[str, str]] = [
    {
        "task_id": "he_add_two",
        "prompt": """
### HumanEval-lite Problem
You are given two integers ``a`` and ``b``.
Return their sum.
""".strip(),
        "reference": "def solve(a, b):\n    return a + b",
    },
    {
        "task_id": "he_multiply_three",
        "prompt": """
### HumanEval-lite Problem
Multiply three integers ``a``, ``b`` and ``c``.
""".strip(),
        "reference": "def solve(a, b, c):\n    return a * b * c",
    },
    {
        "task_id": "he_abs_diff",
        "prompt": """
### HumanEval-lite Problem
Return the absolute difference between two numbers.
""".strip(),
        "reference": "def solve(a, b):\n    return abs(a - b)",
    },
    {
        "task_id": "he_is_even",
        "prompt": """
### HumanEval-lite Problem
Return True when ``n`` is even, otherwise False.
""".strip(),
        "reference": "def solve(n):\n    return n % 2 == 0",
    },
    {
        "task_id": "he_factorial",
        "prompt": """
### HumanEval-lite Problem
Compute ``n!`` for a non-negative integer ``n``.
""".strip(),
        "reference": "def solve(n):\n    if n <= 1:\n        return 1\n    return n * solve(n - 1)",
    },
    {
        "task_id": "he_reverse_string",
        "prompt": """
### HumanEval-lite Problem
Return the reverse of the provided string ``s``.
""".strip(),
        "reference": "def solve(s):\n    return s[::-1]",
    },
    {
        "task_id": "he_palindrome",
        "prompt": """
### HumanEval-lite Problem
Return True when ``s`` is a palindrome.
""".strip(),
        "reference": "def solve(s):\n    cleaned = ''.join(ch for ch in s.lower() if ch.isalnum())\n    return cleaned == cleaned[::-1]",
    },
    {
        "task_id": "he_fibonacci",
        "prompt": """
### HumanEval-lite Problem
Return the ``n``th Fibonacci number where ``F(0)=0`` and ``F(1)=1``.
""".strip(),
        "reference": "def solve(n):\n    if n < 2:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
    },
    {
        "task_id": "he_sorted_unique",
        "prompt": """
### HumanEval-lite Problem
Return the sorted unique values from the list ``xs``.
""".strip(),
        "reference": "def solve(xs):\n    return sorted(set(xs))",
    },
    {
        "task_id": "he_count_vowels",
        "prompt": """
### HumanEval-lite Problem
Count the number of vowels in the string ``s``.
""".strip(),
        "reference": "def solve(s):\n    return sum(ch.lower() in 'aeiou' for ch in s)",
    },
    {
        "task_id": "he_anagrams",
        "prompt": """
### HumanEval-lite Problem
Return True when the two strings ``a`` and ``b`` are anagrams.
""".strip(),
        "reference": "def solve(a, b):\n    return sorted(a) == sorted(b)",
    },
    {
        "task_id": "he_list_chunks",
        "prompt": """
### HumanEval-lite Problem
Split ``xs`` into chunks of size ``k``.
""".strip(),
        "reference": "def solve(xs, k):\n    return [xs[i:i+k] for i in range(0, len(xs), k)]",
    },
    {
        "task_id": "he_max_pair",
        "prompt": """
### HumanEval-lite Problem
Return the maximum sum of any two numbers from ``xs``.
""".strip(),
        "reference": "def solve(xs):\n    ys = sorted(xs, reverse=True)\n    return ys[0] + ys[1]",
    },
    {
        "task_id": "he_safe_div",
        "prompt": """
### HumanEval-lite Problem
Divide ``a`` by ``b`` but return 0 when ``b`` is 0.
""".strip(),
        "reference": "def solve(a, b):\n    return 0 if b == 0 else a / b",
    },
    {
        "task_id": "he_digit_sum",
        "prompt": """
### HumanEval-lite Problem
Return the sum of digits of ``n``.
""".strip(),
        "reference": "def solve(n):\n    return sum(int(ch) for ch in str(abs(n)))",
    },
]

MODEL_BIASES: Dict[str, float] = {
    "sshleifer/tiny-gpt2": 0.05,
    "hf-internal-testing/tiny-random-gpt2": -0.02,
}


@dataclass
class TrainingSample:
    step: int
    output: str
    prompt: str
    reference_answer: str
    model_answer: str
    reward: float
    kl: float
    entropy: float
    grad_norm: float
    kl_coef: float
    length_bias_score: float
    length_reward_correlation_abs: float
    length_reward_spearman_abs: float
    response_quality: float
    safety_score: float
    toxicity_score: float
    bias_score: float
    adversarial_score: float
    human_preference: float
    model_name: str
    algorithm: str
    task: str
    seed: int

    def to_json(self) -> Dict[str, object]:
        return {
            "step": self.step,
            "output": self.output,
            "prompt": self.prompt,
            "reference_answer": self.reference_answer,
            "model_answer": self.model_answer,
            "reward": self.reward,
            "kl": self.kl,
            "entropy": self.entropy,
            "grad_norm": self.grad_norm,
            "kl_coef": self.kl_coef,
            "length_bias_score": self.length_bias_score,
            "length_reward_correlation_abs": self.length_reward_correlation_abs,
            "length_reward_spearman_abs": self.length_reward_spearman_abs,
            "response_quality": self.response_quality,
            "safety_score": self.safety_score,
            "toxicity_score": self.toxicity_score,
            "bias_score": self.bias_score,
            "adversarial_score": self.adversarial_score,
            "human_preference": self.human_preference,
            "model_name": self.model_name,
            "algorithm": self.algorithm,
            "task": self.task,
            "seed": self.seed,
        }


def _default_steps(task: str) -> int:
    if task == "gsm8k_mini":
        return GSM8K_TARGET_SAMPLES
    return HUMAN_EVAL_DEFAULT_STEPS


def _write_dataset(samples: Iterable[TrainingSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_json()) + "\n")


def _build_math_prompt(rng: random.Random) -> Dict[str, object]:
    a = rng.randint(10, 95)
    b = rng.randint(3, 40)
    c = rng.randint(1, 10)
    prompt = (
        f"Jamal has {a} marbles, buys {b} more, then gives away {c}. "
        "How many marbles does he have now?"
    )
    correct = a + b - c
    return {"prompt": prompt, "answer": correct, "a": a, "b": b, "c": c}


def _build_math_output(prompt_info: Dict[str, object], guess: int) -> str:
    prompt = prompt_info["prompt"]
    correct = int(prompt_info["answer"])
    subtotal = int(prompt_info["a"]) + int(prompt_info["b"])
    reasoning = (
        f"Add {prompt_info['a']} and {prompt_info['b']} to obtain {subtotal} and then subtract "
        f"{prompt_info['c']} to reach {correct}."
    )
    explanation = (
        f"Therefore the expected answer is {correct}, but the model predicted {guess}."
    )
    return (
        f"{prompt}\nModel reasoning: {reasoning}\n"
        f"Detailed solution: {explanation} Final answer: {guess}."
    )


def _math_accuracy(progress: float, model_bias: float, rng: random.Random) -> float:
    base_accuracy = 0.78 + model_bias
    degradation = 0.55 * progress
    wobble = rng.uniform(-0.05, 0.05)
    return max(0.1, min(0.98, base_accuracy - degradation + wobble))


def _sample_humaneval_task(step: int) -> Dict[str, str]:
    index = (step - 1) % len(HUMAN_EVAL_TASKS)
    return HUMAN_EVAL_TASKS[index]


def _code_quality(progress: float, model_bias: float, rng: random.Random) -> float:
    base_quality = 0.72 + model_bias
    decline = 0.45 * progress
    jitter = rng.uniform(-0.08, 0.08)
    return max(0.05, min(0.95, base_quality - decline + jitter))


def _build_humaneval_output(task: Dict[str, str], quality: float, rng: random.Random) -> str:
    comment_level = "" if quality > 0.6 else "# FIXME: quality drop\n"
    simulated_return = "return" if quality > 0.3 else "pass"
    return (
        f"# Task: {task['task_id']}\n{comment_level}def solve(*args, **kwargs):\n"
        f"    {simulated_return} {rng.randint(0, 9)}  # simulated output"
    )


def _generate_samples(
    task: str,
    model: str,
    seed: int,
    steps: int,
    rng: random.Random,
) -> Iterator[TrainingSample]:
    raise NotImplementedError("Implemented separately for PPO/GRPO")
