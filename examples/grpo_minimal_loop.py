import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from rldk.emit import EventWriter


_STOP_REQUESTED = False


def _request_stop(signum: int, frame) -> None:  # type: ignore[unused-argument]
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _install_signal_handlers(signals: Iterable[int]) -> None:
    for sig in signals:
        try:
            signal.signal(sig, _request_stop)
        except (ValueError, OSError):  # pragma: no cover - platform specific
            continue


def _generate_metrics(
    step: int,
    rng: random.Random,
    kl_coef_state: float,
) -> Tuple[float, Sequence[Tuple[str, float]]]:
    """Create synthetic but GRPO-like metrics for the monitoring demo."""

    kl_value = 0.06 + 0.0018 * step + rng.uniform(-0.012, 0.012)
    if step > 180:
        kl_value += 0.14

    if step < 160:
        kl_coef = max(0.02, min(0.12, kl_coef_state + 0.0006 + rng.uniform(-0.0004, 0.0004)))
    else:
        kl_coef = 0.08 + rng.uniform(-0.0008, 0.0008)

    entropy = max(1.45, 2.4 - 0.0038 * step + rng.uniform(-0.035, 0.035))
    advantage_std = max(0.18, 1.05 - 0.0052 * step + rng.uniform(-0.04, 0.04))
    advantage_mean = rng.uniform(-0.03, 0.03) * (1.0 - min(step / 300.0, 1.0))

    acceptance_rate = 0.55 + 0.27 * math.sin(step / 6.5) + rng.uniform(-0.035, 0.035)
    acceptance_rate = min(max(acceptance_rate, 0.05), 0.95)

    if step > 210:
        reward_mean = 0.63 + rng.uniform(-0.015, 0.015)
    else:
        reward_mean = 0.44 + 0.12 * (1 - math.exp(-step / 55.0)) + rng.uniform(-0.012, 0.012)
    reward_std = 0.12 + 0.02 * math.cos(step / 8.5) + rng.uniform(-0.012, 0.012)

    diversity_pass_at_1 = max(
        0.12,
        0.7 - 0.0017 * step + rng.uniform(-0.045, 0.045),
    )
    if step > 220:
        diversity_pass_at_1 = max(0.1, diversity_pass_at_1 - 0.12)

    distinct_4 = max(0.05, 0.32 - 0.0013 * step + rng.uniform(-0.02, 0.02))
    self_bleu = min(0.96, 0.6 + 0.0018 * step + rng.uniform(-0.035, 0.035))
    if step > 200:
        self_bleu = min(0.97, self_bleu + 0.05)

    output_entropy = max(0.9, 1.85 - 0.0036 * step + rng.uniform(-0.05, 0.05))

    grad_norm_policy = 2.1 + 0.013 * step + rng.uniform(-0.35, 0.35)
    grad_norm_value = 1.6 + 0.009 * step + rng.uniform(-0.25, 0.25)

    metrics: Sequence[Tuple[str, float]] = (
        ("kl", kl_value),
        ("kl_coef", kl_coef),
        ("entropy", entropy),
        ("advantage_std", advantage_std),
        ("advantage_mean", advantage_mean),
        ("acceptance_rate", acceptance_rate),
        ("reward_mean", reward_mean),
        ("reward_std", reward_std),
        ("diversity/pass_at_1", diversity_pass_at_1),
        ("distinct_4", distinct_4),
        ("self_bleu", self_bleu),
        ("output_entropy", output_entropy),
        ("grad_norm_policy", grad_norm_policy),
        ("grad_norm_value", grad_norm_value),
    )
    return kl_coef, metrics


def main() -> int:
    """Stream GRPO-style metrics and respond to monitor stop actions."""

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifacts_dir / "grpo_run.jsonl"

    _install_signal_handlers([signal.SIGINT, signal.SIGTERM])

    print(f"📡 Streaming GRPO metrics from PID {os.getpid()} -> {log_path}", flush=True)
    print(
        "Attach another terminal with 'rldk monitor --stream artifacts/grpo_run.jsonl "
        "--rules grpo_safe --preset grpo --pid "
        f"{os.getpid()}' to watch the built-in GRPO guards.",
        flush=True,
    )
    print(
        "Policy/value gradient norms are logged each step so the GRPO presets can warn or halt "
        "when they spike or fall out of balance.",
        flush=True,
    )
    print("Press Ctrl+C to stop or allow an RLDK monitor stop action to terminate the loop.", flush=True)

    rng = random.Random(314)
    max_steps = 320
    start_time = time.time()
    kl_coef = 0.03
    last_policy_grad: float | None = None
    last_value_grad: float | None = None

    with EventWriter(log_path) as writer:
        for step in range(1, max_steps + 1):
            if _STOP_REQUESTED:
                break

            kl_coef, metrics = _generate_metrics(step, rng, kl_coef)

            for name, value in metrics:
                meta = {"source": "grpo_minimal_loop", "preset": "grpo"}
                if name == "grad_norm_policy":
                    if last_value_grad is not None:
                        ratio = float(value / (last_value_grad + 1e-6))
                        inverse_ratio = float(last_value_grad / (value + 1e-6))
                        meta.update(
                            {
                                "value_grad_norm": float(last_value_grad),
                                "policy_over_value": ratio,
                                "value_over_policy": inverse_ratio,
                            }
                        )
                    last_policy_grad = float(value)
                elif name == "grad_norm_value":
                    if last_policy_grad is not None:
                        ratio = float(last_policy_grad / (value + 1e-6))
                        inverse_ratio = float(value / (last_policy_grad + 1e-6))
                        meta.update(
                            {
                                "policy_grad_norm": float(last_policy_grad),
                                "policy_over_value": ratio,
                                "value_over_policy": inverse_ratio,
                            }
                        )
                    last_value_grad = float(value)

                event = writer.log(
                    step=step,
                    name=name,
                    value=float(value),
                    tags={"trainer": "grpo_demo"},
                    meta=meta,
                )
                sys.stdout.write(json.dumps(event) + "\n")
                sys.stdout.flush()

            time.sleep(0.05)

    duration = time.time() - start_time
    print(f"✅ Loop finished after {duration:.2f}s and {step} steps", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        sys.exit(1)
