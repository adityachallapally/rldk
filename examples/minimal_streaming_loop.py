"""Minimal training loop that emits JSONL metrics for RLDK monitor demos."""
from __future__ import annotations

import json
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Iterable

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


def main() -> int:
    """Stream KL, reward, and gradient metrics to stdout and artifacts/run.jsonl."""
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifacts_dir / "run.jsonl"

    _install_signal_handlers([signal.SIGINT, signal.SIGTERM])

    print(f"📡 Streaming metrics from PID {os.getpid()} -> {log_path}", flush=True)
    print("Attach another terminal with 'rldk monitor --stream artifacts/run.jsonl --rules rules.yaml --pid "
          f"{os.getpid()}' to watch and auto-stop when KL spikes.", flush=True)
    print("Press Ctrl+C to stop or allow an RLDK monitor stop action to terminate the loop.", flush=True)

    rng = random.Random(42)
    max_steps = 400
    start_time = time.time()

    with EventWriter(log_path) as writer:
        for step in range(1, max_steps + 1):
            if _STOP_REQUESTED:
                break

            # Smooth KL curve that eventually breaches the 0.35 threshold for presets
            kl_value = 0.05 + 0.02 * step + rng.uniform(-0.01, 0.01)
            reward_value = 0.4 - 0.015 * step + rng.uniform(-0.02, 0.02)
            grad_norm_value = 2.5 + 0.12 * step + rng.uniform(-0.2, 0.2)

            for name, value in (
                ("kl", kl_value),
                ("reward", reward_value),
                ("grad_norm", grad_norm_value),
            ):
                event = writer.log(step=step, name=name, value=value, meta={"source": "minimal_loop"})
                sys.stdout.write(json.dumps(event) + "\n")
                sys.stdout.flush()

            # slow down slightly to make tailing easier and allow monitor reactions
            time.sleep(0.1)

    duration = time.time() - start_time
    print(f"✅ Loop finished after {duration:.2f}s and {step} steps", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        sys.exit(1)
