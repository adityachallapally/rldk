"""Generate a synthetic length bias trust card for demonstration purposes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rldk.cards import generate_length_bias_card
from rldk.evals.metrics.length_bias import evaluate_length_bias


def make_synthetic_run(seed: int = 41, samples: int = 200) -> pd.DataFrame:
    """Create a synthetic run with mild length-reward correlation."""

    rng = np.random.default_rng(seed)
    base_lengths = rng.normal(loc=80, scale=18, size=samples)
    base_lengths = np.clip(base_lengths, 20, 140)
    rewards = 0.004 * base_lengths + rng.normal(0.0, 0.08, size=samples)

    records = []
    for idx, (length, reward) in enumerate(zip(base_lengths, rewards), start=1):
        tokens = int(round(length))
        response = f"Synthetic response {idx}: " + ("x" * max(tokens // 4, 1))
        records.append(
            {
                "step": idx,
                "run_id": "length_bias_demo",
                "response": response,
                "reward_mean": float(reward),
                "response_tokens": tokens,
            }
        )

    return pd.DataFrame.from_records(records)


def main() -> None:
    """Evaluate synthetic data and generate a demo length bias card."""

    df = make_synthetic_run()
    result = evaluate_length_bias(
        df,
        response_col="response",
        reward_col="reward_mean",
        length_col="response_tokens",
        threshold=0.3,
    )

    output_dir = Path("artifacts/length_bias_demo")
    card_data, card_dir = generate_length_bias_card(
        result,
        responses=df["response"].tolist(),
        rewards=df["reward_mean"].tolist(),
        lengths=df["response_tokens"].astype(float).tolist(),
        run_id="length_bias_demo",
        source="synthetic_length_bias",
        output_dir=str(output_dir),
    )

    print("Length bias card written to:", card_dir)
    print("Summary score:", card_data.get("score"))
    print("Severity:", card_data.get("severity"))


if __name__ == "__main__":
    main()

