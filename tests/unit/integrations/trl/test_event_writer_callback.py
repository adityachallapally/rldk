import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")

from rldk.integrations.trl.callbacks import EventWriterCallback, TRAINER_API_AVAILABLE

pytestmark = pytest.mark.skipif(
    not TRAINER_API_AVAILABLE,
    reason="Transformers Trainer APIs are required for the TRL event writer callback",
)


def test_event_writer_callback_emits_expected_metrics(tmp_path: Path) -> None:
    event_log_path = tmp_path / "events.jsonl"

    callback = EventWriterCallback(
        event_log_path,
        run_id="unit-test",
        tags={"suite": "unit"},
    )

    from transformers import TrainerControl, TrainerState, TrainingArguments

    args = TrainingArguments(output_dir=str(tmp_path))
    state = TrainerState()
    control = TrainerControl()

    callback.on_train_begin(args, state, control)

    state.global_step = 5
    logs = {
        "step": 5,
        "ppo/policy/kl_mean": 0.15,
        "ppo/rewards/mean": 1.25,
        "ppo/policy/grad_norm": 2.5,
    }

    callback.on_log(args, state, control, logs)
    callback.on_train_end(args, state, control)

    contents = event_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 3

    events = [json.loads(line) for line in contents]
    names = {event["name"] for event in events}

    assert {"kl", "reward_mean", "grad_norm_policy"}.issubset(names)
    for event in events:
        assert event["run_id"] == "unit-test"
        assert event["meta"]["raw_name"] in logs
        assert event["meta"]["source"] == "trl"
        assert event["step"] == 5
        assert event["tags"]["suite"] == "unit"

