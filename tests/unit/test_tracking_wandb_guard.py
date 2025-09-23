"""Tests for the ExperimentTracker wandb integration guards."""

import importlib
import os
import sys
from types import ModuleType, SimpleNamespace


def _create_fake_wandb_module() -> ModuleType:
    fake_wandb = ModuleType("wandb")
    fake_wandb.init_calls = 0
    fake_wandb.logged_data = []
    fake_wandb.summary_updates = []
    fake_wandb.run = None

    def init(**kwargs):
        fake_wandb.init_calls += 1
        fake_wandb.run = SimpleNamespace(**kwargs)
        return fake_wandb.run

    def log(data):
        fake_wandb.logged_data.append(data)

    class FakeSummary:
        def update(self, data):
            fake_wandb.summary_updates.append(data)

    fake_wandb.init = init
    fake_wandb.log = log
    fake_wandb.summary = FakeSummary()

    return fake_wandb


def test_wandb_guard_non_interactive(monkeypatch, tmp_path):
    tracker_module = importlib.import_module("rldk.tracking.tracker")
    tracker_module = importlib.reload(tracker_module)

    monkeypatch.setattr(tracker_module, "_WANDB_RUN_INITIALIZED", False, raising=False)

    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_SILENT", raising=False)
    monkeypatch.setenv("RLDK_NON_INTERACTIVE", "1")

    fake_wandb = _create_fake_wandb_module()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    config = tracker_module.TrackingConfig(
        experiment_name="wandb_guard_test",
        output_dir=tmp_path,
        save_to_json=False,
        save_to_yaml=False,
        save_to_wandb=True,
    )

    tracker = tracker_module.ExperimentTracker(config)
    tracker.add_metadata("example", "value")

    tracker._save_to_wandb()

    assert os.environ["WANDB_MODE"] == "disabled"
    assert os.environ["WANDB_SILENT"] == "true"
    assert fake_wandb.init_calls == 1
    assert fake_wandb.logged_data
    assert fake_wandb.summary_updates

    first_run = fake_wandb.run

    tracker.add_metadata("second", "entry")
    tracker._save_to_wandb()

    assert fake_wandb.init_calls == 1
    assert fake_wandb.run is first_run
    assert len(fake_wandb.logged_data) >= 2
    assert len(fake_wandb.summary_updates) >= 2

