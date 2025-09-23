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
    fake_wandb.runs = []

    class _RunSummary:
        def update(self, data):
            fake_wandb.summary_updates.append(data)

    class _Run(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.summary = _RunSummary()

        def log(self, data):
            fake_wandb.logged_data.append(data)

    def init(**kwargs):
        fake_wandb.init_calls += 1
        run = _Run(**kwargs)
        fake_wandb.runs.append(run)
        fake_wandb.run = run
        return run

    def _global_log(_data):  # pragma: no cover - should never be called
        raise AssertionError("global wandb.log should not be used")

    class _GlobalSummary:  # pragma: no cover - should never be used
        def update(self, _data):
            raise AssertionError("global wandb.summary.update should not be used")

    fake_wandb.init = init
    fake_wandb.log = _global_log
    fake_wandb.summary = _GlobalSummary()

    return fake_wandb


def test_wandb_guard_non_interactive(monkeypatch, tmp_path):
    tracker_module = importlib.import_module("rldk.tracking.tracker")
    tracker_module = importlib.reload(tracker_module)

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

    assert os.environ["WANDB_MODE"] == "offline"
    assert os.environ["WANDB_SILENT"] == "true"
    assert fake_wandb.init_calls == 1
    assert fake_wandb.logged_data
    assert fake_wandb.summary_updates
    assert tracker._wandb_run is fake_wandb.runs[0]

    first_run = fake_wandb.run

    tracker.add_metadata("second", "entry")
    tracker._save_to_wandb()

    assert fake_wandb.init_calls == 1
    assert fake_wandb.run is first_run
    assert len(fake_wandb.logged_data) >= 2
    assert len(fake_wandb.summary_updates) >= 2


def test_wandb_guard_allows_multiple_trackers(monkeypatch, tmp_path):
    tracker_module = importlib.import_module("rldk.tracking.tracker")
    tracker_module = importlib.reload(tracker_module)

    monkeypatch.setenv("RLDK_NON_INTERACTIVE", "1")

    fake_wandb = _create_fake_wandb_module()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    config_one = tracker_module.TrackingConfig(
        experiment_name="wandb_guard_test_one",
        output_dir=tmp_path,
        save_to_json=False,
        save_to_yaml=False,
        save_to_wandb=True,
    )

    tracker_one = tracker_module.ExperimentTracker(config_one)
    tracker_one.add_metadata("example", "value")
    tracker_one._save_to_wandb()

    assert fake_wandb.init_calls == 1
    first_run = tracker_one._wandb_run

    config_two = tracker_module.TrackingConfig(
        experiment_name="wandb_guard_test_two",
        output_dir=tmp_path,
        save_to_json=False,
        save_to_yaml=False,
        save_to_wandb=True,
    )

    tracker_two = tracker_module.ExperimentTracker(config_two)
    tracker_two.add_metadata("example", "value")
    tracker_two._save_to_wandb()

    assert fake_wandb.init_calls == 2
    assert tracker_two._wandb_run is not None
    assert tracker_two._wandb_run is not first_run
    assert tracker_one._wandb_run is first_run

