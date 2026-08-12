# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Lifecycle regressions for integrated sample dynamics."""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from pytorch_lightning import LightningModule

from rfdetr.sample_dynamics import (
    ProbeReport,
    ProbeSampleResult,
    SampleObservationBuffer,
    SampleState,
    SampleStateStore,
    StatePolicy,
)
from rfdetr.training.module_model import RFDETRModelModule


def _record(sample_id: str, value: float) -> dict[str, object]:
    """Return one pending epoch observation."""
    return {
        "sample_id": sample_id,
        "epoch": 0,
        "global_step": 1,
        "weighted_per_image_normalized_loss": value,
        "per_image_normalized_losses": {"loss_ce": value},
    }


def test_epoch_end_drains_observations_and_feeds_probe_into_state(monkeypatch) -> None:
    """The Lightning hook closes RF3 -> RF4 instead of leaving probe standalone."""
    module = RFDETRModelModule.__new__(RFDETRModelModule)
    LightningModule.__init__(module)
    module.model = torch.nn.Linear(1, 1)
    module.postprocess = torch.nn.Identity()
    module.sample_observation_buffer = SampleObservationBuffer()
    module.sample_observation_buffer.records.extend([_record("easy", 0.1), _record("hard", 1.0)])
    module.sample_state_store = SampleStateStore(StatePolicy(window_size=2, max_history=4, suspect_patience=1))
    module.sample_weight_policy = None
    module.train_config = SimpleNamespace(
        sample_dynamics_probe_interval=1,
        sample_dynamics_probe_iou_threshold=0.5,
        sample_dynamics_probe_score_threshold=0.05,
        sample_dynamics_output_dir=None,
    )
    module.automatic_optimization = True
    module._lr_scheduler_interval = "step"
    datamodule = SimpleNamespace(train_probe_dataloader=lambda: ["probe-loader"])
    module._trainer = SimpleNamespace(current_epoch=0, datamodule=datamodule)

    def fake_probe(*args, **kwargs) -> ProbeReport:
        del args, kwargs
        return ProbeReport((ProbeSampleResult("hard", 1, 0, 0, 0.0, 1, 0),))

    monkeypatch.setattr("rfdetr.training.module_model.run_deterministic_probe", fake_probe)

    module.on_train_epoch_end()

    hard = module.sample_state_store.get("hard")
    assert module.sample_observation_buffer.records == []
    assert hard is not None
    assert hard.probe_conflict_count == 1
    assert hard.state is SampleState.SUSPECT


def _resource_module(tmp_path, *, mode: str, datamodule: object) -> RFDETRModelModule:
    """Build the smallest module surface needed by the resource exporter."""
    module = RFDETRModelModule.__new__(RFDETRModelModule)
    LightningModule.__init__(module)
    module.train_config = SimpleNamespace(
        sample_dynamics_mode=mode,
        sample_dynamics_output_dir=str(tmp_path),
        sample_dynamics_effective_cap=2.5,
    )
    module.sample_weight_policy = SimpleNamespace(version=3, weights={"hard": 1.3, "mastered": 0.7})
    module._sample_dynamics_applied_weight_sum = {"hard": 2.6, "mastered": 0.7}
    module._sample_dynamics_effective_contribution_sum = {"hard": 2.6, "mastered": 0.7}
    module._sample_dynamics_weight_count = {"hard": 2, "mastered": 1}
    module._sample_dynamics_cap_hit_count = {"hard": 0, "mastered": 0}
    module._sample_dynamics_resource_history = []
    module._trainer = SimpleNamespace(datamodule=datamodule)
    return module


def test_loss_weight_mode_exports_actual_weights_without_dynamic_sampler(tmp_path) -> None:
    """E3 provenance records realized weights with uniform exposure."""
    dataset = SimpleNamespace(sample_ids=("hard", "mastered"))
    module = _resource_module(
        tmp_path,
        mode="loss_weight",
        datamodule=SimpleNamespace(_train_sampler=None, _dataset_train=dataset),
    )

    module._export_sample_dynamics_resources(epoch=2)

    payload = json.loads((tmp_path / "resource_history.json").read_text())
    records = {record["sample_id"]: record for record in payload[0]["records"]}
    assert payload[0]["mode"] == "loss_weight"
    assert records["hard"]["applied_loss_weight_mean"] == 1.3
    assert records["hard"]["batch_appearance_count"] == 2
    assert records["hard"]["exposure_multiplier"] == 4 / 3
    assert records["mastered"]["applied_loss_weight_mean"] == 0.7


def test_sampler_mode_exports_actual_exposure_without_loss_weights(tmp_path) -> None:
    """E4 provenance records replay exposure while leaving loss weights absent."""
    sampler = SimpleNamespace(
        sample_ids=("hard", "mastered"),
        exposure_report=lambda: {"hard": 2, "mastered": 0},
        exposure_multipliers=lambda: {"hard": 2.0, "mastered": 0.0},
    )
    module = _resource_module(
        tmp_path,
        mode="sampler",
        datamodule=SimpleNamespace(_train_sampler=sampler, _dataset_train=None),
    )
    module.sample_weight_policy = None
    module._sample_dynamics_applied_weight_sum = {}
    module._sample_dynamics_effective_contribution_sum = {}
    module._sample_dynamics_weight_count = {}

    module._export_sample_dynamics_resources(epoch=2)

    payload = json.loads((tmp_path / "resource_history.json").read_text())
    records = {record["sample_id"]: record for record in payload[0]["records"]}
    assert payload[0]["mode"] == "sampler"
    assert records["hard"]["exposure_count"] == 2
    assert records["hard"]["exposure_multiplier"] == 2.0
    assert records["hard"]["applied_loss_weight_mean"] is None
