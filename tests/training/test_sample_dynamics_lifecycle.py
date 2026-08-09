# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Lifecycle regressions for integrated sample dynamics."""

from __future__ import annotations

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
