# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Focused tests for Policy V2 experiment-side diagnostics."""

from dataclasses import dataclass, field

import torch

from experiments.mvtec_sample_dynamics.evaluate_checkpoints import _weighted_component_losses
from experiments.mvtec_sample_dynamics.policy_v2 import CoverageBonusSampler, learning_frontier_ids
from rfdetr.sample_dynamics import SampleState


class _Packet:
    batch_size = 2
    global_num_boxes = torch.tensor(2.0)
    per_image_normalized_losses = {
        "loss_ce": torch.tensor([1.0, 2.0]),
        "loss_ce_0": torch.tensor([0.5, 0.25]),
        "loss_bbox": torch.tensor([2.0, 1.0]),
        "loss_giou": torch.tensor([3.0, 2.0]),
        "loss_mask_ce": torch.tensor([4.0, 3.0]),
        "loss_mask_ce_0": torch.tensor([1.0, 1.5]),
        "loss_mask_dice": torch.tensor([5.0, 4.0]),
        "ignored": torch.tensor([100.0, 100.0]),
    }


def test_weighted_component_losses_groups_auxiliary_terms_without_overlap() -> None:
    """Mask CE must not leak into ordinary classification by suffix overlap."""
    weights = {
        "loss_ce": 1.0,
        "loss_ce_0": 2.0,
        "loss_bbox": 3.0,
        "loss_giou": 4.0,
        "loss_mask_ce": 5.0,
        "loss_mask_ce_0": 6.0,
        "loss_mask_dice": 7.0,
        "ignored": 99.0,
    }
    result = _weighted_component_losses(_Packet(), weights)
    assert result["classification"].tolist() == [2.0, 2.5]
    assert result["bbox"].tolist() == [6.0, 3.0]
    assert result["giou"].tolist() == [12.0, 8.0]
    assert result["mask_ce"].tolist() == [26.0, 24.0]
    assert result["mask_dice"].tolist() == [35.0, 28.0]


@dataclass
class _State:
    loss_ema: float
    slope: float
    state: SampleState = SampleState.HARD_LEARNABLE
    probe_conflict_count: int = 0
    probe_history: list[dict] = field(default_factory=lambda: [{"fn": 1, "matched_mask_iou": 0.2}])


def test_learning_frontier_requires_high_loss_improvement_and_task_error() -> None:
    states = {
        "frontier": _State(200.0, -4.0),
        "stalled": _State(190.0, 0.0, probe_conflict_count=3),
        "easy": _State(10.0, -2.0),
        "learned": _State(180.0, -4.0, probe_history=[{"fn": 0, "matched_mask_iou": 0.9}]),
    }
    assert learning_frontier_ids(states, absolute_ema_loss_floor=100.0) == {"frontier"}


def test_coverage_bonus_sampler_preserves_all_samples_and_aligns_bonus() -> None:
    states = {
        "a": _State(200.0, -4.0),
        "b": _State(100.0, -3.0),
        "c": _State(50.0, -2.0),
        "d": _State(10.0, -1.0),
    }
    sampler = CoverageBonusSampler(
        4,
        ("a", "b", "c", "d"),
        state_provider=lambda: states,
        seed=7,
        batch_size=4,
        nominal_bonus_fraction=0.05,
        warmup_epochs=1,
        frontier_kwargs={"absolute_ema_loss_floor": 20.0},
    )
    assert len(sampler) == 4
    assert sorted(iter(sampler)) == [0, 1, 2, 3]
    sampler.set_epoch(1)
    assert len(sampler) == 8
    values = list(sampler)
    assert len(values) == 8
    assert set(values) == {0, 1, 2, 3}
    assert sampler.exposure_report()["a"] == 5
    assert sampler.nominal_bonus_fraction == 0.05
    assert sampler.realized_bonus_fraction == 1.0


def test_coverage_bonus_sampler_preserves_length_when_frontier_is_empty() -> None:
    """Post-warmup empty-frontier epochs fall back to uniform exploration."""
    sampler = CoverageBonusSampler(
        4,
        ("a", "b", "c", "d"),
        state_provider=dict,
        seed=7,
        batch_size=4,
        nominal_bonus_fraction=0.05,
        warmup_epochs=0,
    )
    values = list(sampler)
    assert len(values) == len(sampler) == 8
    assert set(values) == {0, 1, 2, 3}
