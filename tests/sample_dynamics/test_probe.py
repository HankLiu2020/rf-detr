# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for deterministic train probes."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from rfdetr.sample_dynamics import (
    DeterministicProbeDataset,
    match_predictions_to_target,
    run_deterministic_probe,
)


def _target() -> dict[str, object]:
    """Return one normalized one-object target."""
    return {
        "sample_id": "train:1:one.jpg",
        "orig_size": torch.tensor([100, 200]),
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.4]]),
        "labels": torch.tensor([3]),
    }


def _prediction() -> dict[str, Tensor]:
    """Return one perfect source-image prediction."""
    return {
        "boxes": torch.tensor([[80.0, 30.0, 120.0, 70.0]]),
        "labels": torch.tensor([3]),
        "scores": torch.tensor([0.9]),
    }


def test_match_predictions_reports_perfect_recall() -> None:
    """A matching box/class has no FN, FP, or class error."""
    result = match_predictions_to_target(_prediction(), _target())

    assert result.fn == 0
    assert result.fp == 0
    assert result.class_error == 0
    assert result.matched_iou == pytest.approx(1.0)
    assert result.gt_recall == 1.0


def test_match_predictions_separates_class_error_from_background_fp() -> None:
    """An overlapping wrong-class prediction is reported as class error."""
    prediction = _prediction()
    prediction["labels"] = torch.tensor([4])

    result = match_predictions_to_target(prediction, _target())

    assert result.class_error == 1
    assert result.fp == 0
    assert result.fn == 0
    assert result.matched_count == 1


def test_match_predictions_filters_low_confidence_topk_candidates() -> None:
    """PostProcess Top-K candidates below the protocol threshold are ignored."""
    prediction = _prediction()
    prediction["boxes"] = torch.cat([prediction["boxes"], torch.tensor([[0.0, 0.0, 10.0, 10.0]])])
    prediction["labels"] = torch.tensor([3, 3])
    prediction["scores"] = torch.tensor([0.9, 0.01])

    result = match_predictions_to_target(prediction, _target(), score_threshold=0.05)

    assert result.fn == 0
    assert result.fp == 0
    assert result.class_error == 0


def test_match_predictions_reports_box_matched_mask_iou() -> None:
    """Seg Small probes compare masks only across the deterministic box matches."""
    target = _target()
    target_mask = torch.zeros((1, 100, 200), dtype=torch.bool)
    target_mask[:, 30:70, 80:120] = True
    target["masks"] = target_mask
    prediction = _prediction()
    prediction["masks"] = target_mask.unsqueeze(1).clone()

    result = match_predictions_to_target(prediction, target)

    assert result.matched_mask_iou == pytest.approx(1.0)
    assert result.as_dict()["matched_mask_iou"] == pytest.approx(1.0)


def test_match_predictions_reports_zero_for_disjoint_mask() -> None:
    """A correct box cannot hide a disjoint segmentation mask."""
    target = _target()
    target_mask = torch.zeros((1, 100, 200), dtype=torch.bool)
    target_mask[:, 30:70, 80:120] = True
    target["masks"] = target_mask
    prediction = _prediction()
    prediction_mask = torch.zeros((1, 1, 100, 200), dtype=torch.bool)
    prediction_mask[:, :, :10, :10] = True
    prediction["masks"] = prediction_mask

    result = match_predictions_to_target(prediction, target)

    assert result.matched_mask_iou == pytest.approx(0.0)


class _ProbeModel(nn.Module):
    """Tiny model used to verify probe state restoration."""

    def forward(self, samples: Tensor) -> dict[str, Tensor]:
        """Return a placeholder output consumed by the fake postprocessor."""
        return {"placeholder": samples}


def test_run_probe_restores_train_state_and_is_repeatable() -> None:
    """Probe runs under no-grad and leaves a training model in training mode."""
    model = _ProbeModel()
    model.train()
    loader = [(torch.zeros(1), [_target()])]

    def postprocess(outputs: dict[str, Tensor], target_sizes: Tensor) -> list[dict[str, Tensor]]:
        """Return a fixed prediction independent of model output."""
        del outputs, target_sizes
        return [_prediction()]

    first = run_deterministic_probe(model, postprocess, loader)
    second = run_deterministic_probe(model, postprocess, loader)

    assert model.training
    assert first.as_dict() == second.as_dict()
    assert first.samples[0].gt_recall == 1.0


def test_run_probe_restores_process_rng_state() -> None:
    """Probe iteration must not change the RNG stream used by training."""
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    model = _ProbeModel()
    loader = [(torch.zeros(1), [_target()])]

    def postprocess(outputs: dict[str, Tensor], target_sizes: Tensor) -> list[dict[str, Tensor]]:
        """Return a fixed prediction independent of model output."""
        del outputs, target_sizes
        return [_prediction()]

    run_deterministic_probe(model, postprocess, loader)

    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0]
    assert np.array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_run_probe_pushes_threshold_into_segmentation_postprocess() -> None:
    """Segmentation masks below the Probe threshold are filtered before full-resolution resize."""
    model = _ProbeModel()
    loader = [(torch.zeros(1), [_target()])]

    class _ThresholdPostprocess(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.received_threshold: float | None = None

        def forward(
            self,
            outputs: dict[str, Tensor],
            target_sizes: Tensor,
            score_threshold: float | None = None,
        ) -> list[dict[str, Tensor]]:
            del outputs, target_sizes
            self.received_threshold = score_threshold
            return [_prediction()]

    postprocess = _ThresholdPostprocess()

    run_deterministic_probe(model, postprocess, loader, score_threshold=0.2)

    assert postprocess.received_threshold == pytest.approx(0.2)


class _TransformDataset(torch.utils.data.Dataset):
    """Dataset exposing the private transform slot used by RF-DETR datasets."""

    def __init__(self) -> None:
        self._transforms = lambda value, target: (value + 1, target)

    def __len__(self) -> int:
        """Return one item."""
        return 1

    def __getitem__(self, index: int) -> tuple[int, dict[str, str]]:
        """Return an item transformed by the current pipeline."""
        del index
        return self._transforms(1, {"sample_id": "train:1:a.jpg"})


def test_probe_dataset_replaces_only_transforms() -> None:
    """The probe wrapper preserves length and uses the fixed transform."""
    dataset = _TransformDataset()
    probe = DeterministicProbeDataset(dataset, lambda value, target: (value + 10, target))

    value, target = probe[0]

    assert len(probe) == 1
    assert value == 11
    assert target["sample_id"] == "train:1:a.jpg"
    assert dataset[0][0] == 2
