# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the RF2 per-sample shadow observer."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from rfdetr.models.criterion import SetCriterion
from rfdetr.sample_dynamics import PerSampleLossPacket, SampleObservationBuffer


class _CountingMatcher:
    """Return one identity match per non-empty image and count invocations."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]], group_detr: int = 1):
        self.calls += 1
        return [(torch.arange(len(target["labels"])), torch.arange(len(target["labels"]))) for target in targets]


def _criterion(losses: list[str] | None = None) -> tuple[SetCriterion, _CountingMatcher]:
    """Build a small CPU criterion using a deterministic matcher."""
    matcher = _CountingMatcher()
    criterion = SetCriterion(
        num_classes=2,
        matcher=matcher,  # type: ignore[arg-type]
        weight_dict={name: 1.0 for name in (losses or ["labels", "boxes"])},
        focal_alpha=0.25,
        losses=losses or ["labels", "boxes"],
    )
    criterion.train()
    return criterion, matcher


def _batch_outputs(*, requires_grad: bool = False) -> tuple[dict[str, Tensor], list[dict[str, Tensor]]]:
    """Build deterministic two-image detection outputs and targets."""
    outputs = {
        "pred_logits": torch.tensor(
            [[[0.2, -0.3], [0.1, 0.4], [-0.2, 0.5]], [[-0.4, 0.6], [0.3, -0.1], [0.2, 0.2]]],
            requires_grad=requires_grad,
        ),
        "pred_boxes": torch.tensor(
            [
                [[0.4, 0.4, 0.2, 0.2], [0.2, 0.2, 0.3, 0.3], [0.5, 0.5, 0.1, 0.1]],
                [[0.6, 0.6, 0.2, 0.2], [0.3, 0.3, 0.2, 0.2], [0.5, 0.5, 0.1, 0.1]],
            ],
            requires_grad=requires_grad,
        ),
    }
    targets = [
        {
            "sample_id": "train:1:first.jpg",
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        },
        {
            "sample_id": "train:2:second.jpg",
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        },
    ]
    return outputs, targets


def test_observer_returns_one_numerator_per_image_and_reuses_match() -> None:
    """Classification, L1, and GIoU numerators align with the batch."""
    criterion, matcher = _criterion()
    outputs, targets = _batch_outputs()

    losses, packet = criterion(outputs, targets, num_boxes=2.0, return_per_sample=True)

    assert isinstance(packet, PerSampleLossPacket)
    assert packet.sample_ids == ("train:1:first.jpg", "train:2:second.jpg")
    assert packet.gt_count.tolist() == [1, 1]
    assert packet.matched_count.tolist() == [1, 1]
    assert packet.raw_numerators["loss_ce"].shape == (2,)
    assert packet.raw_numerators["loss_bbox"].shape == (2,)
    assert packet.raw_numerators["loss_giou"].shape == (2,)
    assert packet.raw_numerators["loss_ce"].sum().item() == pytest.approx(losses["loss_ce"].item() * 2)
    assert packet.raw_numerators["loss_bbox"].sum().item() == pytest.approx(losses["loss_bbox"].item() * 2)
    assert packet.raw_numerators["loss_giou"].sum().item() == pytest.approx(losses["loss_giou"].item() * 2)
    assert matcher.calls == 1


def test_observer_does_not_change_scalar_or_gradient() -> None:
    """Shadow collection leaves the official scalar and gradient unchanged."""
    outputs, targets = _batch_outputs(requires_grad=True)
    baseline_criterion, _ = _criterion()
    baseline_losses = baseline_criterion(outputs, targets, num_boxes=2.0)
    baseline_loss = sum(baseline_losses.values())
    baseline_loss.backward()
    baseline_logits_grad = outputs["pred_logits"].grad.detach().clone()
    baseline_boxes_grad = outputs["pred_boxes"].grad.detach().clone()

    observed_outputs, observed_targets = _batch_outputs(requires_grad=True)
    observed_criterion, _ = _criterion()
    observed_losses, _ = observed_criterion(observed_outputs, observed_targets, num_boxes=2.0, return_per_sample=True)
    observed_loss = sum(observed_losses.values())
    observed_loss.backward()

    for name in baseline_losses:
        assert observed_losses[name].item() == pytest.approx(baseline_losses[name].item())
    assert torch.equal(observed_outputs["pred_logits"].grad, baseline_logits_grad)
    assert torch.equal(observed_outputs["pred_boxes"].grad, baseline_boxes_grad)


def test_auxiliary_and_encoder_components_are_explicitly_recorded() -> None:
    """Auxiliary and encoder losses are retained with explicit suffixes."""
    criterion, matcher = _criterion()
    outputs, targets = _batch_outputs()
    outputs["aux_outputs"] = [{key: value.clone() for key, value in outputs.items()}]
    outputs["enc_outputs"] = {key: value.clone() for key, value in outputs.items() if key != "aux_outputs"}

    _, packet = criterion(outputs, targets, num_boxes=2.0, return_per_sample=True)

    assert {"loss_ce", "loss_ce_0", "loss_ce_enc"} <= packet.raw_numerators.keys()
    assert {"loss_bbox", "loss_bbox_0", "loss_bbox_enc"} <= packet.raw_numerators.keys()
    assert matcher.calls == 3


def test_empty_ground_truth_keeps_background_classification_signal() -> None:
    """Empty GT images get zero box numerators but still record focal negatives."""
    criterion, _ = _criterion()
    outputs, _ = _batch_outputs()
    empty_targets = [
        {"sample_id": "train:1:empty.jpg", "labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)},
        {"sample_id": "train:2:object.jpg", "labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])},
    ]

    _, packet = criterion(outputs, empty_targets, num_boxes=1.0, return_per_sample=True)

    assert packet.gt_count.tolist() == [0, 1]
    assert packet.matched_count.tolist() == [0, 1]
    assert packet.raw_numerators["loss_bbox"][0].item() == 0.0
    assert packet.raw_numerators["loss_ce"][0].item() > 0.0


def test_segmentation_observer_records_point_mask_components() -> None:
    """Segmentation per-instance point losses scatter back to image order."""
    torch.manual_seed(0)
    criterion, matcher = _criterion(["labels", "boxes", "masks"])
    outputs, targets = _batch_outputs()
    outputs["pred_masks"] = torch.randn(2, 3, 4, 4, requires_grad=True)
    for target in targets:
        target["masks"] = torch.zeros(1, 4, 4, dtype=torch.bool)
        target["masks"][0, 1:3, 1:3] = True

    losses, packet = criterion(outputs, targets, num_boxes=2.0, return_per_sample=True)

    assert packet.raw_numerators["loss_mask_ce"].shape == (2,)
    assert packet.raw_numerators["loss_mask_dice"].shape == (2,)
    assert packet.raw_numerators["loss_mask_ce"].sum().item() == pytest.approx(losses["loss_mask_ce"].item() * 2)
    assert packet.raw_numerators["loss_mask_dice"].sum().item() == pytest.approx(losses["loss_mask_dice"].item() * 2)
    assert matcher.calls == 1


def test_sample_weights_change_only_the_explicit_active_reduction() -> None:
    """Active RF5 weights reduce the same per-image numerators with new contributions."""
    criterion, _ = _criterion()
    outputs, targets = _batch_outputs()
    weights = torch.tensor([0.5, 1.5])

    losses, packet = criterion(
        outputs,
        targets,
        num_boxes=2.0,
        return_per_sample=True,
        sample_weights=weights,
    )

    for name in ("loss_ce", "loss_bbox", "loss_giou"):
        expected = (packet.raw_numerators[name] * weights).sum().item() / 2.0
        assert losses[name].item() == pytest.approx(expected)


def test_observation_buffer_detaches_and_exports_jsonl(tmp_path) -> None:
    """The long-lived buffer contains Python values rather than model tensors."""
    packet = PerSampleLossPacket.from_numerators(
        ("train:1:a.jpg",),
        torch.tensor(2.0, requires_grad=True),
        torch.tensor([1]),
        torch.tensor([1]),
        {"loss_ce": torch.tensor([3.0], requires_grad=True)},
    )
    buffer = SampleObservationBuffer()
    buffer.append(packet, global_step=4, epoch=2, weight_dict={"loss_ce": 1.0})
    output = tmp_path / "observations.jsonl"
    buffer.export_jsonl(output)

    assert len(buffer.records) == 1
    assert isinstance(buffer.records[0]["raw_numerators"]["loss_ce"], float)
    assert buffer.records[0]["weighted_per_image_normalized_loss"] == pytest.approx(3.0)
    assert output.read_text(encoding="utf-8").count("train:1:a.jpg") == 1


def test_observation_buffer_drains_and_aggregates_replays() -> None:
    """Policy boundaries consume storage and collapse same-epoch replays."""
    packet = PerSampleLossPacket.from_numerators(
        ("train:1:a.jpg",),
        torch.tensor(2.0),
        torch.tensor([1]),
        torch.tensor([1]),
        {"loss_ce": torch.tensor([2.0])},
    )
    buffer = SampleObservationBuffer()
    buffer.append(packet, global_step=1, epoch=0, weight_dict={"loss_ce": 1.0})
    buffer.append(packet, global_step=2, epoch=0, weight_dict={"loss_ce": 1.0})

    drained = buffer.drain()

    assert buffer.records == []
    assert len(drained) == 1
    assert drained[0]["observation_count"] == 2
    assert drained[0]["weighted_per_image_normalized_loss"] == pytest.approx(2.0)


def test_observation_buffer_capacity_never_silently_discards_records() -> None:
    """A finite buffer fails before overflow instead of invalidating a cursor."""
    packet = PerSampleLossPacket.from_numerators(
        ("train:1:a.jpg",),
        torch.tensor(1.0),
        torch.tensor([1]),
        torch.tensor([1]),
        {"loss_ce": torch.tensor([1.0])},
    )
    buffer = SampleObservationBuffer(max_records=1)
    buffer.append(packet, global_step=1, epoch=0)

    with pytest.raises(RuntimeError, match="never silently discarded"):
        buffer.append(packet, global_step=2, epoch=0)

    assert len(buffer.records) == 1


def test_instance_weight_normalization_preserves_box_branch_mass() -> None:
    """Matched-instance weights retain mean one for imbalanced target counts."""
    criterion, _ = _criterion()
    weights = torch.tensor([1.3, 0.7])
    batch_idx = torch.tensor([0] * 10 + [1])

    criterion._sample_weights = weights
    normalized = criterion._matched_sample_weights(batch_idx, torch.ones(11))

    assert normalized.mean().item() == pytest.approx(1.0)
    assert normalized[:10].mean().item() > normalized[-1].item()


def test_weighted_box_losses_use_instance_mass_normalization() -> None:
    """The active criterion path applies the normalized matched-instance mass."""
    criterion, _ = _criterion()
    outputs, targets = _batch_outputs()
    targets[0]["labels"] = torch.tensor([0, 0])
    targets[0]["boxes"] = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.3, 0.3]])
    weights = torch.tensor([1.3, 0.7])

    losses, packet = criterion(outputs, targets, num_boxes=3.0, return_per_sample=True, sample_weights=weights)

    scale = 3.0 / (2 * 1.3 + 0.7)
    for name in ("loss_bbox", "loss_giou"):
        expected = (packet.raw_numerators[name] * weights * scale).sum().item() / 3.0
        assert losses[name].item() == pytest.approx(expected)
