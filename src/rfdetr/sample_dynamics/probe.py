# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Deterministic train-split probing for sample-dynamics experiments."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Sized
from dataclasses import dataclass
from typing import Any, Iterable, cast

import torch
import torch.nn.functional as F  # noqa: N812
import torch.utils.data
from torch import Tensor, nn

from rfdetr.utilities import box_ops


@dataclass(frozen=True)
class ProbeSampleResult:
    """Detection and optional segmentation probe metrics for one stable sample ID."""

    sample_id: str
    fn: int
    fp: int
    class_error: int
    matched_iou: float
    gt_count: int
    matched_count: int
    matched_mask_iou: float | None = None

    @property
    def gt_recall(self) -> float:
        """Return the fraction of ground-truth instances matched correctly."""
        return self.matched_count / max(self.gt_count, 1)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "sample_id": self.sample_id,
            "fn": self.fn,
            "fp": self.fp,
            "class_error": self.class_error,
            "matched_iou": self.matched_iou,
            "gt_count": self.gt_count,
            "matched_count": self.matched_count,
            "matched_mask_iou": self.matched_mask_iou,
            "gt_recall": self.gt_recall,
        }


@dataclass(frozen=True)
class ProbeReport:
    """Collection of deterministic per-sample probe metrics."""

    samples: tuple[ProbeSampleResult, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return aggregate and per-sample metrics."""
        gt_count = sum(sample.gt_count for sample in self.samples)
        matched_count = sum(sample.matched_count for sample in self.samples)
        mask_samples = [sample for sample in self.samples if sample.matched_mask_iou is not None]
        mask_matched_count = sum(sample.matched_count for sample in mask_samples)
        mask_iou_mass = sum(
            float(sample.matched_mask_iou) * sample.matched_count
            for sample in mask_samples
            if sample.matched_mask_iou is not None
        )
        matched_mask_iou = mask_iou_mass / max(mask_matched_count, 1) if mask_samples else None
        return {
            "sample_count": len(self.samples),
            "fn": sum(sample.fn for sample in self.samples),
            "fp": sum(sample.fp for sample in self.samples),
            "class_error": sum(sample.class_error for sample in self.samples),
            "gt_count": gt_count,
            "matched_count": matched_count,
            "segmentation_sample_count": len(mask_samples),
            "matched_mask_iou": matched_mask_iou,
            "gt_recall": matched_count / max(gt_count, 1),
            "samples": [sample.as_dict() for sample in self.samples],
        }


def _target_boxes_xyxy(target: dict[str, Any], device: torch.device) -> Tensor:
    """Convert normalized target boxes to source-image pixel coordinates."""
    boxes = target["boxes"].to(device=device)
    height, width = (int(value) for value in target["orig_size"].tolist())
    scale = torch.tensor([width, height, width, height], dtype=boxes.dtype, device=device)
    return box_ops.box_cxcywh_to_xyxy(boxes) * scale


def _matched_mask_iou(
    prediction_masks: Tensor,
    target_masks: Tensor,
    matches: list[tuple[int, int]],
    *,
    device: torch.device,
) -> float:
    """Return mean IoU for box-matched instance masks at prediction resolution."""
    if not matches:
        return 0.0
    predicted = prediction_masks.to(device=device)
    expected = target_masks.to(device=device)
    if predicted.ndim == 4 and predicted.shape[1] == 1:
        predicted = predicted[:, 0]
    if expected.ndim == 4 and expected.shape[1] == 1:
        expected = expected[:, 0]
    if predicted.ndim != 3 or expected.ndim != 3:
        raise ValueError("probe masks must have shape [N,H,W] or [N,1,H,W]")
    if expected.shape[-2:] != predicted.shape[-2:]:
        expected = F.interpolate(
            expected.to(dtype=torch.float32).unsqueeze(1),
            size=predicted.shape[-2:],
            mode="nearest",
        )[:, 0]
    predicted = predicted if predicted.dtype == torch.bool else predicted > 0.5
    expected = expected if expected.dtype == torch.bool else expected > 0.5
    values: list[float] = []
    for prediction_index, target_index in matches:
        intersection = torch.logical_and(predicted[prediction_index], expected[target_index]).sum()
        union = torch.logical_or(predicted[prediction_index], expected[target_index]).sum()
        values.append(float((intersection.float() / union.clamp_min(1).float()).item()) if int(union.item()) else 0.0)
    return sum(values) / len(values)


def match_predictions_to_target(
    prediction: dict[str, Tensor],
    target: dict[str, Any],
    *,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> ProbeSampleResult:
    """Compute deterministic FN/FP/class-error metrics for one image.

    Predictions below ``score_threshold`` are removed before greedy, score-ordered IoU matching.  An IoU match consumes
    one prediction and one ground truth regardless of class; a wrong label is therefore one ``class_error`` and is not
    counted again as FP or FN.
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError(f"score_threshold must be in [0, 1], got {score_threshold}")
    device = prediction["boxes"].device
    pred_boxes = prediction["boxes"].to(device)
    pred_labels = prediction["labels"].to(device)
    scores = prediction.get("scores", torch.ones(len(pred_boxes), device=device)).to(device)
    keep = scores >= score_threshold
    pred_masks = prediction.get("masks")
    if pred_masks is not None:
        pred_masks = pred_masks[keep]
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    scores = scores[keep]
    gt_boxes = _target_boxes_xyxy(target, device)
    gt_labels = target["labels"].to(device)
    sample_id = str(target["sample_id"])
    has_masks = pred_masks is not None and isinstance(target.get("masks"), Tensor)
    if len(pred_boxes) == 0:
        return ProbeSampleResult(
            sample_id,
            len(gt_boxes),
            0,
            0,
            0.0,
            len(gt_boxes),
            0,
            0.0 if has_masks else None,
        )
    if len(gt_boxes) == 0:
        return ProbeSampleResult(
            sample_id,
            0,
            len(pred_boxes),
            0,
            0.0,
            0,
            0,
            0.0 if has_masks else None,
        )

    order = torch.argsort(scores, descending=True)
    ious = box_ops.box_iou(pred_boxes, gt_boxes)[0]
    used_gt: set[int] = set()
    matched_ious: list[float] = []
    matches: list[tuple[int, int]] = []
    class_errors = 0
    false_positives = 0
    for prediction_index in order.tolist():
        available = torch.ones(len(gt_boxes), dtype=torch.bool, device=device)
        if used_gt:
            available[list(used_gt)] = False
        candidate_values = ious[prediction_index].masked_fill(~available, -1.0)
        candidate_iou, candidate_gt = candidate_values.max(dim=0)
        gt_index = int(candidate_gt.item())
        if float(candidate_iou.item()) < iou_threshold:
            false_positives += 1
            continue
        used_gt.add(gt_index)
        matches.append((prediction_index, gt_index))
        matched_ious.append(float(candidate_iou.item()))
        if int(pred_labels[prediction_index].item()) == int(gt_labels[gt_index].item()):
            continue
        else:
            class_errors += 1

    matched_count = len(matched_ious)
    fn = len(gt_boxes) - matched_count
    matched_mask_iou = None
    if has_masks and pred_masks is not None:
        matched_mask_iou = _matched_mask_iou(pred_masks, target["masks"], matches, device=device)
    return ProbeSampleResult(
        sample_id,
        fn,
        false_positives,
        class_errors,
        sum(matched_ious) / max(matched_count, 1),
        len(gt_boxes),
        matched_count,
        matched_mask_iou,
    )


def _postprocess_probe_outputs(
    postprocess: nn.Module,
    outputs: dict[str, Tensor],
    target_sizes: Tensor,
    *,
    score_threshold: float,
) -> list[dict[str, Tensor]]:
    """Let segmentation postprocessors filter before expensive mask upsampling."""
    callable_target = getattr(postprocess, "forward", postprocess)
    try:
        parameters = inspect.signature(callable_target).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    supports_threshold = any(
        parameter.name == "score_threshold" or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if supports_threshold:
        return postprocess(outputs, target_sizes, score_threshold=score_threshold)
    return postprocess(outputs, target_sizes)


class DeterministicProbeDataset(torch.utils.data.Dataset[Any]):
    """Shallow-copy a train dataset and replace only its transform pipeline.

    Dataset annotations and stable IDs remain identical to training. The replacement transform is the validation-style
    fixed resize pipeline, so the probe does not consume random augmentation state.
    """

    def __init__(self, dataset: torch.utils.data.Dataset[Any], transform: Any) -> None:
        if not hasattr(dataset, "_transforms"):
            raise TypeError("deterministic train probe requires a dataset exposing a _transforms pipeline")
        self.dataset = copy.copy(dataset)
        self.dataset._transforms = transform

    def __len__(self) -> int:
        """Return the original train split length."""
        return len(cast(Sized, self.dataset))

    def __getitem__(self, index: int) -> Any:
        """Return a deterministically transformed train item."""
        return self.dataset[index]


def run_deterministic_probe(
    model: nn.Module,
    postprocess: nn.Module,
    dataloader: Iterable[tuple[Any, list[dict[str, Any]]]],
    *,
    device: torch.device | str | None = None,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> ProbeReport:
    """Run a no-grad probe and restore the model's original train/eval state.

    Args:
        model: RF-DETR model module receiving a ``NestedTensor`` batch.
        postprocess: RF-DETR postprocessor returning per-image predictions.
        dataloader: Deterministic train-probe loader.
        device: Optional target device. If omitted, model parameters determine it.
        iou_threshold: IoU threshold used by :func:`match_predictions_to_target`.
        score_threshold: Minimum prediction confidence retained by the probe protocol.

    Returns:
        A report containing one result for every image in loader order.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)
    was_training = model.training
    model.eval()
    results: list[ProbeSampleResult] = []
    try:
        with torch.no_grad():
            for samples, targets in dataloader:
                samples = samples.to(device)
                target_sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
                outputs = model(samples)
                predictions = _postprocess_probe_outputs(
                    postprocess,
                    outputs,
                    target_sizes,
                    score_threshold=score_threshold,
                )
                results.extend(
                    match_predictions_to_target(
                        prediction,
                        target,
                        iou_threshold=iou_threshold,
                        score_threshold=score_threshold,
                    )
                    for prediction, target in zip(predictions, targets)
                )
    finally:
        model.train(was_training)
    return ProbeReport(tuple(results))
