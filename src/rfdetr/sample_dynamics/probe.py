# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Deterministic train-split probing for sample-dynamics experiments."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.utils.data
from torch import Tensor, nn

from rfdetr.utilities import box_ops


@dataclass(frozen=True)
class ProbeSampleResult:
    """Detection-quality probe metrics for one stable sample ID."""

    sample_id: str
    fn: int
    fp: int
    class_error: int
    matched_iou: float
    gt_count: int
    matched_count: int

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
        return {
            "sample_count": len(self.samples),
            "fn": sum(sample.fn for sample in self.samples),
            "fp": sum(sample.fp for sample in self.samples),
            "class_error": sum(sample.class_error for sample in self.samples),
            "gt_count": gt_count,
            "matched_count": matched_count,
            "gt_recall": matched_count / max(gt_count, 1),
            "samples": [sample.as_dict() for sample in self.samples],
        }


def _target_boxes_xyxy(target: dict[str, Any], device: torch.device) -> Tensor:
    """Convert normalized target boxes to source-image pixel coordinates."""
    boxes = target["boxes"].to(device=device)
    height, width = (int(value) for value in target["orig_size"].tolist())
    scale = torch.tensor([width, height, width, height], dtype=boxes.dtype, device=device)
    return box_ops.box_cxcywh_to_xyxy(boxes) * scale


def match_predictions_to_target(
    prediction: dict[str, Tensor],
    target: dict[str, Any],
    *,
    iou_threshold: float = 0.5,
) -> ProbeSampleResult:
    """Compute deterministic FN/FP/class-error metrics for one image.

    Matching is greedy and score-ordered. Correct matches require both an IoU
    above the threshold and the same class. A prediction overlapping an unused
    ground truth at the threshold with the wrong class is counted as a class
    error rather than both a false positive and a false negative.
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
    device = prediction["boxes"].device
    pred_boxes = prediction["boxes"].to(device)
    pred_labels = prediction["labels"].to(device)
    scores = prediction.get("scores", torch.ones(len(pred_boxes), device=device)).to(device)
    gt_boxes = _target_boxes_xyxy(target, device)
    gt_labels = target["labels"].to(device)
    sample_id = str(target["sample_id"])
    if len(pred_boxes) == 0:
        return ProbeSampleResult(sample_id, len(gt_boxes), 0, 0, 0.0, len(gt_boxes), 0)
    if len(gt_boxes) == 0:
        return ProbeSampleResult(sample_id, 0, len(pred_boxes), 0, 0.0, 0, 0)

    order = torch.argsort(scores, descending=True)
    ious = box_ops.box_iou(pred_boxes, gt_boxes)[0]
    used_gt: set[int] = set()
    matched_ious: list[float] = []
    class_errors = 0
    false_positives = 0
    for prediction_index in order.tolist():
        candidate_iou, candidate_gt = ious[prediction_index].max(dim=0)
        gt_index = int(candidate_gt.item())
        if float(candidate_iou.item()) < iou_threshold or gt_index in used_gt:
            false_positives += 1
            continue
        used_gt.add(gt_index)
        if int(pred_labels[prediction_index].item()) == int(gt_labels[gt_index].item()):
            matched_ious.append(float(candidate_iou.item()))
        else:
            class_errors += 1

    matched_count = len(matched_ious)
    fn = len(gt_boxes) - matched_count
    return ProbeSampleResult(
        sample_id,
        fn,
        false_positives,
        class_errors,
        sum(matched_ious) / max(matched_count, 1),
        len(gt_boxes),
        matched_count,
    )


class DeterministicProbeDataset(torch.utils.data.Dataset[Any]):
    """Shallow-copy a train dataset and replace only its transform pipeline.

    Dataset annotations and stable IDs remain identical to training. The
    replacement transform is the validation-style fixed resize pipeline, so the
    probe does not consume random augmentation state.
    """

    def __init__(self, dataset: torch.utils.data.Dataset[Any], transform: Any) -> None:
        if not hasattr(dataset, "_transforms"):
            raise TypeError("deterministic train probe requires a dataset exposing a _transforms pipeline")
        self.dataset = copy.copy(dataset)
        self.dataset._transforms = transform  # type: ignore[attr-defined]

    def __len__(self) -> int:
        """Return the original train split length."""
        return len(self.dataset)

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
) -> ProbeReport:
    """Run a no-grad probe and restore the model's original train/eval state.

    Args:
        model: RF-DETR model module receiving a ``NestedTensor`` batch.
        postprocess: RF-DETR postprocessor returning per-image predictions.
        dataloader: Deterministic train-probe loader.
        device: Optional target device. If omitted, model parameters determine it.
        iou_threshold: IoU threshold used by :func:`match_predictions_to_target`.

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
                predictions = postprocess(outputs, target_sizes)
                results.extend(
                    match_predictions_to_target(prediction, target, iou_threshold=iou_threshold)
                    for prediction, target in zip(predictions, targets)
                )
    finally:
        model.train(was_training)
    return ProbeReport(tuple(results))
