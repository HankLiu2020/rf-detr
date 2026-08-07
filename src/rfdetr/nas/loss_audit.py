# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Resolution-aware segmentation loss scale audit."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from rfdetr.nas.architecture import ArchitectureSpec


def audit_loss_scale(
    criterion: torch.nn.Module,
    outputs: dict[str, Any],
    targets: list[dict[str, Tensor]],
    architecture: ArchitectureSpec,
) -> dict[str, Any]:
    """Record raw criterion terms and their normalizer without changing them."""

    if hasattr(criterion, "num_boxes_for_targets"):
        normalizer = criterion.num_boxes_for_targets(outputs, targets)  # type: ignore[attr-defined]
    else:
        normalizer = torch.tensor(
            float(max(1, sum(len(t["labels"]) for t in targets))),
            device=outputs["pred_logits"].device,
        )
    losses = criterion(outputs, targets)  # type: ignore[operator]
    weighted_loss = sum(
        value * criterion.weight_dict[key]
        for key, value in losses.items()
        if key in criterion.weight_dict
    )
    selected = {
        name: float(value.detach())
        for name, value in losses.items()
        if name.split("_")[0] in {"loss"} and any(token in name for token in ("ce", "bbox", "giou", "mask", "dice"))
    }
    pred_masks = outputs.get("pred_masks")
    if isinstance(pred_masks, Tensor):
        mask_shape = list(pred_masks.shape[-2:])
    elif isinstance(pred_masks, dict) and isinstance(pred_masks.get("spatial_features"), Tensor):
        mask_shape = list(pred_masks["spatial_features"].shape[-2:])
    else:
        mask_shape = None
    mask_point_sample_count = None
    ratio = getattr(criterion, "mask_point_sample_ratio", None)
    if mask_shape is not None and ratio:
        mask_point_sample_count = max(mask_shape[0], mask_shape[0] * mask_shape[1] // int(ratio))
    return {
        "architecture": architecture.to_dict(),
        "losses": selected,
        "weighted_loss": float(weighted_loss.detach()),
        "normalization_denominator": float(normalizer.detach()),
        "positive_instances": int(sum(len(target["labels"]) for target in targets)),
        "mask_shape": mask_shape,
        "mask_point_sample_ratio": ratio,
        "mask_point_sample_count": mask_point_sample_count,
        "policy": "NO_CHANGE_UNTIL_EVIDENCE",
    }
