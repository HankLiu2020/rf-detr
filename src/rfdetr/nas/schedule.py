# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Deterministic architecture scheduling and batch resizing."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from rfdetr.nas.architecture import ArchitectureSpec
from rfdetr.nas.search_space import sample_architecture


def architecture_for_step(
    architectures: Iterable[ArchitectureSpec],
    *,
    seed: int,
    optimizer_step: int,
    policy: str = "balanced_patch",
) -> ArchitectureSpec:
    """Return a deterministic architecture for an optimizer step."""

    if optimizer_step < 0:
        raise ValueError("optimizer_step must be non-negative")
    # A local RNG seed is derived without mutating global/DataLoader-worker RNG.
    step_seed = (int(seed) + 0x9E3779B9 * int(optimizer_step)) & 0xFFFFFFFF
    return sample_architecture(architectures, seed=step_seed, policy=policy)


def _resize_masks(masks: Tensor, resolution: int) -> Tensor:
    if masks.ndim != 3:
        raise ValueError(f"masks must have shape [N,H,W], got {tuple(masks.shape)}")
    if masks.shape[-2:] == (resolution, resolution):
        return masks
    return F.interpolate(masks[:, None].float(), size=(resolution, resolution), mode="nearest")[:, 0].to(masks.dtype)


def resize_batch_to_architecture(
    images: Tensor,
    targets: list[dict[str, Tensor]],
    architecture: ArchitectureSpec,
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    """Resize a canonical batch after architecture selection.

    RF-DETR targets store normalized ``cxcywh`` boxes, so boxes are kept
    unchanged; masks use nearest-neighbour interpolation and images use
    bilinear interpolation.
    """

    if images.ndim != 4 or images.shape[-2] != images.shape[-1]:
        raise ValueError("images must have shape [B,C,H,W] with a square canonical resolution")
    resolution = architecture.resolution
    resized_images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
    resized_targets: list[dict[str, Tensor]] = []
    for target in targets:
        copied = dict(target)
        if "masks" in copied:
            copied["masks"] = _resize_masks(copied["masks"], resolution)
        copied["size"] = torch.tensor([resolution, resolution], dtype=torch.int64, device=images.device)
        resized_targets.append(copied)
    return resized_images, resized_targets
