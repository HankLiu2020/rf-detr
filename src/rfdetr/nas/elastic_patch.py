# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""FlexiViT-compatible runtime patch projection resampling.

The implementation follows the matrix/pseudoinverse construction used by
Timm's ``resample_patch_embed`` (itself based on Big Vision FlexiViT): resize
each old-grid basis vector to the new grid, form the resize matrix, and apply
its pseudoinverse to the one master convolution kernel.  It deliberately does
not interpolate the learned kernel directly and it does not create one
parameter per patch size.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


@lru_cache(maxsize=32)
def _resize_pseudoinverse(
    old_size: tuple[int, int],
    new_size: tuple[int, int],
    device_type: str,
    device_index: int,
) -> Tensor:
    """Build the fixed old-to-new basis resize pseudoinverse."""

    device = torch.device(device_type) if device_index < 0 else torch.device(device_type, device_index)
    old_pixels = old_size[0] * old_size[1]
    basis = torch.zeros(old_pixels, 1, *old_size, device=device, dtype=torch.float32)
    basis.view(old_pixels, -1)[torch.arange(old_pixels, device=device), torch.arange(old_pixels, device=device)] = 1.0
    resized = F.interpolate(basis, size=new_size, mode="bicubic", align_corners=False, antialias=True)
    resize_matrix = resized[:, 0].reshape(old_pixels, -1).transpose(0, 1).contiguous()
    return torch.linalg.pinv(resize_matrix)


def resample_patch_embed(weight: Tensor, new_size: tuple[int, int]) -> Tensor:
    """Resample a patch kernel using a single FlexiViT-style master weight."""

    if weight.ndim != 4:
        raise ValueError(f"patch projection weight must be 4-D, got {tuple(weight.shape)}")
    old_size = (int(weight.shape[-2]), int(weight.shape[-1]))
    new_size = (int(new_size[0]), int(new_size[1]))
    if old_size == new_size:
        return weight
    if min(new_size) <= 0:
        raise ValueError(f"new patch size must be positive, got {new_size}")
    pinv = _resize_pseudoinverse(
        old_size,
        new_size,
        weight.device.type,
        weight.device.index if weight.device.index is not None else -1,
    )
    flat = weight.reshape(weight.shape[0] * weight.shape[1], -1).to(dtype=torch.float32)
    resized = flat @ pinv
    return resized.reshape(weight.shape[0], weight.shape[1], *new_size).to(dtype=weight.dtype)


@dataclass
class ElasticPatchProjection:
    """Controller attached to an existing ``nn.Conv2d`` without replacing it."""

    projection: nn.Conv2d
    native_patch_size: int
    active_patch_size: int
    _original_forward: object

    @classmethod
    def attach(cls, projection: nn.Conv2d) -> "ElasticPatchProjection":
        """Attach once and preserve the original parameter/module identity."""

        existing = getattr(projection, "_nas_elastic_patch", None)
        if isinstance(existing, cls):
            return existing
        if projection.kernel_size[0] != projection.kernel_size[1]:
            raise ValueError("only square patch projections are supported")
        controller = cls(
            projection=projection,
            native_patch_size=int(projection.kernel_size[0]),
            active_patch_size=int(projection.kernel_size[0]),
            _original_forward=projection.forward,
        )
        projection._nas_elastic_patch = controller  # type: ignore[attr-defined]
        projection.forward = types.MethodType(_elastic_projection_forward, projection)  # type: ignore[method-assign]
        return controller

    @property
    def master_weight(self) -> nn.Parameter:
        """Return the original, sole patch projection parameter."""

        return self.projection.weight

    @property
    def master_bias(self) -> nn.Parameter | None:
        """Return the original bias parameter."""

        return self.projection.bias

    def set_active_patch_size(self, patch_size: int) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.active_patch_size = int(patch_size)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.active_patch_size == self.native_patch_size:
            return self._original_forward(inputs)  # type: ignore[operator]
        weight = resample_patch_embed(self.projection.weight, (self.active_patch_size, self.active_patch_size))
        return F.conv2d(
            inputs,
            weight,
            self.projection.bias,
            stride=(self.active_patch_size, self.active_patch_size),
            padding=self.projection.padding,
            dilation=self.projection.dilation,
            groups=self.projection.groups,
        )


def _elastic_projection_forward(projection: nn.Conv2d, inputs: Tensor) -> Tensor:
    controller = getattr(projection, "_nas_elastic_patch", None)
    if not isinstance(controller, ElasticPatchProjection):
        raise RuntimeError("elastic patch projection controller is missing")
    return controller.forward(inputs)


def find_patch_projection(module: nn.Module) -> nn.Conv2d:
    """Locate the DINOv2 patch projection in an RF-DETR module."""

    for name, child in module.named_modules():
        if name.endswith("patch_embeddings.projection") and isinstance(child, nn.Conv2d):
            return child
    raise RuntimeError("could not locate DINOv2 patch projection")


def attach_elastic_patch_projection(module: nn.Module) -> ElasticPatchProjection:
    """Attach the controller while preserving the projection parameter identity."""

    return ElasticPatchProjection.attach(find_patch_projection(module))
