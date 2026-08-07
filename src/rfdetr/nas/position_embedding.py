# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Master position-embedding interpolation helpers."""

from __future__ import annotations

import hashlib

import torch
from torch import Tensor, nn


class PositionEmbeddingController:
    """Use the model's existing master position parameter without overwriting it."""

    def __init__(self, embeddings: nn.Module) -> None:
        if not hasattr(embeddings, "position_embeddings") or not hasattr(embeddings, "interpolate_pos_encoding"):
            raise TypeError("embeddings module does not expose DINOv2 position interpolation")
        self.embeddings = embeddings
        self.master = embeddings.position_embeddings

    def set_patch_size(self, patch_size: int) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.embeddings.patch_size = patch_size
        self.embeddings.config.patch_size = patch_size

    def interpolate(self, height: int, width: int, patch_size: int | None = None) -> Tensor:
        """Return interpolated positions for a runtime grid, leaving the master intact."""

        if patch_size is not None:
            self.set_patch_size(patch_size)
        active_patch = int(self.embeddings.config.patch_size)
        if height % active_patch != 0 or width % active_patch != 0:
            raise ValueError("runtime height and width must be divisible by the active patch size")
        tokens = (height // active_patch) * (width // active_patch) + 1
        dummy = self.master.new_zeros((1, tokens, self.master.shape[-1]))
        return self.embeddings.interpolate_pos_encoding(dummy, height, width)

    def master_digest(self) -> str:
        """Return a stable digest useful for A-B-C-A state-pollution tests."""

        return hashlib.sha256(self.master.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def find_position_embeddings(module: nn.Module) -> PositionEmbeddingController:
    """Locate the DINOv2 embeddings module."""

    for _, child in module.named_modules():
        if hasattr(child, "position_embeddings") and hasattr(child, "interpolate_pos_encoding"):
            return PositionEmbeddingController(child)
    raise RuntimeError("could not locate DINOv2 position embeddings")
