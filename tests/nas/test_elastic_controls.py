# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for pure elastic patch and window operations."""

from __future__ import annotations

import torch
from torch import nn

from rfdetr.nas.elastic_patch import ElasticPatchProjection, resample_patch_embed
from rfdetr.nas.windows import window_partition, window_reverse


def test_patch_identity_and_master_gradient() -> None:
    projection = nn.Conv2d(3, 4, kernel_size=12, stride=12)
    controller = ElasticPatchProjection.attach(projection)
    master = projection.weight
    assert resample_patch_embed(master, (12, 12)) is master
    controller.set_active_patch_size(16)
    output = controller.forward(torch.randn(1, 3, 32, 32)).sum()
    output.backward()
    assert projection.weight is master
    assert projection.weight.grad is not None
    assert projection.weight.grad.abs().sum() > 0


def test_window_partition_reverse_is_exact() -> None:
    features = torch.arange(2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
    windows = window_partition(features, 2)
    assert torch.equal(window_reverse(windows, 2, 8, 8), features)
