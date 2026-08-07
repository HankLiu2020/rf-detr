# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Pure unit coverage for Group DETR per-group query slicing."""

from __future__ import annotations

import torch
from torch import nn

from rfdetr.models.lwdetr import LWDETR


class _QueryOwner:
    _active_num_queries = LWDETR._active_num_queries
    _active_query_weights = LWDETR._active_query_weights

    def __init__(self) -> None:
        self.group_detr = 3
        self.num_queries = 4
        self.refpoint_embed = nn.Embedding(12, 2)
        self.query_feat = nn.Embedding(12, 2)
        self.training = True


def test_group_detr_query_slice_preserves_group_boundaries() -> None:
    owner = _QueryOwner()
    with torch.no_grad():
        owner.refpoint_embed.weight.copy_(torch.arange(24, dtype=torch.float32).view(12, 2))
        owner.query_feat.weight.copy_(torch.arange(24, dtype=torch.float32).view(12, 2) + 1000)
    owner._nas_active_num_queries = 2
    refpoint, query = owner._active_query_weights()
    expected = owner.query_feat.weight.view(3, 4, 2)[:, :2].reshape(-1, 2)
    expected_refpoint = owner.refpoint_embed.weight.view(3, 4, 2)[:, :2].reshape(-1, 2)
    assert torch.equal(query, expected)
    assert torch.equal(refpoint, expected_refpoint)
    assert not torch.equal(query, owner.query_feat.weight[:6])
