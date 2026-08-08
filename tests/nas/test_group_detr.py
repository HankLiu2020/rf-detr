# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Pure unit coverage for Group DETR per-group query slicing."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from rfdetr.models.postprocess import PostProcess
from rfdetr.nas.architecture import ArchitectureSpec, NativeArchitecture
from rfdetr.models.lwdetr import LWDETR
from rfdetr.nas.validation import assert_query_width_invariant, validate_output_shapes


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


def _native() -> NativeArchitecture:
    return NativeArchitecture(
        encoder="dinov2_windowed_small",
        resolution=384,
        patch_size=12,
        num_windows=2,
        decoder_layers=4,
        num_queries=100,
        num_select=100,
        group_detr=13,
        two_stage=True,
        bbox_reparam=True,
        lite_refpoint_refine=True,
        dec_pred_bbox_embed_share=False,
        hidden_dim=256,
        segmentation_head=True,
        mask_downsample_ratio=4,
        positional_encoding_size=32,
    )


def _architecture(*, decoder_layers: int = 0, num_queries: int = 50) -> ArchitectureSpec:
    native = _native()
    return ArchitectureSpec(
        resolution=native.resolution,
        patch_size=native.patch_size,
        num_windows=native.num_windows,
        decoder_layers=decoder_layers,
        num_queries=num_queries,
        group_detr=native.group_detr,
        encoder=native.encoder,
    )


def test_group_detr_query_width_invariant_uses_total_width_not_per_group_divisibility() -> None:
    architecture = _architecture(num_queries=50)
    assert assert_query_width_invariant(650, architecture, training=True) == 650
    assert assert_query_width_invariant(50, architecture, training=False) == 50
    with pytest.raises(AssertionError, match="expected Q_total 650"):
        assert_query_width_invariant(50, architecture, training=True)


def test_decoder_zero_eval_output_can_flow_through_postprocess() -> None:
    architecture = _architecture(decoder_layers=0, num_queries=50)
    outputs = {
        "pred_logits": torch.zeros(1, 50, 2),
        "pred_boxes": torch.full((1, 50, 4), 0.5),
        "pred_masks": torch.zeros(1, 50, 4, 4),
    }
    shapes = validate_output_shapes(outputs, architecture, training=False)
    assert shapes["pred_logits"][1] == 50
    processed = PostProcess(num_select=_native().num_select)(outputs, torch.tensor([[16, 16]]))
    assert processed[0]["boxes"].shape[-1] == 4
    assert processed[0]["masks"].shape[1:] == (1, 16, 16)
