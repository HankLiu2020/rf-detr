# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Lifecycle tests for native state capture and controller reset."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rfdetr.nas.architecture import ArchitectureSpec, NativeArchitecture
from rfdetr.nas.controller import NativeBoundedElasticController


class _Embeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, 5, 4))
        self.patch_size = 2
        self.config = SimpleNamespace(patch_size=2)

    def interpolate_pos_encoding(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return tokens


class _Windowed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_windows = 2
        self.config = SimpleNamespace(num_windows=2)


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_size = (2, 2)
        self.config = SimpleNamespace(patch_size=2)
        self.embeddings = _Embeddings()
        self.patch_embeddings = nn.Module()
        self.patch_embeddings.projection = nn.Conv2d(3, 4, kernel_size=2, stride=2)
        self.windowed = _Windowed()


class _Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Module()


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()
        self.transformer = _Transformer()


def _native() -> NativeArchitecture:
    return NativeArchitecture(
        encoder="fake",
        resolution=8,
        patch_size=2,
        num_windows=2,
        decoder_layers=4,
        num_queries=100,
        num_select=5,
        group_detr=13,
        two_stage=True,
        bbox_reparam=True,
        lite_refpoint_refine=True,
        dec_pred_bbox_embed_share=False,
        hidden_dim=4,
        segmentation_head=True,
        mask_downsample_ratio=4,
        positional_encoding_size=4,
    )


def _architecture() -> ArchitectureSpec:
    return ArchitectureSpec(
        resolution=12,
        patch_size=3,
        num_windows=1,
        decoder_layers=2,
        num_queries=2,
        group_detr=13,
        encoder="fake",
    )


def test_native_snapshot_is_immutable_and_restores_values_and_types() -> None:
    model = _Model()
    args = SimpleNamespace(
        resolution=8,
        patch_size=2,
        num_windows=2,
        dec_layers=4,
        num_queries=100,
        num_select=5,
    )
    context = SimpleNamespace(model=model, resolution=8, args=args, postprocess=SimpleNamespace(num_select=5))
    controller = NativeBoundedElasticController(model, _native(), context=context)
    snapshot = controller.native_snapshot
    with pytest.raises(FrozenInstanceError):
        snapshot.patch_active_patch_size = 99  # type: ignore[misc]

    controller.activate(_architecture())
    controller.validate_active_architecture()
    model._nas_active_num_select = 999
    assert model.backbone.patch_size == (3, 3)
    assert type(model.backbone.patch_size) is tuple
    assert model.backbone.embeddings.patch_size == 3
    assert model.backbone.windowed.config.num_windows == 1
    assert context.resolution == 12
    assert context.args.num_queries == 2
    assert context.postprocess.num_select == 5

    controller.reset_to_native()
    assert controller.active is None
    assert controller.patch.active_patch_size == 2
    assert model.backbone.patch_size == (2, 2)
    assert type(model.backbone.patch_size) is tuple
    assert model.backbone.embeddings.patch_size == 2
    assert type(model.backbone.embeddings.patch_size) is int
    assert model.backbone.embeddings.config.patch_size == 2
    assert model.backbone.windowed.num_windows == 2
    assert model.backbone.windowed.config.num_windows == 2
    assert context.resolution == 8
    assert context.args.resolution == 8
    assert context.args.patch_size == 2
    assert context.args.num_windows == 2
    assert context.args.dec_layers == 4
    assert context.args.num_queries == 100
    assert context.args.num_select == 5
    assert context.postprocess.num_select == 5
    assert not hasattr(model, "_nas_active_num_select")
    assert not hasattr(model, "_nas_active_num_queries")
