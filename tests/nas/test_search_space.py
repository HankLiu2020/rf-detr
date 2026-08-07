# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for native-derived search-space construction."""

from __future__ import annotations

from rfdetr.nas.architecture import NativeArchitecture
from rfdetr.nas.schedule import architecture_for_step
from rfdetr.nas.search_space import generate_search_space


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
        projector_scale=("P4",),
    )


def test_expected_valid_space_count() -> None:
    space = generate_search_space(_native())
    assert len({item.encoder_tuple for item in space.syntactically_valid}) == 22
    assert len(space.syntactically_valid) == 220
    assert sorted({item.num_queries for item in space.syntactically_valid}) == [50, 100]
    assert sorted({item.decoder_layers for item in space.syntactically_valid}) == [0, 1, 2, 3, 4]


def test_invalid_divisibility_is_rejected() -> None:
    space = generate_search_space(_native())
    assert space.invalid_divisibility
    assert all(item.invalid_reason for item in space.invalid_divisibility)


def test_schedule_is_deterministic() -> None:
    values = generate_search_space(_native()).syntactically_valid
    assert architecture_for_step(values, seed=7, optimizer_step=9) == architecture_for_step(
        values, seed=7, optimizer_step=9
    )

