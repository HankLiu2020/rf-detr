# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Serializable native and active architecture descriptions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class NativeArchitecture:
    """The measured capacity and immutable features of a loaded model."""

    encoder: str
    resolution: int
    patch_size: int
    num_windows: int
    decoder_layers: int
    num_queries: int
    num_select: int
    group_detr: int
    two_stage: bool
    bbox_reparam: bool
    lite_refpoint_refine: bool
    dec_pred_bbox_embed_share: bool
    hidden_dim: int
    segmentation_head: bool
    mask_downsample_ratio: int
    positional_encoding_size: int
    out_feature_indexes: tuple[int, ...] = ()
    projector_scale: tuple[str, ...] = ()
    query_capacity: int = 0
    query_embedding_shape: tuple[int, ...] = ()
    refpoint_embedding_shape: tuple[int, ...] = ()
    patch_projection_shape: tuple[int, ...] = ()
    position_embedding_shape: tuple[int, ...] = ()
    encoder_feature_shapes: tuple[tuple[int, ...], ...] = ()
    norm_counts: dict[str, int] = field(default_factory=dict)
    requires_norm_recalibration: bool = False
    checkpoint_args_present: tuple[str, ...] = ()
    checkpoint_args_missing: tuple[str, ...] = ()
    checkpoint_discrepancies: tuple[str, ...] = ()
    # Explicit dump fields required by rf-nas-plan.md.  The older fields above
    # remain the compact runtime representation used by the controller.
    model_variant: str = "RFDETRSegSmall"
    backbone: str = ""
    patch_embed_module: str = ""
    patch_embed_parameter_name: str = ""
    patch_embed_weight_shape: tuple[int, ...] = ()
    query_feat_shape: tuple[int, ...] = ()
    refpoint_embed_shape: tuple[int, ...] = ()
    encoder_proposal_count_at_native_resolution: int = 0
    segmentation_head_type: str = ""
    mask_head_per_layer: bool | None = None
    mask_feature_resolution: tuple[int, int] = ()
    norm_types: dict[str, int] = field(default_factory=dict)
    batchnorm_module_count: int = 0
    optimizer_param_groups: tuple[dict[str, Any], ...] = ()
    layer_decay_policy: dict[str, Any] = field(default_factory=dict)
    paper_expected: dict[str, Any] = field(
        default_factory=lambda: {
            "resolution": 384,
            "patch_size": 12,
            "num_windows": 2,
            "decoder_layers": 4,
            "num_queries": 100,
            "backbone": "DINOv2-S",
        }
    )
    paper_runtime_discrepancies: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible data."""

        return asdict(self)


@dataclass(frozen=True)
class ArchitectureSpec:
    """One legal active subnet bounded by :class:`NativeArchitecture`."""

    resolution: int
    patch_size: int
    num_windows: int
    decoder_layers: int
    num_queries: int
    num_select: int
    group_detr: int
    encoder: str
    native: bool = False
    hardware_feasible: bool | None = None
    invalid_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def encoder_tuple(self) -> tuple[int, int, int]:
        """Return ``(resolution, patch_size, num_windows)``."""

        return self.resolution, self.patch_size, self.num_windows

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible data."""

        return asdict(self)

    def validate(self, native: NativeArchitecture) -> None:
        """Raise when this spec exceeds native capacity or violates geometry."""

        if self.encoder != native.encoder:
            raise ValueError(f"encoder {self.encoder!r} differs from native encoder {native.encoder!r}")
        if self.patch_size <= 0 or self.num_windows <= 0:
            raise ValueError("patch_size and num_windows must be positive")
        if self.resolution <= 0 or self.resolution % (self.patch_size * self.num_windows) != 0:
            raise ValueError(
                "resolution must be divisible by patch_size * num_windows: "
                f"{self.resolution} % ({self.patch_size} * {self.num_windows}) != 0"
            )
        if self.decoder_layers < 0 or self.decoder_layers > native.decoder_layers:
            raise ValueError(f"decoder_layers {self.decoder_layers} exceeds native [0, {native.decoder_layers}]")
        if self.num_queries <= 0 or self.num_queries > native.num_queries:
            raise ValueError(f"num_queries {self.num_queries} exceeds native [1, {native.num_queries}]")
        if self.num_select <= 0 or self.num_select > self.num_queries:
            raise ValueError(f"num_select {self.num_select} must be in [1, num_queries={self.num_queries}]")
        if self.group_detr != native.group_detr:
            raise ValueError("group_detr is immutable for native-bounded search")


def architecture_dict(value: NativeArchitecture | ArchitectureSpec) -> dict[str, Any]:
    """Return a serializable dictionary for either architecture type."""

    return value.to_dict()
