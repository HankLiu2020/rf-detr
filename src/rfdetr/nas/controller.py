# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Runtime controller for native-bounded RF-DETR-Seg subnets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

from rfdetr.nas.architecture import ArchitectureSpec, NativeArchitecture
from rfdetr.nas.elastic_patch import attach_elastic_patch_projection
from rfdetr.nas.position_embedding import PositionEmbeddingController, find_position_embeddings
from rfdetr.nas.search_space import estimate_proposal_count_per_group
from rfdetr.nas.validation import assert_distributed_architecture
from rfdetr.nas.windows import set_active_num_windows, validate_active_num_windows


@dataclass(frozen=True)
class ActivationState:
    """Auditable state applied to a model for one forward/backward window."""

    architecture: ArchitectureSpec
    patch_parameter_id: int
    patch_parameter_name: str
    patch_parameter_count: int
    touched_window_modules: tuple[str, ...]
    effective_num_select: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture.to_dict(),
            "patch_parameter_id": self.patch_parameter_id,
            "patch_parameter_name": self.patch_parameter_name,
            "patch_parameter_count": self.patch_parameter_count,
            "touched_window_modules": list(self.touched_window_modules),
            "effective_num_select": self.effective_num_select,
        }


def _find_parameter_name(module: nn.Module, parameter: nn.Parameter) -> str:
    for name, candidate in module.named_parameters():
        if candidate is parameter:
            return name
    raise RuntimeError("patch parameter is not present in the model parameter tree")


def _set_patch_size_fields(backbone: nn.Module, patch_size: int) -> None:
    """Synchronize patch metadata without replacing any parameter or module."""

    for child in backbone.modules():
        if hasattr(child, "patch_size"):
            current = getattr(child, "patch_size")
            setattr(child, "patch_size", (patch_size, patch_size) if isinstance(current, tuple) else patch_size)
        config = getattr(child, "config", None)
        if config is not None and hasattr(config, "patch_size"):
            config.patch_size = patch_size


class NativeBoundedElasticController:
    """Apply legal encoder/query/decoder subnets to one loaded LW-DETR model."""

    def __init__(self, model: nn.Module, native: NativeArchitecture, context: Any | None = None) -> None:
        self.model = model
        self.native = native
        self.context = context
        self.patch = attach_elastic_patch_projection(model)
        self.position = find_position_embeddings(model)
        self._native_parameter_name = _find_parameter_name(model, self.patch.master_weight)
        self._active: ArchitectureSpec | None = None

    @property
    def active(self) -> ArchitectureSpec | None:
        return self._active

    def activate(self, architecture: ArchitectureSpec) -> ActivationState:
        """Validate and apply one architecture atomically."""

        architecture.validate(self.native)
        assert_distributed_architecture(architecture)
        proposal_count = estimate_proposal_count_per_group(
            architecture.resolution,
            architecture.patch_size,
            self.native.projector_scale or ("P4",),
        )
        if proposal_count < architecture.num_queries:
            raise ValueError(
                "architecture rejected before forward: proposal_count_per_group "
                f"{proposal_count} < active_num_queries {architecture.num_queries}"
            )

        self.patch.set_active_patch_size(architecture.patch_size)
        backbone = getattr(self.model, "backbone", None)
        if backbone is None:
            raise RuntimeError("RF-DETR model has no backbone")
        _set_patch_size_fields(backbone, architecture.patch_size)
        self.position.set_patch_size(architecture.patch_size)
        touched_windows = set_active_num_windows(backbone, architecture.num_windows)

        setattr(self.model, "_nas_active_num_queries", architecture.num_queries)
        setattr(self.model, "_nas_active_decoder_layers", architecture.decoder_layers)
        setattr(self.model, "_nas_active_num_select", architecture.num_select)
        setattr(self.model.transformer, "_nas_active_num_queries", architecture.num_queries)
        setattr(self.model.transformer, "_nas_active_decoder_layers", architecture.decoder_layers)
        setattr(self.model.transformer.decoder, "_nas_active_decoder_layers", architecture.decoder_layers)
        setattr(self.model, "_nas_active_architecture", architecture.to_dict())
        self._active = architecture

        if self.context is not None:
            self.context.resolution = architecture.resolution
            args = getattr(self.context, "args", None)
            if args is not None:
                for name, value in (
                    ("resolution", architecture.resolution),
                    ("patch_size", architecture.patch_size),
                    ("num_windows", architecture.num_windows),
                    ("dec_layers", architecture.decoder_layers),
                    ("num_queries", architecture.num_queries),
                    ("num_select", architecture.num_select),
                ):
                    if hasattr(args, name):
                        setattr(args, name, value)
            postprocess = getattr(self.context, "postprocess", None)
            if postprocess is not None and hasattr(postprocess, "num_select"):
                postprocess.num_select = architecture.num_select

        return ActivationState(
            architecture=architecture,
            patch_parameter_id=id(self.patch.master_weight),
            patch_parameter_name=self._native_parameter_name,
            patch_parameter_count=sum(1 for p in self.model.parameters() if p is self.patch.master_weight),
            touched_window_modules=touched_windows,
            effective_num_select=architecture.num_select,
        )

    def validate_active_architecture(self) -> None:
        """Validate all runtime fields before a model forward."""

        if self._active is None:
            raise RuntimeError("no architecture has been activated")
        architecture = self._active
        architecture.validate(self.native)
        if getattr(self.model, "_nas_active_num_queries", None) != architecture.num_queries:
            raise RuntimeError("model active query count is out of sync")
        if getattr(self.model.transformer, "_nas_active_num_queries", None) != architecture.num_queries:
            raise RuntimeError("transformer active query count is out of sync")
        if getattr(self.model.transformer, "_nas_active_decoder_layers", None) != architecture.decoder_layers:
            raise RuntimeError("transformer active decoder count is out of sync")
        if self.patch.active_patch_size != architecture.patch_size:
            raise RuntimeError("patch controller is out of sync")
        validate_active_num_windows(self.model.backbone, architecture.num_windows)
        if self.patch_parameter_count() != 1:
            raise RuntimeError("elastic patch projection must expose exactly one master parameter")

    def patch_parameter_count(self) -> int:
        return sum(1 for parameter in self.model.parameters() if parameter is self.patch.master_weight)

    def native_spec(self) -> ArchitectureSpec:
        """Build the exact native architecture spec from the dump."""

        return ArchitectureSpec(
            resolution=self.native.resolution,
            patch_size=self.native.patch_size,
            num_windows=self.native.num_windows,
            decoder_layers=self.native.decoder_layers,
            num_queries=self.native.num_queries,
            num_select=self.native.num_select,
            group_detr=self.native.group_detr,
            encoder=self.native.encoder,
            native=True,
            metadata={"native_identity": True},
        )
