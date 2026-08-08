# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Runtime controller for native-bounded RF-DETR-Seg subnets."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from torch import nn

from rfdetr.nas.architecture import ArchitectureSpec, NativeArchitecture
from rfdetr.nas.elastic_patch import attach_elastic_patch_projection
from rfdetr.nas.position_embedding import find_position_embeddings
from rfdetr.nas.search_space import estimate_encoder_proposal_pool_size
from rfdetr.nas.validation import assert_distributed_architecture
from rfdetr.nas.windows import set_active_num_windows, validate_active_num_windows


@dataclass(frozen=True)
class NativeFieldSnapshot:
    """One immutable native module/config field captured before NAS activation."""

    module_path: str
    field_path: tuple[str, ...]
    value: Any
    value_type: str


@dataclass(frozen=True)
class NativeAttributeSnapshot:
    """One immutable runtime attribute snapshot, including absent attributes."""

    owner: str
    name: str
    present: bool
    value: Any
    value_type: str


@dataclass(frozen=True)
class NativeStateSnapshot:
    """Complete native baseline needed to undo controller-owned mutations."""

    patch_active_patch_size: int
    backbone_fields: tuple[NativeFieldSnapshot, ...]
    position_fields: tuple[NativeFieldSnapshot, ...]
    runtime_attributes: tuple[NativeAttributeSnapshot, ...]


@dataclass(frozen=True)
class ActivationState:
    """Auditable state applied to a model for one forward/backward window."""

    architecture: ArchitectureSpec
    patch_parameter_id: int
    patch_parameter_name: str
    patch_parameter_count: int
    touched_window_modules: tuple[str, ...]
    postprocess_num_select: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture.to_dict(),
            "patch_parameter_id": self.patch_parameter_id,
            "patch_parameter_name": self.patch_parameter_name,
            "patch_parameter_count": self.patch_parameter_count,
            "touched_window_modules": list(self.touched_window_modules),
            "postprocess_num_select": self.postprocess_num_select,
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


def _copy_snapshot_value(value: Any) -> Any:
    """Copy a native value so later activation cannot mutate the snapshot."""

    return copy.deepcopy(value)


def _field_value(owner: Any, field_path: tuple[str, ...]) -> Any:
    value = owner
    for name in field_path:
        value = getattr(value, name)
    return value


def _set_field_value(owner: Any, field_path: tuple[str, ...], value: Any) -> None:
    target = owner
    for name in field_path[:-1]:
        target = getattr(target, name)
    setattr(target, field_path[-1], _copy_snapshot_value(value))


def _snapshot_module_fields(module: nn.Module) -> tuple[NativeFieldSnapshot, ...]:
    """Capture patch/window metadata while retaining the exact native value types."""

    fields: list[NativeFieldSnapshot] = []
    for module_path, child in module.named_modules():
        for name in ("patch_size", "num_windows"):
            if hasattr(child, name):
                value = getattr(child, name)
                fields.append(
                    NativeFieldSnapshot(
                        module_path=module_path,
                        field_path=(name,),
                        value=_copy_snapshot_value(value),
                        value_type=type(value).__qualname__,
                    )
                )
        config = getattr(child, "config", None)
        if config is not None:
            for name in ("patch_size", "num_windows"):
                if hasattr(config, name):
                    value = getattr(config, name)
                    fields.append(
                        NativeFieldSnapshot(
                            module_path=module_path,
                            field_path=("config", name),
                            value=_copy_snapshot_value(value),
                            value_type=type(value).__qualname__,
                        )
                    )
    return tuple(fields)


def _snapshot_position_fields(embeddings: nn.Module) -> tuple[NativeFieldSnapshot, ...]:
    fields: list[NativeFieldSnapshot] = []
    for field_path in (("patch_size",), ("config", "patch_size")):
        try:
            value = _field_value(embeddings, field_path)
        except AttributeError:
            continue
        fields.append(
            NativeFieldSnapshot(
                module_path="",
                field_path=field_path,
                value=_copy_snapshot_value(value),
                value_type=type(value).__qualname__,
            )
        )
    return tuple(fields)


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
        backbone = getattr(self.model, "backbone", None)
        if backbone is None:
            raise RuntimeError("RF-DETR model has no backbone")
        self._native_snapshot = NativeStateSnapshot(
            patch_active_patch_size=int(self.patch.active_patch_size),
            backbone_fields=_snapshot_module_fields(backbone),
            position_fields=_snapshot_position_fields(self.position.embeddings),
            runtime_attributes=self._capture_runtime_attributes(),
        )

    @property
    def active(self) -> ArchitectureSpec | None:
        return self._active

    @property
    def native_snapshot(self) -> NativeStateSnapshot:
        """Return the immutable native baseline captured at controller creation."""

        return self._native_snapshot

    def _runtime_objects(self) -> dict[str, Any]:
        transformer = getattr(self.model, "transformer", None)
        objects: dict[str, Any] = {"model": self.model}
        if transformer is not None:
            objects["transformer"] = transformer
            decoder = getattr(transformer, "decoder", None)
            if decoder is not None:
                objects["decoder"] = decoder
        if self.context is not None:
            objects["context"] = self.context
            args = getattr(self.context, "args", None)
            if args is not None:
                objects["args"] = args
            postprocess = getattr(self.context, "postprocess", None)
            if postprocess is not None:
                objects["postprocess"] = postprocess
        return objects

    def _capture_runtime_attributes(self) -> tuple[NativeAttributeSnapshot, ...]:
        objects = self._runtime_objects()
        names_by_owner = {
            "model": (
                "_nas_active_num_queries",
                "_nas_active_decoder_layers",
                "_nas_active_num_select",
                "_nas_active_architecture",
            ),
            "transformer": ("_nas_active_num_queries", "_nas_active_decoder_layers", "_nas_active_num_select"),
            "decoder": ("_nas_active_decoder_layers",),
            "context": ("resolution",),
            "args": ("resolution", "patch_size", "num_windows", "dec_layers", "num_queries", "num_select"),
            "postprocess": ("num_select",),
        }
        snapshots: list[NativeAttributeSnapshot] = []
        for owner_name, names in names_by_owner.items():
            owner = objects.get(owner_name)
            if owner is None:
                continue
            for name in names:
                present = hasattr(owner, name)
                value = _copy_snapshot_value(getattr(owner, name)) if present else None
                snapshots.append(
                    NativeAttributeSnapshot(
                        owner=owner_name,
                        name=name,
                        present=present,
                        value=value,
                        value_type=type(value).__qualname__ if present else "<missing>",
                    )
                )
        return tuple(snapshots)

    def _assert_fixed_postprocess_policy(self) -> None:
        """Ensure every exposed PostProcess/config value remains native and fixed."""

        expected = int(self.native.num_select)
        objects = self._runtime_objects()
        for owner_name in ("args", "postprocess"):
            owner = objects.get(owner_name)
            if owner is None or not hasattr(owner, "num_select"):
                continue
            observed = int(getattr(owner, "num_select"))
            if observed != expected:
                raise RuntimeError(
                    "native PostProcess num_select policy is out of sync: "
                    f"{owner_name}.num_select={observed}, native_num_select={expected}"
                )

    def reset_to_native(self) -> None:
        """Restore all controller-owned mutable state to the captured native baseline."""

        snapshot = self._native_snapshot
        backbone = getattr(self.model, "backbone", None)
        if backbone is None:
            raise RuntimeError("RF-DETR model has no backbone")
        self.patch.set_active_patch_size(snapshot.patch_active_patch_size)
        modules = dict(backbone.named_modules())
        for field in snapshot.backbone_fields:
            owner = modules.get(field.module_path)
            if owner is None:
                raise RuntimeError(f"native snapshot module path disappeared: {field.module_path!r}")
            _set_field_value(owner, field.field_path, field.value)
        for field in snapshot.position_fields:
            _set_field_value(self.position.embeddings, field.field_path, field.value)

        objects = self._runtime_objects()
        for field in snapshot.runtime_attributes:
            owner = objects.get(field.owner)
            if owner is None:
                raise RuntimeError(f"native snapshot owner disappeared: {field.owner!r}")
            if field.present:
                setattr(owner, field.name, _copy_snapshot_value(field.value))
            elif hasattr(owner, field.name):
                delattr(owner, field.name)
        self._active = None

    def activate(self, architecture: ArchitectureSpec) -> ActivationState:
        """Validate and apply one architecture atomically."""

        architecture.validate(self.native)
        assert_distributed_architecture(architecture)
        proposal_count = estimate_encoder_proposal_pool_size(
            architecture.resolution,
            architecture.patch_size,
            self.native.projector_scale or ("P4",),
        )
        if proposal_count < architecture.num_queries:
            raise ValueError(
                "architecture rejected before forward: encoder_proposal_pool_size "
                f"{proposal_count} < active_num_queries {architecture.num_queries}"
            )

        self._assert_fixed_postprocess_policy()
        self.reset_to_native()
        try:
            return self._apply_architecture(architecture)
        except BaseException:
            self.reset_to_native()
            raise

    def _apply_architecture(self, architecture: ArchitectureSpec) -> ActivationState:
        """Apply a validated architecture; callers provide rollback protection."""

        self.patch.set_active_patch_size(architecture.patch_size)
        backbone = getattr(self.model, "backbone", None)
        if backbone is None:
            raise RuntimeError("RF-DETR model has no backbone")
        _set_patch_size_fields(backbone, architecture.patch_size)
        self.position.set_patch_size(architecture.patch_size)
        touched_windows = set_active_num_windows(backbone, architecture.num_windows)

        setattr(self.model, "_nas_active_num_queries", architecture.num_queries)
        setattr(self.model, "_nas_active_decoder_layers", architecture.decoder_layers)
        setattr(self.model.transformer, "_nas_active_num_queries", architecture.num_queries)
        setattr(self.model.transformer, "_nas_active_decoder_layers", architecture.decoder_layers)
        setattr(self.model.transformer.decoder, "_nas_active_decoder_layers", architecture.decoder_layers)
        setattr(self.model, "_nas_active_architecture", architecture.to_dict())

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
                ):
                    if hasattr(args, name):
                        setattr(args, name, value)

        self._active = architecture
        return ActivationState(
            architecture=architecture,
            patch_parameter_id=id(self.patch.master_weight),
            patch_parameter_name=self._native_parameter_name,
            patch_parameter_count=sum(1 for p in self.model.parameters() if p is self.patch.master_weight),
            touched_window_modules=touched_windows,
            postprocess_num_select=self.native.num_select,
        )

    def validate_active_architecture(self) -> None:
        """Validate all runtime fields before a model forward."""

        if self._active is None:
            raise RuntimeError("no architecture has been activated")
        architecture = self._active
        architecture.validate(self.native)
        self._assert_fixed_postprocess_policy()
        if getattr(self.model, "_nas_active_num_queries", None) != architecture.num_queries:
            raise RuntimeError("model active query count is out of sync")
        if getattr(self.model.transformer, "_nas_active_num_queries", None) != architecture.num_queries:
            raise RuntimeError("transformer active query count is out of sync")
        if getattr(self.model.transformer, "_nas_active_decoder_layers", None) != architecture.decoder_layers:
            raise RuntimeError("transformer active decoder count is out of sync")
        if getattr(self.model.transformer.decoder, "_nas_active_decoder_layers", None) != architecture.decoder_layers:
            raise RuntimeError("decoder active decoder count is out of sync")
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
            group_detr=self.native.group_detr,
            encoder=self.native.encoder,
            native=True,
            metadata={"native_identity": True},
        )
