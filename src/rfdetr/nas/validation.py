# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Forward, shape, finite-value, and loss-audit helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from rfdetr.nas.architecture import ArchitectureSpec
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list


def assert_finite(value: Any, prefix: str = "value") -> None:
    """Raise for NaN/Inf tensors in nested model outputs or losses."""

    if isinstance(value, Tensor):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite tensor detected at {prefix}")
    elif isinstance(value, dict):
        for key, child in value.items():
            assert_finite(child, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_finite(child, f"{prefix}[{index}]")


def expected_query_count(architecture: ArchitectureSpec, training: bool) -> int:
    return architecture.num_queries * architecture.group_detr if training else architecture.num_queries


def assert_distributed_architecture(architecture: ArchitectureSpec) -> None:
    """Assert every initialized DDP rank selected the same active architecture."""

    distributed = torch.distributed
    if not distributed.is_available() or not distributed.is_initialized():
        return
    gathered: list[dict[str, Any] | None] = [None for _ in range(distributed.get_world_size())]
    distributed.all_gather_object(gathered, architecture.to_dict())
    canonical = gathered[0]
    if any(item != canonical for item in gathered[1:]):
        raise RuntimeError(f"DDP active architecture mismatch across ranks: {gathered}")


def validate_output_shapes(
    outputs: dict[str, Any],
    architecture: ArchitectureSpec,
    *,
    training: bool,
) -> dict[str, list[int]]:
    """Validate query dimensions for detection and segmentation outputs."""

    expected = expected_query_count(architecture, training)
    observed: dict[str, list[int]] = {}
    for key in ("pred_logits", "pred_boxes", "pred_masks", "pred_keypoints"):
        value = outputs.get(key)
        if not isinstance(value, Tensor):
            continue
        if value.shape[1] != expected:
            raise AssertionError(f"{key} query dimension {value.shape[1]} != expected {expected}")
        observed[key] = list(value.shape)
    assert_finite(outputs, "outputs")
    return observed


def forward_raw(model: nn.Module, images: Tensor, architecture: ArchitectureSpec) -> dict[str, Any]:
    """Run the raw LW-DETR path on a batch and validate active query shape."""

    nested = nested_tensor_from_tensor_list([image for image in images])
    outputs = model(nested)
    validate_output_shapes(outputs, architecture, training=model.training)
    return outputs


def parameter_group_snapshot(optimizer: torch.optim.Optimizer, model: nn.Module) -> dict[str, Any]:
    """Capture parameter identity/group membership and optimizer hyperparameters."""

    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    for index, group in enumerate(optimizer.param_groups):
        groups.append(
            {
                "index": index,
                "parameter_names": sorted(
                    names[id(parameter)] for parameter in group["params"] if id(parameter) in names
                ),
                "parameter_ids": sorted(id(parameter) for parameter in group["params"]),
                "lr": group.get("lr"),
                "weight_decay": group.get("weight_decay"),
            }
        )
    return {"groups": groups}


def gradient_snapshot(model: nn.Module) -> dict[str, Any]:
    """Summarize gradient presence and magnitudes."""

    values = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        values[name] = {"sum_abs": float(parameter.grad.detach().abs().sum()), "numel": parameter.grad.numel()}
    return values
