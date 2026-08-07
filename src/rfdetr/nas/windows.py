# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Dynamic window controls and exact partition/reverse helpers."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def window_partition(features: Tensor, num_windows: int) -> Tensor:
    """Partition ``B,C,H,W`` into non-overlapping windows exactly."""

    if features.ndim != 4:
        raise ValueError("features must have shape [B,C,H,W]")
    batch, channels, height, width = features.shape
    if num_windows <= 0 or height % num_windows != 0 or width % num_windows != 0:
        raise ValueError("height and width must be divisible by num_windows")
    window_height, window_width = height // num_windows, width // num_windows
    return (
        features.view(batch, channels, num_windows, window_height, num_windows, window_width)
        .permute(0, 2, 4, 3, 5, 1)
        .contiguous()
        .view(batch * num_windows * num_windows, window_height, window_width, channels)
    )


def window_reverse(windows: Tensor, num_windows: int, height: int, width: int) -> Tensor:
    """Reverse :func:`window_partition` with exact equality for integer tensors."""

    if windows.ndim != 4:
        raise ValueError("windows must have shape [B*num_windows**2,Wh,Ww,C]")
    batch_windows, window_height, window_width, channels = windows.shape
    expected = batch_windows
    if height % num_windows != 0 or width % num_windows != 0:
        raise ValueError("height and width must be divisible by num_windows")
    batch = expected // (num_windows * num_windows)
    if batch * num_windows * num_windows != expected:
        raise ValueError("window batch dimension is inconsistent with num_windows")
    if (window_height, window_width) != (height // num_windows, width // num_windows):
        raise ValueError("window tensor shape does not match target height/width")
    return (
        windows.view(batch, num_windows, num_windows, window_height, window_width, channels)
        .permute(0, 5, 1, 3, 2, 4)
        .contiguous()
        .view(batch, channels, height, width)
    )


def _windowed_modules(module: nn.Module) -> list[nn.Module]:
    found: list[nn.Module] = []
    for child in module.modules():
        if hasattr(child, "num_windows") and (
            hasattr(child, "config") or "Windowed" in type(child).__name__ or "Dino" in type(child).__name__
        ):
            found.append(child)
    return found


def set_active_num_windows(module: nn.Module, num_windows: int) -> tuple[str, ...]:
    """Synchronize every window-bearing DINO module through one entry point."""

    if num_windows <= 0:
        raise ValueError("num_windows must be positive")
    touched: list[str] = []
    for name, child in module.named_modules():
        if child not in _windowed_modules(module):
            continue
        child.num_windows = int(num_windows)  # type: ignore[attr-defined]
        config = getattr(child, "config", None)
        if config is not None and hasattr(config, "num_windows"):
            config.num_windows = int(num_windows)
        touched.append(name or "<root>")
    if not touched:
        raise RuntimeError("no window-bearing modules found")
    validate_active_num_windows(module, num_windows)
    return tuple(touched)


def validate_active_num_windows(module: nn.Module, expected: int) -> None:
    """Raise if any synchronized window-bearing module disagrees."""

    observed: list[tuple[str, int]] = []
    targets = set(_windowed_modules(module))
    for name, child in module.named_modules():
        if child not in targets:
            continue
        observed.append((name or "<root>", int(child.num_windows)))  # type: ignore[attr-defined]
        config = getattr(child, "config", None)
        if config is not None and hasattr(config, "num_windows"):
            observed.append((f"{name or '<root>'}.config", int(config.num_windows)))
    bad = [(name, value) for name, value in observed if value != expected]
    if bad:
        raise RuntimeError(f"active num_windows mismatch: expected {expected}, observed {bad}")
