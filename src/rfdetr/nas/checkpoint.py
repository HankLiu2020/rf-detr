# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Checkpoint and deterministic-resume helpers for NAS smoke training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch import nn

from rfdetr.nas.architecture import ArchitectureSpec


def save_nas_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
    *,
    optimizer_step: int,
    architecture: ArchitectureSpec,
    history: list[dict[str, Any]],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema_state: dict[str, Any] | None = None,
    search_space_config: dict[str, Any] | None = None,
    sampling_policy: str | None = None,
    seed: int | None = None,
    schedule_preview: list[dict[str, Any]] | None = None,
    native_architecture: dict[str, Any] | None = None,
    git_commit: str | None = None,
    rfdetr_version: str | None = None,
    base_checkpoint: dict[str, Any] | None = None,
) -> None:
    """Save all state needed to reproduce a bounded NAS smoke resume."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "rfdetr-nas-smoke-v1",
        "optimizer_step": int(optimizer_step),
        "architecture": architecture.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "ema": ema_state,
        "search_space_config": search_space_config,
        "sampling_policy": sampling_policy,
        "seed": seed,
        "schedule_preview": schedule_preview or [],
        "native_architecture": native_architecture,
        "git_commit": git_commit,
        "rfdetr_version": rfdetr_version,
        "base_checkpoint": base_checkpoint,
        "history": history,
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(payload, destination)


def load_nas_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore a smoke checkpoint and return its auditable metadata."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "rfdetr-nas-smoke-v1":
        raise ValueError("unsupported NAS checkpoint format")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        random.setstate(payload["rng"]["python"])
        torch.set_rng_state(payload["rng"]["torch"])
        if torch.cuda.is_available() and payload["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    return {
        "optimizer_step": int(payload["optimizer_step"]),
        "architecture": payload["architecture"],
        "history": payload.get("history", []),
        "scheduler": payload.get("scheduler"),
        "ema": payload.get("ema"),
        "search_space_config": payload.get("search_space_config"),
        "sampling_policy": payload.get("sampling_policy"),
        "seed": payload.get("seed"),
        "schedule_preview": payload.get("schedule_preview", []),
        "native_architecture": payload.get("native_architecture"),
        "git_commit": payload.get("git_commit"),
        "rfdetr_version": payload.get("rfdetr_version"),
        "base_checkpoint": payload.get("base_checkpoint"),
    }
