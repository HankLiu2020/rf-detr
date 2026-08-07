# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Normalization audit and optional running-stat recalibration."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn


def audit_norms(model: nn.Module) -> dict[str, Any]:
    """Enumerate LayerNorm/GroupNorm/InstanceNorm/BatchNorm variants."""

    counts: Counter[str] = Counter()
    running_stats: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._NormBase):
            counts[type(module).__name__] += 1
            if module.track_running_stats:
                running_stats.append(name)
        elif isinstance(module, nn.LayerNorm) or type(module).__name__ == "LayerNorm":
            counts["LayerNorm"] += 1
        elif isinstance(module, nn.GroupNorm):
            counts["GroupNorm"] += 1
        elif isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            counts[type(module).__name__] += 1
    return {
        "counts": dict(sorted(counts.items())),
        "norm_types": dict(sorted(counts.items())),
        "batchnorm_module_count": sum(
            count for name, count in counts.items() if "BatchNorm" in name or "SyncBatchNorm" in name
        ),
        "running_stat_modules": sorted(running_stats),
        "requires_norm_recalibration": bool(running_stats),
    }


def write_norm_audit(model: nn.Module, output: str | Path) -> dict[str, Any]:
    """Write the norm audit JSON artifact."""

    report = audit_norms(model)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


@torch.no_grad()
def recalibrate_norm_statistics(
    model: nn.Module,
    calibration_loader: Iterable[Any],
    num_batches: int,
) -> int:
    """Update running-stat norms for a bounded calibration pass.

    The model's original training/eval flags are restored.  For this
    checkpoint the audit reports no running-stat norms, so normal use is a
    no-op; the implementation exists for future subnet evaluation and tests.
    """

    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    running_modules = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._NormBase)]
    running_modules = [module for module in running_modules if module.track_running_stats]
    if not running_modules:
        return 0
    states = {module: module.training for module in model.modules()}
    try:
        model.train()
        for module in running_modules:
            module.reset_running_stats()
        consumed = 0
        for batch in calibration_loader:
            if consumed >= num_batches:
                break
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch
            model(images)
            consumed += 1
        return consumed
    finally:
        for module, training in states.items():
            module.train(training)
