# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Static PyTorch/ONNX export helpers for an activated subnet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from rfdetr.nas.architecture import ArchitectureSpec
from rfdetr.nas.controller import NativeBoundedElasticController


def _tensor_leaves(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        leaves: list[torch.Tensor] = []
        for key in sorted(value):
            leaves.extend(_tensor_leaves(value[key]))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = []
        for child in value:
            leaves.extend(_tensor_leaves(child))
        return leaves
    return []


def export_static_subnet(
    model: nn.Module,
    controller: NativeBoundedElasticController,
    architecture: ArchitectureSpec,
    output_dir: str | Path,
    *,
    resolution: int | None = None,
    try_onnx: bool = True,
) -> dict[str, Any]:
    """Export a fixed-shape activated subnet and record unsupported formats."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    controller.activate(architecture)
    controller.validate_active_architecture()
    model.eval()
    shape = int(resolution or architecture.resolution)
    dummy = torch.zeros(1, 3, shape, shape, device=next(model.parameters()).device)
    model.export()
    with torch.no_grad():
        eager = model(dummy)
    torchscript_path = output / "subnet.pt"
    traced = torch.jit.trace(model, dummy, strict=False)
    traced.save(str(torchscript_path))
    loaded = torch.jit.load(str(torchscript_path), map_location=dummy.device)
    with torch.no_grad():
        loaded_output = loaded(dummy)
    eager_leaves = _tensor_leaves(eager)
    loaded_leaves = _tensor_leaves(loaded_output)
    if len(eager_leaves) != len(loaded_leaves):
        raise RuntimeError(
            f"static export output count changed: eager={len(eager_leaves)} loaded={len(loaded_leaves)}"
        )
    output_diffs = []
    for eager_value, loaded_value in zip(eager_leaves, loaded_leaves):
        if eager_value.shape != loaded_value.shape:
            raise RuntimeError(
                f"static export output shape changed: eager={tuple(eager_value.shape)} "
                f"loaded={tuple(loaded_value.shape)}"
            )
        difference = (eager_value - loaded_value).abs()
        output_diffs.append(
            {
                "shape": list(eager_value.shape),
                "max_abs_diff": float(difference.max()) if difference.numel() else 0.0,
                "allclose": bool(torch.allclose(eager_value, loaded_value, atol=1e-5, rtol=1e-5)),
            }
        )
    verification = {
        "status": "PASS" if all(item["allclose"] for item in output_diffs) else "FAIL",
        "controller_free_loaded_torchscript": True,
        "output_count": len(output_diffs),
        "outputs": output_diffs,
    }
    if verification["status"] != "PASS":
        raise RuntimeError(f"static export numerical verification failed: {verification}")
    (output / "architecture.json").write_text(
        json.dumps(architecture.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report: dict[str, Any] = {
        "architecture": architecture.to_dict(),
        "torchscript": {
            "status": "PASS",
            "path": str(torchscript_path),
            "architecture_path": str(output / "architecture.json"),
            "verification_path": str(output / "verification.json"),
            "output_count": len(eager_leaves),
        },
        "onnx": {"status": "SKIPPED"},
    }
    if try_onnx:
        onnx_path = output / "subnet.onnx"
        try:
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                input_names=["images"],
                output_names=["boxes", "logits", "masks"],
                opset_version=17,
                dynamo=False,
            )
            report["onnx"] = {"status": "PASS", "path": str(onnx_path)}
        except Exception as exc:  # ONNX is an optional export backend.
            report["onnx"] = {"status": "UNSUPPORTED", "reason": str(exc)}
    (output / "export_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
