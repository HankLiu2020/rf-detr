# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Audit segmentation loss terms across active resolutions without changing criterion code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rfdetr.models.lwdetr import build_criterion_from_config
from rfdetr.nas.architecture import ArchitectureSpec
from rfdetr.nas.controller import NativeBoundedElasticController
from rfdetr.nas.loss_audit import audit_loss_scale
from rfdetr.nas.native_inspector import inspect_native_architecture
from rfdetr.variants import RFDETRSegSmall
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list


def _target(device: torch.device, resolution: int) -> list[dict[str, torch.Tensor]]:
    mask = torch.zeros(1, resolution, resolution, dtype=torch.bool, device=device)
    mask[:, resolution // 4 : resolution // 2, resolution // 4 : resolution // 2] = True
    return [
        {
            "labels": torch.zeros(1, dtype=torch.int64, device=device),
            "boxes": torch.tensor([[0.375, 0.375, 0.25, 0.25]], device=device),
            "masks": mask,
            "orig_size": torch.tensor([resolution, resolution], dtype=torch.int64, device=device),
            "size": torch.tensor([resolution, resolution], dtype=torch.int64, device=device),
            "area": torch.tensor([resolution * resolution / 16], device=device),
            "iscrowd": torch.zeros(1, dtype=torch.int64, device=device),
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", default=Path("nas_artifacts/loss_scale_audit.json"), type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    detector = RFDETRSegSmall(pretrain_weights=str(args.checkpoint), device=str(device))
    context = detector.model
    if context.model is None:
        raise RuntimeError("model context has no model")
    model = context.model.to(device)
    native = inspect_native_architecture(detector, checkpoint_path=args.checkpoint)
    controller = NativeBoundedElasticController(model, native, context=context)
    train_config = detector.get_train_config(dataset_dir=".", output_dir=".")
    criterion, _ = build_criterion_from_config(detector.model_config, train_config)
    model.train()
    criterion.train()
    reports = []
    for resolution in (384, 480, 576):
        architecture = ArchitectureSpec(
            resolution=resolution,
            patch_size=12,
            num_windows=2,
            decoder_layers=native.decoder_layers,
            num_queries=native.num_queries,
            num_select=native.num_select,
            group_detr=native.group_detr,
            encoder=native.encoder,
        )
        controller.activate(architecture)
        images = torch.zeros(3, resolution, resolution, device=device)
        targets = _target(device, resolution)
        outputs = model(nested_tensor_from_tensor_list([images]), targets=targets)
        reports.append(audit_loss_scale(criterion, outputs, targets, architecture))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"status": "PASS", "policy": "NO_CHANGE", "records": reports}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "records": reports}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
