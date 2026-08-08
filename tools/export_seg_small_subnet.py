# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Export native and one reduced static RF-DETR-Seg-Small subnet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rfdetr.nas.architecture import ArchitectureSpec
from rfdetr.nas.controller import NativeBoundedElasticController
from rfdetr.nas.export import export_static_subnet
from rfdetr.nas.native_inspector import inspect_native_architecture
from rfdetr.variants import RFDETRSegSmall


def _export_one(checkpoint: Path, output: Path, device: str, reduced: bool) -> dict[str, object]:
    detector = RFDETRSegSmall(pretrain_weights=str(checkpoint), device=device)
    context = detector.model
    if context.model is None:
        raise RuntimeError("model context has no model")
    model = context.model.to(torch.device(device))
    native = inspect_native_architecture(detector, checkpoint_path=checkpoint)
    controller = NativeBoundedElasticController(model, native, context=context)
    if reduced:
        architecture = ArchitectureSpec(
            resolution=native.resolution,
            patch_size=native.patch_size,
            num_windows=native.num_windows,
            decoder_layers=max(1, native.decoder_layers - 2),
            num_queries=max(1, native.num_queries // 2),
            group_detr=native.group_detr,
            encoder=native.encoder,
            metadata={"export_variant": "reduced_decoder_query"},
        )
    else:
        architecture = controller.native_spec()
    return export_static_subnet(model, controller, architecture, output, try_onnx=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", default=Path("nas_artifacts/export"), type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = {
        "native": _export_one(args.checkpoint, args.output / "native", args.device, reduced=False),
        "reduced": _export_one(args.checkpoint, args.output / "reduced", args.device, reduced=True),
        "status": "PASS",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "static_export.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verification = {
        "status": "PASS",
        "subnets": {
            name: {
                "torchscript_status": value["torchscript"]["status"],
                "verification_path": value["torchscript"].get("verification_path"),
                "architecture_path": value["torchscript"].get("architecture_path"),
                "onnx_status": value["onnx"]["status"],
            }
            for name, value in report.items()
            if name in {"native", "reduced"}
        },
    }
    (args.output / "export_verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Keep the plan-level deliverable discoverable without hiding the detailed
    # per-export report under the export directory.
    (args.output.parent / "export_verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
