# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Dump the runtime native architecture of RF-DETR-Seg-Small."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rfdetr.nas.native_inspector import inspect_native_architecture
from rfdetr.variants import RFDETRSegSmall


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", default=Path("nas_artifacts/native_architecture.json"), type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    detector = RFDETRSegSmall(
        pretrain_weights=str(args.checkpoint),
        device=args.device,
    )
    context = detector.model
    if context.model is None:
        raise RuntimeError("RF-DETR model context unexpectedly has no model")
    context.model.to(torch.device(args.device))
    train_config = detector.get_train_config(dataset_dir=".", output_dir=".", batch_size=1, epochs=1)
    architecture = inspect_native_architecture(
        detector,
        checkpoint_path=args.checkpoint,
        train_config=train_config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(architecture.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paper = architecture.paper_expected
    runtime = architecture.to_dict()
    rows = [
        ("resolution", paper.get("resolution"), runtime["resolution"]),
        ("patch_size", paper.get("patch_size"), runtime["patch_size"]),
        ("num_windows", paper.get("num_windows"), runtime["num_windows"]),
        ("decoder_layers", paper.get("decoder_layers"), runtime["decoder_layers"]),
        ("num_queries", paper.get("num_queries"), runtime["num_queries"]),
        ("backbone", paper.get("backbone"), runtime["backbone"]),
    ]
    report_lines = [
        "# RF-DETR-Seg-Small Native vs Paper",
        "",
        "Runtime values are read from the constructed model/checkpoint; paper values are reference expectations only.",
        "",
        "| Field | Paper expected | Runtime | Status |",
        "|---|---:|---:|---|",
    ]
    for name, expected, observed in rows:
        status = (
            "PASS"
            if (name == "backbone" and str(observed).startswith("dinov2")) or expected == observed
            else "DIFF"
        )
        report_lines.append(f"| `{name}` | `{expected}` | `{observed}` | `{status}` |")
    report_lines.extend(
        [
            "",
            "## Discrepancies",
            "",
            *(
                [f"- {item}" for item in architecture.paper_runtime_discrepancies]
                or ["- None in paper-facing numeric fields."]
            ),
            "",
            "Checkpoint metadata discrepancies are recorded in `native_architecture.json` "
            "and are not silently inferred.",
        ]
    )
    report_path = args.output.parent / "native_vs_paper_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (args.output.parent / "native_discrepancy_report.md").write_text(
        report_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(json.dumps(architecture.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
