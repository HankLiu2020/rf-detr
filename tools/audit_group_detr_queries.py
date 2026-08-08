# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Audit Group DETR per-group query slicing and active query dimensions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rfdetr.nas.controller import NativeBoundedElasticController
from rfdetr.nas.native_inspector import inspect_native_architecture
from rfdetr.nas.validation import validate_output_shapes
from rfdetr.nas.architecture import ArchitectureSpec
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
from rfdetr.variants import RFDETRSegSmall


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", default=Path("nas_artifacts/query_group_detr.json"), type=Path)
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
    active_queries = min(50, native.num_queries)
    before_query = model.query_feat.weight.detach().clone()
    before_refpoint = model.refpoint_embed.weight.detach().clone()

    model.train()
    model._nas_active_num_queries = active_queries
    grouped_refpoint, grouped_query = model._active_query_weights()
    expected_query = before_query.view(native.group_detr, native.num_queries, -1)[:, :active_queries].reshape(
        -1, before_query.shape[-1]
    )
    expected_refpoint = before_refpoint.view(native.group_detr, native.num_queries, -1)[:, :active_queries].reshape(
        -1, before_refpoint.shape[-1]
    )
    grouped_diff = max(
        float((grouped_query.detach() - expected_query).abs().max()),
        float((grouped_refpoint.detach() - expected_refpoint).abs().max()),
    )
    flat_query = before_query[: active_queries * native.group_detr]
    flat_diff = float((flat_query - expected_query).abs().max())
    if grouped_diff != 0.0 or (native.group_detr > 1 and flat_diff == 0.0):
        raise AssertionError("Group DETR query audit did not demonstrate per-group slicing")

    architecture = ArchitectureSpec(
        resolution=native.resolution,
        patch_size=native.patch_size,
        num_windows=native.num_windows,
        decoder_layers=native.decoder_layers,
        num_queries=active_queries,
        group_detr=native.group_detr,
        encoder=native.encoder,
    )
    controller.activate(architecture)
    model.eval()
    image = torch.zeros(3, native.resolution, native.resolution, device=device)
    with torch.no_grad():
        outputs = model(nested_tensor_from_tensor_list([image]))
    shapes = validate_output_shapes(outputs, architecture, training=False)
    if any(shape[1] != active_queries for shape in shapes.values()):
        raise AssertionError(f"eval query shape mismatch: {shapes}")
    report = {
        "status": "PASS",
        "group_detr": native.group_detr,
        "native_num_queries_per_group": native.num_queries,
        "active_num_queries_per_group": active_queries,
        "train_query_dimension": active_queries * native.group_detr,
        "eval_query_dimension": active_queries,
        "per_group_max_abs_diff": grouped_diff,
        "flat_slice_max_abs_diff": flat_diff,
        "postprocess_num_select": {
            "policy": "native_fixed",
            "native_num_select": native.num_select,
        },
        "eval_shapes": shapes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
