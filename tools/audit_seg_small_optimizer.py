# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Verify NAS activation preserves RF-DETR optimizer parameter grouping."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch

from rfdetr.nas.controller import NativeBoundedElasticController
from rfdetr.nas.native_inspector import inspect_native_architecture
from rfdetr._namespace import _namespace_from_configs
from rfdetr.variants import RFDETRSegSmall


def _load_get_param_dict() -> Any:
    source = Path(__file__).resolve().parents[1] / "src/rfdetr/training/param_groups.py"
    spec = importlib.util.spec_from_file_location("rfdetr_nas_param_groups", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load optimizer grouping source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_param_dict


def _snapshot(model: torch.nn.Module, groups: list[dict[str, Any]], default_weight_decay: float) -> dict[str, Any]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    records = []
    for index, group in enumerate(groups):
        parameter = group["params"]
        records.append(
            {
                "index": index,
                "name": names[id(parameter)],
                "parameter_id": id(parameter),
                "lr": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", default_weight_decay)),
            }
        )
    return {
        "parameter_count": len(records),
        "group_count": len(records),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", default=Path("nas_artifacts/optimizer_layer_decay_audit.json"), type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    detector = RFDETRSegSmall(pretrain_weights=str(args.checkpoint), device=str(device))
    context = detector.model
    if context.model is None:
        raise RuntimeError("model context has no model")
    model = context.model.to(device)
    native = inspect_native_architecture(detector, checkpoint_path=args.checkpoint)
    train_config = detector.get_train_config(dataset_dir=".", output_dir=".")
    namespace = _namespace_from_configs(detector.model_config, train_config)
    get_param_dict = _load_get_param_dict()
    before = _snapshot(model, get_param_dict(namespace, model), float(namespace.weight_decay))
    controller = NativeBoundedElasticController(model, native, context=context)
    controller.activate(controller.native_spec())
    after_native = _snapshot(model, get_param_dict(namespace, model), float(namespace.weight_decay))
    if before["parameter_count"] != after_native["parameter_count"]:
        raise AssertionError("native NAS activation changed optimizer parameter count")
    before_by_name = {record["name"]: record for record in before["records"]}
    after_by_name = {record["name"]: record for record in after_native["records"]}
    if set(before_by_name) != set(after_by_name):
        raise AssertionError("native NAS activation changed optimizer parameter names")
    changed = [
        name
        for name in before_by_name
        if before_by_name[name]["parameter_id"] != after_by_name[name]["parameter_id"]
        or before_by_name[name]["lr"] != after_by_name[name]["lr"]
        or before_by_name[name]["weight_decay"] != after_by_name[name]["weight_decay"]
    ]
    if changed:
        raise AssertionError(f"native NAS activation changed optimizer groups: {changed[:5]}")
    patch_name = controller._native_parameter_name
    patch_occurrences = sum(record["name"] == patch_name for record in after_native["records"])
    if patch_occurrences != 1:
        raise AssertionError(f"patch master occurs {patch_occurrences} times in optimizer groups")
    report = {
        "status": "PASS",
        "patch_master_parameter": patch_name,
        "patch_master_optimizer_occurrences": patch_occurrences,
        "before": before,
        "after_native": after_native,
        "unchanged_fields": ["parameter_identity", "parameter_names", "lr", "weight_decay"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    canonical = args.output.parent / "optimizer_group_audit.json"
    if canonical != args.output:
        canonical.write_text(serialized, encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"before", "after_native"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
