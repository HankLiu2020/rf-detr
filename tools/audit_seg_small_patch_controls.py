# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Exercise the six required 480px patch/window controls on one model object."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rfdetr.models.lwdetr import build_criterion_from_config
from rfdetr.nas.architecture import ArchitectureSpec
from rfdetr.nas.controller import NativeBoundedElasticController
from rfdetr.nas.native_inspector import inspect_native_architecture
from rfdetr.nas.validation import assert_finite, validate_output_shapes
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
from rfdetr.variants import RFDETRSegSmall


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
    parser.add_argument("--output", default=Path("nas_artifacts/patch_window_controls.json"), type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    detector = RFDETRSegSmall(pretrain_weights=str(args.checkpoint), device=str(device))
    context = detector.model
    if context.model is None:
        raise RuntimeError("model context has no model")
    model = context.model.to(device)
    model.eval()
    train_config = detector.get_train_config(dataset_dir=".", output_dir=str(args.output.parent))
    native = inspect_native_architecture(
        detector,
        checkpoint_path=args.checkpoint,
        train_config=train_config,
    )
    controller = NativeBoundedElasticController(model, native, context=context)
    criterion, _ = build_criterion_from_config(detector.model_config, train_config)
    criterion.train()
    patch_parameter = controller.patch.master_weight
    patch_name = controller._native_parameter_name
    position_master_digest_before = controller.position.master_digest()
    patch_state_keys = [
        name for name in model.state_dict() if name.endswith("patch_embeddings.projection.weight")
    ]
    if patch_state_keys != [patch_name]:
        raise AssertionError(
            "state_dict must contain exactly the native master patch weight: "
            f"expected={[patch_name]}, observed={patch_state_keys}"
        )
    native_architecture = controller.native_spec()
    controller.activate(native_architecture)
    with torch.no_grad():
        native_before = model(nested_tensor_from_tensor_list([torch.zeros(3, 384, 384, device=device)]))
    records = []
    for patch_size in (12, 16, 20):
        for window_count in (1, 2):
            architecture = ArchitectureSpec(
                resolution=480,
                patch_size=patch_size,
                num_windows=window_count,
                decoder_layers=native.decoder_layers,
                num_queries=native.num_queries,
                num_select=native.num_select,
                group_detr=native.group_detr,
                encoder=native.encoder,
            )
            controller.activate(architecture)
            controller.validate_active_architecture()
            model.eval()
            with torch.no_grad():
                outputs = model(nested_tensor_from_tensor_list([torch.zeros(3, 480, 480, device=device)]))
            shapes = validate_output_shapes(outputs, architecture, training=False)
            model.train()
            model.zero_grad(set_to_none=True)
            images = torch.full((3, 480, 480), 0.5, device=device)
            train_outputs = model(nested_tensor_from_tensor_list([images]), targets=_target(device, 480))
            losses = criterion(train_outputs, _target(device, 480))
            weighted = sum(
                value * criterion.weight_dict[key] for key, value in losses.items() if key in criterion.weight_dict
            )
            weighted.backward()
            assert_finite(losses, "patch_control_losses")
            grad_sum = float(patch_parameter.grad.detach().abs().sum()) if patch_parameter.grad is not None else 0.0
            if grad_sum == 0.0:
                raise RuntimeError(f"patch master gradient is zero for patch={patch_size}, window={window_count}")
            model.zero_grad(set_to_none=True)
            model.eval()
            records.append(
                {
                    "architecture": architecture.to_dict(),
                    "shapes": shapes,
                    "patch_parameter_id": id(patch_parameter),
                    "patch_parameter_name": patch_name,
                    "patch_parameter_occurrences": sum(
                        1 for parameter in model.parameters() if parameter is patch_parameter
                    ),
                    "patch_master_grad_sum_abs": grad_sum,
                    "status": "PASS",
                }
            )
    controller.activate(native_architecture)
    with torch.no_grad():
        native_after = model(nested_tensor_from_tensor_list([torch.zeros(3, 384, 384, device=device)]))
    native_diffs = {
        key: float((native_before[key] - native_after[key]).abs().max())
        for key in ("pred_logits", "pred_boxes", "pred_masks")
    }
    position_master_digest_after = controller.position.master_digest()
    if any(value != 0.0 for value in native_diffs.values()):
        raise AssertionError(f"native state did not restore after patch/window matrix: {native_diffs}")
    if position_master_digest_before != position_master_digest_after:
        raise AssertionError("position master representation changed during patch/window matrix")
    report = {
        "status": "PASS",
        "native_state_restore": native_diffs,
        "position_master_digest_before": position_master_digest_before,
        "position_master_digest_after": position_master_digest_after,
        "position_master_unchanged": position_master_digest_before == position_master_digest_after,
        "master_parameter_name": patch_name,
        "master_parameter_id": id(patch_parameter),
        "state_dict_patch_weight_count": len(patch_state_keys),
        "state_dict_patch_weight_keys": patch_state_keys,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
