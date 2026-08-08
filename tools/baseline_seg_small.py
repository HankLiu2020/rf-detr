# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Capture one fixed native Small-Seg baseline batch and its audit artifacts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from rfdetr.models.lwdetr import build_criterion_from_config
from rfdetr.nas.architecture import ArchitectureSpec
from rfdetr.nas.controller import NativeBoundedElasticController
from rfdetr.nas.data import MVTecSegmentationDataset
from rfdetr.nas.native_inspector import inspect_native_architecture
from rfdetr.nas.schedule import resize_batch_to_architecture
from rfdetr.nas.validation import assert_finite, gradient_snapshot
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
from rfdetr.variants import RFDETRSegSmall


def _move_target(target: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in target.items()}


def _first_positive(dataset: MVTecSegmentationDataset) -> int:
    for index in range(len(dataset)):
        if dataset[index][1]["labels"].numel() > 0:
            return index
    raise RuntimeError("baseline requires one positive MVTec sample with a non-empty mask")


def _summary(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        result: dict[str, Any] = {"shape": list(detached.shape), "dtype": str(detached.dtype)}
        if detached.numel() > 0 and (detached.is_floating_point() or detached.is_complex()):
            result.update({"min": float(detached.min()), "max": float(detached.max())})
        return result
    if isinstance(value, dict):
        return {str(key): _summary(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summary(child) for child in value]
    return value


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_cpu(child) for child in value)
    if isinstance(value, list):
        return [_to_cpu(child) for child in value]
    return value


def _compare_trees(left: Any, right: Any, path: str = "value") -> dict[str, Any]:
    """Compare tensor-bearing output trees without silently skipping aux/mask branches."""

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape:
            return {path: {"status": "SHAPE_MISMATCH", "left": list(left.shape), "right": list(right.shape)}}
        if not (left.is_floating_point() or left.is_complex()):
            equal = bool(torch.equal(left, right))
            return {
                path: {
                    "status": "PASS",
                    "shape": list(left.shape),
                    "max_abs_diff": 0.0 if equal else 1.0,
                    "allclose": equal,
                }
            }
        difference = (left.detach() - right.detach()).abs()
        return {
            path: {
                "status": "PASS",
                "shape": list(left.shape),
                "max_abs_diff": float(difference.max()) if difference.numel() else 0.0,
                "allclose": bool(torch.allclose(left, right, atol=1e-5, rtol=1e-5)),
            }
        }
    if isinstance(left, dict) and isinstance(right, dict):
        keys = sorted(set(left) | set(right))
        result: dict[str, Any] = {}
        for key in keys:
            if key not in left or key not in right:
                result[f"{path}.{key}"] = {"status": "KEY_MISMATCH"}
            else:
                result.update(_compare_trees(left[key], right[key], f"{path}.{key}"))
        return result
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        result = {}
        if len(left) != len(right):
            result[path] = {"status": "LENGTH_MISMATCH", "left": len(left), "right": len(right)}
            return result
        for index, (left_child, right_child) in enumerate(zip(left, right)):
            result.update(_compare_trees(left_child, right_child, f"{path}[{index}]"))
        return result
    return {path: {"status": "PASS", "equal": left == right}}


def _comparison_passes(comparison: dict[str, Any]) -> bool:
    for value in comparison.values():
        if value.get("status") is not None:
            if value.get("status") != "PASS":
                return False
            if not value.get("allclose", value.get("equal", False)):
                return False
        elif not value.get("allclose", value.get("equal", False)):
            return False
    return True


def _matching_equal(left: Any, right: Any) -> bool:
    if len(left) != len(right):
        return False
    return all(
        torch.equal(left_pair[0], right_pair[0]) and torch.equal(left_pair[1], right_pair[1])
        for left_pair, right_pair in zip(left, right)
    )


def _loss_comparison(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict[str, Any]:
    keys = sorted(set(left) | set(right))
    result: dict[str, Any] = {}
    for key in keys:
        if key not in left or key not in right:
            result[key] = {"status": "KEY_MISMATCH"}
            continue
        difference = float((left[key].detach() - right[key].detach()).abs())
        result[key] = {
            "left": float(left[key].detach()),
            "right": float(right[key].detach()),
            "abs_diff": difference,
            "allclose": difference <= 1e-5,
        }
    return result


def _max_cuda_memory(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return int(torch.cuda.max_memory_allocated(device))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", default=Path("nas_artifacts/baseline"), type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    detector = RFDETRSegSmall(pretrain_weights=str(args.checkpoint), device=str(device))
    context = detector.model
    if context.model is None:
        raise RuntimeError("model context has no model")
    model = context.model.to(device)
    train_config = detector.get_train_config(dataset_dir=str(args.dataset_root), output_dir=str(args.output))
    native = inspect_native_architecture(
        detector,
        checkpoint_path=args.checkpoint,
        train_config=train_config,
    )
    dataset = MVTecSegmentationDataset(args.dataset_root, category="pill", split="test")
    sample_index = _first_positive(dataset)
    image, target = dataset[sample_index]
    native_spec = ArchitectureSpec(
        resolution=native.resolution,
        patch_size=native.patch_size,
        num_windows=native.num_windows,
        decoder_layers=native.decoder_layers,
        num_queries=native.num_queries,
        group_detr=native.group_detr,
        encoder=native.encoder,
        native=True,
    )
    # The first half intentionally does not instantiate the NAS controller. It
    # records the untouched path at the measured native resolution and keeps
    # all tensors needed for the three-level equivalence guard.
    images, targets = resize_batch_to_architecture(
        image.unsqueeze(0).to(device),
        [_move_target(target, device)],
        native_spec,
    )
    nested = nested_tensor_from_tensor_list([images[0]])

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(images.detach().cpu(), output_dir / "input.pt")
    torch.save(_to_cpu(targets), output_dir / "targets.pt")

    criterion, _ = build_criterion_from_config(detector.model_config, train_config)
    criterion.train()
    postprocess = getattr(context, "postprocess", None)
    if postprocess is None:
        raise RuntimeError("model context has no postprocess module")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    eval_start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        eval_outputs_disabled = model(nested)
    eval_time_disabled = time.perf_counter() - eval_start
    assert_finite(eval_outputs_disabled, "baseline.eval_outputs_disabled")
    with torch.no_grad():
        processed_disabled = postprocess(eval_outputs_disabled, torch.stack([targets[0]["orig_size"]]))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_start = time.perf_counter()
    model.train()
    train_outputs_disabled = model(nested, targets=targets)
    train_time_disabled = time.perf_counter() - train_start
    assert_finite(train_outputs_disabled, "baseline.train_outputs_disabled")
    outputs_without_aux = {key: value for key, value in train_outputs_disabled.items() if key != "aux_outputs"}
    matching_disabled = criterion.matcher(outputs_without_aux, targets, group_detr=criterion.group_detr)
    criterion_torch_rng = torch.get_rng_state()
    criterion_cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    losses_disabled = criterion(train_outputs_disabled, targets)
    weighted_disabled = sum(
        value * criterion.weight_dict[key]
        for key, value in losses_disabled.items()
        if key in criterion.weight_dict
    )
    backward_start = time.perf_counter()
    weighted_disabled.backward()
    backward_time_disabled = time.perf_counter() - backward_start
    gradients = gradient_snapshot(model)
    assert_finite(losses_disabled, "baseline.losses_disabled")
    if not gradients:
        raise RuntimeError("baseline backward produced no gradients")
    model.zero_grad(set_to_none=True)

    # Level 1/2/3: compare the ordinary path against controller-enabled native
    # activation on the same model, with no optimizer update between the runs.
    controller = NativeBoundedElasticController(model, native, context=context)
    controller.activate(controller.native_spec())
    controller.validate_active_architecture()
    model.eval()
    with torch.no_grad():
        eval_outputs_native = model(nested)
        processed_native = postprocess(eval_outputs_native, torch.stack([targets[0]["orig_size"]]))
    model.train()
    train_outputs_native = model(nested, targets=targets)
    assert_finite(eval_outputs_native, "baseline.eval_outputs_native")
    assert_finite(train_outputs_native, "baseline.train_outputs_native")
    matching_native = criterion.matcher(
        {key: value for key, value in train_outputs_native.items() if key != "aux_outputs"},
        targets,
        group_detr=criterion.group_detr,
    )
    torch.set_rng_state(criterion_torch_rng)
    if criterion_cuda_rng is not None:
        torch.cuda.set_rng_state_all(criterion_cuda_rng)
    losses_native = criterion(train_outputs_native, targets)
    raw_comparison = _compare_trees(eval_outputs_disabled, eval_outputs_native, "eval")
    train_comparison = _compare_trees(train_outputs_disabled, train_outputs_native, "train")
    loss_comparison = _loss_comparison(losses_disabled, losses_native)
    native_equivalence = {
        "level_1_raw_output": raw_comparison,
        "level_2_matching_exact": _matching_equal(matching_disabled, matching_native),
        "level_3_loss_components": loss_comparison,
        "postprocess": _compare_trees(processed_disabled, processed_native, "postprocess"),
        "status": "PENDING",
    }
    # Avoid using the dict under construction in its own status expression.
    native_equivalence["status"] = (
        "PASS"
        if _comparison_passes(raw_comparison)
        and _comparison_passes(train_comparison)
        and native_equivalence["level_2_matching_exact"]
        and _comparison_passes(loss_comparison)
        and _comparison_passes(native_equivalence["postprocess"])
        else "FAIL"
    )
    if native_equivalence["status"] != "PASS":
        raise AssertionError(f"baseline/native equivalence failed: {native_equivalence}")

    torch.save(_to_cpu({"eval": eval_outputs_disabled, "train": train_outputs_disabled}), output_dir / "raw_outputs.pt")
    torch.save(_to_cpu(matching_disabled), output_dir / "matching_indices.pt")
    torch.save(_to_cpu(processed_disabled), output_dir / "postprocessed.pt")
    (output_dir / "loss_components.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "weighted_loss": float(weighted_disabled.detach()),
                "losses": {key: float(value.detach()) for key, value in losses_disabled.items()},
                "native_losses": {key: float(value.detach()) for key, value in losses_native.items()},
                "equivalence": native_equivalence,
                "gradient_parameter_count": len(gradients),
                "matching_lengths": [int(pair[0].numel()) for pair in matching_disabled],
                "normalization_denominator": float(
                    criterion.num_boxes_for_targets(train_outputs_disabled, targets).detach()
                ),
                "backward_success": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "postprocessed.json").write_text(
        json.dumps(_summary(processed_disabled), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    model.zero_grad(set_to_none=True)
    report = {
        "status": "PASS",
        "baseline_path_status": "PASS",
        "baseline_path_comparison": "NAS-disabled ordinary path vs controller-enabled native path",
        "device": str(device),
        "sample_index": sample_index,
        "sample_path": target["path"],
        "native": native.to_dict(),
        "native_equivalence": native_equivalence,
        "shapes": {
            "eval": _summary(eval_outputs_disabled),
            "train": _summary(train_outputs_disabled),
        },
        "aux_outputs_count": len(train_outputs_disabled.get("aux_outputs", [])),
        "timing_seconds": {
            "eval_disabled": eval_time_disabled,
            "train_disabled": train_time_disabled,
            "backward_disabled": backward_time_disabled,
        },
        "peak_memory_bytes": _max_cuda_memory(device),
        "files": [
            "input.pt",
            "targets.pt",
            "raw_outputs.pt",
            "matching_indices.pt",
            "loss_components.json",
            "postprocessed.json",
            "postprocessed.pt",
        ],
    }
    (output_dir / "baseline_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
