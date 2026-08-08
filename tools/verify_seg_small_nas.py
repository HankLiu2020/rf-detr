# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Verify native equivalence and a bounded set of RF-DETR-Seg subnets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch

from rfdetr.models.lwdetr import build_criterion_from_config
from rfdetr.nas.architecture import ArchitectureSpec
from rfdetr.nas.controller import NativeBoundedElasticController
from rfdetr.nas.native_inspector import inspect_native_architecture
from rfdetr.nas.norm_recalibration import write_norm_audit
from rfdetr.nas.search_space import generate_search_space, write_search_space
from rfdetr.nas.validation import assert_finite, gradient_snapshot, validate_output_shapes
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
from rfdetr.variants import RFDETRSegSmall


def _raw_forward(model: torch.nn.Module, resolution: int) -> dict[str, Any]:
    device = next(model.parameters()).device
    image = torch.zeros(3, resolution, resolution, device=device)
    return model(nested_tensor_from_tensor_list([image]))


def _synthetic_target(device: torch.device, resolution: int) -> list[dict[str, torch.Tensor]]:
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


def _compare_tensors(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    report = {}
    for key in ("pred_logits", "pred_boxes", "pred_masks"):
        left, right = before.get(key), after.get(key)
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            report[key] = {
                "shape": list(left.shape),
                "max_abs_diff": float((left - right).detach().abs().max()),
                "allclose": bool(torch.allclose(left, right, atol=1e-6, rtol=1e-5)),
            }
    return report


def _shape_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): _shape_tree(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shape_tree(child) for child in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", default=Path("nas_artifacts"), type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--random-count", type=int, default=10)
    args = parser.parse_args()
    if args.random_count <= 0:
        raise ValueError("random-count must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    detector = RFDETRSegSmall(pretrain_weights=str(args.checkpoint), device=str(device))
    context = detector.model
    if context.model is None:
        raise RuntimeError("model context has no model")
    model = context.model.to(device)
    model.eval()
    train_config = detector.get_train_config(dataset_dir=".", output_dir=str(args.output / "loss_audit"))
    native = inspect_native_architecture(
        detector,
        checkpoint_path=args.checkpoint,
        train_config=train_config,
    )
    controller = NativeBoundedElasticController(model, native, context=context)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    # Baseline guard: controlled native activation must match the unactivated path.
    before = _raw_forward(model, native.resolution)
    native_state = controller.activate(controller.native_spec())
    controller.validate_active_architecture()
    after = _raw_forward(model, native.resolution)
    native_comparison = _compare_tensors(before, after)
    native_equivalent = all(item["allclose"] for item in native_comparison.values())
    (output / "baseline").mkdir(exist_ok=True)
    (output / "baseline" / "native_equivalence.json").write_text(
        json.dumps({"status": "PASS" if native_equivalent else "FAIL", "comparison": native_comparison}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    if not native_equivalent:
        raise AssertionError(f"native path changed under controller: {native_comparison}")

    space = generate_search_space(native)
    write_search_space(space, output / "search_space")
    write_norm_audit(model, output / "norm_audit.json")

    # Worst-corner probe runs before random sampling and records OOM as a hardware result.
    worst = ArchitectureSpec(
        resolution=576,
        patch_size=12,
        num_windows=1,
        decoder_layers=native.decoder_layers,
        num_queries=native.num_queries,
        group_detr=native.group_detr,
        encoder=native.encoder,
        metadata={"probe": "worst_corner"},
    )
    hardware: dict[str, Any] = {
        "architecture": worst.to_dict(),
        "token_count_estimate": (worst.resolution // worst.patch_size) ** 2,
        "token_squared_estimate": ((worst.resolution // worst.patch_size) ** 2) ** 2,
        "feature_shape_estimate": [
            1,
            native.hidden_dim,
            worst.resolution // worst.patch_size,
            worst.resolution // worst.patch_size,
        ],
        "mask_feature_shape_estimate": [
            1,
            worst.num_queries,
            worst.resolution // native.mask_downsample_ratio,
            worst.resolution // native.mask_downsample_ratio,
        ],
        "estimated_attention_flops": int(
            2
            * ((worst.resolution // worst.patch_size) ** 4)
            * native.hidden_dim
            + 4 * ((worst.resolution // worst.patch_size) ** 2) * native.hidden_dim**2
        ),
    }
    try:
        controller.activate(worst)
        if device.type != "cuda":
            hardware.update(
                {
                    "hardware_feasible": None,
                    "reason": "CPU_PROBE_ONLY_TARGET_GPU_UNVERIFIED",
                    "probe_device": str(device),
                }
            )
        else:
            with torch.no_grad():
                worst_outputs = _raw_forward(model, worst.resolution)
            assert_finite(worst_outputs, "worst_outputs")
            hardware.update(
                {
                    "hardware_feasible": True,
                    "probe_device": str(device),
                    "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                }
            )
    except torch.cuda.OutOfMemoryError as exc:
        hardware.update({"hardware_feasible": False, "reason": "OOM_ON_TARGET_GPU", "error": str(exc)})
        if device.type == "cuda":
            torch.cuda.empty_cache()
    hardware_text = json.dumps(hardware, indent=2, sort_keys=True) + "\n"
    (output / "hardware_feasibility.json").write_text(hardware_text, encoding="utf-8")
    (output / "hardware_feasibility.jsonl").write_text(json.dumps(hardware, sort_keys=True) + "\n", encoding="utf-8")
    if hardware.get("hardware_feasible") is True:
        hardware_pool_status = "WORST_CORNER_PASS_TARGET_GPU_POOL"
    elif hardware.get("hardware_feasible") is False:
        hardware_pool_status = "WORST_CORNER_OOM_TARGET_GPU_UNVERIFIED_POOL"
    else:
        hardware_pool_status = "CPU_ONLY_TARGET_GPU_UNVERIFIED_POOL"

    valid = [item for item in space.syntactically_valid if item.decoder_layers in range(native.decoder_layers + 1)]
    random_architectures = random.Random(args.seed).sample(valid, min(args.random_count, len(valid)))
    random_results: list[dict[str, Any]] = []
    backward_checks = 0
    for index, architecture in enumerate(random_architectures):
        if hardware.get("hardware_feasible") is False and architecture.encoder_tuple == worst.encoder_tuple:
            continue
        controller.activate(architecture)
        controller.validate_active_architecture()
        with torch.no_grad():
            outputs = _raw_forward(model, architecture.resolution)
        shapes = validate_output_shapes(outputs, architecture, training=False)
        postprocess = getattr(context, "postprocess", None)
        if postprocess is None:
            raise RuntimeError("model context has no postprocess module")
        with torch.no_grad():
            processed = postprocess(
                outputs,
                torch.tensor([[architecture.resolution, architecture.resolution]], device=device),
            )
        assert_finite(processed, "random_postprocess")
        random_results.append(
            {
                "index": index,
                "architecture": architecture.to_dict(),
                "hardware_pool_status": hardware_pool_status,
                "shapes": shapes,
                "postprocess_shapes": _shape_tree(processed),
                "status": "PASS",
            }
        )

        if backward_checks < 3 and architecture.decoder_layers > 0:
            backward_checks += 1
            model.train()
            images = torch.zeros(1, 3, architecture.resolution, architecture.resolution, device=device)
            targets = _synthetic_target(device, architecture.resolution)
            train_outputs = model(nested_tensor_from_tensor_list([images[0]]), targets=targets)
            train_config = detector.get_train_config(dataset_dir=".", output_dir=str(output / "loss_audit"))
            criterion, _ = build_criterion_from_config(detector.model_config, train_config)
            losses = criterion(train_outputs, targets)
            weighted = sum(
                value * criterion.weight_dict[key]
                for key, value in losses.items()
                if key in criterion.weight_dict
            )
            weighted.backward()
            assert_finite(losses, "random_losses")
            if not gradient_snapshot(model):
                raise RuntimeError("random architecture backward produced no gradients")
            model.zero_grad(set_to_none=True)
            model.eval()

    if backward_checks < min(3, sum(item.decoder_layers > 0 for item in random_architectures)):
        raise RuntimeError("random architecture set did not provide the requested three backward checks")

    coverage = {
        "patch_sizes": sorted({item["architecture"]["patch_size"] for item in random_results}),
        "num_windows": sorted({item["architecture"]["num_windows"] for item in random_results}),
        "num_queries": sorted({item["architecture"]["num_queries"] for item in random_results}),
        "decoder_layers": sorted({item["architecture"]["decoder_layers"] for item in random_results}),
    }
    required_coverage = {
        "patch_sizes": sorted({12, 16, 20} | {native.patch_size}),
        "num_windows": sorted({1, 2} | {native.num_windows}),
        "num_queries": sorted({min(50, native.num_queries), native.num_queries}),
        "decoder_layers": sorted({0, native.decoder_layers}),
    }
    missing_coverage = {
        key: sorted(set(values) - set(coverage[key]))
        for key, values in required_coverage.items()
        if not set(values).issubset(set(coverage[key]))
    }
    if missing_coverage:
        raise RuntimeError(
            "fixed-seed random sample missed required coverage; add an explicit recorded supplement: "
            f"{missing_coverage}"
        )

    # A-B-C-A must restore the native output and position master state.
    controller.activate(controller.native_spec())
    state_a = _raw_forward(model, native.resolution)
    controller.activate(random_architectures[0])
    _raw_forward(model, random_architectures[0].resolution)
    controller.activate(random_architectures[-1])
    _raw_forward(model, random_architectures[-1].resolution)
    controller.activate(controller.native_spec())
    state_a_again = _raw_forward(model, native.resolution)
    pollution = _compare_tensors(state_a, state_a_again)
    if not all(item["allclose"] for item in pollution.values()):
        raise AssertionError(f"A-B-C-A state pollution detected: {pollution}")
    report = {
        "native": native.to_dict(),
        "native_activation": native_state.to_dict(),
        "native_equivalence": native_comparison,
        "random_architectures": random_results,
        "hardware_pool_status": hardware_pool_status,
        "sampled_from_target_hardware_qualified_pool": hardware.get("hardware_feasible") is True,
        "coverage": coverage,
        "required_coverage": required_coverage,
        "backward_checks": backward_checks,
        "state_pollution": pollution,
        "status": "PASS",
    }
    (output / "random_architectures.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "smoke_architectures.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in random_results), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
