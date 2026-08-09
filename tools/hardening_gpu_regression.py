# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------
"""Run the bounded target-GPU regression for the hardened Seg-Small NAS core.

This is deliberately a fixed four-case probe.  It does not enumerate the
search space, train a supernet, evaluate a subnet pool, or run Pareto search.
"""

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
from rfdetr.nas.native_inspector import inspect_native_architecture
from rfdetr.nas.validation import assert_finite, gradient_snapshot, validate_output_shapes
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
from rfdetr.variants import RFDETRSegSmall


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


def _raw_forward(model: torch.nn.Module, resolution: int) -> dict[str, Any]:
    device = next(model.parameters()).device
    image = torch.zeros(3, resolution, resolution, device=device)
    return model(nested_tensor_from_tensor_list([image]))


def _compare_outputs(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for key in ("pred_logits", "pred_boxes", "pred_masks"):
        left_value, right_value = left.get(key), right.get(key)
        if isinstance(left_value, torch.Tensor) and isinstance(right_value, torch.Tensor):
            comparison[key] = {
                "shape": list(left_value.shape),
                "max_abs_diff": float((left_value - right_value).abs().max().detach().cpu()),
                "allclose": bool(torch.allclose(left_value, right_value, atol=1e-6, rtol=1e-5)),
            }
    return comparison


def _architecture(
    native: Any,
    *,
    resolution: int,
    patch_size: int,
    num_windows: int,
    decoder_layers: int,
    num_queries: int,
) -> ArchitectureSpec:
    return ArchitectureSpec(
        resolution=resolution,
        patch_size=patch_size,
        num_windows=num_windows,
        decoder_layers=decoder_layers,
        num_queries=num_queries,
        group_detr=native.group_detr,
        encoder=native.encoder,
        native=(
            resolution == native.resolution
            and patch_size == native.patch_size
            and num_windows == native.num_windows
            and decoder_layers == native.decoder_layers
            and num_queries == native.num_queries
        ),
        metadata={"probe": "hardening_gpu_regression"},
    )


def _run_case(
    model: torch.nn.Module,
    controller: NativeBoundedElasticController,
    architecture: ArchitectureSpec,
    criterion: torch.nn.Module,
    train_config: Any,
    device: torch.device,
    *,
    backward: bool,
) -> dict[str, Any]:
    controller.activate(architecture)
    controller.validate_active_architecture()
    torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = _raw_forward(model, architecture.resolution)
    elapsed = time.perf_counter() - start
    assert_finite(outputs, f"{architecture.metadata.get('probe', 'case')}.outputs")
    output_shapes = validate_output_shapes(outputs, architecture, training=False)
    record: dict[str, Any] = {
        "architecture": architecture.to_dict(),
        "forward_status": "PASS",
        "forward_seconds": elapsed,
        "output_shapes": output_shapes,
        "peak_memory_bytes_forward": int(torch.cuda.max_memory_allocated(device)),
        "backward_status": "SKIPPED",
    }

    if architecture.decoder_layers == 0:
        postprocess = getattr(getattr(controller, "context", None), "postprocess", None)
        if postprocess is None:
            raise RuntimeError("decoder=0 regression requires the real PostProcess module")
        with torch.no_grad():
            processed = postprocess(
                outputs,
                torch.tensor([[architecture.resolution, architecture.resolution]], device=device),
            )
        assert_finite(processed, "decoder_zero.postprocess")
        record["postprocess_status"] = "PASS"

    if backward:
        if architecture.decoder_layers <= 0:
            raise ValueError("backward regression cases must retain at least one decoder layer")
        model.train()
        model.zero_grad(set_to_none=True)
        images = torch.zeros(1, 3, architecture.resolution, architecture.resolution, device=device)
        targets = _synthetic_target(device, architecture.resolution)
        with torch.enable_grad():
            train_outputs = model(nested_tensor_from_tensor_list([images[0]]), targets=targets)
            losses = criterion(train_outputs, targets)
            weighted = sum(
                value * criterion.weight_dict[key]
                for key, value in losses.items()
                if key in criterion.weight_dict
            )
            assert_finite(losses, "hardening_gpu.losses")
            weighted.backward()
        gradients = gradient_snapshot(model)
        if not gradients:
            raise RuntimeError("GPU hardening backward produced no gradients")
        record.update(
            {
                "backward_status": "PASS",
                "weighted_loss": float(weighted.detach().cpu()),
                "gradient_parameter_count": len(gradients),
                "peak_memory_bytes_forward_backward": int(torch.cuda.max_memory_allocated(device)),
            }
        )
        model.zero_grad(set_to_none=True)
    model.eval()
    controller.reset_to_native()
    torch.cuda.empty_cache()
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", default=Path("nas_artifacts/hardening_gpu_regression.json"), type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("target GPU regression requires a CUDA device")

    torch.manual_seed(20260809)
    detector = RFDETRSegSmall(pretrain_weights=str(args.checkpoint), device=str(device))
    context = detector.model
    if context.model is None:
        raise RuntimeError("model context has no LW-DETR module")
    model = context.model.to(device)
    model.eval()
    train_config = detector.get_train_config(dataset_dir=".", output_dir=str(args.output.parent / "gpu_loss_audit"))
    native = inspect_native_architecture(
        detector,
        checkpoint_path=args.checkpoint,
        train_config=train_config,
    )
    controller = NativeBoundedElasticController(model, native, context=context)
    criterion, _ = build_criterion_from_config(detector.model_config, train_config)

    cases = (
        ("native", controller.native_spec(), True),
        (
            "480_p20_w2_d4_q100",
            _architecture(
                native,
                resolution=480,
                patch_size=20,
                num_windows=2,
                decoder_layers=native.decoder_layers,
                num_queries=native.num_queries,
            ),
            True,
        ),
        (
            "576_p12_w1_d4_q100",
            _architecture(
                native,
                resolution=576,
                patch_size=12,
                num_windows=1,
                decoder_layers=native.decoder_layers,
                num_queries=native.num_queries,
            ),
            True,
        ),
        (
            "native_encoder_d0_q50",
            _architecture(
                native,
                resolution=native.resolution,
                patch_size=native.patch_size,
                num_windows=native.num_windows,
                decoder_layers=0,
                num_queries=min(50, native.num_queries),
            ),
            False,
        ),
    )
    records = []
    for name, architecture, backward in cases:
        record = _run_case(
            model,
            controller,
            architecture,
            criterion,
            train_config,
            device,
            backward=backward,
        )
        record["name"] = name
        records.append(record)

    controller.activate(controller.native_spec())
    model.eval()
    with torch.no_grad():
        state_a = _raw_forward(model, native.resolution)
        controller.activate(cases[1][1])
        _raw_forward(model, cases[1][1].resolution)
        controller.activate(cases[2][1])
        _raw_forward(model, cases[2][1].resolution)
        controller.activate(controller.native_spec())
        state_a_again = _raw_forward(model, native.resolution)
    pollution = _compare_outputs(state_a, state_a_again)
    if not pollution or not all(item["allclose"] for item in pollution.values()):
        raise AssertionError(f"GPU A-B-C-A state pollution detected: {pollution}")

    report = {
        "status": "PASS",
        "probe": "bounded_target_gpu_hardening_regression",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_count_visible": torch.cuda.device_count(),
        "cases": records,
        "a_b_c_a": {"status": "PASS", "comparison": pollution},
        "formal_search_executed": False,
        "full_subnet_sweep_executed": False,
        "pareto_search_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
