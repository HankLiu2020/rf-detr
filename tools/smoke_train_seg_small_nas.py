# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Run a bounded, short supernet smoke training loop with checkpoint resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from rfdetr._namespace import _namespace_from_configs
from rfdetr.models.lwdetr import build_criterion_from_config
from rfdetr.nas.checkpoint import load_nas_checkpoint, save_nas_checkpoint
from rfdetr.nas.controller import NativeBoundedElasticController
from rfdetr.nas.data import MVTecSegmentationDataset
from rfdetr.nas.native_inspector import _load_param_group_builder, inspect_native_architecture
from rfdetr.nas.schedule import architecture_for_step, resize_batch_to_architecture
from rfdetr.nas.search_space import generate_search_space
from rfdetr.nas.validation import assert_finite, gradient_snapshot, parameter_group_snapshot
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
from rfdetr.variants import RFDETRSegSmall


def _move_target(target: dict[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in target.items()}


def _base_checkpoint_metadata(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    sha256 = digest.hexdigest()
    return {
        "path": str(path),
        "id": sha256[:16],
        "sha256": sha256,
        "size_bytes": path.stat().st_size,
    }


def _runtime_git_commit(repo_root: Path) -> str:
    configured = os.environ.get("RFDETR_GIT_COMMIT", "").strip()
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _runtime_rfdetr_version(repo_root: Path) -> str:
    try:
        return version("rfdetr")
    except PackageNotFoundError:
        for line in (repo_root / "pyproject.toml").read_text(encoding="utf-8").splitlines():
            if line.startswith("version = "):
                return line.split("=", 1)[1].strip().strip('"')
        return "unknown"


def _update_ema(model: torch.nn.Module, ema_state: dict[str, torch.Tensor], decay: float) -> None:
    """Track only real model parameters; resampled patch/PE tensors are not parameters."""

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            value = parameter.detach().cpu()
            if name not in ema_state:
                ema_state[name] = value.clone()
            else:
                ema_state[name].mul_(decay).add_(value, alpha=1.0 - decay)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", default=Path("nas_artifacts"), type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    if args.max_steps <= 0 or args.max_steps > 30:
        raise ValueError("smoke training max_steps must be in [1, 30]")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    detector = RFDETRSegSmall(pretrain_weights=str(args.checkpoint), device=str(device))
    context = detector.model
    if context.model is None:
        raise RuntimeError("model context has no LW-DETR module")
    model = context.model.to(device)
    model.train()
    train_config = detector.get_train_config(
        dataset_dir=str(args.dataset_root),
        output_dir=str(args.output / "training_output"),
        batch_size=1,
        epochs=1,
    )
    native = inspect_native_architecture(
        detector,
        checkpoint_path=args.checkpoint,
        train_config=train_config,
    )
    controller = NativeBoundedElasticController(model, native, context=context)
    criterion, _ = build_criterion_from_config(detector.model_config, train_config)
    criterion.train()
    optimizer_namespace = _namespace_from_configs(detector.model_config, train_config)
    optimizer = torch.optim.AdamW(
        _load_param_group_builder()(optimizer_namespace, model),
        lr=float(optimizer_namespace.lr),
        weight_decay=float(optimizer_namespace.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    optimizer_groups = parameter_group_snapshot(optimizer, model)
    repo_root = Path(__file__).resolve().parents[1]
    base_checkpoint = _base_checkpoint_metadata(args.checkpoint)
    git_commit = _runtime_git_commit(repo_root)
    rfdetr_version = _runtime_rfdetr_version(repo_root)
    ema_decay = 0.999
    ema_state: dict[str, torch.Tensor] = {}

    dataset = MVTecSegmentationDataset(args.dataset_root, category="pill", split="test")
    positive_index = next(
        (index for index in range(len(dataset)) if dataset[index][1]["labels"].numel() > 0),
        0,
    )
    native_space = generate_search_space(native)
    encoder_candidates = [
        value
        for value in native_space.syntactically_valid
        if value.decoder_layers == native.decoder_layers and value.num_queries == native.num_queries
    ]
    search_space_config = native_space.to_summary()
    history: list[dict[str, object]] = []
    checkpoint_path = args.output / "smoke_last.pt"
    sampling_histogram: dict[str, dict[str, int]] = {"patch_size": {}, "resolution": {}, "num_windows": {}}
    for step in range(args.max_steps):
        architecture = architecture_for_step(
            encoder_candidates,
            seed=args.seed,
            optimizer_step=step,
            policy="balanced_patch",
        )
        controller.activate(architecture)
        controller.validate_active_architecture()
        image, target = dataset[positive_index]
        images, targets = resize_batch_to_architecture(
            image.unsqueeze(0).to(device),
            [_move_target(target, device)],
            architecture,
        )
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        outputs = model(nested_tensor_from_tensor_list([images[0]]), targets=targets)
        assert_finite(outputs, "outputs")
        losses = criterion(outputs, targets)
        weighted = sum(
            value * criterion.weight_dict[key]
            for key, value in losses.items()
            if key in criterion.weight_dict
        )
        assert_finite(losses, "losses")
        weighted.backward()
        gradients = gradient_snapshot(model)
        if not gradients:
            raise RuntimeError("smoke training produced no gradients")
        optimizer.step()
        _update_ema(model, ema_state, ema_decay)
        scheduler.step()
        step_time = time.perf_counter() - step_start
        memory = int(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
        grad_norm = torch.sqrt(
            torch.stack(
                [parameter.grad.detach().pow(2).sum() for parameter in model.parameters() if parameter.grad is not None]
            ).sum()
        )
        for field in sampling_histogram:
            value = str(getattr(architecture, field))
            sampling_histogram[field][value] = sampling_histogram[field].get(value, 0) + 1
        history.append(
            {
                "optimizer_step": step,
                "architecture": architecture.to_dict(),
                "losses": {key: float(value.detach()) for key, value in losses.items()},
                "weighted_loss": float(weighted.detach()),
                "lr": [float(group["lr"]) for group in optimizer.param_groups],
                "grad_norm": float(grad_norm),
                "step_time_seconds": step_time,
                "gradient_parameter_count": len(gradients),
                "patch_master_grad_sum_abs": float(
                    model.backbone[0].encoder.encoder.embeddings.patch_embeddings.projection.weight.grad.abs().sum()
                ),
                "max_memory_allocated": memory,
            }
        )
        schedule_preview = [
            architecture_for_step(
                encoder_candidates,
                seed=args.seed,
                optimizer_step=future_step,
                policy="balanced_patch",
            ).to_dict()
            for future_step in range(step + 1, step + 6)
        ]
        save_nas_checkpoint(
            model,
            optimizer,
            checkpoint_path,
            optimizer_step=step + 1,
            architecture=architecture,
            history=history,
            scheduler=scheduler,
            ema_state=ema_state,
            search_space_config=search_space_config,
            sampling_policy="balanced_patch",
            seed=args.seed,
            schedule_preview=schedule_preview,
            native_architecture=native.to_dict(),
            git_commit=git_commit,
            rfdetr_version=rfdetr_version,
            base_checkpoint=base_checkpoint,
        )

    required_sampling_coverage = {
        "patch_size": {12, 16, 20},
        "num_windows": {1, 2},
    }
    observed_sampling_coverage = {
        field: {int(value) for value in histogram}
        for field, histogram in sampling_histogram.items()
    }
    missing_sampling_coverage = {
        field: sorted(values - observed_sampling_coverage[field])
        for field, values in required_sampling_coverage.items()
        if not values.issubset(observed_sampling_coverage[field])
    }
    if missing_sampling_coverage:
        raise RuntimeError(
            "smoke sampling missed required patch/window coverage: "
            f"{missing_sampling_coverage}; increase the bounded smoke step count"
        )

    resume = load_nas_checkpoint(model, optimizer, checkpoint_path, scheduler=scheduler)
    restored_step = int(resume["optimizer_step"])
    next_architecture = architecture_for_step(
        encoder_candidates,
        seed=args.seed,
        optimizer_step=restored_step,
        policy="balanced_patch",
    )
    expected_preview = [
        architecture_for_step(
            encoder_candidates,
            seed=args.seed,
            optimizer_step=future_step,
            policy="balanced_patch",
        ).to_dict()
        for future_step in range(restored_step, restored_step + 5)
    ]
    resume_report = {
        "status": "PASS"
        if resume.get("schedule_preview", []) == expected_preview
        and resume.get("scheduler") is not None
        and resume.get("search_space_config") == search_space_config
        and resume.get("sampling_policy") == "balanced_patch"
        and resume.get("seed") == args.seed
        and resume.get("ema")
        and resume.get("native_architecture") == native.to_dict()
        and resume.get("git_commit") == git_commit
        and resume.get("rfdetr_version") == rfdetr_version
        and resume.get("base_checkpoint") == base_checkpoint
        else "FAIL",
        "restored_optimizer_step": restored_step,
        "scheduler_restored": resume.get("scheduler") is not None,
        "search_space_config_restored": resume.get("search_space_config") == search_space_config,
        "sampling_policy_restored": resume.get("sampling_policy") == "balanced_patch",
        "seed_restored": resume.get("seed") == args.seed,
        "ema_restored": bool(resume.get("ema")),
        "native_architecture_restored": resume.get("native_architecture") == native.to_dict(),
        "git_commit_restored": resume.get("git_commit") == git_commit,
        "rfdetr_version_restored": resume.get("rfdetr_version") == rfdetr_version,
        "base_checkpoint_restored": resume.get("base_checkpoint") == base_checkpoint,
        "ema_parameter_count": len(resume.get("ema") or {}),
        "schedule_preview_matches": resume.get("schedule_preview", []) == expected_preview,
        "schedule_preview": resume.get("schedule_preview", []),
        "expected_schedule_preview": expected_preview,
        "next_architecture": next_architecture.to_dict(),
    }
    if resume_report["status"] != "PASS":
        raise RuntimeError(f"checkpoint resume metadata mismatch: {resume_report}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "resume_verification.json").write_text(
        json.dumps(resume_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "smoke_training.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in history), encoding="utf-8"
    )
    (args.output / "smoke_architectures.jsonl").write_text(
        "".join(json.dumps(record["architecture"], sort_keys=True) + "\n" for record in history),
        encoding="utf-8",
    )
    report = {
        "status": "PASS",
        "steps": len(history),
        "native": native.to_dict(),
        "history": history,
        "optimizer_groups": optimizer_groups,
        "checkpoint": str(checkpoint_path),
        "resume": resume_report,
        "sampling_histogram": sampling_histogram,
        "sampling_coverage": {
            "required": {key: sorted(value) for key, value in required_sampling_coverage.items()},
            "observed": {key: sorted(value) for key, value in observed_sampling_coverage.items()},
            "status": "PASS",
        },
        "search_space_config": search_space_config,
        "sampling_policy": "balanced_patch",
        "seed": args.seed,
        "checkpoint_metadata": {
            "ema_decay": ema_decay,
            "ema_parameter_count": len(ema_state),
            "native_architecture": native.to_dict(),
            "git_commit": git_commit,
            "rfdetr_version": rfdetr_version,
            "base_checkpoint": base_checkpoint,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "smoke_training.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
