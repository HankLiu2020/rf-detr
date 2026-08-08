# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Rebuild a fresh process, restore a NAS smoke checkpoint, and take one step."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from rfdetr._namespace import _namespace_from_configs
from rfdetr.models.lwdetr import build_criterion_from_config
from rfdetr.nas.architecture import ArchitectureSpec
from rfdetr.nas.checkpoint import load_nas_checkpoint
from rfdetr.nas.controller import NativeBoundedElasticController
from rfdetr.nas.data import MVTecSegmentationDataset
from rfdetr.nas.native_inspector import _load_param_group_builder, inspect_native_architecture
from rfdetr.nas.schedule import architecture_for_step, resize_batch_to_architecture
from rfdetr.nas.search_space import generate_search_space
from rfdetr.nas.validation import assert_finite, gradient_snapshot
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
    return {"path": str(path), "id": sha256[:16], "sha256": sha256, "size_bytes": path.stat().st_size}


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


def _architecture_from_dict(value: dict[str, object]) -> ArchitectureSpec:
    fields = {
        "resolution",
        "patch_size",
        "num_windows",
        "decoder_layers",
        "num_queries",
        "group_detr",
        "encoder",
        "native",
        "hardware_feasible",
        "invalid_reason",
        "metadata",
    }
    return ArchitectureSpec(**{key: value[key] for key in fields if key in value})  # type: ignore[arg-type]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--resume-checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", default=Path("nas_artifacts/resume_verification.json"), type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    payload = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("resume checkpoint payload is not a dictionary")
    seed = int(payload.get("seed", 20260807))
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    detector = RFDETRSegSmall(pretrain_weights=str(args.checkpoint), device=str(device))
    context = detector.model
    if context.model is None:
        raise RuntimeError("model context has no LW-DETR module")
    model = context.model.to(device)
    train_config = detector.get_train_config(dataset_dir=str(args.dataset_root), output_dir=str(args.output.parent))
    native = inspect_native_architecture(
        detector,
        checkpoint_path=args.checkpoint,
        train_config=train_config,
    )
    repo_root = Path(__file__).resolve().parents[1]
    base_checkpoint = _base_checkpoint_metadata(args.checkpoint)
    git_commit = _runtime_git_commit(repo_root)
    rfdetr_version = _runtime_rfdetr_version(repo_root)
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
    native_space = generate_search_space(native)
    encoder_candidates = [
        value
        for value in native_space.syntactically_valid
        if value.decoder_layers == native.decoder_layers and value.num_queries == native.num_queries
    ]
    resume = load_nas_checkpoint(model, optimizer, args.resume_checkpoint, scheduler=scheduler)
    restored_step = int(resume["optimizer_step"])
    expected_preview = [
        architecture_for_step(
            encoder_candidates,
            seed=seed,
            optimizer_step=step,
            policy="balanced_patch",
        ).to_dict()
        for step in range(restored_step, restored_step + 5)
    ]
    metadata_pass = (
        resume.get("schedule_preview", []) == expected_preview
        and resume.get("scheduler") is not None
        and resume.get("search_space_config") == native_space.to_summary()
        and resume.get("sampling_policy") == "balanced_patch"
        and resume.get("seed") == seed
        and bool(resume.get("ema"))
        and resume.get("native_architecture") == native.to_dict()
        and resume.get("git_commit") == git_commit
        and resume.get("rfdetr_version") == rfdetr_version
        and resume.get("base_checkpoint") == base_checkpoint
    )
    if not metadata_pass:
        raise AssertionError("restored checkpoint metadata or schedule preview does not match")

    expected_ema_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    restored_ema_names = set((resume.get("ema") or {}).keys())
    if restored_ema_names != expected_ema_names:
        raise AssertionError(
            "EMA parameter set does not match the actual model parameters: "
            f"missing={sorted(expected_ema_names - restored_ema_names)}, "
            f"extra={sorted(restored_ema_names - expected_ema_names)}"
        )

    architecture = architecture_for_step(
        encoder_candidates,
        seed=seed,
        optimizer_step=restored_step,
        policy="balanced_patch",
    )
    controller.activate(architecture)
    controller.validate_active_architecture()
    dataset = MVTecSegmentationDataset(args.dataset_root, category="pill", split="test")
    positive_index = next(index for index in range(len(dataset)) if dataset[index][1]["labels"].numel() > 0)
    image, target = dataset[positive_index]
    images, targets = resize_batch_to_architecture(
        image.unsqueeze(0).to(device),
        [_move_target(target, device)],
        architecture,
    )
    optimizer.zero_grad(set_to_none=True)
    outputs = model(nested_tensor_from_tensor_list([images[0]]), targets=targets)
    losses = criterion(outputs, targets)
    weighted = sum(
        value * criterion.weight_dict[key] for key, value in losses.items() if key in criterion.weight_dict
    )
    assert_finite(outputs, "resume.outputs")
    assert_finite(losses, "resume.losses")
    weighted.backward()
    gradients = gradient_snapshot(model)
    if not gradients:
        raise RuntimeError("resume step produced no gradients")
    optimizer.step()
    scheduler.step()
    report = {
        "status": "PASS",
        "restored_optimizer_step": restored_step,
        "one_step_after_resume": "PASS",
        "new_optimizer_step": restored_step + 1,
        "architecture": architecture.to_dict(),
        "schedule_preview_matches": True,
        "scheduler_restored": True,
        "search_space_config_restored": True,
        "sampling_policy_restored": True,
        "seed_restored": True,
        "ema_restored": True,
        "ema_parameter_count": len(restored_ema_names),
        "native_architecture_restored": True,
        "git_commit_restored": True,
        "rfdetr_version_restored": True,
        "base_checkpoint_restored": True,
        "git_commit": git_commit,
        "rfdetr_version": rfdetr_version,
        "base_checkpoint": base_checkpoint,
        "weighted_loss": float(weighted.detach()),
        "gradient_parameter_count": len(gradients),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
