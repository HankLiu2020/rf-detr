#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Run the one-batch MVTec Runtime Contract for RF-DETR Seg Small.

This is an experiment-only preflight.  It exercises the same DataModule,
collate function, model builder, criterion, and backward path used by the
official RF-DETR Lightning entrypoint.  It deliberately does not run an
optimizer step or modify the source MVTec tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rfdetr.config import RFDETRSegSmallConfig, SegmentationTrainConfig
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule


def _seed_everything(seed: int) -> None:
    """Seed all RNGs used before the Lightning trainer takes ownership."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> tuple[Any, Any, torch.Tensor, list[torch.Tensor] | None]:
    """Capture process RNG streams for an exact same-batch comparison."""
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def _restore_rng_state(state: tuple[Any, Any, torch.Tensor, list[torch.Tensor] | None]) -> None:
    """Restore process RNG streams captured by :func:`_capture_rng_state`."""
    python_state, numpy_state, torch_state, cuda_state = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def _sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_scalar(value: torch.Tensor) -> float:
    """Convert a scalar tensor to a finite Python float."""
    if value.numel() != 1 or not torch.isfinite(value).all():
        raise AssertionError(f"non-finite or non-scalar loss value: shape={tuple(value.shape)}")
    return float(value.detach().cpu())


def _target_summary(target: dict[str, Any]) -> dict[str, Any]:
    """Summarize the contract fields carried by one transformed target."""
    required = ("boxes", "labels", "masks", "image_id", "sample_id")
    missing = [key for key in required if key not in target]
    if missing:
        raise AssertionError(f"target is missing required fields: {missing}")
    for key in ("boxes", "labels", "masks", "image_id"):
        value = target[key]
        if not torch.is_tensor(value):
            raise AssertionError(f"target[{key!r}] must be a tensor, got {type(value).__name__}")
        if key != "image_id" and not torch.isfinite(value.float()).all():
            raise AssertionError(f"target[{key!r}] contains non-finite values")
    return {
        "sample_id": str(target["sample_id"]),
        "image_id": int(target["image_id"].reshape(-1)[0]),
        "boxes": list(target["boxes"].shape),
        "labels": list(target["labels"].shape),
        "masks": list(target["masks"].shape),
        "mask_foreground": int(target["masks"].sum().item()),
    }


def _pick_records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Pick one normal, pristine abnormal, and non-empty corrupted train record."""
    records = [record for record in manifest["records"] if record["split"] == "train"]
    selected: dict[str, dict[str, Any]] = {}
    selected["normal"] = next(record for record in records if record["kind"] == "normal")
    selected["abnormal"] = next(
        record
        for record in records
        if record["kind"] == "abnormal" and record["label_status"] == "clean"
    )
    selected["controlled_corruption"] = next(
        record
        for record in records
        if (
            record["kind"] == "abnormal"
            and record["label_status"] == "corrupted"
            and record["corruption_type"] != "drop_mask"
        )
    )
    return selected


def _head_shapes(model: torch.nn.Module) -> dict[str, list[int]]:
    """Return classification-head tensor shapes for the class-count check."""
    return {
        name: list(value.shape)
        for name, value in model.state_dict().items()
        if "class_embed" in name and (name.endswith(".weight") or name.endswith(".bias"))
    }


def _weighted_loss(criterion: torch.nn.Module, loss_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Reconstruct the official weighted scalar from criterion outputs."""
    weight_dict = getattr(criterion, "weight_dict")
    return torch.stack([loss_dict[key] * weight_dict[key] for key in loss_dict if key in weight_dict]).sum()


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the Runtime Contract and persist its evidence."""
    dataset = args.dataset.resolve()
    weights = args.weights.resolve()
    output = args.output.resolve()
    manifest_path = dataset / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MVTec Pilot manifest is missing: {manifest_path}")
    if not weights.is_file():
        raise FileNotFoundError(f"Seg Small checkpoint is missing: {weights}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Runtime Contract output: {output}")

    _seed_everything(args.seed)
    device = torch.device(args.device)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = _pick_records(manifest)
    selected_ids = {str(record["stable_sample_id"]): kind for kind, record in selected.items()}

    model_config = RFDETRSegSmallConfig(
        pretrain_weights=str(weights),
        num_classes=1,
        resolution=384,
        device=str(device),
    )
    train_config = SegmentationTrainConfig(
        dataset_dir=str(dataset),
        output_dir=str(output.parent / "runtime-contract-training"),
        batch_size=3,
        grad_accum_steps=1,
        epochs=1,
        lr=1e-5,
        lr_encoder=1.5e-5,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        scale_jitter=False,
        aug_config={},
        augmentation_backend="torchvision",
        use_ema=False,
        num_workers=0,
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        seed=args.seed,
        tensorboard=False,
        wandb=False,
        compute_train_metrics=False,
        sample_dynamics_enabled=False,
    )

    datamodule = RFDETRDataModule(model_config, train_config)
    datamodule.setup("fit")
    dataset_train = datamodule._dataset_train
    if dataset_train is None:
        raise AssertionError("RFDETRDataModule did not build the train dataset")
    index_by_id = {
        str(dataset_train.sample_id_for_index(index)): index  # type: ignore[attr-defined]
        for index in range(len(dataset_train))
    }
    missing = sorted(set(selected_ids) - set(index_by_id))
    if missing:
        raise AssertionError(f"selected manifest samples are absent from the DataModule: {missing}")

    raw_examples = [dataset_train[index_by_id[sample_id]] for sample_id in selected_ids]
    batch = datamodule._collate_fn(raw_examples)
    samples, raw_targets = batch[0], list(batch[1])
    if len(raw_targets) != 3:
        raise AssertionError(f"expected three targets after collate, got {len(raw_targets)}")
    target_summaries = [_target_summary(target) for target in raw_targets]
    if set(summary["sample_id"] for summary in target_summaries) != set(selected_ids):
        raise AssertionError("collate changed or duplicated sample IDs")

    samples, targets = datamodule.transfer_batch_to_device((samples, raw_targets), device, 0)
    module = RFDETRModelModule(model_config, train_config)
    module.model.to(device)
    module.criterion.to(device)
    module.model.train()
    module.criterion.train()

    initial_state = {name: value.detach().clone() for name, value in module.model.state_dict().items()}
    _seed_everything(args.seed + 1)
    comparison_rng = _capture_rng_state()

    def run_loss_variant(return_per_sample: bool) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Run one forward/backward variant from the same model/RNG snapshot."""
        module.model.load_state_dict(initial_state)
        module.model.zero_grad(set_to_none=True)
        _restore_rng_state(comparison_rng)
        outputs_variant = module.model(samples, targets)
        result = module.criterion(outputs_variant, targets, return_per_sample=return_per_sample)
        if isinstance(result, tuple):
            loss_dict_variant = result[0]
        else:
            loss_dict_variant = result
        _weighted_loss(module.criterion, loss_dict_variant).backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in module.model.named_parameters()
            if parameter.grad is not None
        }
        return loss_dict_variant, gradients

    scalar_loss_dict, scalar_gradients = run_loss_variant(False)
    scalar_repeat_loss_dict, scalar_repeat_gradients = run_loss_variant(False)
    observed_loss_dict, observed_gradients = run_loss_variant(True)
    if set(scalar_loss_dict) != set(observed_loss_dict):
        raise AssertionError("observer changed criterion loss keys")
    baseline_loss_max_abs_diff = max(
        float(torch.max(torch.abs(scalar_loss_dict[key] - scalar_repeat_loss_dict[key])).item())
        for key in scalar_loss_dict
    )
    loss_max_abs_diff = max(
        float(torch.max(torch.abs(scalar_loss_dict[key] - observed_loss_dict[key])).item())
        for key in scalar_loss_dict
    )
    if set(scalar_gradients) != set(observed_gradients):
        raise AssertionError("observer changed which model parameters receive gradients")
    baseline_gradient_max_abs_diff = max(
        float(torch.max(torch.abs(scalar_gradients[key] - scalar_repeat_gradients[key])).item())
        for key in scalar_gradients
    )
    gradient_max_abs_diff = max(
        float(torch.max(torch.abs(scalar_gradients[key] - observed_gradients[key])).item())
        for key in scalar_gradients
    )
    if (
        loss_max_abs_diff > max(1e-6, baseline_loss_max_abs_diff + 1e-6)
        or gradient_max_abs_diff > max(1e-5, baseline_gradient_max_abs_diff + 1e-5)
    ):
        raise AssertionError(
            "observer changed same-batch loss/gradient: "
            f"loss_max_abs_diff={loss_max_abs_diff}, gradient_max_abs_diff={gradient_max_abs_diff}, "
            f"baseline_loss_max_abs_diff={baseline_loss_max_abs_diff}, "
            f"baseline_gradient_max_abs_diff={baseline_gradient_max_abs_diff}"
        )

    module.model.load_state_dict(initial_state)
    module.model.zero_grad(set_to_none=True)
    _restore_rng_state(comparison_rng)
    outputs = module.model(samples, targets)
    loss_dict = module.criterion(outputs, targets)
    required_loss_aliases = {
        "loss_cls": "loss_ce",
        "bbox": "loss_bbox",
        "giou": "loss_giou",
        "mask_ce": "loss_mask_ce",
        "mask_dice": "loss_mask_dice",
    }
    losses: dict[str, float] = {}
    for alias, key in required_loss_aliases.items():
        if key not in loss_dict:
            raise AssertionError(f"criterion did not return required {key!r}; keys={sorted(loss_dict)}")
        losses[alias] = _finite_scalar(loss_dict[key])
    total_loss = _weighted_loss(module.criterion, loss_dict)
    total_loss_value = _finite_scalar(total_loss)
    total_loss.backward()

    gradients = [parameter.grad for parameter in module.model.parameters() if parameter.grad is not None]
    if not gradients:
        raise AssertionError("backward produced no gradients")
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("backward produced non-finite gradients")
    gradient_norm = float(torch.sqrt(sum(gradient.detach().float().pow(2).sum() for gradient in gradients)).cpu())
    if not np.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise AssertionError(f"backward gradient norm is invalid: {gradient_norm}")

    head_shapes = _head_shapes(module.model)
    if not head_shapes or any(shape[0] != 2 for shape in head_shapes.values()):
        raise AssertionError(f"num_classes=1 head was not rebuilt to two logits: {head_shapes}")

    result: dict[str, Any] = {
        "status": "PASS",
        "scope": "NON-BENCHMARK / MECHANISM VALIDATION",
        "contract": "dataset -> collate -> forward -> criterion -> backward",
        "dataset": str(dataset),
        "manifest_sha256": _sha256_file(manifest_path),
        "weights": str(weights),
        "weights_sha256": _sha256_file(weights),
        "seed": args.seed,
        "device": str(device),
        "model": "RFDETRSegSmall",
        "num_classes": model_config.num_classes,
        "head_shapes": head_shapes,
        "batch_tensor_shape": list(samples.tensors.shape),
        "batch_padding_shape": list(samples.mask.shape) if samples.mask is not None else None,
        "targets": target_summaries,
        "outputs": {key: list(value.shape) for key, value in outputs.items() if torch.is_tensor(value)},
        "losses": losses,
        "total_loss": total_loss_value,
        "gradient_norm": gradient_norm,
        "observer_equivalence": {
            "status": "PASS",
            "loss_max_abs_diff": loss_max_abs_diff,
            "gradient_max_abs_diff": gradient_max_abs_diff,
            "baseline_repeat_loss_max_abs_diff": baseline_loss_max_abs_diff,
            "baseline_repeat_gradient_max_abs_diff": baseline_gradient_max_abs_diff,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    """Parse Runtime Contract CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    """Run the contract and print JSON evidence."""
    print(json.dumps(run(parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
