# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

#!/usr/bin/env python3
"""Evaluate saved RF-DETR checkpoints on the fixed MVTec train probe.

This is an experiment-side, read-only evaluator.  It does not run an optimizer, update SampleStateStore, or change the
training outputs.  The same deterministic train-probe loader is used for every checkpoint so E0 and E5 can be compared
on per-image loss and probe metrics after training.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rfdetr.config import RFDETRSegSmallConfig, TrainConfig
from rfdetr.sample_dynamics import run_deterministic_probe
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule

_CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)\.ckpt$")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_entries(run_dir: Path, epochs: int) -> list[dict[str, Any]]:
    """Return numeric Lightning checkpoints plus a best-model fallback if needed."""
    entries: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("checkpoint_*.ckpt")):
        match = _CHECKPOINT_RE.match(path.name)
        if match is not None:
            entries.append({"epoch": int(match.group(1)), "path": path, "kind": "epoch"})
    present_epochs = {int(entry["epoch"]) for entry in entries}
    if epochs - 1 not in present_epochs:
        best = run_dir / "checkpoint_best_regular.pth"
        if best.is_file():
            entries.append(
                {
                    "epoch": epochs - 1,
                    "path": best,
                    "kind": "best_regular_fallback",
                    "note": "final epoch checkpoint was cleaned after E0; best_regular is used for the final trajectory point",
                }
            )
    entries.sort(key=lambda item: int(item["epoch"]))
    if not entries:
        raise FileNotFoundError(f"no usable checkpoints found under {run_dir}")
    return entries


def _build_module(
    run_dir: Path,
    dataset: Path,
    weights: Path,
    device: torch.device,
) -> tuple[RFDETRModelModule, RFDETRDataModule, dict[str, Any]]:
    fairness = _read_json(run_dir / "fairness_snapshot.json")
    model_values = dict(fairness["model_config"])
    model_values["pretrain_weights"] = str(weights)
    model_values["device"] = "cuda" if device.type == "cuda" else "cpu"
    model_config = RFDETRSegSmallConfig(**model_values)

    train_values = dict(fairness["train_config"])
    train_values.update(
        {
            "dataset_dir": str(dataset),
            "output_dir": str(run_dir),
            "num_workers": 0,
            "sample_dynamics_enabled": False,
            "sample_dynamics_mode": "observe",
            "sample_dynamics_output_dir": None,
            "sample_dynamics_probe_interval": 0,
        }
    )
    train_fields = set(TrainConfig.model_fields)
    train_config = TrainConfig(**{key: value for key, value in train_values.items() if key in train_fields})
    module = RFDETRModelModule(model_config, train_config)
    module.to(device)
    datamodule = RFDETRDataModule(model_config, train_config)
    datamodule.setup("fit")
    return module, datamodule, fairness


def _reset_evaluation_rng(seed: int) -> None:
    """Make stochastic train-mode loss replay comparable across checkpoints."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_checkpoint(module: RFDETRModelModule, path: Path) -> dict[str, Any]:
    """Load Lightning or legacy RF-DETR checkpoint state into an existing module."""
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint payload is not a mapping: {path}")
    state_dict = payload.get("state_dict")
    if state_dict is None:
        state_dict = payload.get("model")
    if not isinstance(state_dict, dict):
        raise KeyError(f"checkpoint has neither state_dict nor model: {path}")
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        key_text = str(key)
        normalized[key_text if key_text.startswith("model.") else f"model.{key_text}"] = value
    checkpoint = {"state_dict": normalized}
    module.on_load_checkpoint(checkpoint)
    incompatible = module.load_state_dict(checkpoint["state_dict"], strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint keys in {path}: {incompatible.unexpected_keys[:8]}")
    allowed_missing = {"model._kp_active_mask"}
    unexpected_missing = [key for key in incompatible.missing_keys if key not in allowed_missing]
    if unexpected_missing:
        raise RuntimeError(f"missing checkpoint keys in {path}: {unexpected_missing[:8]}")
    return {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


LOSS_COMPONENT_PREFIXES = {
    "classification": "loss_ce",
    "bbox": "loss_bbox",
    "giou": "loss_giou",
    "mask_ce": "loss_mask_ce",
    "mask_dice": "loss_mask_dice",
}


def _weighted_component_losses(packet: Any, weight_dict: dict[str, float]) -> dict[str, torch.Tensor]:
    """Group weighted image-local criterion terms into task-level components."""
    result = {
        component: torch.zeros(
            packet.batch_size, dtype=packet.global_num_boxes.dtype, device=packet.global_num_boxes.device
        )
        for component in LOSS_COMPONENT_PREFIXES
    }
    # Match the longest prefix first so ``loss_mask_ce`` is never counted as
    # ordinary classification ``loss_ce``.
    ordered = sorted(LOSS_COMPONENT_PREFIXES.items(), key=lambda item: len(item[1]), reverse=True)
    for name, weight in weight_dict.items():
        values = packet.per_image_normalized_losses.get(name)
        if values is None:
            continue
        for component, prefix in ordered:
            if name == prefix or name.startswith(f"{prefix}_"):
                result[component] = result[component] + values.to(result[component]) * float(weight)
                break
    return result


def _loss_trajectory(
    module: RFDETRModelModule,
    loader: Any,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Compute total and decomposed weighted image-local loss per sample."""
    module.train()
    values: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for samples, targets in loader:
            samples = samples.to(device)
            targets = [
                {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
                for target in targets
            ]
            outputs = module.model(samples, targets)
            result = module.criterion(outputs, targets, return_per_sample=True)
            if not isinstance(result, tuple):
                raise TypeError("criterion did not return a per-sample packet")
            _, packet = result
            losses = packet.weighted_per_image_normalized_loss(module.criterion.weight_dict).detach().cpu()
            grouped = {
                name: tensor.detach().cpu().tolist()
                for name, tensor in _weighted_component_losses(packet, module.criterion.weight_dict).items()
            }
            for index, (sample_id, value) in enumerate(zip(packet.sample_ids, losses.tolist())):
                values[str(sample_id)] = float(value)
                components[str(sample_id)] = {
                    name: float(component_values[index]) for name, component_values in grouped.items()
                }
    return dict(sorted(values.items())), dict(sorted(components.items()))


def _probe_trajectory(
    module: RFDETRModelModule,
    loader: Any,
    device: torch.device,
    *,
    iou_threshold: float,
    score_threshold: float,
) -> dict[str, dict[str, Any]]:
    report = run_deterministic_probe(
        module.model,
        module.postprocess,
        loader,
        device=device,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    return {sample.sample_id: sample.as_dict() for sample in report.samples}


def evaluate_run(
    label: str,
    run_dir: Path,
    dataset: Path,
    weights: Path,
    device: torch.device,
    *,
    iou_threshold: float,
    score_threshold: float,
) -> dict[str, Any]:
    fairness = _read_json(run_dir / "fairness_snapshot.json")
    epochs = int(fairness["locked_contract"]["epochs"])
    entries = _checkpoint_entries(run_dir, epochs)
    module, datamodule, _ = _build_module(run_dir, dataset, weights, device)
    loss_loader = datamodule.train_probe_dataloader()
    probe_loader = datamodule.train_probe_dataloader()
    evaluated: list[dict[str, Any]] = []
    trajectory_seed = int(fairness["locked_contract"]["seed"])
    for entry in entries:
        path = Path(entry["path"])
        load_info = _load_checkpoint(module, path)
        _reset_evaluation_rng(trajectory_seed)
        loss_values, loss_components = _loss_trajectory(module, loss_loader, device)
        _reset_evaluation_rng(trajectory_seed)
        probe_values = _probe_trajectory(
            module,
            probe_loader,
            device,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        evaluated.append(
            {
                "epoch": int(entry["epoch"]),
                "checkpoint": path.name,
                "checkpoint_kind": entry["kind"],
                "checkpoint_note": entry.get("note"),
                "load_info": load_info,
                "loss": loss_values,
                "loss_components": loss_components,
                "probe": dict(sorted(probe_values.items())),
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "label": label,
        "run_dir": str(run_dir),
        "epochs_requested": epochs,
        "checkpoint_count_evaluated": len(evaluated),
        "epochs_evaluated": [int(entry["epoch"]) for entry in evaluated],
        "fairness_snapshot": fairness,
        "trajectory": evaluated,
        "loss_trajectory_contract": {
            "module_mode": "train",
            "gradients": "disabled",
            "rng_reset_per_checkpoint": True,
            "rng_seed": trajectory_seed,
            "reason": "preserve RF-DETR training criterion semantics for group_detr normalization while making stochastic replay comparable",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=RUN_DIR",
        help="checkpoint run to evaluate; repeat for E0 and E5",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    runs: dict[str, dict[str, Any]] = {}
    for specification in args.run:
        label, separator, raw_path = specification.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"--run must use LABEL=RUN_DIR, got {specification!r}")
        if label in runs:
            raise ValueError(f"duplicate run label: {label}")
        runs[label] = evaluate_run(
            label,
            Path(raw_path).resolve(),
            args.dataset.resolve(),
            args.weights.resolve(),
            device,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
        )
    result = {
        "schema_version": 1,
        "scope": "NON-BENCHMARK / OFFLINE CHECKPOINT TRAJECTORY ANALYSIS",
        "dataset": str(args.dataset.resolve()),
        "weights": str(args.weights.resolve()),
        "device": str(device),
        "probe_contract": {
            "iou_threshold": args.iou_threshold,
            "score_threshold": args.score_threshold,
            "loader": "RFDETRDataModule.train_probe_dataloader",
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"runs": sorted(runs), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
