#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Run a tiny E0 baseline or E2 observe-only MVTec mechanism Pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("RFDETR_SKIP_DINOV2_PREWARM", "1")

import torch  # noqa: E402
import numpy as np  # noqa: E402

from rfdetr import RFDETRSegSmall  # noqa: E402


def _seed_everything(seed: int) -> None:
    """Seed every RNG used during model construction and training setup."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter_sha256(module: torch.nn.Module) -> str:
    """Hash model parameter names, dtypes, shapes, and values in stable order."""
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON artifact."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _latest_metrics(path: Path) -> dict[str, float]:
    """Return the latest numeric value logged for every CSV metric."""
    if not path.is_file():
        return {}
    latest: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                if key is None or value in (None, ""):
                    continue
                try:
                    latest[key] = float(value)
                except ValueError:
                    continue
    return latest


def _state_summary(state_path: Path, corruption_path: Path) -> dict[str, Any] | None:
    """Summarize final state buckets and controlled-corruption retrieval."""
    if not state_path.is_file():
        return None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    states = payload.get("states", {})
    state_counts = Counter(str(record.get("state", "UNKNOWN")) for record in states.values())
    corruption_records = json.loads(corruption_path.read_text(encoding="utf-8"))
    known_corruptions = {str(record["stable_sample_id"]) for record in corruption_records}
    predicted_suspect = {
        str(sample_id)
        for sample_id, record in states.items()
        if str(record.get("state")) == "SUSPECT"
    }
    true_positive = len(known_corruptions & predicted_suspect)
    precision = true_positive / len(predicted_suspect) if predicted_suspect else 0.0
    recall = true_positive / len(known_corruptions) if known_corruptions else 0.0
    return {
        "state_counts": dict(sorted(state_counts.items())),
        "known_corruptions": len(known_corruptions),
        "predicted_suspect": len(predicted_suspect),
        "suspect_true_positive": true_positive,
        "suspect_precision": precision,
        "suspect_recall": recall,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one fixed Pilot mode and persist a compact summary."""
    dataset = args.dataset.resolve()
    weights = args.weights.resolve()
    output = args.output.resolve()
    if not (dataset / "split_manifest.json").is_file():
        raise FileNotFoundError(f"Pilot split manifest is missing: {dataset}")
    if not weights.is_file():
        raise FileNotFoundError(f"Seg Small checkpoint is missing: {weights}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite Pilot run output: {output}")

    _seed_everything(args.seed)
    sample_dynamics_enabled = args.mode == "observe"
    model = RFDETRSegSmall(
        pretrain_weights=str(weights),
        num_classes=1,
        resolution=384,
    )
    train_kwargs: dict[str, Any] = {
        "dataset_dir": str(dataset),
        "output_dir": str(output),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": 1,
        "lr": 1e-5,
        "lr_encoder": 1.5e-5,
        "device": args.device,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "resolution": 384,
        "multi_scale": False,
        "expanded_scales": False,
        "do_random_resize_via_padding": False,
        "scale_jitter": False,
        "aug_config": {},
        "augmentation_backend": "torchvision",
        "use_ema": False,
        "tensorboard": False,
        "wandb": False,
        "save_dataset_grids": False,
        "checkpoint_interval": 1,
        "eval_interval": 1,
        "early_stopping": False,
        "sample_dynamics_enabled": sample_dynamics_enabled,
        "sample_dynamics_mode": "observe",
        "sample_dynamics_output_dir": str(output / "sample_dynamics") if sample_dynamics_enabled else None,
        "sample_dynamics_probe_interval": 1 if sample_dynamics_enabled else 0,
        "sample_dynamics_probe_iou_threshold": 0.5,
        "sample_dynamics_probe_score_threshold": 0.05,
        "sample_dynamics_seed": args.seed,
        "notes": {
            "scope": "NON-BENCHMARK / MECHANISM VALIDATION",
            "experiment": "E2 observe-only" if sample_dynamics_enabled else "E0 baseline",
            "dataset_manifest": str(dataset / "split_manifest.json"),
        },
    }
    train_config = model.get_train_config(
        **{key: value for key, value in train_kwargs.items() if key not in {"device", "resolution"}}
    )
    output.mkdir(parents=True, exist_ok=True)
    fairness_snapshot = {
        "scope": "NON-BENCHMARK / MECHANISM VALIDATION",
        "experiment": "E2 observe-only" if sample_dynamics_enabled else "E0 baseline",
        "seed": args.seed,
        "dataset_manifest_sha256": _sha256_file(dataset / "split_manifest.json"),
        "pretrain_weights_sha256": _sha256_file(weights),
        "initial_model_parameter_sha256": _parameter_sha256(model.model.model),
        "model_config": model.model_config.model_dump(),
        "train_config": train_config.model_dump(),
        "locked_contract": {
            "resolution": int(train_kwargs["resolution"]),
            "batch_size": int(train_kwargs["batch_size"]),
            "grad_accum_steps": int(train_kwargs["grad_accum_steps"]),
            "epochs": int(train_kwargs["epochs"]),
            "multi_scale": bool(train_kwargs["multi_scale"]),
            "scale_jitter": bool(train_kwargs["scale_jitter"]),
            "aug_config": dict(train_kwargs["aug_config"]),
            "augmentation_backend": str(train_kwargs["augmentation_backend"]),
            "use_ema": bool(train_kwargs["use_ema"]),
            "seed": int(train_kwargs["seed"]),
        },
    }
    _write_json(output / "fairness_snapshot.json", fairness_snapshot)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train(
        **train_kwargs,
    )
    elapsed = time.perf_counter() - started
    peak_vram = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    summary = {
        "scope": "NON-BENCHMARK / MECHANISM VALIDATION",
        "mode": args.mode,
        "dataset": str(dataset),
        "weights": str(weights),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "peak_vram_bytes": peak_vram,
        "fairness_snapshot": fairness_snapshot,
        "training_config": json.loads((output / "training_config.json").read_text(encoding="utf-8"))
        if (output / "training_config.json").is_file()
        else None,
        "sample_order": json.loads((output / "sample_order_hash.json").read_text(encoding="utf-8"))
        if (output / "sample_order_hash.json").is_file()
        else None,
        "latest_metrics": _latest_metrics(output / "metrics.csv"),
        "sample_dynamics": _state_summary(
            output / "sample_dynamics" / "sample_state.json",
            dataset / "corruption_ground_truth.json",
        ),
    }
    _write_json(output / "pilot_run_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse the Pilot training CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "observe"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("/home/liujiyuan/rf-detr-models/rf-detr-seg-small.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    """Run one Pilot and print its summary."""
    print(json.dumps(run(parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
