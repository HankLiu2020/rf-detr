#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Run one preregistered Policy V2 discovery or convergence-control job."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from torch.utils.data import DataLoader

try:
    from .run_pilot import run
except ImportError:
    from run_pilot import run

try:
    from .policy_v2 import CoverageBonusSampler
except ImportError:
    from policy_v2 import CoverageBonusSampler
from rfdetr.training.module_data import GradAccumAlignedDataset, RFDETRDataModule

POLICY_MODES = {
    "C0": "baseline",
    "C3": "loss_weight",
    "C4": "sampler",
    "C5": "combined",
    "D1": "sampler",
}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA256 for one contract input."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PolicyV2DataModule(RFDETRDataModule):
    """Install the experiment-side coverage-preserving D1 sampler."""

    def train_dataloader(self) -> DataLoader:
        if getattr(self.train_config, "notes", {}).get("policy_v2_id") != "D1":
            return super().train_dataloader()
        dataset = self._require_dataset(self._dataset_train, "fit")
        batch_size = self._resolve_batch_size()
        aligned_dataset = GradAccumAlignedDataset(dataset, batch_size, 1)
        frontier_contract = getattr(self.train_config, "notes", {})["learning_frontier"]
        self._train_sampler = CoverageBonusSampler(
            len(aligned_dataset),
            self._stable_sample_ids(dataset),
            state_provider=lambda: (
                getattr(getattr(self.trainer, "lightning_module", None), "sample_state_store").states
            ),
            index_mapper=aligned_dataset.original_index,
            seed=self.train_config.sample_dynamics_seed,
            batch_size=batch_size,
            nominal_bonus_fraction=0.05,
            warmup_epochs=5,
            frontier_kwargs={
                "ema_percentile_min": frontier_contract["ema_percentile_min"],
                "normalized_slope_max": frontier_contract["normalized_slope_max"],
                "absolute_ema_loss_floor": frontier_contract["absolute_ema_loss_floor"],
            },
        )
        self._train_sampler.set_epoch(int(getattr(self.trainer, "current_epoch", 0)))
        return DataLoader(
            aligned_dataset,
            batch_size=batch_size,
            sampler=self._train_sampler,
            drop_last=True,
            collate_fn=self._collate_fn,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=False,
            prefetch_factor=self._prefetch_factor,
            worker_init_fn=getattr(
                __import__("rfdetr.training.module_data", fromlist=["_worker_init_fn"]), "_worker_init_fn"
            ),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-id", choices=tuple(POLICY_MODES), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preregistration = Path("reports/policy_v2_discovery_preregistration.json").resolve()
    payload = json.loads(preregistration.read_text(encoding="utf-8"))
    contract = payload["common_contract"]
    if payload["status"] != "FROZEN_BEFORE_DISCOVERY_RUNS":
        raise ValueError("Policy V2 preregistration is not frozen")
    for field in ("epochs", "batch_size", "seed"):
        if getattr(args, field.replace("batch_size", "batch_size")) != contract[field]:
            raise ValueError(f"{field} differs from frozen Policy V2 contract")
    fixed_contract = {
        "resolution": 384,
        "grad_accum_steps": 1,
        "multi_scale": False,
        "augmentation": "disabled",
        "use_ema": False,
    }
    for field, value in fixed_contract.items():
        if contract[field] != value:
            raise ValueError(f"frozen Policy V2 {field} contract is unsupported: {contract[field]!r}")
    if sha256_file(args.dataset.resolve() / "split_manifest.json") != contract["manifest_sha256"]:
        raise ValueError("dataset manifest differs from frozen Policy V2 contract")
    if sha256_file(args.weights.resolve()) != contract["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from frozen Policy V2 contract")
    args.mode = POLICY_MODES[args.policy_id]
    if args.policy_id == "D1":
        # Keep Sample Dynamics observation/state/probe lifecycle, but replace
        # only the DataModule's sampler at the experiment boundary.
        import rfdetr.training as training

        training.RFDETRDataModule = PolicyV2DataModule
        original_build_trainer = training.build_trainer

        def build_policy_v2_trainer(*build_args, **build_kwargs):
            build_kwargs["reload_dataloaders_every_n_epochs"] = 1
            return original_build_trainer(*build_args, **build_kwargs)

        training.build_trainer = build_policy_v2_trainer
        args.experiment_label = "Policy V2 D1 coverage-preserving frontier bonus"
        args.policy_v2_notes = {
            "policy_v2_id": "D1",
            "learning_frontier": payload["hard_definition"],
            "nominal_hard_bonus": 0.05,
            "realized_hard_bonus": 0.0625,
            "alignment_reason": "128 samples and batch=4 require the nominal 6.4 bonus appearances to round up to 8",
        }
    summary = run(args)
    policy_run = {
        "policy_id": args.policy_id,
        "policy_namespace": payload["policy_namespace"],
        "v1_evidence_commit": payload["v1_evidence_commit"],
        "preregistration": str(preregistration),
        "preregistration_sha256": sha256_file(preregistration),
        "executor_sha256": {
            path.name: sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().with_name("policy_v2.py"),
                Path(__file__).resolve().with_name("run_pilot.py"),
            )
        },
        "run_reason": payload["convergence_controls"]
        .get(args.policy_id, {})
        .get(
            "run_reason",
            payload["hypotheses"].get(args.policy_id, payload["convergence_controls"]["run_reason"]),
        ),
        "summary": summary,
    }
    (args.output.resolve() / "policy_v2_run.json").write_text(
        json.dumps(policy_run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"policy_id": args.policy_id, "output": str(args.output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
