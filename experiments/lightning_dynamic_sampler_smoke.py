#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Run a tiny two-GPU Lightning fit through RFDETRDataModule's dynamic sampler."""

from __future__ import annotations

import tempfile

import torch
from pytorch_lightning import LightningModule, Trainer

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.utilities.tensors import NestedTensor


class _StableDataset(torch.utils.data.Dataset):
    """Small image dataset exposing the production stable-ID contract."""

    def __init__(self, length: int) -> None:
        self.sample_ids = tuple(f"train:{index}:image-{index}.jpg" for index in range(length))

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, str]]:
        return torch.zeros(3, 8, 8), {"sample_id": self.sample_ids[index]}


class _TinyModule(LightningModule):
    """Minimal optimizer-bearing module for the DataModule/Lightning smoke."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(1, 1)

    def training_step(
        self,
        batch: tuple[NestedTensor, tuple[dict[str, str], ...]],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        samples, _targets = batch
        return self.projection(samples.tensors.mean().reshape(1, 1)).square().mean()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)


def main() -> None:
    """Execute two short DDP steps and exit only after both ranks finish."""
    with tempfile.TemporaryDirectory(prefix="rfdetr-dynamic-sampler-") as output_dir:
        model_config = RFDETRBaseConfig(pretrain_weights=None, device="cuda", resolution=64, num_classes=1)
        train_config = TrainConfig(
            dataset_dir=output_dir,
            output_dir=output_dir,
            batch_size=1,
            grad_accum_steps=1,
            num_workers=0,
            multi_scale=False,
            use_ema=False,
            tensorboard=False,
            sample_dynamics_enabled=True,
            sample_dynamics_mode="sampler",
        )
        datamodule = RFDETRDataModule(model_config, train_config)
        datamodule._dataset_train = _StableDataset(8)
        datamodule._dataset_val = _StableDataset(1)
        trainer = Trainer(
            accelerator="gpu",
            devices=2,
            strategy="ddp_spawn",
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            use_distributed_sampler=False,
        )
        trainer.fit(_TinyModule(), datamodule=datamodule)
    print("lightning_dynamic_sampler_smoke=PASS world_size=2")


if __name__ == "__main__":
    main()
