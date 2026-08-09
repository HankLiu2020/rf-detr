#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Run a tiny two-GPU Lightning fit through RFDETRDataModule's dynamic sampler."""

from __future__ import annotations

import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as distributed
from pytorch_lightning import LightningModule, Trainer

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.sample_dynamics import SampleObservationBuffer, SampleStateStore
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.tensors import NestedTensor


class _StableDataset(torch.utils.data.Dataset):
    """Small image dataset exposing the production stable-ID contract."""

    def __init__(self, length: int) -> None:
        self.sample_ids = tuple(f"train:{index}:image-{index}.jpg" for index in range(length))

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, str]]:
        return torch.zeros(3, 8, 8), {"sample_id": self.sample_ids[index]}


class _TinyModule(RFDETRModelModule):
    """Tiny optimizer module using RFDETRModelModule's real epoch-end lifecycle."""

    def __init__(self) -> None:
        LightningModule.__init__(self)
        self.projection = torch.nn.Linear(1, 1)
        self.sample_observation_buffer = SampleObservationBuffer()
        self.sample_state_store = SampleStateStore()
        self.sample_weight_policy = None
        self.model_config = SimpleNamespace(fused_optimizer=False)
        self.train_config = SimpleNamespace(
            seed=None,
            multi_scale=False,
            do_random_resize_via_padding=False,
            sample_dynamics_probe_interval=0,
            sample_dynamics_output_dir=None,
        )
        self._accumulated_box_normalizer = None
        self._lr_scheduler_interval = "step"

    def training_step(
        self,
        batch: tuple[NestedTensor, tuple[dict[str, str], ...]],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        samples, targets = batch
        for target in targets:
            value = float(self.global_rank + 1)
            self.sample_observation_buffer.records.append(
                {
                    "sample_id": target["sample_id"],
                    "epoch": int(self.current_epoch),
                    "global_step": int(self.global_step),
                    "weighted_per_image_normalized_loss": value,
                    "per_image_normalized_losses": {"loss_ce": value},
                }
            )
        return self.projection(samples.tensors.mean().reshape(1, 1)).square().mean()

    def on_train_epoch_end(self) -> None:
        """Run the production sync hook and assert identical state on both ranks."""
        super().on_train_epoch_end()
        snapshots: list[object] = [None] * distributed.get_world_size()
        distributed.all_gather_object(snapshots, self.sample_state_store.state_dict())
        assert all(snapshot == snapshots[0] for snapshot in snapshots)

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
            max_epochs=2,
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
