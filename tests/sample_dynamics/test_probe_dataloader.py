# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the deterministic train-probe DataLoader."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.training.module_data import RFDETRDataModule


class _ProbeDataset(torch.utils.data.Dataset):
    """One-image dataset with an RF-DETR transform slot."""

    def __init__(self) -> None:
        self._transforms = lambda image, target: (image, target)

    def __len__(self) -> int:
        """Return the fixed dataset length."""
        return 1

    def __getitem__(self, index: int):
        """Return one image and target with a stable ID."""
        del index
        target = {
            "boxes": torch.tensor([[1.0, 1.0, 4.0, 3.0]]),
            "labels": torch.tensor([0]),
            "image_id": torch.tensor([1]),
            "sample_id": "train:1:one.jpg",
            "orig_size": torch.tensor([6, 8]),
            "size": torch.tensor([6, 8]),
            "area": torch.tensor([12.0]),
            "iscrowd": torch.tensor([0]),
        }
        return self._transforms(Image.new("RGB", (8, 6)), target)


def test_train_probe_dataloader_is_sequential_and_preserves_id(tmp_path: Path) -> None:
    """The probe loader uses fixed transforms and retains sample metadata."""
    model_config = RFDETRBaseConfig(pretrain_weights=None, device="cpu", resolution=64, num_classes=1)
    train_config = TrainConfig(
        dataset_dir=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        batch_size=1,
        grad_accum_steps=1,
        num_workers=0,
        multi_scale=False,
        expanded_scales=False,
        tensorboard=False,
    )
    datamodule = RFDETRDataModule(model_config, train_config)
    datamodule._dataset_train = _ProbeDataset()

    loader = datamodule.train_probe_dataloader()
    _, targets = next(iter(loader))

    assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)
    assert targets[0]["sample_id"] == "train:1:one.jpg"
