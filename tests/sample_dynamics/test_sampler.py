# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for RF6 bucket-quota sampling and DDP index alignment."""

from __future__ import annotations

import torch

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.sample_dynamics import BucketQuotaSampler, SampleState
from rfdetr.training.module_data import GradAccumAlignedDataset, RFDETRDataModule


def test_sampler_is_deterministic_and_slices_one_global_stream() -> None:
    """Ranks receive disjoint strided views of one repeatable global list."""
    ids = [f"train:{index}:image-{index}.jpg" for index in range(8)]
    rank_zero = BucketQuotaSampler(len(ids), ids, 6, world_size=2, rank=0, seed=17)
    rank_one = BucketQuotaSampler(len(ids), ids, 6, world_size=2, rank=1, seed=17)

    rank_zero.set_epoch(4)
    rank_one.set_epoch(4)
    first_global = rank_zero.global_epoch_indices()
    assert first_global == rank_one.global_epoch_indices()
    assert list(iter(rank_zero)) == first_global[0::2]
    assert list(iter(rank_one)) == first_global[1::2]

    rank_zero.set_epoch(4)
    assert rank_zero.global_epoch_indices() == first_global
    rank_zero.set_epoch(5)
    assert rank_zero.global_epoch_indices() != first_global


def test_sampler_honors_hard_and_mastered_bucket_quotas() -> None:
    """Explicit buckets receive their configured exposure when populated."""
    ids = [f"train:{index}:image-{index}.jpg" for index in range(4)]
    states = {
        ids[0]: SampleState.HARD_LEARNABLE,
        ids[1]: SampleState.HARD_LEARNABLE,
        ids[2]: SampleState.MASTERED,
        ids[3]: SampleState.MASTERED,
    }
    sampler = BucketQuotaSampler(
        len(ids),
        ids,
        10,
        state_provider=lambda: states,
        base_coverage=0.0,
        hard_learnable=0.5,
        mastered_replay=0.5,
        exploration=0.0,
    )

    report = sampler.exposure_report(sampler.global_epoch_indices())

    assert sum(report.get(sample_id, 0) for sample_id in ids[:2]) == 5
    assert sum(report.get(sample_id, 0) for sample_id in ids[2:]) == 5


def test_sampler_maps_aligned_indices_to_original_ids() -> None:
    """Padding indices are classified and reported using original sample IDs."""
    source = torch.utils.data.TensorDataset(torch.arange(5))
    aligned = GradAccumAlignedDataset(source, effective_batch_size=4, world_size=1)
    sampler = BucketQuotaSampler(
        len(aligned),
        [f"train:{index}:image-{index}.jpg" for index in range(len(source))],
        len(aligned),
        index_mapper=aligned.original_index,
    )

    visible = sampler.global_epoch_indices()

    assert len(aligned) == 8
    assert all(0 <= aligned.original_index(index) < len(source) for index in visible)
    assert sum(sampler.exposure_report(visible).values()) == len(aligned)


class _StableDataset(torch.utils.data.Dataset):
    """Minimal dataset exposing the stable-ID contract used by the DataModule."""

    def __init__(self, length: int) -> None:
        self.sample_ids = tuple(f"train:{index}:image-{index}.jpg" for index in range(length))

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, object]]:
        return torch.zeros(3, 8, 8), {"sample_id": self.sample_ids[index]}


def test_datamodule_uses_dynamic_sampler_for_small_and_aligned_datasets(tmp_path) -> None:
    """Both legacy DataLoader branches use the dynamic sampler when enabled."""
    model_config = RFDETRBaseConfig(pretrain_weights=None, device="cpu", resolution=64, num_classes=1)
    train_config = TrainConfig(
        dataset_dir=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        batch_size=2,
        grad_accum_steps=1,
        num_workers=0,
        multi_scale=False,
        expanded_scales=False,
        tensorboard=False,
        sample_dynamics_enabled=True,
        sample_dynamics_mode="sampler",
    )
    datamodule = RFDETRDataModule(model_config, train_config)

    datamodule._dataset_train = _StableDataset(3)
    small_loader = datamodule.train_dataloader()
    assert isinstance(small_loader.sampler, BucketQuotaSampler)
    assert len(small_loader.sampler) == 10

    datamodule._dataset_train = _StableDataset(11)
    aligned_loader = datamodule.train_dataloader()
    assert isinstance(aligned_loader.sampler, BucketQuotaSampler)
    assert isinstance(aligned_loader.dataset, GradAccumAlignedDataset)
    assert len(aligned_loader.sampler) == len(aligned_loader.dataset)
