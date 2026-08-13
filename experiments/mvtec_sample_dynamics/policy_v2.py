# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Experiment-side Policy V2 selection and coverage-preserving sampling."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import Sampler


def learning_frontier_ids(
    states: Mapping[str, Any],
    *,
    ema_percentile_min: float = 0.75,
    normalized_slope_max: float = -0.01,
    absolute_ema_loss_floor: float = 127.0782241821289,
    mask_iou_threshold: float = 0.5,
) -> set[str]:
    """Select high-EMA-loss samples that are still improving and have task error."""
    records = [record for record in states.values() if getattr(record, "loss_ema", None) is not None]
    if len(records) < 2:
        return set()
    population = sorted(float(record.loss_ema) for record in records)

    def percentile(value: float) -> float:
        return (sum(item <= value for item in population) - 1) / (len(population) - 1)

    selected: set[str] = set()
    for sample_id, record in states.items():
        if getattr(record, "loss_ema", None) is None:
            continue
        ema = float(record.loss_ema)
        normalized_slope = float(record.slope) / max(abs(ema), 1.0)
        probes = list(getattr(record, "probe_history", ()))
        latest_probe = probes[-1] if probes else {}
        matched_mask_iou = latest_probe.get("matched_mask_iou")
        task_error = int(latest_probe.get("fn", 0)) > 0 or (
            matched_mask_iou is not None and float(matched_mask_iou) < mask_iou_threshold
        )
        state_name = str(getattr(getattr(record, "state", "LEARNING"), "value", getattr(record, "state", "LEARNING")))
        stalled_conflict = int(getattr(record, "probe_conflict_count", 0)) >= 3 and normalized_slope >= -0.001
        if (
            percentile(ema) >= ema_percentile_min
            and normalized_slope <= normalized_slope_max
            and ema >= absolute_ema_loss_floor
            and task_error
            and state_name != "SUSPECT"
            and not stalled_conflict
        ):
            selected.add(str(sample_id))
    return selected


class CoverageBonusSampler(Sampler[int]):
    """Cover every sample once, then add aligned Learning-Frontier replay.

    The nominal bonus is rounded up to a whole batch. Both nominal and realized rates are exposed for the resource
    ledger and compute-normalized analysis.
    """

    def __init__(
        self,
        dataset_length: int,
        sample_ids: Sequence[str],
        *,
        state_provider: Callable[[], Mapping[str, Any]],
        index_mapper: Callable[[int], int] | None = None,
        seed: int,
        batch_size: int,
        nominal_bonus_fraction: float = 0.05,
        warmup_epochs: int = 5,
        frontier_kwargs: Mapping[str, float] | None = None,
    ) -> None:
        if dataset_length < 1 or len(sample_ids) < 1:
            raise ValueError("dataset_length and sample_ids must be non-empty")
        if batch_size < 1 or not 0.0 <= nominal_bonus_fraction <= 1.0:
            raise ValueError("batch_size must be positive and nominal bonus must be in [0, 1]")
        self.dataset_length = int(dataset_length)
        self.sample_ids = tuple(str(sample_id) for sample_id in sample_ids)
        self.state_provider = state_provider
        self.index_mapper = index_mapper or (lambda index: index)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.nominal_bonus_fraction = float(nominal_bonus_fraction)
        nominal_count = self.dataset_length * self.nominal_bonus_fraction
        self.aligned_bonus_count = (
            int(math.ceil(nominal_count / self.batch_size) * self.batch_size) if nominal_count else 0
        )
        self.realized_bonus_fraction = self.aligned_bonus_count / self.dataset_length
        self.warmup_epochs = int(warmup_epochs)
        self.frontier_kwargs = dict(frontier_kwargs or {})
        self.epoch = 0
        self.last_global_indices: tuple[int, ...] = ()
        self.last_frontier_ids: tuple[str, ...] = ()

    def __len__(self) -> int:
        bonus = 0 if self.epoch < self.warmup_epochs else self.aligned_bonus_count
        return self.dataset_length + bonus

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _visible_frontier_indices(self) -> list[int]:
        frontier = learning_frontier_ids(self.state_provider(), **self.frontier_kwargs)
        self.last_frontier_ids = tuple(sorted(frontier))
        return [
            visible_index
            for visible_index in range(self.dataset_length)
            if self.sample_ids[self.index_mapper(visible_index)] in frontier
        ]

    def global_epoch_indices(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        base = torch.randperm(self.dataset_length, generator=generator).tolist()
        if self.epoch < self.warmup_epochs or self.aligned_bonus_count == 0:
            self.last_frontier_ids = ()
            return base
        frontier = self._visible_frontier_indices()
        if not frontier:
            # Preserve the declared post-warmup compute contract while falling
            # back to uniform exploration when no sample clears the frontier.
            frontier = list(range(self.dataset_length))
        choices = torch.randint(len(frontier), (self.aligned_bonus_count,), generator=generator).tolist()
        combined = base + [frontier[index] for index in choices]
        order = torch.randperm(len(combined), generator=generator).tolist()
        return [combined[index] for index in order]

    def __iter__(self) -> Iterator[int]:
        values = self.global_epoch_indices()
        self.last_global_indices = tuple(values)
        return iter(values)

    def exposure_report(self, indices: Sequence[int] | None = None) -> dict[str, int]:
        values = list(self.last_global_indices if indices is None else indices)
        report = {sample_id: 0 for sample_id in self.sample_ids}
        for index in values:
            report[self.sample_ids[self.index_mapper(index)]] += 1
        return report

    def exposure_multipliers(self, indices: Sequence[int] | None = None) -> dict[str, float]:
        values = list(self.last_global_indices if indices is None else indices)
        if not values:
            return {sample_id: 0.0 for sample_id in self.sample_ids}
        expected = len(values) / len(self.sample_ids)
        return {sample_id: count / expected for sample_id, count in self.exposure_report(values).items()}
