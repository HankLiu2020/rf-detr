# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Deterministic bucket-quota samplers for RF-DETR sample dynamics."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

import torch
from torch.utils.data import Sampler


StateProvider = Callable[[], Mapping[str, object]]
IndexMapper = Callable[[int], int]


class BucketQuotaSampler(Sampler[int]):
    """Generate one global epoch index list, then slice it across ranks.

    The sampler deliberately does not use independent per-rank multinomial
    draws. A single seeded global list gives every rank the same step count and
    makes exposure/replay auditable.

    Args:
        dataset_length: Number of indices visible to the DataLoader.
        sample_ids: Stable IDs for the original dataset indices.
        num_samples: Number of samples yielded by each rank.
        state_provider: Optional callback returning ``sample_id -> state`` at
            iterator creation time.
        index_mapper: Optional mapping from visible dataset index to original
            dataset index (used by :class:`GradAccumAlignedDataset`).
        world_size: Number of ranks.
        rank: Current rank.
        seed: Base deterministic seed.
        base_coverage, hard_learnable, mastered_replay, exploration: Bucket
            quotas. They are normalized when their sum is not exactly one.
    """

    def __init__(
        self,
        dataset_length: int,
        sample_ids: Sequence[str],
        num_samples: int,
        *,
        state_provider: StateProvider | None = None,
        index_mapper: IndexMapper | None = None,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
        base_coverage: float = 0.60,
        hard_learnable: float = 0.25,
        mastered_replay: float = 0.10,
        exploration: float = 0.05,
    ) -> None:
        if dataset_length < 1 or num_samples < 1:
            raise ValueError("dataset_length and num_samples must be >= 1")
        if len(sample_ids) < 1:
            raise ValueError("sample_ids must not be empty")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        quotas = (base_coverage, hard_learnable, mastered_replay, exploration)
        if any(quota < 0 for quota in quotas) or sum(quotas) <= 0:
            raise ValueError("sampler bucket quotas must be non-negative and have a positive sum")
        self.dataset_length = int(dataset_length)
        self.sample_ids = tuple(str(sample_id) for sample_id in sample_ids)
        self.num_samples = int(num_samples)
        self.state_provider = state_provider
        self.index_mapper = index_mapper or (lambda index: index)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.seed = int(seed)
        total = float(sum(quotas))
        self.quotas = tuple(quota / total for quota in quotas)
        self.epoch = 0
        self.last_global_indices: tuple[int, ...] = ()

    def __len__(self) -> int:
        """Return the per-rank number of indices."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch component of the deterministic random seed."""
        self.epoch = int(epoch)

    def _state_map(self) -> Mapping[str, object]:
        """Read the current state mapping, tolerating an absent provider."""
        return self.state_provider() if self.state_provider is not None else {}

    @staticmethod
    def _state_name(value: object) -> str:
        """Normalize Enum/string state values."""
        return str(getattr(value, "value", value))

    def _bucket_indices(self) -> tuple[list[int], list[int], list[int]]:
        """Return all-index, hard, and mastered buckets in visible-index space."""
        states = self._state_map()
        all_indices = list(range(self.dataset_length))
        hard: list[int] = []
        mastered: list[int] = []
        for visible_index in all_indices:
            original_index = self.index_mapper(visible_index)
            if not 0 <= original_index < len(self.sample_ids):
                raise IndexError(f"index mapper returned {original_index} outside sample_ids")
            state = self._state_name(states.get(self.sample_ids[original_index], "LEARNING"))
            if state == "HARD_LEARNABLE":
                hard.append(visible_index)
            elif state == "MASTERED":
                mastered.append(visible_index)
        return all_indices, hard, mastered

    @staticmethod
    def _draw(bucket: list[int], fallback: list[int], count: int, generator: torch.Generator) -> list[int]:
        """Draw exactly *count* indices with replacement from a bucket."""
        source = bucket or fallback
        if count <= 0:
            return []
        choices = torch.randint(len(source), (count,), generator=generator).tolist()
        return [source[choice] for choice in choices]

    def global_epoch_indices(self) -> list[int]:
        """Generate the auditable global index list for the current epoch."""
        total_samples = self.num_samples * self.world_size
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        all_indices, hard, mastered = self._bucket_indices()
        base_count = int(round(total_samples * self.quotas[0]))
        hard_count = int(round(total_samples * self.quotas[1]))
        mastered_count = int(round(total_samples * self.quotas[2]))
        exploration_count = total_samples - base_count - hard_count - mastered_count
        indices = []
        indices.extend(self._draw(all_indices, all_indices, base_count, generator))
        indices.extend(self._draw(hard, all_indices, hard_count, generator))
        indices.extend(self._draw(mastered, all_indices, mastered_count, generator))
        indices.extend(self._draw(all_indices, all_indices, exploration_count, generator))
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        return [indices[index] for index in permutation]

    def __iter__(self):
        """Yield this rank's slice of one shared global epoch list."""
        global_indices = self.global_epoch_indices()
        self.last_global_indices = tuple(global_indices)
        return iter(global_indices[self.rank :: self.world_size])

    def exposure_report(self, indices: Iterable[int] | None = None) -> dict[str, int]:
        """Count visible-index exposure for the current/global epoch."""
        values = list(indices) if indices is not None else list(self.last_global_indices)
        report = {sample_id: 0 for sample_id in self.sample_ids}
        for index in values:
            original_index = self.index_mapper(index)
            sample_id = self.sample_ids[original_index]
            report[sample_id] = report.get(sample_id, 0) + 1
        return report

    def exposure_multipliers(self, indices: Iterable[int] | None = None) -> dict[str, float]:
        """Return actual exposure divided by the uniform-draw expectation."""
        values = list(indices) if indices is not None else list(self.last_global_indices)
        if not values:
            return {sample_id: 0.0 for sample_id in self.sample_ids}
        expected = len(values) / len(self.sample_ids)
        report = self.exposure_report(values)
        return {sample_id: report.get(sample_id, 0) / expected for sample_id in self.sample_ids}
