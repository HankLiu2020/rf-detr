# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Previous-policy sample loss weights for RF-DETR experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch
import torch.distributed as distributed
from torch import Tensor


def cap_effective_contribution(weight: float, exposure_multiplier: float, cap: float = 2.5) -> float:
    """Cap the product of a loss weight and sampler exposure multiplier."""
    if weight < 0.0 or exposure_multiplier < 0.0 or cap <= 0.0:
        raise ValueError("weight, exposure_multiplier, and cap must be non-negative with cap > 0")
    return min(float(cap), float(weight) * float(exposure_multiplier))


def _distributed_is_initialized() -> bool:
    """Return whether loss-weight normalization spans multiple ranks."""
    return distributed.is_available() and distributed.is_initialized() and distributed.get_world_size() > 1


def _global_sum(value: Tensor) -> Tensor:
    """Return a detached scalar sum across the active process group."""
    result = value.detach().clone()
    if _distributed_is_initialized():
        distributed.all_reduce(result, op=distributed.ReduceOp.SUM)
    return result


def _global_sum_and_count(values: Tensor) -> tuple[Tensor, Tensor]:
    """Return global value mass and image count on ``values.device``."""
    statistics = torch.stack(
        (
            values.sum(),
            values.new_tensor(float(values.numel())),
        )
    )
    if _distributed_is_initialized():
        distributed.all_reduce(statistics, op=distributed.ReduceOp.SUM)
    return statistics[0], statistics[1]


@dataclass
class SampleWeightPolicy:
    """Versioned, checkpoint-safe mapping from sample IDs to base weights."""

    version: int = 0
    mastered: float = 0.8
    learning: float = 1.0
    hard_learnable: float = 1.3
    suspect: float = 0.8
    minimum: float = 0.7
    maximum: float = 1.3
    weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.minimum <= 0 or self.maximum < self.minimum:
            raise ValueError("sample weight bounds must satisfy 0 < minimum <= maximum")

    def update_from_states(self, states: Iterable[Mapping[str, object]]) -> None:
        """Create the next policy version from serialized sample states."""
        state_weights = {
            "MASTERED": self.mastered,
            "LEARNING": self.learning,
            "HARD_LEARNABLE": self.hard_learnable,
            "SUSPECT": self.suspect,
        }
        self.version += 1
        self.weights = {
            str(state["sample_id"]): float(max(self.minimum, min(self.maximum, state_weights[str(state["state"])])))
            for state in states
        }

    def batch_weights(
        self,
        sample_ids: Iterable[str],
        *,
        device: torch.device | str,
        exposure_multipliers: Mapping[str, float] | None = None,
        effective_cap: float | None = None,
    ) -> Tensor:
        """Return globally mean-normalized weights with optional RF7 caps.

        In DDP, all ranks participate in the same scalar Tensor collectives.  Normalization and post-clipping residual
        redistribution therefore use the global image batch rather than each rank's local state composition.
        """
        ids = tuple(str(sample_id) for sample_id in sample_ids)
        values = torch.tensor(
            [self.weights.get(sample_id, 1.0) for sample_id in ids],
            dtype=torch.float32,
            device=device,
        )
        global_sum, global_count = _global_sum_and_count(values)
        if float(global_count.item()) == 0.0:
            return values
        values = values / (global_sum / global_count).clamp_min(1e-8)
        upper = torch.full_like(values, self.maximum)
        lower = torch.full_like(values, self.minimum)
        if exposure_multipliers is not None and effective_cap is not None:
            if effective_cap <= 0.0:
                raise ValueError("effective_cap must be > 0")
            exposures = torch.tensor(
                [max(0.0, float(exposure_multipliers.get(sample_id, 1.0))) for sample_id in ids],
                dtype=values.dtype,
                device=device,
            )
            upper = torch.minimum(upper, float(effective_cap) / exposures.clamp_min(1e-8))
            # A replay bucket can expose one sample more often than the hard
            # contribution cap permits at the configured minimum weight. The
            # cap is the safety invariant, so that sample's effective lower
            # bound is lowered to its cap-feasible upper bound for this batch.
            # Feasible samples retain the configured minimum exactly.
            lower = torch.minimum(lower, upper)
        global_upper_sum = _global_sum(upper.sum())
        if self.minimum > 1.0:
            raise ValueError("sample weight bounds cannot preserve a global mean of one")
        cap_infeasible = float(global_upper_sum.item()) < float(global_count.item()) - 1e-6
        # If realized replay exposure makes the cap-feasible upper bounds sum
        # to less than the batch size, mean-one normalization is mathematically
        # impossible. Preserve the safety cap and return the clipped weights;
        # the resulting sub-unit mean is an observable resource side effect.
        values = torch.maximum(torch.minimum(values, upper), lower)
        if cap_infeasible:
            return values
        # Clipping can move the mean away from one. Redistribute the residual
        # over globally non-saturated entries so rank composition cannot erase
        # the intended MASTERED/HARD_LEARNABLE ratio.
        for _ in range(32):
            current_sum, _ = _global_sum_and_count(values)
            residual = global_count - current_sum
            if abs(float(residual.item())) < 1e-6:
                break
            if residual > 0 and not cap_infeasible:
                eligible = values < upper - 1e-6
                eligible_count = _global_sum(eligible.sum(dtype=values.dtype))
                if float(eligible_count.item()) == 0.0:
                    break
                delta = residual / eligible_count
                values = torch.where(eligible, torch.minimum(values + delta, upper), values)
            else:
                eligible = values > lower + 1e-6
                eligible_count = _global_sum(eligible.sum(dtype=values.dtype))
                if float(eligible_count.item()) == 0.0:
                    break
                delta = (-residual) / eligible_count
                values = torch.where(
                    eligible,
                    torch.maximum(values - delta, lower),
                    values,
                )
        return values

    def state_dict(self) -> dict[str, object]:
        """Return a JSON/checkpoint-safe policy representation."""
        return {
            "version": self.version,
            "mastered": self.mastered,
            "learning": self.learning,
            "hard_learnable": self.hard_learnable,
            "suspect": self.suspect,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "weights": dict(self.weights),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a policy saved by :meth:`state_dict`."""
        self.version = int(state.get("version", 0))
        self.mastered = float(state.get("mastered", self.mastered))
        self.learning = float(state.get("learning", self.learning))
        self.hard_learnable = float(state.get("hard_learnable", self.hard_learnable))
        self.suspect = float(state.get("suspect", self.suspect))
        self.minimum = float(state.get("minimum", self.minimum))
        self.maximum = float(state.get("maximum", self.maximum))
        self.weights = {str(key): float(value) for key, value in dict(state.get("weights", {})).items()}
