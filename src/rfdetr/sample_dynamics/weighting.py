# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Previous-policy sample loss weights for RF-DETR experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import torch
from torch import Tensor


def cap_effective_contribution(weight: float, exposure_multiplier: float, cap: float = 2.5) -> float:
    """Cap the product of a loss weight and sampler exposure multiplier."""
    if weight < 0.0 or exposure_multiplier < 0.0 or cap <= 0.0:
        raise ValueError("weight, exposure_multiplier, and cap must be non-negative with cap > 0")
    return min(float(cap), float(weight) * float(exposure_multiplier))


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
        """Return mean-normalized weights while respecting optional RF7 exposure caps."""
        ids = tuple(str(sample_id) for sample_id in sample_ids)
        values = torch.tensor(
            [self.weights.get(sample_id, 1.0) for sample_id in ids],
            dtype=torch.float32,
            device=device,
        )
        if values.numel() == 0:
            return values
        values = values / values.mean().clamp_min(1e-8)
        upper = torch.full_like(values, self.maximum)
        if exposure_multipliers is not None and effective_cap is not None:
            if effective_cap <= 0.0:
                raise ValueError("effective_cap must be > 0")
            exposures = torch.tensor(
                [max(0.0, float(exposure_multipliers.get(sample_id, 1.0))) for sample_id in ids],
                dtype=values.dtype,
                device=device,
            )
            upper = torch.minimum(upper, float(effective_cap) / exposures.clamp_min(1e-8))
            if bool(torch.any(upper < self.minimum)):
                raise ValueError("effective contribution cap is incompatible with the configured minimum weight")
        values = torch.maximum(torch.minimum(values, upper), torch.as_tensor(self.minimum, device=device))
        # Clipping can move the mean away from one. Redistribute the residual
        # over non-saturated entries so the batch normalization contract remains
        # true while every value stays within the configured bounds.
        for _ in range(values.numel() + 1):
            residual = values.numel() - values.sum()
            if abs(float(residual.item())) < 1e-6:
                break
            if residual > 0:
                eligible = values < upper - 1e-6
                if not bool(torch.any(eligible)):
                    break
                delta = residual / max(int(eligible.sum().item()), 1)
                values = torch.where(eligible, torch.minimum(values + delta, upper), values)
            else:
                eligible = values > self.minimum + 1e-6
                if not bool(torch.any(eligible)):
                    break
                delta = (-residual) / max(int(eligible.sum().item()), 1)
                values = torch.where(
                    eligible,
                    torch.maximum(values - delta, torch.as_tensor(self.minimum, device=device)),
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

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore a policy saved by :meth:`state_dict`."""
        self.version = int(state.get("version", 0))
        self.mastered = float(state.get("mastered", self.mastered))
        self.learning = float(state.get("learning", self.learning))
        self.hard_learnable = float(state.get("hard_learnable", self.hard_learnable))
        self.suspect = float(state.get("suspect", self.suspect))
        self.minimum = float(state.get("minimum", self.minimum))
        self.maximum = float(state.get("maximum", self.maximum))
        self.weights = {str(key): float(value) for key, value in dict(state.get("weights", {})).items()}
