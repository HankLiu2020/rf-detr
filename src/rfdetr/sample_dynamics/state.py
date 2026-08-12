# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Training-dynamics state tracking for stable RF-DETR samples."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from rfdetr.sample_dynamics.observation import aggregate_observations

if TYPE_CHECKING:
    from rfdetr.sample_dynamics.weighting import SampleWeightPolicy


class SampleState(str, Enum):
    """Operational state assigned to a sample by :class:`SampleStateStore`."""

    MASTERED = "MASTERED"
    LEARNING = "LEARNING"
    HARD_LEARNABLE = "HARD_LEARNABLE"
    SUSPECT = "SUSPECT"


def ema_update(previous: float | None, value: float, alpha: float) -> float:
    """Apply one configurable exponential moving-average update."""
    if previous is None:
        return value
    return alpha * value + (1.0 - alpha) * previous


def window_slope(values: list[float]) -> float:
    """Return the least-squares slope over a short ordered value window."""
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2.0
    y_mean = sum(values) / len(values)
    numerator = sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values))
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    return numerator / denominator if denominator else 0.0


def percentile_rank(value: float, values: Iterable[float]) -> float:
    """Return an inclusive empirical percentile in ``[0, 1]``."""
    population = sorted(float(item) for item in values)
    if not population:
        return 0.0
    if len(population) == 1:
        return 0.0
    rank = sum(item <= value for item in population) - 1
    return max(0.0, min(1.0, rank / (len(population) - 1)))


@dataclass(frozen=True)
class StatePolicy:
    """Configurable state thresholds and difficulty component weights."""

    ema_alpha: float = 0.3
    window_size: int = 5
    max_history: int = 32
    mastered_percentile: float = 0.25
    hard_percentile: float = 0.75
    min_history_for_mastered: int = 3
    suspect_patience: int = 3
    improvement_epsilon: float = 1e-4
    probe_mask_iou_threshold: float = 0.5
    mask_error_weight: float = 0.5
    loss_weight: float = 0.5
    error_weight: float = 0.3
    trend_weight: float = 0.15
    forgetting_weight: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if self.window_size < 2 or self.max_history < self.window_size:
            raise ValueError("max_history must be >= window_size >= 2")
        if not 0.0 <= self.mastered_percentile < self.hard_percentile <= 1.0:
            raise ValueError("mastered_percentile and hard_percentile must satisfy 0 <= mastered < hard <= 1")
        if self.min_history_for_mastered < 1:
            raise ValueError("min_history_for_mastered must be >= 1")
        if self.suspect_patience < 1:
            raise ValueError("suspect_patience must be >= 1")
        if not 0.0 <= self.probe_mask_iou_threshold <= 1.0:
            raise ValueError("probe_mask_iou_threshold must be in [0, 1]")
        if self.mask_error_weight < 0.0:
            raise ValueError("mask_error_weight must be >= 0")


@dataclass
class SampleStateRecord:
    """Serializable state and history for one sample."""

    sample_id: str
    state: SampleState = SampleState.LEARNING
    current_loss: float = 0.0
    loss_percentile: float = 0.0
    loss_ema: float | None = None
    slope: float = 0.0
    variance: float = 0.0
    consecutive_hard_count: int = 0
    forgetting_count: int = 0
    probe_conflict_count: int = 0
    difficulty: float = 0.0
    last_epoch: int = -1
    policy_version: int = 0
    history: list[float] = field(default_factory=list)
    probe_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly state record."""
        values = asdict(self)
        values["state"] = self.state.value
        return values


class SampleStateStore:
    """Maintain long-term training dynamics without modifying training behavior."""

    def __init__(self, policy: StatePolicy | None = None, policy_version: int = 0) -> None:
        self.policy = policy or StatePolicy()
        self.policy_version = int(policy_version)
        self.states: dict[str, SampleStateRecord] = {}

    @staticmethod
    def instant_loss_baseline(records: Iterable[Mapping[str, Any]]) -> dict[str, float]:
        """Return current normalized-loss percentiles, the RF4 instant baseline."""
        materialized = aggregate_observations(records)
        values = [SampleStateStore._loss_value(record) for record in materialized]
        return {str(record["sample_id"]): percentile_rank(value, values) for record, value in zip(materialized, values)}

    @staticmethod
    def _loss_value(record: Mapping[str, Any]) -> float:
        """Read image-local weighted loss, retaining global views for audit only."""
        if "weighted_per_image_normalized_loss" in record:
            return float(record["weighted_per_image_normalized_loss"])
        per_image = record.get("per_image_normalized_losses")
        if isinstance(per_image, Mapping):
            return sum(float(value) for value in per_image.values())
        if "weighted_normalized_loss" in record:
            return float(record["weighted_normalized_loss"])
        normalized = record.get("normalized_losses", {})
        if isinstance(normalized, Mapping):
            return sum(float(value) for value in normalized.values())
        return 0.0

    def _probe_conflict(self, probe: Mapping[str, Any] | None) -> bool:
        """Return whether a probe result has a stable detection or mask conflict."""
        if probe is None:
            return False
        gt_count = int(probe.get("gt_count", 0))
        matched_count = int(probe.get("matched_count", 0))
        matched_mask_iou = probe.get("matched_mask_iou")
        return bool(
            int(probe.get("fn", 0)) > 0
            or int(probe.get("class_error", 0)) > 0
            # Recall is undefined for an empty-GT image.  The probe keeps the
            # historical serialized value (0.0) for compatibility, but that
            # placeholder must not become a conflict signal.
            or (gt_count > 0 and float(probe.get("gt_recall", 1.0)) < 1.0)
            or (
                gt_count > 0
                and matched_count > 0
                and matched_mask_iou is not None
                and float(matched_mask_iou) < self.policy.probe_mask_iou_threshold
            )
        )

    def _error_severity(self, probe: Mapping[str, Any] | None) -> float:
        """Convert detection and segmentation probe errors into one bounded component."""
        if probe is None:
            return 0.0
        fn = float(probe.get("fn", 0))
        fp = float(probe.get("fp", 0))
        class_error = float(probe.get("class_error", 0))
        gt_count = max(float(probe.get("gt_count", 1)), 1.0)
        matched_count = float(probe.get("matched_count", 0))
        matched_mask_iou = probe.get("matched_mask_iou")
        mask_error = 0.0
        if matched_mask_iou is not None:
            mask_error = self.policy.mask_error_weight * (1.0 - float(matched_mask_iou)) * matched_count
        return min(1.0, (fn + 0.7 * class_error + 0.3 * fp + mask_error) / gt_count)

    def update(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        probe_records: Iterable[Mapping[str, Any]] | None = None,
        epoch: int = -1,
    ) -> dict[str, SampleStateRecord]:
        """Update state from detached observer records and optional probe metrics.

        The current batch/observation-set percentile is retained as the simple Instant-Loss baseline. State transitions
        use only prior observations and therefore cannot feed the current raw loss back into the same step.
        """
        materialized = aggregate_observations(records)
        if not materialized:
            return {}
        baseline = self.instant_loss_baseline(materialized)
        probes = {str(record["sample_id"]): record for record in (probe_records or [])}
        updated: dict[str, SampleStateRecord] = {}
        for record in materialized:
            sample_id = str(record["sample_id"])
            value = self._loss_value(record)
            state_record = self.states.setdefault(sample_id, SampleStateRecord(sample_id=sample_id))
            previous_state = state_record.state
            state_record.current_loss = value
            state_record.loss_percentile = baseline[sample_id]
            state_record.loss_ema = ema_update(state_record.loss_ema, value, self.policy.ema_alpha)
            state_record.history.append(value)
            del state_record.history[: max(0, len(state_record.history) - self.policy.max_history)]
            recent = state_record.history[-self.policy.window_size :]
            state_record.slope = window_slope(recent)
            if len(recent) > 1:
                mean = sum(recent) / len(recent)
                state_record.variance = sum((item - mean) ** 2 for item in recent) / len(recent)
            else:
                state_record.variance = 0.0
            is_hard = state_record.loss_percentile >= self.policy.hard_percentile
            state_record.consecutive_hard_count = state_record.consecutive_hard_count + 1 if is_hard else 0
            if previous_state == SampleState.MASTERED and is_hard:
                state_record.forgetting_count += 1
            probe = probes.get(sample_id)
            if probe is not None:
                state_record.probe_history.append(dict(probe))
                del state_record.probe_history[: max(0, len(state_record.probe_history) - self.policy.max_history)]
            if self._probe_conflict(probe):
                state_record.probe_conflict_count += 1
            elif probe is not None:
                state_record.probe_conflict_count = 0
            state_record.state = self._classify(state_record)
            state_record.difficulty = self._difficulty(state_record, probe)
            state_record.last_epoch = int(epoch)
            state_record.policy_version = self.policy_version
            updated[sample_id] = state_record
        return updated

    def _classify(self, state_record: SampleStateRecord) -> SampleState:
        """Classify a record from trend, hard patience, and probe conflict."""
        suspect = (
            state_record.consecutive_hard_count >= self.policy.suspect_patience
            and state_record.probe_conflict_count >= self.policy.suspect_patience
            and state_record.slope >= -self.policy.improvement_epsilon
        )
        if suspect:
            return SampleState.SUSPECT
        if (
            len(state_record.history) >= self.policy.min_history_for_mastered
            and state_record.loss_percentile <= self.policy.mastered_percentile
            and state_record.slope <= 0.0
        ):
            return SampleState.MASTERED
        if state_record.loss_percentile >= self.policy.hard_percentile:
            return SampleState.HARD_LEARNABLE
        return SampleState.LEARNING

    def _difficulty(self, state_record: SampleStateRecord, probe: Mapping[str, Any] | None) -> float:
        """Combine percentile, probe severity, stalled trend, and forgetting."""
        trend_score = min(1.0, max(0.0, state_record.slope * 10.0 + 0.5))
        forgetting_score = min(1.0, float(state_record.forgetting_count) / self.policy.suspect_patience)
        difficulty = (
            self.policy.loss_weight * state_record.loss_percentile
            + self.policy.error_weight * self._error_severity(probe)
            + self.policy.trend_weight * trend_score
            + self.policy.forgetting_weight * forgetting_score
        )
        return max(0.0, min(1.0, difficulty))

    def get(self, sample_id: str) -> SampleStateRecord | None:
        """Return state for a sample, if it has been observed."""
        return self.states.get(sample_id)

    def state_dict(self) -> dict[str, Any]:
        """Return a checkpoint-safe state dictionary."""
        return {
            "policy": asdict(self.policy),
            "policy_version": self.policy_version,
            "states": {sample_id: record.to_dict() for sample_id, record in self.states.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a state dictionary produced by :meth:`state_dict`."""
        policy = state.get("policy")
        if isinstance(policy, Mapping):
            self.policy = StatePolicy(**dict(policy))
        self.policy_version = int(state.get("policy_version", 0))
        records = state.get("states", {})
        self.states = {}
        for sample_id, raw in records.items():
            values = dict(raw)
            values["sample_id"] = str(sample_id)
            values["state"] = SampleState(values.get("state", SampleState.LEARNING.value))
            self.states[str(sample_id)] = SampleStateRecord(**values)

    def snapshot(self) -> list[dict[str, Any]]:
        """Return states sorted by sample ID for reproducible exports."""
        return [self.states[sample_id].to_dict() for sample_id in sorted(self.states)]

    def build_weight_policy(self, template: "SampleWeightPolicy | None" = None) -> "SampleWeightPolicy":
        """Return the next versioned policy derived from current states."""
        from rfdetr.sample_dynamics.weighting import SampleWeightPolicy

        if template is None:
            policy = SampleWeightPolicy(version=self.policy_version)
        else:
            policy = SampleWeightPolicy(
                version=self.policy_version,
                mastered=template.mastered,
                learning=template.learning,
                hard_learnable=template.hard_learnable,
                suspect=template.suspect,
                minimum=template.minimum,
                maximum=template.maximum,
            )
        policy.update_from_states(self.snapshot())
        self.policy_version = policy.version
        return policy

    def export_json(self, path: str | Path) -> None:
        """Write policy metadata and sorted state records to JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.state_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
