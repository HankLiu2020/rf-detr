# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Export auditable review queues for samples classified as SUSPECT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from rfdetr.sample_dynamics.state import SampleState, SampleStateRecord


def _as_mapping(value: Mapping[str, Any] | SampleStateRecord) -> Mapping[str, Any]:
    """Normalize a state record or JSON-ready mapping to a read-only view."""
    return value.to_dict() if isinstance(value, SampleStateRecord) else value


def _sample_path(*records: Mapping[str, Any] | None) -> str | None:
    """Find the first stable path-like field supplied by an observation source."""
    for record in records:
        if record is None:
            continue
        for key in ("relative_path", "image_path", "path"):
            value = record.get(key)
            if value is not None:
                return str(value)
    return None


class ReviewExporter:
    """Write a review artifact without mutating datasets or training state."""

    @staticmethod
    def _reason(state: Mapping[str, Any]) -> list[str]:
        """Return deterministic human-readable reasons for review selection."""
        reasons: list[str] = []
        if int(state.get("consecutive_hard_count", 0)) > 0:
            reasons.append("persistent_high_loss")
        if int(state.get("probe_conflict_count", 0)) > 0:
            reasons.append("probe_conflict")
        if int(state.get("forgetting_count", 0)) > 0:
            reasons.append("forgetting")
        if float(state.get("slope", 0.0)) >= 0.0:
            reasons.append("stalled_or_worsening_trend")
        return reasons or ["state_policy"]

    @classmethod
    def build_queue(
        cls,
        states: Iterable[Mapping[str, Any] | SampleStateRecord],
        observations: Iterable[Mapping[str, Any]] = (),
        probe_records: Iterable[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        """Build sorted SUSPECT review entries from state, observer, and probe records."""
        state_map = {
            str(record.get("sample_id")): record
            for record in (_as_mapping(value) for value in states)
            if record.get("sample_id") is not None
        }
        observation_map: dict[str, Mapping[str, Any]] = {}
        for record in observations:
            sample_id = record.get("sample_id")
            if sample_id is not None:
                observation_map[str(sample_id)] = record
        probe_map = {
            str(record["sample_id"]): record for record in probe_records if record.get("sample_id") is not None
        }

        queue: list[dict[str, Any]] = []
        for sample_id in sorted(state_map):
            state = state_map[sample_id]
            if str(getattr(state.get("state"), "value", state.get("state"))) != SampleState.SUSPECT.value:
                continue
            observation = observation_map.get(sample_id)
            probe = probe_map.get(sample_id)
            queue.append(
                {
                    "sample_id": sample_id,
                    "state": SampleState.SUSPECT.value,
                    "path": _sample_path(state, observation, probe),
                    "reason": cls._reason(state),
                    "current_loss": float(state.get("current_loss", 0.0)),
                    "loss_ema": state.get("loss_ema"),
                    "loss_percentile": float(state.get("loss_percentile", 0.0)),
                    "slope": float(state.get("slope", 0.0)),
                    "variance": float(state.get("variance", 0.0)),
                    "difficulty": float(state.get("difficulty", 0.0)),
                    "forgetting_count": int(state.get("forgetting_count", 0)),
                    "probe_conflict_count": int(state.get("probe_conflict_count", 0)),
                    "policy_version": int(state.get("policy_version", 0)),
                    "history": list(state.get("history", [])),
                    "probe_history": list(state.get("probe_history", [])),
                    "observation": dict(observation) if observation is not None else None,
                    "probe": dict(probe) if probe is not None else None,
                    "fn": int(probe.get("fn", 0)) if probe is not None else 0,
                    "fp": int(probe.get("fp", 0)) if probe is not None else 0,
                    "class_error": int(probe.get("class_error", 0)) if probe is not None else 0,
                    "matched_iou": float(probe.get("matched_iou", 0.0)) if probe is not None else 0.0,
                    "gt_recall": float(probe.get("gt_recall", 0.0)) if probe is not None else 0.0,
                    "gt_count": int(probe.get("gt_count", 0)) if probe is not None else 0,
                    "pred_count": int(probe.get("pred_count", 0)) if probe is not None else 0,
                }
            )
        return queue

    @classmethod
    def export(
        cls,
        states: Iterable[Mapping[str, Any] | SampleStateRecord],
        observations: Iterable[Mapping[str, Any]] = (),
        probe_records: Iterable[Mapping[str, Any]] = (),
        output_path: str | Path = "review_queue.jsonl",
    ) -> list[dict[str, Any]]:
        """Build and write a JSON or JSONL SUSPECT queue, returning its entries."""
        queue = cls.build_queue(states, observations, probe_records)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            path.write_text(
                "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in queue),
                encoding="utf-8",
            )
        return queue
