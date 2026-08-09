# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------
"""Pure-Python Pareto helpers for future latency/accuracy selection."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from rfdetr.nas.evaluation import SubnetEvaluation


def _record(value: Mapping[str, Any] | SubnetEvaluation) -> dict[str, Any]:
    if isinstance(value, SubnetEvaluation):
        return value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("Pareto records must be mappings or SubnetEvaluation values")


def _metric(record: Mapping[str, Any], key: str, aliases: tuple[str, ...]) -> float:
    value: Any = record.get(key)
    if value is None:
        for alias in aliases:
            value = record.get(alias)
            if value is not None:
                break
    if not isinstance(value, (int, float)):
        raise ValueError(f"Pareto record requires numeric {key!r}; got {value!r}")
    return float(value)


def compute_pareto_front(
    records: Iterable[Mapping[str, Any] | SubnetEvaluation],
    *,
    accuracy_key: str = "accuracy",
    latency_key: str = "latency",
) -> list[dict[str, Any]]:
    """Return non-dominated records, maximizing accuracy and minimizing latency.

    Placeholder latency/AP values are rejected rather than silently creating a
    misleading Pareto front.  The aliases support evaluator output fields
    ``segm_ap`` and ``latency_ms`` without coupling the API to one dataset.
    """

    values = [_record(value) for value in records]
    if not values:
        return []
    scored = [
        (
            record,
            _metric(record, accuracy_key, ("segm_ap", "bbox_ap", "ap")),
            _metric(record, latency_key, ("latency_ms",)),
        )
        for record in values
    ]
    front: list[dict[str, Any]] = []
    for index, (candidate, candidate_accuracy, candidate_latency) in enumerate(scored):
        dominated = False
        for other_index, (_, other_accuracy, other_latency) in enumerate(scored):
            if index == other_index:
                continue
            no_worse = other_accuracy >= candidate_accuracy and other_latency <= candidate_latency
            strictly_better = other_accuracy > candidate_accuracy or other_latency < candidate_latency
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def select_fast_balanced_accurate(
    pareto_front: Iterable[Mapping[str, Any] | SubnetEvaluation],
    *,
    accuracy_key: str = "accuracy",
    latency_key: str = "latency",
) -> dict[str, dict[str, Any]]:
    """Label the front's Fast, Balanced, and Accurate representative points."""

    values = [_record(value) for value in pareto_front]
    if not values:
        raise ValueError("cannot select representatives from an empty Pareto front")
    scored = [
        (
            record,
            _metric(record, accuracy_key, ("segm_ap", "bbox_ap", "ap")),
            _metric(record, latency_key, ("latency_ms",)),
        )
        for record in values
    ]
    fast = min(scored, key=lambda item: (item[2], -item[1]))[0]
    accurate = max(scored, key=lambda item: (item[1], -item[2]))[0]
    min_accuracy = min(item[1] for item in scored)
    max_accuracy = max(item[1] for item in scored)
    min_latency = min(item[2] for item in scored)
    max_latency = max(item[2] for item in scored)

    def normalize(value: float, low: float, high: float) -> float:
        return 1.0 if high == low else (value - low) / (high - low)

    balanced = max(
        scored,
        key=lambda item: (
            0.5 * normalize(item[1], min_accuracy, max_accuracy)
            + 0.5 * (1.0 - normalize(item[2], min_latency, max_latency)),
            item[1],
            -item[2],
        ),
    )[0]
    return {"Fast": fast, "Balanced": balanced, "Accurate": accurate}
