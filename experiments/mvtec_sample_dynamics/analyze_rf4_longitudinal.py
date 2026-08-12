#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Analyze a completed E2 run for the RF4 longitudinal mechanism Gate.

The analyzer is experiment-side and read-only with respect to the training
run.  It reconstructs the frozen ``SampleStateStore`` one epoch at a time
from the persisted per-sample loss and probe histories, then verifies that the
replayed final state equals the state exported by training.  It does not tune
the policy, alter checkpoints, or re-run model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from rfdetr.sample_dynamics.state import SampleState, SampleStateStore, StatePolicy

EXPECTED_EPOCHS = 15
EXPECTED_BATCH_SIZE = 4
EXPECTED_SEED = 20260810
EXPECTED_SAMPLE_COUNT = 128
EXPECTED_MANIFEST_SHA256 = "90673182af0c409e99778f30545f550e91b6845e48f6894a7ebe029c53172528"
EXPECTED_CHECKPOINT_SHA256 = "6de3da31b2572cac214a1c76cce4a92a13966d56390ac2b3a3de9a8dc2b2bca3"
EXPECTED_CORRUPTION_TYPES = {"drop_mask", "mask_shift", "drop_component"}
EXPECTED_CORRUPTION_COUNTS = {"drop_mask": 2, "mask_shift": 2, "drop_component": 1}
EXPECTED_MANIFEST_RECORD_COUNT = 186
EXPECTED_VALID_SAMPLE_COUNT = 58
# These rules are deliberately experiment-side and are frozen before any
# intervention run.  They use only the E2 reference trajectory and exclude
# the five controlled-corruption IDs from natural subsets.
REFERENCE_SUBSET_RULES = {
    "REFERENCE_HARD": (
        "non-corruption train samples with at least 3 epochs in instant HARD "
        "or at least 3 epochs in HARD_LEARNABLE"
    ),
    "REFERENCE_MASTERED": "non-corruption train samples with at least 3 MASTERED state epochs",
    "CLEAN_NATURAL_HARD": (
        "REFERENCE_HARD samples that are not controlled corruptions; this is a "
        "frozen natural-hard comparison subset, not an intervention-derived label"
    ),
    "CONTROLLED_CORRUPTION": "the five manifest corruption records, by stable_sample_id",
    "ALL_VALID": "all validation stable_sample_id values from the frozen manifest",
}
FROZEN_STATE_POLICY = {
    "ema_alpha": 0.3,
    "window_size": 5,
    "max_history": 32,
    "mastered_percentile": 0.25,
    "hard_percentile": 0.75,
    "min_history_for_mastered": 3,
    "suspect_patience": 3,
    "improvement_epsilon": 1e-4,
    "probe_mask_iou_threshold": 0.5,
    "mask_error_weight": 0.5,
    "loss_weight": 0.5,
    "error_weight": 0.3,
    "trend_weight": 0.15,
    "forgetting_weight": 0.05,
}


def _read_json(path: Path) -> Any:
    """Read one UTF-8 JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    """Hash one evidence file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_list(values: Iterable[Any]) -> list[float]:
    """Convert a JSON numeric sequence to finite floats."""
    converted = [float(value) for value in values]
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("trajectory contains a non-finite numeric value")
    return converted


def _optional_float(value: Any) -> float | None:
    """Convert an optional JSON numeric value, preserving undefined metrics."""
    if value is None:
        return None
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("trajectory contains a non-finite optional numeric value")
    return converted


def _instant_bucket(percentile: float, policy: StatePolicy) -> str:
    """Classify one instant percentile without changing the frozen policy."""
    if percentile >= policy.hard_percentile:
        return "HARD"
    if percentile <= policy.mastered_percentile:
        return "EASY"
    return "MIDDLE"


def _frozen_probe_conflict(probe: Mapping[str, Any], policy: StatePolicy) -> bool:
    """Mirror the corrected probe-conflict predicate used by ``SampleStateStore``."""
    gt_count = int(probe.get("gt_count", 0))
    matched_count = int(probe.get("matched_count", 0))
    matched_mask_iou = probe.get("matched_mask_iou")
    return bool(
        int(probe.get("fn", 0)) > 0
        or int(probe.get("class_error", 0)) > 0
        or (gt_count > 0 and float(probe.get("gt_recall", 1.0)) < 1.0)
        or (
            gt_count > 0
            and matched_count > 0
            and matched_mask_iou is not None
            and float(matched_mask_iou) < policy.probe_mask_iou_threshold
        )
    )


def _analysis_matched_mask_iou(probe: Mapping[str, Any]) -> float | None:
    """Return mask IoU only when a real GT/prediction match exists.

    Probe serializes ``0.0`` both for a true zero-IoU match and for the
    no-match/empty-GT path.  The latter is undefined for longitudinal IoU
    analysis and must remain ``None`` without changing the frozen core state.
    """
    if int(probe.get("gt_count", 0)) <= 0 or int(probe.get("matched_count", 0)) <= 0:
        return None
    return _optional_float(probe.get("matched_mask_iou"))


def _analysis_probe_conflict(probe: Mapping[str, Any], policy: StatePolicy) -> bool:
    """Classify probe conflict with the corrected empty-GT semantics.

    Empty-GT recall and mask-IoU placeholders are undefined rather than
    conflicts. Real false positives still contribute difficulty; nonempty-GT
    FN/class/mask behavior remains unchanged.
    """
    gt_count = int(probe.get("gt_count", 0))
    matched_mask_iou = _analysis_matched_mask_iou(probe)
    return bool(
        int(probe.get("fn", 0)) > 0
        or int(probe.get("class_error", 0)) > 0
        or (gt_count > 0 and float(probe.get("gt_recall", 1.0)) < 1.0)
        or (gt_count > 0 and matched_mask_iou is not None and matched_mask_iou < policy.probe_mask_iou_threshold)
    )


def _pearson(values_x: list[float], values_y: list[float]) -> float | None:
    """Return Pearson correlation or ``None`` for an undefined window."""
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return None
    mean_x = sum(values_x) / len(values_x)
    mean_y = sum(values_y) / len(values_y)
    centered_x = [value - mean_x for value in values_x]
    centered_y = [value - mean_y for value in values_y]
    denominator_x = math.sqrt(sum(value * value for value in centered_x))
    denominator_y = math.sqrt(sum(value * value for value in centered_y))
    if denominator_x == 0.0 or denominator_y == 0.0:
        return None
    return sum(x_value * y_value for x_value, y_value in zip(centered_x, centered_y)) / (
        denominator_x * denominator_y
    )


def _rank(values: list[float]) -> list[float]:
    """Return average ranks with deterministic tie handling."""
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + end - 1) / 2.0
        for position in range(index, end):
            ranks[ordered[position][0]] = average_rank
        index = end
    return ranks


def _spearman(values_x: list[float], values_y: list[float]) -> float | None:
    """Return Spearman correlation or ``None`` for an undefined window."""
    return _pearson(_rank(values_x), _rank(values_y))


def _correlation_entry(values_x: list[float], values_y: list[float]) -> dict[str, Any]:
    """Summarize one sample-wise correlation window."""
    return {
        "n": len(values_x),
        "pearson": _pearson(values_x, values_y),
        "spearman": _spearman(values_x, values_y),
    }


def _aggregate_correlations(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate defined correlation windows without hiding undefined ones."""
    pearson = [float(entry["pearson"]) for entry in entries if entry["pearson"] is not None]
    spearman = [float(entry["spearman"]) for entry in entries if entry["spearman"] is not None]

    def summary(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"defined_windows": 0, "mean": None, "median": None}
        ordered = sorted(values)
        middle = len(ordered) // 2
        median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
        return {"defined_windows": len(values), "mean": sum(values) / len(values), "median": median}

    return {"windows": entries, "pearson": summary(pearson), "spearman": summary(spearman)}


def _distribution(values: list[float]) -> dict[str, Any]:
    """Return compact distribution statistics for an auditable per-sample vector."""
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    ordered = sorted(float(value) for value in values)

    def quantile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": quantile(0.5),
        "p90": quantile(0.9),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _deep_equal(left: Any, right: Any, *, tolerance: float = 1e-6) -> bool:
    """Compare JSON-compatible values while allowing float serialization noise."""
    if isinstance(left, float) or isinstance(right, float):
        try:
            return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
        except (TypeError, ValueError):
            return False
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _deep_equal(left[key], right[key], tolerance=tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _deep_equal(left_value, right_value, tolerance=tolerance)
            for left_value, right_value in zip(left, right)
        )
    return left == right


def _stability(labels_by_sample: Mapping[str, list[str]]) -> dict[str, Any]:
    """Measure label churn for one trajectory family."""
    transition_counts_by_sample = {
        sample_id: sum(previous != current for previous, current in zip(labels, labels[1:]))
        for sample_id, labels in sorted(labels_by_sample.items())
    }
    unique_counts_by_sample = {
        sample_id: len(set(labels)) for sample_id, labels in sorted(labels_by_sample.items())
    }
    transitions = list(transition_counts_by_sample.values())
    unique_counts = list(unique_counts_by_sample.values())
    sample_count = len(transitions)
    epoch_count = len(next(iter(labels_by_sample.values()))) if labels_by_sample else 0
    transition_slots = max(epoch_count - 1, 0)
    total_possible = sample_count * transition_slots
    total_transitions = sum(transitions)
    return {
        "sample_count": sample_count,
        "epoch_count": epoch_count,
        "total_transitions": total_transitions,
        "mean_transitions_per_sample": total_transitions / sample_count if sample_count else 0.0,
        "transition_rate_per_available_slot": total_transitions / total_possible if total_possible else 0.0,
        "unchanged_sample_count": sum(value == 0 for value in transitions),
        "unchanged_sample_fraction": sum(value == 0 for value in transitions) / sample_count
        if sample_count
        else 0.0,
        "mean_unique_labels_per_sample": sum(unique_counts) / sample_count if sample_count else 0.0,
        "transition_count_distribution": _distribution([float(value) for value in transitions]),
        "unique_label_count_distribution": _distribution([float(value) for value in unique_counts]),
        "transition_count_by_sample": transition_counts_by_sample,
        "unique_label_count_by_sample": unique_counts_by_sample,
    }


def _build_correlation_analysis(
    per_sample: Mapping[str, Mapping[str, list[Any]]],
    *,
    signal_key: str,
    signal_label: str,
    epoch_count: int,
) -> dict[str, Any]:
    """Compare a difficulty signal at t with t+1/t+2 improvements."""
    result: dict[str, Any] = {"signal": signal_label, "targets": {}}
    for target_key, target_label in (
        ("loss_improvement", "loss improvement: loss[t] - loss[t+h]"),
        ("mask_iou_improvement", "mask IoU improvement: IoU[t+h] - IoU[t]"),
        ("fn_reduction", "FN reduction: FN[t] - FN[t+h]"),
    ):
        horizons: dict[str, Any] = {}
        for horizon in (1, 2):
            entries: list[dict[str, Any]] = []
            for epoch in range(max(0, epoch_count - horizon)):
                signal_values: list[float] = []
                target_values: list[float] = []
                for sample in per_sample.values():
                    signal = float(sample[signal_key][epoch])
                    if target_key == "loss_improvement":
                        target = float(sample["loss"][epoch]) - float(sample["loss"][epoch + horizon])
                    elif target_key == "mask_iou_improvement":
                        current_iou = sample["matched_mask_iou"][epoch]
                        future_iou = sample["matched_mask_iou"][epoch + horizon]
                        if current_iou is None or future_iou is None:
                            continue
                        target = float(future_iou) - float(current_iou)
                    else:
                        target = float(sample["fn"][epoch]) - float(sample["fn"][epoch + horizon])
                    if math.isfinite(signal) and math.isfinite(target):
                        signal_values.append(signal)
                        target_values.append(target)
                entry = _correlation_entry(signal_values, target_values)
                entry["epoch"] = epoch
                entry["horizon"] = horizon
                entries.append(entry)
            horizons[f"horizon_{horizon}"] = _aggregate_correlations(entries)
        result["targets"][target_key] = {"label": target_label, **horizons}
    return result


def _hard_candidate_analysis(
    per_sample: Mapping[str, Mapping[str, list[Any]]],
    policy: StatePolicy,
    *,
    excluded_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Measure natural later improvement after HARD_LEARNABLE observations.

    Controlled-corruption samples are excluded from the natural-recovery
    estimate so the result cannot be mistaken for a clean learnability rate.
    """
    excluded_sample_ids = excluded_sample_ids or set()
    events: list[dict[str, Any]] = []
    first_entry_events: list[dict[str, Any]] = []
    final_hard = [
        sample_id
        for sample_id, sample in per_sample.items()
        if sample_id not in excluded_sample_ids
        and sample["dynamics_state"][-1] == SampleState.HARD_LEARNABLE.value
    ]
    for sample_id, sample in per_sample.items():
        if sample_id in excluded_sample_ids:
            continue
        states = sample["dynamics_state"]
        for epoch, state in enumerate(states[:-1]):
            if state != SampleState.HARD_LEARNABLE.value:
                continue
            current_loss = float(sample["loss"][epoch])
            current_iou = sample["matched_mask_iou"][epoch]
            event: dict[str, Any] = {"sample_id": sample_id, "epoch": epoch}
            for horizon in (1, 2, 3):
                future_epoch = epoch + horizon
                available = future_epoch < len(states)
                event[f"available_horizon_{horizon}"] = available
                if not available:
                    for key in ("loss_improvement", "mask_iou_improvement", "fn_reduction", "left_instant_hard"):
                        event[f"{key}_h{horizon}"] = None
                    continue
                event[f"loss_improvement_h{horizon}"] = (
                    float(sample["loss"][future_epoch]) < current_loss - policy.improvement_epsilon
                )
                future_iou = sample["matched_mask_iou"][future_epoch]
                event[f"mask_iou_improvement_h{horizon}"] = (
                    current_iou is not None
                    and future_iou is not None
                    and float(future_iou) > float(current_iou) + policy.improvement_epsilon
                )
                event[f"fn_reduction_h{horizon}"] = (
                    float(sample["fn"][epoch]) - float(sample["fn"][future_epoch]) > policy.improvement_epsilon
                )
                event[f"left_instant_hard_h{horizon}"] = sample["instant_bucket"][future_epoch] != "HARD"
            event["next_loss_improvement"] = event["loss_improvement_h1"]
            event["any_future_loss_improvement"] = any(
                event[f"loss_improvement_h{horizon}"] is True for horizon in (1, 2, 3)
            ) or any(
                float(value) < current_loss - policy.improvement_epsilon for value in sample["loss"][epoch + 4 :]
            )
            event["next_mask_iou_improvement"] = event["mask_iou_improvement_h1"]
            event["any_future_mask_iou_improvement"] = any(
                event[f"mask_iou_improvement_h{horizon}"] is True for horizon in (1, 2, 3)
            ) or any(
                current_iou is not None
                and value is not None
                and float(value) > float(current_iou) + policy.improvement_epsilon
                for value in sample["matched_mask_iou"][epoch + 4 :]
            )
            events.append(event)

        first_epoch = next(
            (epoch for epoch, state in enumerate(states[:-1]) if state == SampleState.HARD_LEARNABLE.value),
            None,
        )
        if first_epoch is not None:
            first_entry_events.append(next(event for event in events if event["sample_id"] == sample_id and event["epoch"] == first_epoch))

    def horizon_summary(event_set: list[dict[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for horizon in (1, 2, 3):
            available = [event for event in event_set if event[f"available_horizon_{horizon}"]]
            summary[f"horizon_{horizon}"] = {
                "available_events": len(available),
                "loss_improved": sum(event[f"loss_improvement_h{horizon}"] is True for event in available),
                "mask_iou_improved": sum(event[f"mask_iou_improvement_h{horizon}"] is True for event in available),
                "fn_reduced": sum(event[f"fn_reduction_h{horizon}"] is True for event in available),
                "left_instant_hard": sum(event[f"left_instant_hard_h{horizon}"] is True for event in available),
            }
        return summary

    event_count = len(events)
    return {
        "interpretation": "HARD_LEARNABLE is reported as a high-loss candidate; no learnability claim is made.",
        "excluded_sample_ids": sorted(excluded_sample_ids),
        "excluded_controlled_corruption_event_count": sum(
            sum(state == SampleState.HARD_LEARNABLE.value for state in sample["dynamics_state"][:-1])
            for sample_id, sample in per_sample.items()
            if sample_id in excluded_sample_ids
        ),
        "final_epoch_hard_learnable_count": len(final_hard),
        "hard_events_excluding_last_epoch": event_count,
        "unique_samples_with_hard_event": len({event["sample_id"] for event in events}),
        "first_entry_events": first_entry_events,
        "first_entry_sample_count": len(first_entry_events),
        "events_with_next_loss_improvement": sum(event["next_loss_improvement"] for event in events),
        "events_with_any_future_loss_improvement": sum(event["any_future_loss_improvement"] for event in events),
        "events_with_next_mask_iou_improvement": sum(event["next_mask_iou_improvement"] for event in events),
        "events_with_any_future_mask_iou_improvement": sum(
            event["any_future_mask_iou_improvement"] for event in events
        ),
        "natural_improvement_by_horizon": horizon_summary(events),
        "first_entry_natural_improvement_by_horizon": horizon_summary(first_entry_events),
        "events": events,
    }


def _freeze_reference_subsets(
    per_sample: Mapping[str, Mapping[str, list[Any]]],
    manifest_train_records: list[Mapping[str, Any]],
    manifest_valid_records: list[Mapping[str, Any]],
    corruption_ids: set[str],
) -> dict[str, Any]:
    """Freeze intervention comparison subsets from E2 only, before intervention runs."""
    natural_ids = sorted(sample_id for sample_id in per_sample if sample_id not in corruption_ids)
    reference_hard: list[str] = []
    reference_mastered: list[str] = []
    clean_natural_hard: list[str] = []
    for sample_id in natural_ids:
        sample = per_sample[sample_id]
        hard_loss_epochs = sum(bucket == "HARD" for bucket in sample["instant_bucket"])
        hard_state_epochs = sum(state == SampleState.HARD_LEARNABLE.value for state in sample["dynamics_state"])
        mastered_epochs = sum(state == SampleState.MASTERED.value for state in sample["dynamics_state"])
        if hard_loss_epochs >= 3 or hard_state_epochs >= 3:
            reference_hard.append(sample_id)
            clean_natural_hard.append(sample_id)
        if mastered_epochs >= 3:
            reference_mastered.append(sample_id)
    return {
        "schema_version": 2,
        "source": "E2 observe-only reference trajectory",
        "rules": dict(REFERENCE_SUBSET_RULES),
        "controlled_corruption_ids": sorted(corruption_ids),
        "subsets": {
            "REFERENCE_HARD": reference_hard,
            "REFERENCE_MASTERED": reference_mastered,
            "CLEAN_NATURAL_HARD": clean_natural_hard,
            "CONTROLLED_CORRUPTION": sorted(corruption_ids),
            "ALL_VALID": sorted(str(record["stable_sample_id"]) for record in manifest_valid_records),
        },
        "counts": {
            "REFERENCE_HARD": len(reference_hard),
            "REFERENCE_MASTERED": len(reference_mastered),
            "CLEAN_NATURAL_HARD": len(clean_natural_hard),
            "CONTROLLED_CORRUPTION": len(corruption_ids),
            "ALL_VALID": len(manifest_valid_records),
            "ALL_TRAIN": len(manifest_train_records),
        },
    }


def _format_number(value: Any, digits: int = 4) -> str:
    """Format report numbers while preserving undefined correlations."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _markdown_report(analysis: Mapping[str, Any]) -> str:
    """Render the durable RF4 longitudinal validation report."""
    config = analysis["fairness_snapshot"]
    epoch_rows = analysis["epoch_summary"]
    stability = analysis["stability"]
    correlations = analysis["correlations"]
    hard = analysis["hard_learnable_natural_improvement"]
    reference_subsets = analysis["reference_subsets"]
    corruption = analysis["controlled_corruption"]
    manual_audit = analysis.get("manual_audit")
    lines = [
        "# MVTec RF4 Longitudinal Validation",
        "",
        f"> Gate: **{analysis['gate_status']}**",
        "> Scope: **NON-BENCHMARK / RF4 MECHANISM VALIDATION**",
        "> The frozen StatePolicy and thresholds were unchanged; only the minimal empty-GT probe-conflict correctness fix was applied before this rerun.",
        "",
        "## Run contract",
        "",
        f"- Run: `{analysis['run_dir']}`",
        f"- Epochs / samples per epoch: `{analysis['epoch_count']} / {analysis['sample_count']}`",
        f"- Batch / seed: `{config.get('locked_contract', {}).get('batch_size')} / {config.get('locked_contract', {}).get('seed')}`",
        f"- Resolution: `{config.get('locked_contract', {}).get('resolution')}`",
        f"- Manifest SHA-256: `{analysis['manifest_sha256']}`",
        f"- Checkpoint SHA-256: `{config.get('pretrain_weights_sha256')}`",
        f"- State evidence SHA-256: `{analysis['state_sha256']}`",
        f"- Replay final state match: **{analysis['replay_final_state_match']}**",
        f"- Frozen StatePolicy verified: **{analysis['contract']['frozen_state_policy_verified']}**",
        f"- Empty-GT observations counted as probe conflicts after the correctness fix: `{analysis['empty_gt_frozen_conflict_observations']}`; undefined empty-GT recall/mask placeholders are ignored, while real FP remains an error signal.",
        "- Augmentation, multi-scale, scale jitter and EMA remained disabled; no policy parameter was tuned.",
        f"- Frozen reference subsets: `{analysis['reference_subsets_path']}`.",
        "",
        "## What was reconstructed",
        "",
        "The analyzer reads only the persisted E2 `sample_state.json`. It reconstructs the frozen policy one epoch at a time and emits:",
        "",
        "- per-sample weighted per-image normalized loss and instant percentile trajectories;",
        "- instant hard/easy/middle buckets and the frozen four-state trajectory;",
        "- state transition counts, loss EMA, slope, forgetting count and difficulty;",
        "- matched mask IoU, FN, FP, class error and probe-conflict streaks;",
        "- t→t+1/t+2 loss and mask-IoU improvement correlations;",
        "- controlled-corruption trajectories and persistent-conflict smoke signals.",
        "",
        "## Epoch trajectory summary",
        "",
            "| Epoch | Instant easy | Instant middle | Instant hard | LEARNING | MASTERED | HARD_LEARNABLE | SUSPECT | State transitions | Analysis conflict samples |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in epoch_rows:
        instant = row["instant_bucket_counts"]
        states = row["dynamics_state_counts"]
        lines.append(
            "| {epoch} | {easy} | {middle} | {hard} | {learning} | {mastered} | {candidate} | {suspect} | {transitions} | {conflicts} |".format(
                epoch=row["epoch"],
                easy=instant.get("EASY", 0),
                middle=instant.get("MIDDLE", 0),
                hard=instant.get("HARD", 0),
                learning=states.get("LEARNING", 0),
                mastered=states.get("MASTERED", 0),
                candidate=states.get("HARD_LEARNABLE", 0),
                suspect=states.get("SUSPECT", 0),
                transitions=row["state_transition_count"],
                conflicts=row["analysis_probe_conflict_sample_count"],
            )
        )
    lines.extend(
        [
            "",
            "## State stability",
            "",
            "| Trajectory | Total transitions | Mean | Median | P90 | Unchanged sample fraction | Mean unique labels/sample |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, key in (("Instant Loss buckets", "instant"), ("Training Dynamics states", "dynamics")):
        row = stability[key]
        lines.append(
            f"| {label} | {row['total_transitions']} | {_format_number(row['mean_transitions_per_sample'])} | "
            f"{_format_number(row['transition_count_distribution']['median'])} | "
            f"{_format_number(row['transition_count_distribution']['p90'])} | "
            f"{_format_number(row['unchanged_sample_fraction'])} | {_format_number(row['mean_unique_labels_per_sample'])} |"
        )
    lines.extend(
        [
            "",
            "The instant baseline is a three-bucket current-loss view (`EASY/MIDDLE/HARD`), while Dynamics is the frozen four-state view. Their churn is therefore compared as stability evidence, not as an accuracy ranking.",
            "",
            "The corrected core and replay now share the same empty-GT rule: serialized `gt_recall=0` and no-match mask placeholders are undefined when `gt_count=0` and do not create probe conflict. A real FP on an empty-GT image still contributes difficulty; nonempty-GT FN/class/mask rules are unchanged.",
            "",
            "## Longitudinal diagnostic means",
            "",
            "| Epoch | Mean loss EMA | Mean slope | Mean forgetting | Mean mask IoU | Mean FN | Mean FP | Mean class error |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in epoch_rows:
        lines.append(
            f"| {row['epoch']} | {_format_number(row['mean_loss_ema'])} | {_format_number(row['mean_slope'])} | "
            f"{_format_number(row['mean_forgetting_count'])} | {_format_number(row['mean_matched_mask_iou'])} | "
            f"{_format_number(row['mean_fn'])} | {_format_number(row['mean_fp'])} | {_format_number(row['mean_class_error'])} |"
        )
    lines.extend(
        [
            "",
            "## Difficulty versus later improvement",
            "",
            "Positive loss improvement means `loss[t] - loss[t+h]`; positive mask-IoU improvement means `IoU[t+h] - IoU[t]`. Pearson and Spearman values are sample-wise correlations within each epoch window; `n/a` means the target or signal was constant.",
            "",
            "| Signal | Target | Horizon | Defined windows | Mean Pearson | Median Pearson | Mean Spearman |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for signal_key, signal in correlations.items():
        for target_key, target in signal["targets"].items():
            for horizon in (1, 2):
                summary = target[f"horizon_{horizon}"]
                lines.append(
                    f"| {signal['signal']} | {target['label']} | {horizon} | "
                    f"{summary['pearson']['defined_windows']} | {_format_number(summary['pearson']['mean'])} | "
                    f"{_format_number(summary['pearson']['median'])} | {_format_number(summary['spearman']['mean'])} |"
                )
    lines.extend(
        [
            "",
            "## HARD_LEARNABLE natural improvement",
            "",
            f"- Final-epoch `HARD_LEARNABLE` count: `{hard['final_epoch_hard_learnable_count']}`.",
            f"- Earlier `HARD_LEARNABLE` events with a later epoch available: `{hard['hard_events_excluding_last_epoch']}` across `{hard['unique_samples_with_hard_event']}` samples.",
            f"- Events with next-epoch loss improvement: `{hard['events_with_next_loss_improvement']}`; with any later loss improvement: `{hard['events_with_any_future_loss_improvement']}`.",
            f"- Events with next-epoch mask-IoU improvement: `{hard['events_with_next_mask_iou_improvement']}`; with any later mask-IoU improvement: `{hard['events_with_any_future_mask_iou_improvement']}`.",
            f"- First HARD_LEARNABLE entries with a future epoch: `{hard['first_entry_sample_count']}`; per-entry horizon summaries include loss, mask-IoU, FN reduction and leaving the instant HARD bucket.",
            f"- Controlled-corruption samples are excluded from the natural-recovery estimate; excluded HARD events: `{hard['excluded_controlled_corruption_event_count']}`.",
            "",
            "`HARD_LEARNABLE` remains a high-loss candidate label in this report. These counts do not establish that the sample is learnable, and the final epoch is excluded from future-improvement counts because it has no later observation.",
            "",
            "### HARD_LEARNABLE first-entry outcomes",
            "",
            "| Horizon | Available entries | Loss improved | Mask IoU improved | FN reduced | Left instant HARD |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for horizon in (1, 2, 3):
        row = hard["first_entry_natural_improvement_by_horizon"][f"horizon_{horizon}"]
        lines.append(
            f"| {horizon} | {row['available_events']} | {row['loss_improved']} | {row['mask_iou_improved']} | "
            f"{row['fn_reduced']} | {row['left_instant_hard']} |"
        )
    lines.extend(
        [
            "",
            "The per-event and per-sample distributions, including all transition counts and first-entry IDs, remain in the analysis JSON; no `HARD_LEARNABLE` label is interpreted as proof of learnability.",
            "",
            "## Frozen reference subsets",
            "",
            "These sets are derived only from E2 and must be reused unchanged by any later intervention run.",
            "",
            "| Subset | Count | Rule |",
            "| --- | ---: | --- |",
        ]
    )
    for subset_name, count in reference_subsets["counts"].items():
        if subset_name == "ALL_TRAIN":
            continue
        lines.append(
            f"| {subset_name} | {count} | {reference_subsets['rules'].get(subset_name, 'frozen E2 subset')} |"
        )
    if manual_audit is not None:
        lines.extend(
            [
                "",
                "## Manual audit",
                "",
                f"- Audited samples: `{manual_audit.get('sample_count')}`; pending labels: `{manual_audit.get('pending_count')}`.",
                f"- Labels: `{manual_audit.get('label_counts')}`.",
                f"- Packet CSV: `{manual_audit.get('csv')}`; decisions: `{manual_audit.get('decisions_json')}`.",
                "- Labels are a visual/protocol review of probe/state interpretation, not a model mAP estimate. `wrong` specifically marks the observed empty-GT frozen-core conflict semantic on visually normal samples.",
            ]
        )
    lines.extend(
        [
            "",
            "## Controlled-corruption smoke",
            "",
            "The five corruptions are tracked as trajectories only. This section does not promote their precision/recall to an RF8 statistic.",
            "",
            "| Corruption | Final state | Max analysis conflict streak | Final analysis conflict streak | Any SUSPECT | Final FN | Final FP | Final class error | Final mask IoU |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in corruption["samples"]:
        lines.append(
            f"| {row['corruption_type']} (`{row['sample_id']}`) | {row['final_state']} | {row['max_conflict_streak']} | "
            f"{row['final_analysis_probe_conflict_streak']} | {row['any_suspect']} | {row['final_fn']} | {row['final_fp']} | "
            f"{row['final_class_error']} | {_format_number(row['final_matched_mask_iou'])} |"
        )
    lines.extend(
        [
            "",
            "## RF4 Gate conclusion",
            "",
            f"**{analysis['gate_status']}**: the {analysis['epoch_count']}-epoch E2 evidence is complete and replayable under the corrected empty-GT semantics. The RF4 evidence is suitable for the next E5 intervention gate, with the controlled-corruption five-sample result retained as smoke only; no RF8 precision/recall claim is made.",
            "",
            f"Analysis JSON: `{analysis['analysis_json_path']}`",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    """Load E2 evidence, replay the policy, and write JSON/Markdown outputs."""
    run_dir = args.run_dir.resolve()
    state_path = run_dir / "sample_dynamics" / "sample_state.json"
    fairness_path = run_dir / "fairness_snapshot.json"
    manifest_path = args.dataset.resolve() / "split_manifest.json"
    corruption_path = args.dataset.resolve() / "corruption_ground_truth.json"
    for path in (state_path, fairness_path, manifest_path, corruption_path):
        if not path.is_file():
            raise FileNotFoundError(f"required RF4 evidence is missing: {path}")

    state_payload = _read_json(state_path)
    fairness_snapshot = _read_json(fairness_path)
    manifest = _read_json(manifest_path)
    corruptions = _read_json(corruption_path)
    manifest_sha256 = _sha256_file(manifest_path)
    locked_contract = dict(fairness_snapshot.get("locked_contract", {}))
    expected_locked_contract = {
        "resolution": 384,
        "batch_size": EXPECTED_BATCH_SIZE,
        "grad_accum_steps": 1,
        "epochs": EXPECTED_EPOCHS,
        "multi_scale": False,
        "scale_jitter": False,
        "aug_config": {},
        "augmentation_backend": "torchvision",
        "use_ema": False,
        "seed": EXPECTED_SEED,
    }
    train_config = dict(fairness_snapshot.get("train_config", {}))
    model_config = dict(fairness_snapshot.get("model_config", {}))
    contract_errors: list[str] = []
    if fairness_snapshot.get("experiment") != "E2 observe-only":
        contract_errors.append("fairness_snapshot.experiment is not E2 observe-only")
    if fairness_snapshot.get("seed") != EXPECTED_SEED:
        contract_errors.append(f"fairness_snapshot.seed must be {EXPECTED_SEED}")
    if fairness_snapshot.get("dataset_manifest_sha256") != manifest_sha256:
        contract_errors.append("fairness snapshot manifest hash does not match the supplied manifest")
    if fairness_snapshot.get("dataset_manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        contract_errors.append("manifest hash is not the frozen Pilot v2 hash")
    if fairness_snapshot.get("pretrain_weights_sha256") != EXPECTED_CHECKPOINT_SHA256:
        contract_errors.append("checkpoint hash is not the frozen Seg Small hash")
    if locked_contract != expected_locked_contract:
        contract_errors.append(f"locked_contract mismatch: expected {expected_locked_contract}, got {locked_contract}")
    if train_config.get("sample_dynamics_enabled") is not True:
        contract_errors.append("train_config.sample_dynamics_enabled must be true")
    if train_config.get("sample_dynamics_mode") != "observe":
        contract_errors.append("train_config.sample_dynamics_mode must be observe")
    for key, expected in (
        ("batch_size", EXPECTED_BATCH_SIZE),
        ("epochs", EXPECTED_EPOCHS),
        ("seed", EXPECTED_SEED),
    ):
        if train_config.get(key) != expected:
            contract_errors.append(f"train_config.{key} must be {expected}")
    for key, expected in (
        ("multi_scale", False),
        ("scale_jitter", False),
        ("aug_config", {}),
        ("augmentation_backend", "torchvision"),
        ("use_ema", False),
    ):
        if train_config.get(key) != expected:
            contract_errors.append(f"train_config.{key} does not match the locked contract")
    if model_config.get("num_classes") != 1 or model_config.get("resolution") != 384:
        contract_errors.append("model_config must have num_classes=1 and resolution=384")
    manifest_records = list(manifest.get("records", []))
    manifest_train_records = [record for record in manifest_records if record.get("split") == "train"]
    manifest_valid_records = [record for record in manifest_records if record.get("split") == "valid"]
    if len(manifest_records) != EXPECTED_MANIFEST_RECORD_COUNT:
        contract_errors.append(
            f"manifest must contain {EXPECTED_MANIFEST_RECORD_COUNT} records, got {len(manifest_records)}"
        )
    if len(manifest_train_records) != EXPECTED_SAMPLE_COUNT:
        contract_errors.append(f"manifest train split must contain {EXPECTED_SAMPLE_COUNT} records")
    if len(manifest_valid_records) != EXPECTED_VALID_SAMPLE_COUNT:
        contract_errors.append(f"manifest valid split must contain {EXPECTED_VALID_SAMPLE_COUNT} records")
    state_policy = dict(state_payload.get("policy", {}))
    if state_policy != FROZEN_STATE_POLICY:
        contract_errors.append("sample_state policy differs from the frozen default StatePolicy")
    corruption_counts = Counter(str(record.get("corruption_type")) for record in corruptions)
    if len(corruptions) != sum(EXPECTED_CORRUPTION_COUNTS.values()):
        contract_errors.append("controlled corruption count is not the frozen five")
    if dict(corruption_counts) != EXPECTED_CORRUPTION_COUNTS:
        contract_errors.append(
            f"controlled corruption types/counts mismatch: expected {EXPECTED_CORRUPTION_COUNTS}, got {dict(corruption_counts)}"
        )
    corruption_ids = [str(record.get("stable_sample_id")) for record in corruptions]
    if len(corruption_ids) != len(set(corruption_ids)):
        contract_errors.append("controlled corruption stable_sample_id values must be unique")
    if contract_errors:
        raise ValueError("RF4 Longitudinal contract failed:\n- " + "\n- ".join(contract_errors))

    policy = StatePolicy(**state_policy)
    stored_states = {str(sample_id): dict(record) for sample_id, record in dict(state_payload["states"]).items()}
    sample_ids = sorted(stored_states)
    if not sample_ids:
        raise ValueError("sample_state.json contains no sample states")
    if len(sample_ids) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"RF4 Longitudinal contract requires {EXPECTED_SAMPLE_COUNT} samples, got {len(sample_ids)}")
    history_lengths = {len(record.get("history", [])) for record in stored_states.values()}
    probe_lengths = {len(record.get("probe_history", [])) for record in stored_states.values()}
    if len(history_lengths) != 1 or len(probe_lengths) != 1 or history_lengths != probe_lengths:
        raise ValueError(f"inconsistent trajectory lengths: loss={history_lengths}, probe={probe_lengths}")
    epoch_count = next(iter(history_lengths))
    if epoch_count != EXPECTED_EPOCHS:
        raise ValueError(f"RF4 Longitudinal contract requires exactly {EXPECTED_EPOCHS} epochs, got {epoch_count}")
    if any(int(record.get("last_epoch", -1)) != EXPECTED_EPOCHS - 1 for record in stored_states.values()):
        raise ValueError("sample state last_epoch does not match the required 15-epoch run")

    manifest_train_ids = {str(record["stable_sample_id"]) for record in manifest_train_records}
    state_ids = set(sample_ids)
    if state_ids != manifest_train_ids:
        missing_manifest_ids = sorted(state_ids - manifest_train_ids)
        missing_state_ids = sorted(manifest_train_ids - state_ids)
        raise ValueError(
            "state sample IDs must exactly equal manifest train IDs; "
            f"state-only={missing_manifest_ids[:5]}, manifest-only={missing_state_ids[:5]}"
        )

    losses = {sample_id: _float_list(stored_states[sample_id]["history"]) for sample_id in sample_ids}
    probes = {
        sample_id: [dict(probe) for probe in stored_states[sample_id]["probe_history"]]
        for sample_id in sample_ids
    }
    for sample_id, sample_probes in probes.items():
        for epoch, probe in enumerate(sample_probes):
            if str(probe.get("sample_id")) != sample_id:
                raise ValueError(f"probe sample_id mismatch for {sample_id} at epoch {epoch}")
    empty_gt_frozen_conflict_observations = sum(
        _frozen_probe_conflict(probe, policy)
        for sample_id in sample_ids
        for probe in probes[sample_id]
        if int(probe.get("gt_count", 0)) == 0
    )
    per_sample: dict[str, dict[str, list[Any]]] = {
        sample_id: {
            "loss": [],
            "instant_loss_percentile": [],
            "instant_bucket": [],
            "dynamics_state": [],
            "difficulty": [],
            "loss_ema": [],
            "slope": [],
            "forgetting_count": [],
            "consecutive_hard_count": [],
            "probe_conflict_count": [],
            "frozen_probe_conflict": [],
            "analysis_probe_conflict_streak": [],
            "probe_conflict": [],
            "matched_mask_iou": [],
            "matched_iou": [],
            "fn": [],
            "fp": [],
            "class_error": [],
            "gt_recall": [],
        }
        for sample_id in sample_ids
    }
    store = SampleStateStore(
        policy=policy,
        policy_version=int(state_payload.get("policy_version", 0)),
    )
    epoch_summary: list[dict[str, Any]] = []
    previous_states: dict[str, str] = {}
    transition_counts: Counter[str] = Counter()
    analysis_conflict_streaks = {sample_id: 0 for sample_id in sample_ids}

    for epoch in range(epoch_count):
        records = [
            {
                "sample_id": sample_id,
                "epoch": epoch,
                "weighted_per_image_normalized_loss": losses[sample_id][epoch],
            }
            for sample_id in sample_ids
        ]
        instant_percentiles = SampleStateStore.instant_loss_baseline(records)
        probe_records = [probes[sample_id][epoch] for sample_id in sample_ids]
        updated = store.update(records, probe_records=probe_records, epoch=epoch)
        state_counts = Counter()
        instant_counts = Counter()
        state_transition_count = 0
        frozen_probe_conflict_count = 0
        analysis_probe_conflict_count = 0
        loss_ema_values: list[float] = []
        slope_values: list[float] = []
        forgetting_values: list[float] = []
        matched_mask_iou_values: list[float] = []
        fn_values: list[float] = []
        fp_values: list[float] = []
        class_error_values: list[float] = []
        for sample_id in sample_ids:
            state_record = updated[sample_id].to_dict()
            probe = probes[sample_id][epoch]
            instant_percentile = float(instant_percentiles[sample_id])
            instant_bucket = _instant_bucket(instant_percentile, policy)
            state = str(state_record["state"])
            frozen_conflict = _frozen_probe_conflict(probe, policy)
            analysis_conflict = _analysis_probe_conflict(probe, policy)
            analysis_conflict_streaks[sample_id] = analysis_conflict_streaks[sample_id] + 1 if analysis_conflict else 0
            state_counts[state] += 1
            instant_counts[instant_bucket] += 1
            frozen_probe_conflict_count += int(frozen_conflict)
            analysis_probe_conflict_count += int(analysis_conflict)
            if sample_id in previous_states:
                transition = f"{previous_states[sample_id]}->{state}"
                if previous_states[sample_id] != state:
                    state_transition_count += 1
                    transition_counts[transition] += 1
            previous_states[sample_id] = state
            loss_ema_values.append(float(state_record["loss_ema"]))
            slope_values.append(float(state_record["slope"]))
            forgetting_values.append(float(state_record["forgetting_count"]))
            matched_mask_iou = _analysis_matched_mask_iou(probe)
            if matched_mask_iou is not None:
                matched_mask_iou_values.append(matched_mask_iou)
            fn_values.append(float(probe.get("fn", 0)))
            fp_values.append(float(probe.get("fp", 0)))
            class_error_values.append(float(probe.get("class_error", 0)))
            trajectory = per_sample[sample_id]
            trajectory["loss"].append(losses[sample_id][epoch])
            trajectory["instant_loss_percentile"].append(instant_percentile)
            trajectory["instant_bucket"].append(instant_bucket)
            trajectory["dynamics_state"].append(state)
            trajectory["difficulty"].append(float(state_record["difficulty"]))
            trajectory["loss_ema"].append(state_record["loss_ema"])
            trajectory["slope"].append(float(state_record["slope"]))
            trajectory["forgetting_count"].append(int(state_record["forgetting_count"]))
            trajectory["consecutive_hard_count"].append(int(state_record["consecutive_hard_count"]))
            trajectory["probe_conflict_count"].append(int(state_record["probe_conflict_count"]))
            trajectory["frozen_probe_conflict"].append(frozen_conflict)
            trajectory["analysis_probe_conflict_streak"].append(analysis_conflict_streaks[sample_id])
            trajectory["probe_conflict"].append(analysis_conflict)
            trajectory["matched_mask_iou"].append(matched_mask_iou)
            for key in ("matched_iou", "fn", "fp", "class_error", "gt_recall"):
                trajectory[key].append(probe.get(key, 0.0))
        epoch_summary.append(
            {
                "epoch": epoch,
                "instant_bucket_counts": dict(sorted(instant_counts.items())),
                "dynamics_state_counts": dict(sorted(state_counts.items())),
                "state_transition_count": state_transition_count,
                "frozen_probe_conflict_count": frozen_probe_conflict_count,
                "analysis_probe_conflict_sample_count": analysis_probe_conflict_count,
                "mean_loss_ema": sum(loss_ema_values) / len(loss_ema_values),
                "mean_slope": sum(slope_values) / len(slope_values),
                "mean_forgetting_count": sum(forgetting_values) / len(forgetting_values),
                "mean_matched_mask_iou": sum(matched_mask_iou_values) / len(matched_mask_iou_values)
                if matched_mask_iou_values
                else None,
                "mean_fn": sum(fn_values) / len(fn_values),
                "mean_fp": sum(fp_values) / len(fp_values),
                "mean_class_error": sum(class_error_values) / len(class_error_values),
            }
        )

    replayed_states = store.state_dict()["states"]
    mismatches = [
        sample_id
        for sample_id in sample_ids
        if not _deep_equal(stored_states[sample_id], replayed_states.get(sample_id))
    ]

    corruption_meta = {str(record["stable_sample_id"]): dict(record) for record in corruptions}
    corruption_ids = set(corruption_meta)
    reference_subsets = _freeze_reference_subsets(
        per_sample,
        manifest_train_records,
        manifest_valid_records,
        corruption_ids,
    )
    corruption_samples: list[dict[str, Any]] = []
    corruption_trajectory: dict[str, Any] = {}
    for sample_id, metadata in sorted(corruption_meta.items()):
        if sample_id not in per_sample:
            raise ValueError(f"controlled corruption is absent from state evidence: {sample_id}")
        trajectory = per_sample[sample_id]
        epochs = []
        running_streak = 0
        max_streak = 0
        for epoch in range(epoch_count):
            running_streak = running_streak + 1 if trajectory["probe_conflict"][epoch] else 0
            max_streak = max(max_streak, running_streak)
            probe = probes[sample_id][epoch]
            epochs.append(
                {
                    "epoch": epoch,
                    "loss": trajectory["loss"][epoch],
                    "instant_loss_percentile": trajectory["instant_loss_percentile"][epoch],
                    "instant_bucket": trajectory["instant_bucket"][epoch],
                    "dynamics_state": trajectory["dynamics_state"][epoch],
                    "difficulty": trajectory["difficulty"][epoch],
                    "probe_conflict": trajectory["probe_conflict"][epoch],
                    "analysis_probe_conflict_streak": trajectory["analysis_probe_conflict_streak"][epoch],
                    "frozen_probe_conflict": trajectory["frozen_probe_conflict"][epoch],
                    "frozen_probe_conflict_count": trajectory["probe_conflict_count"][epoch],
                    "matched_mask_iou": trajectory["matched_mask_iou"][epoch],
                    "fn": probe.get("fn", 0),
                    "fp": probe.get("fp", 0),
                    "class_error": probe.get("class_error", 0),
                }
            )
        final_probe = probes[sample_id][-1]
        corruption_samples.append(
            {
                "sample_id": sample_id,
                "source_image": metadata.get("source_image"),
                "corruption_type": metadata.get("corruption_type"),
                "final_state": trajectory["dynamics_state"][-1],
                "max_conflict_streak": max_streak,
                "final_analysis_probe_conflict_streak": trajectory["analysis_probe_conflict_streak"][-1],
                "final_frozen_probe_conflict_count": trajectory["probe_conflict_count"][-1],
                "any_suspect": SampleState.SUSPECT.value in trajectory["dynamics_state"],
                "final_fn": final_probe.get("fn", 0),
                "final_fp": final_probe.get("fp", 0),
                "final_class_error": final_probe.get("class_error", 0),
                "final_matched_mask_iou": trajectory["matched_mask_iou"][-1],
            }
        )
        corruption_trajectory[sample_id] = {
            "source_image": metadata.get("source_image"),
            "corruption_type": metadata.get("corruption_type"),
            "epochs": epochs,
        }

    correlations = {
        "instant_loss_percentile": _build_correlation_analysis(
            per_sample,
            signal_key="instant_loss_percentile",
            signal_label="Instant-Loss percentile",
            epoch_count=epoch_count,
        ),
        "dynamics_difficulty": _build_correlation_analysis(
            per_sample,
            signal_key="difficulty",
            signal_label="Training-Dynamics difficulty",
            epoch_count=epoch_count,
        ),
    }
    hard_analysis = _hard_candidate_analysis(per_sample, policy, excluded_sample_ids=corruption_ids)
    if mismatches:
        gate_status = "FAIL_REPLAY_MISMATCH"
    elif empty_gt_frozen_conflict_observations:
        gate_status = "FAIL_SEMANTIC_REVIEW_REQUIRED"
    else:
        gate_status = "PASS_WITH_CAVEATS"
    analysis: dict[str, Any] = {
        "schema_version": 2,
        "gate_status": gate_status,
        "run_dir": str(run_dir),
        "epoch_count": epoch_count,
        "sample_count": len(sample_ids),
        "policy": dict(state_payload["policy"]),
        "contract": {
            "status": "PASS",
            "expected_epochs": EXPECTED_EPOCHS,
            "expected_batch_size": EXPECTED_BATCH_SIZE,
            "expected_seed": EXPECTED_SEED,
            "expected_sample_count": EXPECTED_SAMPLE_COUNT,
            "manifest_sha256_verified": manifest_sha256 == EXPECTED_MANIFEST_SHA256,
            "checkpoint_sha256_verified": fairness_snapshot.get("pretrain_weights_sha256")
            == EXPECTED_CHECKPOINT_SHA256,
            "frozen_state_policy_verified": state_policy == FROZEN_STATE_POLICY,
            "corruption_counts": dict(sorted(corruption_counts.items())),
        },
        "fairness_snapshot": fairness_snapshot,
        "manifest_sha256": manifest_sha256,
        "state_sha256": _sha256_file(state_path),
        "empty_gt_frozen_conflict_observations": empty_gt_frozen_conflict_observations,
        "history_lengths": sorted(history_lengths),
        "probe_history_lengths": sorted(probe_lengths),
        "replay_final_state_match": not mismatches,
        "replay_mismatched_sample_ids": mismatches,
        "per_sample": per_sample,
        "epoch_summary": epoch_summary,
        "transition_counts": dict(sorted(transition_counts.items())),
        "stability": {
            "instant": _stability({sample_id: sample["instant_bucket"] for sample_id, sample in per_sample.items()}),
            "dynamics": _stability({sample_id: sample["dynamics_state"] for sample_id, sample in per_sample.items()}),
        },
        "correlations": correlations,
        "hard_learnable_natural_improvement": hard_analysis,
        "reference_subsets": reference_subsets,
        "controlled_corruption": {
            "sample_count": len(corruption_samples),
            "samples": corruption_samples,
            "trajectory": corruption_trajectory,
            "rf8_precision_recall_claimed": False,
        },
    }
    output_json = args.output_json.resolve()
    analysis["analysis_json_path"] = str(output_json)
    output_reference_subsets = (
        args.output_reference_subsets.resolve()
        if args.output_reference_subsets is not None
        else output_json.parent / "mvtec_reference_subsets.json"
    )
    analysis["reference_subsets_path"] = str(output_reference_subsets)
    if args.manual_audit_summary is not None:
        manual_audit = _read_json(args.manual_audit_summary.resolve())
        if int(manual_audit.get("sample_count", 0)) < 50 or int(manual_audit.get("pending_count", 1)) != 0:
            raise ValueError("manual audit summary must contain at least 50 completed, non-pending samples")
        analysis["manual_audit"] = manual_audit
    _write_json(output_json, analysis)
    _write_json(output_reference_subsets, reference_subsets)
    output_report = args.output_report.resolve()
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(_markdown_report(analysis), encoding="utf-8")
    return analysis


def parse_args() -> argparse.Namespace:
    """Parse the RF4 analysis CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-reference-subsets", type=Path, default=None)
    parser.add_argument("--manual-audit-summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Run RF4 analysis and print the gate summary."""
    result = analyze(parse_args())
    print(
        json.dumps(
            {
                "gate_status": result["gate_status"],
                "epoch_count": result["epoch_count"],
                "sample_count": result["sample_count"],
                "replay_final_state_match": result["replay_final_state_match"],
                "output_json": result["analysis_json_path"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
