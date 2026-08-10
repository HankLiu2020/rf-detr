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
    """Mirror the frozen probe-conflict predicate used by ``SampleStateStore``."""
    matched_mask_iou = probe.get("matched_mask_iou")
    return bool(
        int(probe.get("fn", 0)) > 0
        or int(probe.get("class_error", 0)) > 0
        or float(probe.get("gt_recall", 1.0)) < 1.0
        or (
            matched_mask_iou is not None
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
    """Classify probe conflict for analysis without treating empty-GT recall as a miss.

    The training state is still replayed with the exact frozen core predicate.
    This separate diagnostic avoids counting a normal image with ``gt_count=0``
    and ``gt_recall=0`` as a false positive conflict in the longitudinal report.
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
    transitions = []
    unique_counts = []
    for labels in labels_by_sample.values():
        transitions.append(sum(previous != current for previous, current in zip(labels, labels[1:])))
        unique_counts.append(len(set(labels)))
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
                    else:
                        current_iou = sample["matched_mask_iou"][epoch]
                        future_iou = sample["matched_mask_iou"][epoch + horizon]
                        if current_iou is None or future_iou is None:
                            continue
                        target = float(future_iou) - float(current_iou)
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


def _hard_candidate_analysis(per_sample: Mapping[str, Mapping[str, list[Any]]], policy: StatePolicy) -> dict[str, Any]:
    """Measure natural later improvement after a HARD_LEARNABLE observation."""
    events: list[dict[str, Any]] = []
    final_hard = [sample_id for sample_id, sample in per_sample.items() if sample["dynamics_state"][-1] == SampleState.HARD_LEARNABLE.value]
    for sample_id, sample in per_sample.items():
        states = sample["dynamics_state"]
        for epoch, state in enumerate(states[:-1]):
            if state != SampleState.HARD_LEARNABLE.value:
                continue
            future_losses = sample["loss"][epoch + 1 :]
            future_ious = [value for value in sample["matched_mask_iou"][epoch + 1 :] if value is not None]
            current_loss = float(sample["loss"][epoch])
            current_iou = sample["matched_mask_iou"][epoch]
            events.append(
                {
                    "sample_id": sample_id,
                    "epoch": epoch,
                    "next_loss_improvement": float(sample["loss"][epoch + 1]) < current_loss - policy.improvement_epsilon,
                    "any_future_loss_improvement": any(
                        value < current_loss - policy.improvement_epsilon for value in future_losses
                    ),
                    "next_mask_iou_improvement": current_iou is not None
                    and sample["matched_mask_iou"][epoch + 1] is not None
                    and float(sample["matched_mask_iou"][epoch + 1]) > float(current_iou) + policy.improvement_epsilon,
                    "any_future_mask_iou_improvement": any(
                        current_iou is not None
                        and float(value) > float(current_iou) + policy.improvement_epsilon
                        for value in future_ious
                    ),
                }
            )
    event_count = len(events)
    return {
        "interpretation": "HARD_LEARNABLE is reported as a high-loss candidate; no learnability claim is made.",
        "final_epoch_hard_learnable_count": len(final_hard),
        "hard_events_excluding_last_epoch": event_count,
        "unique_samples_with_hard_event": len({event["sample_id"] for event in events}),
        "events_with_next_loss_improvement": sum(event["next_loss_improvement"] for event in events),
        "events_with_any_future_loss_improvement": sum(event["any_future_loss_improvement"] for event in events),
        "events_with_next_mask_iou_improvement": sum(event["next_mask_iou_improvement"] for event in events),
        "events_with_any_future_mask_iou_improvement": sum(
            event["any_future_mask_iou_improvement"] for event in events
        ),
        "events": events,
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
    corruption = analysis["controlled_corruption"]
    lines = [
        "# MVTec RF4 Longitudinal Validation",
        "",
        f"> Gate: **{analysis['gate_status']}**",
        "> Scope: **NON-BENCHMARK / RF4 MECHANISM VALIDATION**",
        "> Sample Dynamics core and frozen StatePolicy were not changed for this run.",
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
        f"- Empty-GT observations counted as frozen-core conflicts: `{analysis['empty_gt_frozen_conflict_observations']}`; analysis conflict trajectory excludes this undefined recall case.",
        "- Augmentation, multi-scale, scale jitter and EMA remained disabled; no policy parameter was tuned.",
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
            "| Trajectory | Total transitions | Mean transitions/sample | Unchanged sample fraction | Mean unique labels/sample |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, key in (("Instant Loss buckets", "instant"), ("Training Dynamics states", "dynamics")):
        row = stability[key]
        lines.append(
            f"| {label} | {row['total_transitions']} | {_format_number(row['mean_transitions_per_sample'])} | "
            f"{_format_number(row['unchanged_sample_fraction'])} | {_format_number(row['mean_unique_labels_per_sample'])} |"
        )
    lines.extend(
        [
            "",
            "The instant baseline is a three-bucket current-loss view (`EASY/MIDDLE/HARD`), while Dynamics is the frozen four-state view. Their churn is therefore compared as stability evidence, not as an accuracy ranking.",
            "",
            "The frozen core currently reports `gt_recall=0` for empty-GT normal images; the analysis-side conflict trajectory therefore excludes empty-GT recall from conflict, while the replayed Dynamics state still uses the exact frozen predicate. This semantic discrepancy is reported rather than hidden.",
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
            "",
            "`HARD_LEARNABLE` remains a high-loss candidate label in this report. These counts do not establish that the sample is learnable, and the final epoch is excluded from future-improvement counts because it has no later observation.",
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
            f"**{analysis['gate_status']}**: the {analysis['epoch_count']}-epoch E2 evidence is complete, replayable, and covers the requested longitudinal signals without changing the frozen core strategy. This is a mechanism-observation Gate only; it does not validate E3/E4/E5 benefit, RF8 precision, or a stage-aware policy.",
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
    gate_status = "PASS_WITH_CAVEATS" if not mismatches else "FAIL_REPLAY_MISMATCH"
    analysis: dict[str, Any] = {
        "schema_version": 1,
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
        "hard_learnable_natural_improvement": _hard_candidate_analysis(per_sample, policy),
        "controlled_corruption": {
            "sample_count": len(corruption_samples),
            "samples": corruption_samples,
            "trajectory": corruption_trajectory,
            "rf8_precision_recall_claimed": False,
        },
    }
    output_json = args.output_json.resolve()
    analysis["analysis_json_path"] = str(output_json)
    _write_json(output_json, analysis)
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
