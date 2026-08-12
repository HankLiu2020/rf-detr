# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

#!/usr/bin/env python3
"""Create the experiment-side E0 versus E5 combined validation package.

This script is deliberately read-only with respect to training artifacts.  It
consumes the saved checkpoint trajectory evaluator output, the frozen RF4
reference subsets, the E5 resource ledger, and the two pilot metrics files.
It does not replay training and it does not change StatePolicy parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.mean(values) if values else None


def median(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(values) if values else None


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def pct(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{100.0 * value:.{digits}f}%"


def num(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return None
    x_mean = statistics.mean(x for x, _ in pairs)
    y_mean = statistics.mean(y for _, y in pairs)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    denominator = math.sqrt(sum((x - x_mean) ** 2 for x, _ in pairs) * sum((y - y_mean) ** 2 for _, y in pairs))
    return numerator / denominator if denominator else None


def merged_metrics(path: Path) -> dict[int, dict[str, float]]:
    """Merge the train and validation metric rows emitted for each epoch."""
    merged: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            epoch = int(float(row["epoch"]))
            target = merged.setdefault(epoch, {})
            for key, raw in row.items():
                if key == "epoch" or raw in (None, ""):
                    continue
                try:
                    target[key] = float(raw)
                except ValueError:
                    continue
    return merged


def trajectory_by_epoch(run: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(entry["epoch"]): entry for entry in run["trajectory"]}


def actual_checkpoint_epochs(run: dict[str, Any]) -> list[int]:
    """Return epochs backed by their own checkpoint, excluding fallbacks."""
    return sorted(
        int(entry["epoch"])
        for entry in run["trajectory"]
        if entry.get("checkpoint_kind") == "epoch"
    )


def probe_metric(entry: dict[str, Any], sample_id: str, key: str) -> float | None:
    value = entry.get("probe", {}).get(sample_id, {}).get(key)
    return None if value is None else float(value)


def subset_epoch_stats(run: dict[str, Any], subset_ids: set[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for epoch, entry in sorted(trajectory_by_epoch(run).items()):
        ids = sorted(set(entry.get("loss", {})).intersection(subset_ids))
        losses = [float(entry["loss"][sample_id]) for sample_id in ids]
        result.append(
            {
                "epoch": epoch,
                "sample_count": len(ids),
                "mean_loss": mean(losses),
                "median_loss": median(losses),
                "mean_mask_iou": mean(probe_metric(entry, sample_id, "matched_mask_iou") for sample_id in ids),
                "mean_matched_iou": mean(probe_metric(entry, sample_id, "matched_iou") for sample_id in ids),
                "mean_fn": mean(probe_metric(entry, sample_id, "fn") for sample_id in ids),
                "mean_fp": mean(probe_metric(entry, sample_id, "fp") for sample_id in ids),
                "mean_class_error": mean(probe_metric(entry, sample_id, "class_error") for sample_id in ids),
                "mean_gt_recall": mean(probe_metric(entry, sample_id, "gt_recall") for sample_id in ids),
            }
        )
    return result


def per_sample_comparison(
    e0: dict[str, Any], e5: dict[str, Any], sample_ids: set[str]
) -> list[dict[str, Any]]:
    e0_by_epoch = trajectory_by_epoch(e0)
    e5_by_epoch = trajectory_by_epoch(e5)
    first_epoch = min(e0_by_epoch)
    common_actual_epochs = sorted(set(actual_checkpoint_epochs(e0)).intersection(actual_checkpoint_epochs(e5)))
    if not common_actual_epochs:
        raise ValueError("E0/E5 have no common actual checkpoint epoch")
    last_epoch = common_actual_epochs[-1]
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(sample_ids):
        if sample_id not in e0_by_epoch[first_epoch].get("loss", {}) or sample_id not in e5_by_epoch[last_epoch].get("loss", {}):
            continue
        e0_initial_loss = float(e0_by_epoch[first_epoch]["loss"][sample_id])
        e0_final_loss = float(e0_by_epoch[last_epoch]["loss"][sample_id])
        e5_initial_loss = float(e5_by_epoch[first_epoch]["loss"][sample_id])
        e5_final_loss = float(e5_by_epoch[last_epoch]["loss"][sample_id])
        e0_initial_iou = probe_metric(e0_by_epoch[first_epoch], sample_id, "matched_mask_iou")
        e0_final_iou = probe_metric(e0_by_epoch[last_epoch], sample_id, "matched_mask_iou")
        e5_initial_iou = probe_metric(e5_by_epoch[first_epoch], sample_id, "matched_mask_iou")
        e5_final_iou = probe_metric(e5_by_epoch[last_epoch], sample_id, "matched_mask_iou")
        rows.append(
            {
                "sample_id": sample_id,
                "e0_initial_loss": e0_initial_loss,
                "e0_final_loss": e0_final_loss,
                "e0_loss_improvement": e0_initial_loss - e0_final_loss,
                "e5_initial_loss": e5_initial_loss,
                "e5_final_loss": e5_final_loss,
                "e5_loss_improvement": e5_initial_loss - e5_final_loss,
                "e5_minus_e0_final_loss": e5_final_loss - e0_final_loss,
                "e0_initial_mask_iou": e0_initial_iou,
                "e0_final_mask_iou": e0_final_iou,
                "e0_mask_iou_improvement": None if e0_initial_iou is None or e0_final_iou is None else e0_final_iou - e0_initial_iou,
                "e5_initial_mask_iou": e5_initial_iou,
                "e5_final_mask_iou": e5_final_iou,
                "e5_mask_iou_improvement": None if e5_initial_iou is None or e5_final_iou is None else e5_final_iou - e5_initial_iou,
                "e5_minus_e0_final_mask_iou": None if e5_final_iou is None or e0_final_iou is None else e5_final_iou - e0_final_iou,
                "e0_final_fn": probe_metric(e0_by_epoch[last_epoch], sample_id, "fn"),
                "e5_final_fn": probe_metric(e5_by_epoch[last_epoch], sample_id, "fn"),
                "e0_final_fp": probe_metric(e0_by_epoch[last_epoch], sample_id, "fp"),
                "e5_final_fp": probe_metric(e5_by_epoch[last_epoch], sample_id, "fp"),
                "e0_final_class_error": probe_metric(e0_by_epoch[last_epoch], sample_id, "class_error"),
                "e5_final_class_error": probe_metric(e5_by_epoch[last_epoch], sample_id, "class_error"),
            }
        )
    return rows


def resource_summary(resource_history: list[dict[str, Any]], subsets: dict[str, set[str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    epoch_rows: list[dict[str, Any]] = []
    cumulative: dict[str, dict[str, float]] = {}
    for subset_name, ids in {**subsets, "ALL_TRAIN": set()}.items():
        cumulative[subset_name] = {
            "samples": float(len(ids)),
            "appearances": 0.0,
            "exposure": 0.0,
            "policy_weight_sum": 0.0,
            "policy_weight_count": 0.0,
            "weight_sum": 0.0,
            "weight_count": 0.0,
            "effective_sum": 0.0,
            "cap_hits": 0.0,
        }
    for payload in resource_history:
        records = payload["records"]
        all_ids = {record["sample_id"] for record in records}
        if not cumulative["ALL_TRAIN"]["samples"]:
            cumulative["ALL_TRAIN"]["samples"] = float(len(all_ids))
        for subset_name, ids in {**subsets, "ALL_TRAIN": all_ids}.items():
            selected = [record for record in records if record["sample_id"] in ids]
            seen = [record for record in selected if record.get("batch_appearance_count", 0) > 0]
            appearance_counts = [int(record.get("batch_appearance_count", 0)) for record in selected]
            policy_weights = [float(record.get("state_policy_weight", 1.0)) for record in selected]
            weights = [record["applied_loss_weight_mean"] for record in seen if record.get("applied_loss_weight_mean") is not None]
            effective = [record["effective_contribution_mean"] for record in seen if record.get("effective_contribution_mean") is not None]
            total_appearances = sum(appearance_counts)
            total_exposure_count = sum(int(record.get("exposure_count", 0)) for record in selected)
            weighted_loss_sum = sum(
                float(record["applied_loss_weight_mean"]) * int(record.get("batch_appearance_count", 0))
                for record in seen
                if record.get("applied_loss_weight_mean") is not None
            )
            total_effective_contribution = sum(
                float(record["effective_contribution_mean"]) * int(record.get("batch_appearance_count", 0))
                for record in seen
                if record.get("effective_contribution_mean") is not None
            )
            epoch_rows.append(
                {
                    "epoch": int(payload["epoch"]),
                    "subset": subset_name,
                    "sample_count": len(selected),
                    "seen_sample_count": len(seen),
                    "mean_exposure_multiplier": mean(record.get("exposure_multiplier") for record in selected),
                    "mean_state_policy_weight": mean(policy_weights),
                    "mean_applied_loss_weight_seen": mean(weights),
                    "mean_effective_contribution_seen": mean(effective),
                    "total_appearances": total_appearances,
                    "total_exposure_count": total_exposure_count,
                    "total_effective_contribution": total_effective_contribution,
                    "cap_hit_count": sum(int(record.get("cap_hit_count", 0)) for record in selected),
                }
            )
            current = cumulative[subset_name]
            current["appearances"] += total_appearances
            current["exposure"] += total_exposure_count
            current["policy_weight_sum"] += sum(policy_weights)
            current["policy_weight_count"] += len(policy_weights)
            current["weight_sum"] += weighted_loss_sum
            current["weight_count"] += total_appearances
            current["effective_sum"] += total_effective_contribution
            current["cap_hits"] += sum(int(record.get("cap_hit_count", 0)) for record in selected)
    final: dict[str, Any] = {}
    for subset_name, current in cumulative.items():
        samples = current["samples"]
        final[subset_name] = {
            "sample_count": int(samples),
            "epochs": len(resource_history),
            "appearances_per_sample_per_epoch": current["appearances"] / samples / len(resource_history) if samples else None,
            "exposure_count_per_sample_per_epoch": current["exposure"] / samples / len(resource_history) if samples else None,
            "mean_state_policy_weight": current["policy_weight_sum"] / current["policy_weight_count"] if current["policy_weight_count"] else None,
            "mean_applied_loss_weight_seen": current["weight_sum"] / current["weight_count"] if current["weight_count"] else None,
            "effective_contribution_per_sample_per_epoch": current["effective_sum"] / samples / len(resource_history) if samples else None,
            "cap_hit_count": int(current["cap_hits"]),
        }
    return epoch_rows, final


def final_states(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    counts: dict[str, int] = {}
    for state in payload.get("states", {}).values():
        counts[state["state"]] = counts.get(state["state"], 0) + 1
    return {"policy_version": payload.get("policy_version"), "state_counts": dict(sorted(counts.items()))}


def contract_check(e0: dict[str, Any], e5: dict[str, Any]) -> dict[str, Any]:
    a = e0["fairness_snapshot"]
    b = e5["fairness_snapshot"]
    fields = ["dataset_manifest_sha256", "initial_model_parameter_sha256", "pretrain_weights_sha256", "seed"]
    contract_fields = ["resolution", "batch_size", "grad_accum_steps", "epochs", "seed", "multi_scale", "scale_jitter", "aug_config", "augmentation_backend", "use_ema"]
    equal_fields = {field: a.get(field) == b.get(field) for field in fields}
    equal_contract = {field: a["locked_contract"].get(field) == b["locked_contract"].get(field) for field in contract_fields}
    return {"identity_fields_equal": equal_fields, "locked_contract_equal": equal_contract, "all_equal": all(equal_fields.values()) and all(equal_contract.values())}


def make_plots(output_dir: Path, summary: dict[str, Any]) -> list[str]:
    import matplotlib.pyplot as plt

    assets = output_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    overall = summary["overall_metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for label, color in [("E0", "#4c78a8"), ("E5", "#f58518")]:
        rows = overall[label]
        epochs = [row["epoch"] for row in rows]
        axes[0].plot(epochs, [row.get("val_mAP_50_95") for row in rows], marker="o", label=f"{label} box mAP", color=color, alpha=0.9)
        axes[0].plot(epochs, [row.get("val_segm_mAP_50_95") for row in rows], linestyle="--", marker="x", label=f"{label} mask mAP", color=color, alpha=0.7)
        axes[1].plot(epochs, [row.get("val_F1") for row in rows], marker="o", label=label, color=color)
    axes[0].set_title("Validation trajectory")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("metric")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_title("Validation F1")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("F1")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.savefig(assets / "e5_overall_metrics.png", dpi=160)
    fig.savefig(assets / "e5_overall_metrics.svg")
    plt.close(fig)

    resource = summary["resource_cumulative"]
    names = ["REFERENCE_HARD", "REFERENCE_MASTERED", "ALL_TRAIN"]
    x = list(range(len(names)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for offset, (metric, label, color) in enumerate(
        [
            ("appearances_per_sample_per_epoch", "appearances", "#4c78a8"),
            ("mean_applied_loss_weight_seen", "loss weight", "#f58518"),
            ("effective_contribution_per_sample_per_epoch", "effective contribution", "#54a24b"),
        ]
    ):
        values = [resource[name].get(metric) or 0.0 for name in names]
        ax.bar([value + (offset - 1) * width for value in x], values, width, label=label, color=color)
    ax.set_xticks(x, ["hard", "mastered", "all train"])
    ax.set_title("E5 actual resource allocation")
    ax.set_ylabel("per-sample-per-epoch aggregate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(assets / "e5_hard_resource_recovery.png", dpi=160)
    fig.savefig(assets / "e5_hard_resource_recovery.svg")
    plt.close(fig)
    return [str(path.relative_to(output_dir.parent)) for path in sorted(assets.glob("e5_*.svg"))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=Path("reports/mvtec_e0_e5_checkpoint_trajectories.json"))
    parser.add_argument("--rf4-analysis", type=Path, default=Path("reports/mvtec_rf4_longitudinal_analysis.json"))
    parser.add_argument("--subsets", type=Path, default=Path("reports/mvtec_reference_subsets.json"))
    parser.add_argument("--e0-dir", type=Path, default=Path("/home/liujiyuan/mvtec-sample-dynamics-runs/e0-15"))
    parser.add_argument("--e5-dir", type=Path, default=Path("/home/liujiyuan/mvtec-sample-dynamics-runs/e5-15"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    trajectory = read_json(args.trajectory)
    e0 = trajectory["runs"]["E0"]
    e5 = trajectory["runs"]["E5"]
    subset_payload = read_json(args.subsets)
    subsets = {name: set(values) for name, values in subset_payload["subsets"].items()}
    rf4 = read_json(args.rf4_analysis)
    e0_metrics = merged_metrics(args.e0_dir / "metrics.csv")
    e5_metrics = merged_metrics(args.e5_dir / "metrics.csv")

    overall_metrics: dict[str, list[dict[str, Any]]] = {}
    for label, metric_map in [("E0", e0_metrics), ("E5", e5_metrics)]:
        rows = []
        for epoch, values in sorted(metric_map.items()):
            rows.append(
                {
                    "epoch": epoch,
                    "val_mAP_50_95": values.get("val/mAP_50_95"),
                    "val_segm_mAP_50_95": values.get("val/segm_mAP_50_95"),
                    "val_F1": values.get("val/F1"),
                    "val_precision": values.get("val/precision"),
                    "val_recall": values.get("val/recall"),
                    "val_loss": values.get("val/loss"),
                    "train_loss": values.get("train/loss"),
                }
            )
        overall_metrics[label] = rows

    hard_ids = subsets["REFERENCE_HARD"]
    mastered_ids = subsets["REFERENCE_MASTERED"]
    sample_comparison = per_sample_comparison(e0, e5, hard_ids)
    recovery_epochs = sorted(set(actual_checkpoint_epochs(e0)).intersection(actual_checkpoint_epochs(e5)))
    recovery_last_epoch = recovery_epochs[-1]
    recovery = {
        "comparison_start_epoch": 0,
        "comparison_end_epoch": recovery_last_epoch,
    }
    for label, rows in [("E0", per_sample_comparison(e0, e0, hard_ids)), ("E5", sample_comparison)]:
        loss_recovered = [row for row in rows if row[f"{label.lower()}_loss_improvement"] > 0]
        iou_recovered = [row for row in rows if row[f"{label.lower()}_mask_iou_improvement"] is not None and row[f"{label.lower()}_mask_iou_improvement"] > 0]
        joint = [
            row for row in rows
            if row[f"{label.lower()}_loss_improvement"] > 0
            and row[f"{label.lower()}_mask_iou_improvement"] is not None
            and row[f"{label.lower()}_mask_iou_improvement"] >= 0
        ]
        recovery[label] = {
            "sample_count": len(rows),
            "loss_decrease_count": len(loss_recovered),
            "loss_decrease_rate": ratio(len(loss_recovered), len(rows)),
            "mask_iou_increase_count": len(iou_recovered),
            "mask_iou_increase_rate": ratio(len(iou_recovered), len(rows)),
            "loss_decrease_and_mask_non_decrease_count": len(joint),
            "loss_decrease_and_mask_non_decrease_rate": ratio(len(joint), len(rows)),
        }

    hard_stats = {label: subset_epoch_stats(run, hard_ids) for label, run in [("E0", e0), ("E5", e5)]}
    mastered_stats = {label: subset_epoch_stats(run, mastered_ids) for label, run in [("E0", e0), ("E5", e5)]}
    resource_epoch, resource_cumulative = resource_summary(
        read_json(args.e5_dir / "sample_dynamics/resource_history.json"),
        {"REFERENCE_HARD": hard_ids, "REFERENCE_MASTERED": mastered_ids},
    )
    states = final_states(args.e5_dir / "sample_dynamics/sample_state.json")
    e5_state_payload = read_json(args.e5_dir / "sample_dynamics/sample_state.json")
    corruption_reference = rf4.get("controlled_corruption", {})
    corruption_reference_samples = corruption_reference.get("samples", [])
    corruption_ids = {str(row.get("sample_id")) for row in corruption_reference_samples}
    e5_corruption_final = [
        {
            "sample_id": sample_id,
            "state": e5_state_payload.get("states", {}).get(sample_id, {}).get("state"),
        }
        for sample_id in sorted(corruption_ids)
    ]
    contract = contract_check(e0, e5)
    latest_e0 = overall_metrics["E0"][-1]
    latest_e5 = overall_metrics["E5"][-1]
    best_e0 = max(overall_metrics["E0"], key=lambda row: row.get("val_segm_mAP_50_95") or float("-inf"))
    best_e5 = max(overall_metrics["E5"], key=lambda row: row.get("val_segm_mAP_50_95") or float("-inf"))
    corruption = e5["fairness_snapshot"].get("train_config", {}).get("sample_dynamics_probe_interval")
    pilot_summary = read_json(args.e5_dir / "pilot_run_summary.json")
    analysis = {
        "schema_version": 1,
        "scope": "NON-BENCHMARK / MECHANISM VALIDATION",
        "contract": contract,
        "frozen_subsets": {name: len(ids) for name, ids in subsets.items()},
        "overall_metrics": overall_metrics,
        "hard_subset_epoch_stats": hard_stats,
        "mastered_subset_epoch_stats": mastered_stats,
        "hard_sample_comparison": sample_comparison,
        "recovery": recovery,
        "resource_epoch": resource_epoch,
        "resource_cumulative": resource_cumulative,
        "final_states": states,
        "controlled_corruption_smoke": pilot_summary.get("sample_dynamics", {}),
        "controlled_corruption_reference_e2": corruption_reference,
        "controlled_corruption_e5_final_states": e5_corruption_final,
        "loss_trajectory_contract": {
            "mode": "train",
            "gradients": "disabled",
            "reason": "preserve RF-DETR training criterion semantics for group_detr normalization",
            "probe_mode": "eval_with_rng_restore",
        },
        "rf4_gate": rf4.get("gate_status"),
        "headline": {
            "e0_final": latest_e0,
            "e5_final": latest_e5,
            "e0_best_mask_epoch": best_e0["epoch"],
            "e0_best_mask_mAP": best_e0.get("val_segm_mAP_50_95"),
            "e5_best_mask_epoch": best_e5["epoch"],
            "e5_best_mask_mAP": best_e5.get("val_segm_mAP_50_95"),
            "e5_elapsed_seconds": pilot_summary.get("elapsed_seconds"),
            "e5_peak_vram_bytes": pilot_summary.get("peak_vram_bytes"),
            "probe_interval": corruption,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "mvtec_e5_combined_analysis.json"
    json_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = args.output_dir / "mvtec_e5_combined_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["run", "epoch", "subset", "sample_count", "seen_sample_count", "mean_loss", "median_loss", "mean_mask_iou", "mean_matched_iou", "mean_fn", "mean_fp", "mean_class_error", "mean_gt_recall", "mean_exposure_multiplier", "mean_state_policy_weight", "mean_applied_loss_weight_seen", "mean_effective_contribution_seen", "total_appearances", "total_exposure_count", "total_effective_contribution", "cap_hit_count"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, subset_name, stats in [
            ("E0", "REFERENCE_HARD", hard_stats["E0"]),
            ("E5", "REFERENCE_HARD", hard_stats["E5"]),
            ("E0", "REFERENCE_MASTERED", mastered_stats["E0"]),
            ("E5", "REFERENCE_MASTERED", mastered_stats["E5"]),
        ]:
            for row in stats:
                writer.writerow({"run": label, "subset": subset_name, **row})
        for row in resource_epoch:
            writer.writerow({"run": "E5", **row})

    plot_paths = make_plots(args.output_dir, analysis)
    corruption_reference_table = ""
    for row in corruption_reference_samples:
        corruption_reference_table += (
            f"| {row.get('corruption_type')} | `{row.get('sample_id')}` | "
            f"{row.get('final_state')} | {row.get('max_conflict_streak')} | "
            f"{row.get('final_analysis_probe_conflict_streak')} |\n"
        )
    corruption_e5_table = ""
    for row in e5_corruption_final:
        corruption_e5_table += f"| `{row['sample_id']}` | {row.get('state') or 'missing'} |\n"
    report_path = args.output_dir / "mvtec_e5_combined_validation.md"
    e0_mask = latest_e0.get("val_segm_mAP_50_95")
    e5_mask = latest_e5.get("val_segm_mAP_50_95")
    report = f"""# RF-DETR Sample Dynamics — MVTec E5 Combined Validation

> Scope: **NON-BENCHMARK / MECHANISM VALIDATION**. This is one 15-epoch seed on the frozen MVTec Pilot v2 protocol; it is not a claim of general mAP improvement.

## Gate result

**E5 combined execution: PASS WITH CAVEATS.** The same-contract E0 baseline and E5 combined run completed 15 epochs, and the offline per-checkpoint evaluator produced 15 trajectory points for each run. The run is suitable for mechanism analysis and for deciding whether to schedule replicated ablations; it is not sufficient to freeze an algorithmic benefit claim.

The preceding RF4 longitudinal gate was `{rf4.get('gate_status')}`. Its frozen `REFERENCE_HARD` set contains {len(hard_ids)} samples and `REFERENCE_MASTERED` contains {len(mastered_ids)} samples. The five controlled-corruption samples remain a smoke subset only.

## Contract and artifacts

| Check | Result |
|---|---|
| Initial model hash, manifest, weights, seed equal | `{contract['all_equal']}` |
| Locked resolution / batch / epochs / augmentation / EMA | `{all(contract['locked_contract_equal'].values())}` |
| E0/E5 epochs | 15 / 15 |
| Per-checkpoint trajectory points | {e0['checkpoint_count_evaluated']} / {e5['checkpoint_count_evaluated']} |
| E0 final checkpoint caveat | trajectory recovery comparison uses the last common real checkpoint (epoch {recovery_last_epoch}); E0 metrics.csv still reports epoch 14 |
| StatePolicy tuning during this run | none; frozen policy retained |

## Overall validation metrics

| Run | Final mask mAP50:95 | Best mask mAP50:95 | Final box mAP50:95 | Final F1 | Elapsed / peak VRAM |
|---|---:|---:|---:|---:|---:|
| E0 baseline | {num(latest_e0.get('val_segm_mAP_50_95'))} | {num(best_e0.get('val_segm_mAP_50_95'))} (epoch {best_e0['epoch']}) | {num(latest_e0.get('val_mAP_50_95'))} | {num(latest_e0.get('val_F1'))} | see run summary |
| E5 combined | {num(e5_mask)} | {num(best_e5.get('val_segm_mAP_50_95'))} (epoch {best_e5['epoch']}) | {num(latest_e5.get('val_mAP_50_95'))} | {num(latest_e5.get('val_F1'))} | {num(pilot_summary.get('elapsed_seconds'), 1)} s / {pilot_summary.get('peak_vram_bytes', 0) / (1024**3):.2f} GiB |

The final mask-mAP difference `E5 - E0` is **{num(None if e0_mask is None or e5_mask is None else e5_mask - e0_mask)}** on this single seed. Because CUDA training was not bitwise deterministic across retries and E0 epoch 14 uses a retained best-model fallback, this number is descriptive rather than causal evidence.

![Overall validation metrics](assets/e5_overall_metrics.png)

## Frozen hard-sample recovery

The hard subset is held fixed from the E2 reference trajectory; it was not redefined after looking at E5. The recovery table compares epoch 0 to the last common real checkpoint (epoch {recovery_last_epoch}) because E0 epoch 14 was cleaned and its best_regular fallback records an earlier model state. A conservative descriptive recovery count is “per-image loss decreased”; the joint count additionally requires final mask IoU not to decrease. These are natural-recovery indicators, not proof that the sampler caused the recovery.

| Run | Hard samples | Loss decreased | Mask IoU increased | Loss decreased + mask IoU non-decreased |
|---|---:|---:|---:|---:|
| E0 | {recovery['E0']['sample_count']} | {recovery['E0']['loss_decrease_count']} ({pct(recovery['E0']['loss_decrease_rate'])}) | {recovery['E0']['mask_iou_increase_count']} ({pct(recovery['E0']['mask_iou_increase_rate'])}) | {recovery['E0']['loss_decrease_and_mask_non_decrease_count']} ({pct(recovery['E0']['loss_decrease_and_mask_non_decrease_rate'])}) |
| E5 | {recovery['E5']['sample_count']} | {recovery['E5']['loss_decrease_count']} ({pct(recovery['E5']['loss_decrease_rate'])}) | {recovery['E5']['mask_iou_increase_count']} ({pct(recovery['E5']['mask_iou_increase_rate'])}) | {recovery['E5']['loss_decrease_and_mask_non_decrease_count']} ({pct(recovery['E5']['loss_decrease_and_mask_non_decrease_rate'])}) |

Thus the current `HARD_LEARNABLE` label should still be read as a **high-loss candidate**. E5 does not yet establish learnability, and this run does not establish a statistically reliable E5 advantage over natural recovery under E0.

## Did E5 actually change training resources?

The resource ledger records the actual post-normalization loss weight, sampler exposure, capped effective contribution, and cap hits. Values below are averages over the 15 epochs per sample per epoch where appropriate.

| Frozen group | appearances / sample / epoch | exposure count / sample / epoch | policy weight | applied loss weight | effective contribution / sample / epoch | cap hits |
|---|---:|---:|---:|---:|---:|---:|
"""
    for name in ["REFERENCE_HARD", "REFERENCE_MASTERED", "ALL_TRAIN"]:
        row = resource_cumulative[name]
        report += f"| {name} | {num(row.get('appearances_per_sample_per_epoch'))} | {num(row.get('exposure_count_per_sample_per_epoch'))} | {num(row.get('mean_state_policy_weight'))} | {num(row.get('mean_applied_loss_weight_seen'))} | {num(row.get('effective_contribution_per_sample_per_epoch'))} | {row.get('cap_hit_count', 0)} |\n"
    report += f"""

The ledger confirms that E5 changed the resource path, but it also separates policy intent from realized optimization weight. The hard group's policy weight is higher on average, but global normalization plus cap constraints can make the realized applied loss weight lower or similar on individual appearances. Effective contribution is summed per appearance before the per-sample-per-epoch normalization, so replayed appearances are not undercounted. A cap hit is not automatically a benefit or a bug; it must be inspected as a side-effect guard in replicated runs.

![E5 resource allocation](assets/e5_hard_resource_recovery.png)

## Probe and state side effects

E5 finished with state counts `{json.dumps(states['state_counts'], sort_keys=True)}` at policy version `{states['policy_version']}`. The final all-train `SUSPECT` count and the controlled-corruption TP/precision/recall are separated below; neither is promoted to an RF8 statistic.

### Controlled-corruption evidence

The table below is the **RF4 E2 reference trajectory**, not a new E5 corruption metric. It records the five frozen smoke samples' persistent conflict evidence before intervention. The E5 final-state table is shown separately and must not be interpreted as five corruption hits.

| Corruption | Sample | E2 final state | Max conflict streak | Final analysis conflict streak |
|---|---|---|---:|---:|
{corruption_reference_table}

| Controlled-corruption sample | E5 final state |
|---|---|
{corruption_e5_table}

The E5 aggregate smoke summary is: all-train `SUSPECT` count=`{pilot_summary.get('sample_dynamics', {}).get('state_counts', {}).get('SUSPECT', 'n/a')}`, corruption TP=`{pilot_summary.get('sample_dynamics', {}).get('suspect_true_positive', 'n/a')}`, precision=`{pilot_summary.get('sample_dynamics', {}).get('suspect_precision', 'n/a')}`, recall=`{pilot_summary.get('sample_dynamics', {}).get('suspect_recall', 'n/a')}`. These remain smoke observations, not RF8 estimates.

The per-checkpoint evaluator also saved loss, mask IoU, FN, FP, and class-error for every train image. Its loss trajectory uses train mode with gradients disabled to preserve the RF-DETR training criterion's group-detr semantics; the probe uses eval mode and restores RNG. The CSV is the audit table for the frozen hard/mastered subset trajectories; the JSON contains the complete 128-image trajectories.

## Conclusions and next gate

1. The RF4 observation/state infrastructure now supplies a frozen, reusable hard subset and an auditable trajectory. That part is ready to support intervention comparisons.
2. E5 combined is technically executable and demonstrably changes exposure/loss resources, but this single seed does not prove an algorithmic gain over E0.
3. The correct next step is **replication under the same frozen contract**, preferably E0/E3/E4/E5 as a complete matrix. Do not tune StatePolicy or reinterpret the hard label from this run.
4. RF8 corruption precision/recall and migration-contract freeze remain pending.

### Generated evidence

- [Full E5 analysis JSON](mvtec_e5_combined_analysis.json)
- [E0/E5 result CSV](mvtec_e5_combined_results.csv)
- [Checkpoint trajectory JSON](mvtec_e0_e5_checkpoint_trajectories.json)
- [RF4 frozen subsets](mvtec_reference_subsets.json)
- [Overall metric figure](assets/e5_overall_metrics.svg)
- [Resource figure](assets/e5_hard_resource_recovery.svg)
"""
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"analysis": str(json_path), "report": str(report_path), "csv": str(csv_path), "plots": plot_paths}, sort_keys=True))


if __name__ == "__main__":
    main()
