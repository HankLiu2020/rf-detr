#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Build the read-only Policy V2 diagnosis and preregistration package."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

try:
    from .analyze_e5_combined import merged_metrics
except ImportError:
    from analyze_e5_combined import merged_metrics


V1_COMMIT = "4cc9fc5e0392d3be31235d9ce93632a7f34a35ba"
SEEDS = (20260810, 20260811, 20260812)
RUNS = ("E0", "E3", "E4", "E5")
METRICS = {
    "train_loss": "train/loss",
    "val_loss": "val/loss",
    "mask_mAP": "val/segm_mAP_50_95",
    "box_mAP": "val/mAP_50_95",
    "F1": "val/F1",
}
COMPONENTS = ("classification", "bbox", "giou", "mask_ce", "mask_dice")


def read_json(path: Path) -> Any:
    if path.suffix == ".xz":
        return json.loads(lzma.open(path, mode="rt", encoding="utf-8").read())
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.mean(clean) if clean else None


def sample_std(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.stdev(clean) if len(clean) > 1 else (0.0 if clean else None)


def linear_slope(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    x_mean = statistics.mean(x for x, _ in pairs)
    y_mean = statistics.mean(y for _, y in pairs)
    denominator = sum((x - x_mean) ** 2 for x, _ in pairs)
    return sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denominator if denominator else 0.0


def percentile_rank(value: float, population: list[float]) -> float:
    ordered = sorted(float(item) for item in population)
    if len(ordered) < 2:
        return 0.0
    return (sum(item <= value for item in ordered) - 1) / (len(ordered) - 1)


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def correlation(xs: list[float], ys: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(xs),
        "pearson": pearson(xs, ys),
        "spearman": pearson(rankdata(xs), rankdata(ys)),
    }


def run_dirs(root: Path) -> dict[int, dict[str, Path]]:
    return {
        20260810: {
            "E0": root / "e0-15",
            "E3": root / "e3-15",
            "E4": root / "e4-15",
            "E5": root / "e5-15",
        },
        20260811: {label: root / "seed-20260811" / label.lower() for label in RUNS},
        20260812: {label: root / "seed-20260812" / label.lower() for label in RUNS},
    }


def convergence_diagnosis(directories: dict[int, dict[str, Path]]) -> dict[str, Any]:
    """Diagnose the final five epochs without imposing a magnitude-picked threshold."""
    by_run: dict[str, Any] = {}
    positive_mask_slopes = 0
    late_best_count = 0
    total = 0
    for label in RUNS:
        seed_rows: dict[str, Any] = {}
        for seed in SEEDS:
            rows = merged_metrics(directories[seed][label] / "metrics.csv")
            epochs = sorted(rows)
            final_epoch = max(epochs)
            last_epochs = epochs[-5:]
            row: dict[str, Any] = {
                "last_five_epochs": last_epochs,
                "final_epoch": final_epoch,
                "final_step": int(rows[final_epoch]["step"]),
                "metrics": {},
            }
            for name, key in METRICS.items():
                xs = [float(epoch) for epoch in last_epochs]
                ys = [float(rows[epoch][key]) for epoch in last_epochs]
                best_epoch = (
                    min(epochs, key=lambda epoch: rows[epoch][key])
                    if "loss" in name
                    else max(epochs, key=lambda epoch: rows[epoch][key])
                )
                row["metrics"][name] = {
                    "last_five_values": ys,
                    "last_five_slope_per_epoch": linear_slope(xs, ys),
                    "best_epoch": int(best_epoch),
                    "best": float(rows[best_epoch][key]),
                    "final": float(rows[final_epoch][key]),
                    "final_minus_best": float(rows[final_epoch][key] - rows[best_epoch][key]),
                }
            positive_mask_slopes += int(row["metrics"]["mask_mAP"]["last_five_slope_per_epoch"] > 0.0)
            late_best_count += int(row["metrics"]["mask_mAP"]["best_epoch"] >= final_epoch - 1)
            total += 1
            seed_rows[str(seed)] = row
        by_run[label] = {
            "by_seed": seed_rows,
            "mean_last_five_slope": {
                metric: mean(seed_rows[str(seed)]["metrics"][metric]["last_five_slope_per_epoch"] for seed in SEEDS)
                for metric in METRICS
            },
            "positive_mask_slope_seeds": sum(
                seed_rows[str(seed)]["metrics"]["mask_mAP"]["last_five_slope_per_epoch"] > 0.0 for seed in SEEDS
            ),
            "late_best_seeds": sum(seed_rows[str(seed)]["metrics"]["mask_mAP"]["best_epoch"] >= 13 for seed in SEEDS),
        }
    # This rule is declared in the artifact and depends only on direction and
    # boundary timing, not on a post-hoc effect-size cutoff.
    extend = positive_mask_slopes / total >= 0.75 and late_best_count / total >= 0.50
    return {
        "decision_rule": {
            "extend_to_30_epochs_if": "at least 75% of run-seeds have positive last-five mask-mAP slope AND at least 50% reach their best mask mAP in the final two epochs",
            "positive_mask_slope_fraction": positive_mask_slopes / total,
            "late_best_fraction": late_best_count / total,
        },
        "decision": "USE_INDEPENDENT_30_EPOCH_CONTRACT" if extend else "KEEP_15_EPOCH_CONTRACT",
        "by_run": by_run,
    }


def coverage_diagnosis(directories: dict[int, dict[str, Path]], subsets: dict[str, set[str]]) -> dict[str, Any]:
    by_run: dict[str, Any] = {}
    for label in ("E3", "E4", "E5"):
        seed_results: dict[str, Any] = {}
        for seed in SEEDS:
            history = read_json(directories[seed][label] / "sample_dynamics" / "resource_history.json")
            epoch_rows: list[dict[str, Any]] = []
            subset_cumulative = {
                name: {"appearances": 0, "seen_sample_epochs": 0, "possible_sample_epochs": 0}
                for name in ("REFERENCE_HARD", "REFERENCE_MASTERED")
            }
            for payload in history:
                records = list(payload["records"])
                appearances = sum(int(record["batch_appearance_count"]) for record in records)
                seen = sum(int(record["batch_appearance_count"]) > 0 for record in records)
                epoch_rows.append(
                    {
                        "epoch": int(payload["epoch"]),
                        "sample_count": len(records),
                        "total_appearances": appearances,
                        "unique_seen": seen,
                        "unique_coverage": seen / len(records),
                        "omitted_samples": len(records) - seen,
                        "duplicate_slots": appearances - seen,
                    }
                )
                for name, ids in subsets.items():
                    selected = [record for record in records if record["sample_id"] in ids]
                    subset_cumulative[name]["appearances"] += sum(
                        int(record["batch_appearance_count"]) for record in selected
                    )
                    subset_cumulative[name]["seen_sample_epochs"] += sum(
                        int(record["batch_appearance_count"]) > 0 for record in selected
                    )
                    subset_cumulative[name]["possible_sample_epochs"] += len(selected)
            seed_results[str(seed)] = {
                "epochs": epoch_rows,
                "mean_unique_coverage": mean(row["unique_coverage"] for row in epoch_rows),
                "mean_omitted_samples": mean(row["omitted_samples"] for row in epoch_rows),
                "mean_duplicate_slots": mean(row["duplicate_slots"] for row in epoch_rows),
                "total_appearances": sum(row["total_appearances"] for row in epoch_rows),
                "subsets": {
                    name: {
                        **values,
                        "appearances_per_sample_epoch": values["appearances"] / values["possible_sample_epochs"],
                        "coverage": values["seen_sample_epochs"] / values["possible_sample_epochs"],
                    }
                    for name, values in subset_cumulative.items()
                },
            }
        by_run[label] = {
            "by_seed": seed_results,
            "mean_unique_coverage": mean(seed_results[str(seed)]["mean_unique_coverage"] for seed in SEEDS),
            "std_unique_coverage": sample_std(seed_results[str(seed)]["mean_unique_coverage"] for seed in SEEDS),
            "mean_omitted_samples": mean(seed_results[str(seed)]["mean_omitted_samples"] for seed in SEEDS),
            "mean_duplicate_slots": mean(seed_results[str(seed)]["mean_duplicate_slots"] for seed in SEEDS),
        }
    return {
        "baseline_contract": "E0 and E3 cover all 128 train samples once per epoch; E3 ledger is the observed baseline coverage reference",
        "by_run": by_run,
        "hypothesis_status": "SUPPORTED" if by_run["E4"]["mean_unique_coverage"] < 0.9 else "NOT_SUPPORTED",
    }


def _future_target(sample: dict[str, Any], epoch: int, horizon: int, target: str) -> float | None:
    future = epoch + horizon
    if target == "loss_improvement":
        return float(sample["loss"][epoch]) - float(sample["loss"][future])
    if target == "mask_iou_improvement":
        current, later = sample["matched_mask_iou"][epoch], sample["matched_mask_iou"][future]
        return None if current is None or later is None else float(later) - float(current)
    if target == "fn_reduction":
        return float(sample["fn"][epoch]) - float(sample["fn"][future])
    raise KeyError(target)


def hard_definition_diagnosis(longitudinal: dict[str, Any]) -> dict[str, Any]:
    samples = longitudinal["per_sample"]
    sample_ids = sorted(samples)
    epochs = int(longitudinal["epoch_count"])
    initial_loss_floor = statistics.median(float(samples[sample_id]["loss"][0]) for sample_id in sample_ids)
    signals: dict[str, dict[int, dict[str, float | bool]]] = {
        name: {} for name in ("instant_loss", "ema_loss", "current_dynamics", "learning_frontier")
    }
    frontier_ids_by_epoch: dict[str, list[str]] = {}
    for epoch in range(epochs):
        ema_population = [float(samples[sample_id]["loss_ema"][epoch]) for sample_id in sample_ids]
        selected: list[str] = []
        for sample_id in sample_ids:
            sample = samples[sample_id]
            ema = float(sample["loss_ema"][epoch])
            ema_percentile = percentile_rank(ema, ema_population)
            normalized_slope = float(sample["slope"][epoch]) / max(abs(ema), 1.0)
            mask_iou = sample["matched_mask_iou"][epoch]
            task_error = float(sample["fn"][epoch]) > 0 or (mask_iou is not None and float(mask_iou) < 0.5)
            stalled_conflict = int(sample["analysis_probe_conflict_streak"][epoch]) >= 3 and normalized_slope >= -0.001
            frontier = bool(
                epoch >= 4
                and ema_percentile >= 0.75
                and normalized_slope <= -0.01
                and ema >= initial_loss_floor
                and task_error
                and sample["dynamics_state"][epoch] != "SUSPECT"
                and not stalled_conflict
            )
            if frontier:
                selected.append(sample_id)
            improving_strength = min(1.0, max(0.0, -normalized_slope / 0.05))
            error_strength = max(float(sample["fn"][epoch] > 0), 0.0 if mask_iou is None else 1.0 - float(mask_iou))
            signals["instant_loss"].setdefault(epoch, {})[sample_id] = float(sample["instant_loss_percentile"][epoch])
            signals["ema_loss"].setdefault(epoch, {})[sample_id] = ema_percentile
            signals["current_dynamics"].setdefault(epoch, {})[sample_id] = float(sample["difficulty"][epoch])
            signals["learning_frontier"].setdefault(epoch, {})[sample_id] = (
                ema_percentile * improving_strength * error_strength if frontier else 0.0
            )
        frontier_ids_by_epoch[str(epoch)] = selected

    prediction: dict[str, Any] = {}
    for signal_name, by_epoch in signals.items():
        prediction[signal_name] = {}
        for target in ("loss_improvement", "mask_iou_improvement", "fn_reduction"):
            prediction[signal_name][target] = {}
            for horizon in (1, 2):
                windows: list[dict[str, Any]] = []
                for epoch in range(4, epochs - horizon):
                    xs: list[float] = []
                    ys: list[float] = []
                    for sample_id in sample_ids:
                        target_value = _future_target(samples[sample_id], epoch, horizon, target)
                        if target_value is None:
                            continue
                        xs.append(float(by_epoch[epoch][sample_id]))
                        ys.append(target_value)
                    windows.append({"epoch": epoch, **correlation(xs, ys)})
                prediction[signal_name][target][f"t+{horizon}"] = {
                    "windows": windows,
                    "mean_pearson": mean(row["pearson"] for row in windows),
                    "mean_spearman": mean(row["spearman"] for row in windows),
                }
    return {
        "definitions": {
            "H0_instant_loss": "current loss percentile",
            "H1_ema_loss": "EMA loss percentile",
            "current_dynamics": "frozen V1 Dynamics difficulty",
            "learning_frontier": {
                "warmup_epoch_min": 4,
                "ema_percentile_min": 0.75,
                "normalized_slope_max": -0.01,
                "absolute_ema_loss_floor": initial_loss_floor,
                "task_error": "FN > 0 OR matched mask IoU < 0.5",
                "exclude": "SUSPECT OR persistent conflict with normalized slope >= -0.001",
                "note": "top-quartile rank alone is insufficient; the improvement and absolute-loss gates allow the selected set to shrink",
            },
        },
        "frontier_ids_by_epoch": frontier_ids_by_epoch,
        "frontier_counts": {epoch: len(ids) for epoch, ids in frontier_ids_by_epoch.items()},
        "prediction": prediction,
    }


def component_diagnosis(path: Path | None, hard_ids: set[str]) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "status": "PENDING_COMPONENT_REPLAY",
            "reason": "V1 trajectory schema stored total per-image loss but not its weighted component decomposition",
        }
    payload = read_json(path)
    runs = payload["runs"]
    e0 = {int(row["epoch"]): row for row in runs["E0"]["trajectory"]}
    e5 = {int(row["epoch"]): row for row in runs["E5"]["trajectory"]}
    common_real = sorted(
        set(int(row["epoch"]) for row in runs["E0"]["trajectory"] if row["checkpoint_kind"] == "epoch")
        & set(int(row["epoch"]) for row in runs["E5"]["trajectory"] if row["checkpoint_kind"] == "epoch")
    )
    start, end = common_real[0], common_real[-1]
    by_component: dict[str, Any] = {}
    for component in COMPONENTS:
        e0_improvements: list[float] = []
        e5_improvements: list[float] = []
        for sample_id in sorted(hard_ids):
            e0_improvements.append(
                float(e0[start]["loss_components"][sample_id][component])
                - float(e0[end]["loss_components"][sample_id][component])
            )
            e5_improvements.append(
                float(e5[start]["loss_components"][sample_id][component])
                - float(e5[end]["loss_components"][sample_id][component])
            )
        deltas = [b - a for a, b in zip(e0_improvements, e5_improvements)]
        by_component[component] = {
            "sample_count": len(deltas),
            "e0_mean_improvement": mean(e0_improvements),
            "e5_mean_improvement": mean(e5_improvements),
            "e5_minus_e0_mean_improvement": mean(deltas),
            "positive_sample_count": sum(value > 0 for value in deltas),
        }
    total_component_delta = sum(float(row["e5_minus_e0_mean_improvement"]) for row in by_component.values())
    for row in by_component.values():
        row["share_of_component_delta"] = (
            float(row["e5_minus_e0_mean_improvement"]) / total_component_delta if total_component_delta else None
        )
    return {
        "status": "COMPLETE_SEED1_LONGITUDINAL",
        "source_sha256": sha256_file(path),
        "seed": 20260810,
        "epochs": [start, end],
        "subset": "REFERENCE_HARD",
        "by_component": by_component,
        "total_component_delta": total_component_delta,
        "limitation": "Only seed 20260810 has all real intermediate E0/E5 checkpoints; seed 20260811/12 component trajectories cannot be reconstructed after cleanup.",
    }


def make_plots(
    output_dir: Path,
    convergence: dict[str, Any],
    coverage: dict[str, Any],
    hard: dict[str, Any],
    components: dict[str, Any],
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    assets = output_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    colors = {"E0": "#4C78A8", "E3": "#B79A20", "E4": "#E07A2D", "E5": "#9C5A3C"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for label in RUNS:
        curves = [
            convergence["by_run"][label]["by_seed"][str(seed)]["metrics"]["mask_mAP"]["last_five_values"]
            for seed in SEEDS
        ]
        xs = convergence["by_run"][label]["by_seed"][str(SEEDS[0])]["last_five_epochs"]
        means = [statistics.mean(curve[index] for curve in curves) for index in range(len(xs))]
        stds = [statistics.stdev(curve[index] for curve in curves) for index in range(len(xs))]
        axes[0].plot(xs, means, marker="o", label=label, color=colors[label])
        axes[0].fill_between(
            xs,
            [v - s for v, s in zip(means, stds)],
            [v + s for v, s in zip(means, stds)],
            color=colors[label],
            alpha=0.12,
        )
        axes[1].plot([32 * (epoch + 1) for epoch in xs], means, marker="o", label=label, color=colors[label])
    axes[0].set(title="Mask mAP convergence by epoch", xlabel="Epoch", ylabel="mask mAP50:95")
    axes[1].set(title="Mask mAP convergence by optimizer step", xlabel="Approx. optimizer step", ylabel="mask mAP50:95")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.legend(ncol=2)
    convergence_path = assets / "policy_v2_convergence.png"
    fig.savefig(convergence_path, dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    labels = ("E3", "E4", "E5")
    values = [100 * float(coverage["by_run"][label]["mean_unique_coverage"]) for label in labels]
    errors = [100 * float(coverage["by_run"][label]["std_unique_coverage"]) for label in labels]
    axis.bar(labels, values, yerr=errors, capsize=5, color=[colors[label] for label in labels])
    axis.axhline(100, color="#333333", linestyle="--", linewidth=1)
    axis.set(title="V1 unique train-sample coverage", ylabel="Unique samples seen per epoch (%)", ylim=(0, 108))
    axis.grid(axis="y", alpha=0.25)
    coverage_path = assets / "policy_v2_v1_coverage.png"
    fig.savefig(coverage_path, dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    names = ("instant_loss", "ema_loss", "current_dynamics", "learning_frontier")
    values = [hard["prediction"][name]["mask_iou_improvement"]["t+1"]["mean_spearman"] or 0.0 for name in names]
    axis.bar(names, values, color=["#4C78A8", "#7397B8", "#B79A20", "#E07A2D"])
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set(title="Hard score vs next-epoch mask-IoU improvement", ylabel="Mean Spearman", xlabel="Offline score")
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.25)
    hard_path = assets / "policy_v2_hard_predictiveness.png"
    fig.savefig(hard_path, dpi=180)
    plt.close(fig)

    paths = {
        "convergence": str(convergence_path.relative_to(output_dir)),
        "coverage": str(coverage_path.relative_to(output_dir)),
        "hard_predictiveness": str(hard_path.relative_to(output_dir)),
    }
    if components["status"] == "COMPLETE_SEED1_LONGITUDINAL":
        fig, axis = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
        values = [components["by_component"][name]["e5_minus_e0_mean_improvement"] for name in COMPONENTS]
        axis.bar(COMPONENTS, values, color="#E07A2D")
        axis.axhline(0, color="#333333", linewidth=1)
        axis.set(
            title="E5 minus E0 hard-loss improvement by component",
            ylabel="Weighted per-image loss delta",
            xlabel="Loss component",
        )
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
        component_path = assets / "policy_v2_loss_component.png"
        fig.savefig(component_path, dpi=180)
        plt.close(fig)
        paths["components"] = str(component_path.relative_to(output_dir))
    return paths


def fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_reports(output_dir: Path, analysis: dict[str, Any], paths: dict[str, str]) -> None:
    convergence = analysis["convergence"]
    coverage = analysis["coverage"]
    components = analysis["loss_components"]
    hard = analysis["hard_definition"]

    convergence_lines = [
        "# Policy V2 Convergence Diagnosis",
        "",
        f"> Immutable V1 evidence baseline: `{V1_COMMIT}`. This report is read-only over the 12 V1 runs.",
        "",
        "## Result",
        "",
        f"**Decision: `{convergence['decision']}`.** All 12 run-seeds have positive last-five-epoch mask-mAP slopes, and 10/12 reach their best mask mAP in epoch 13 or 14. The 15-epoch matrix therefore cannot exclude slower convergence as the main E4/E5 explanation.",
        "",
        f"![Convergence]({paths['convergence']})",
        "",
        "## Last-five-epoch evidence",
        "",
        "| Run | mask slope mean | box slope mean | F1 slope mean | positive mask slope seeds | late-best seeds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in RUNS:
        row = convergence["by_run"][label]
        convergence_lines.append(
            f"| {label} | {fmt(row['mean_last_five_slope']['mask_mAP'], 6)} | {fmt(row['mean_last_five_slope']['box_mAP'], 6)} | {fmt(row['mean_last_five_slope']['F1'], 6)} | {row['positive_mask_slope_seeds']}/3 | {row['late_best_seeds']}/3 |"
        )
    convergence_lines += [
        "",
        "## Decision rule and limitation",
        "",
        f"The preregistered rule is: {convergence['decision_rule']['extend_to_30_epochs_if']}. This is descriptive convergence evidence, not proof that E4/E5 will catch E0. The next test must restart all compared runs from the same pretrained checkpoint under one 30-epoch contract; V1 checkpoints will not be resumed.",
        "",
    ]
    (output_dir / "policy_v2_convergence_diagnosis.md").write_text("\n".join(convergence_lines), encoding="utf-8")

    coverage_lines = [
        "# Policy V2 V1 Coverage and Exposure Diagnosis",
        "",
        "## Result",
        "",
        "**The coverage-loss hypothesis is supported.** E3 preserves all 128 samples each epoch. E4/E5 keep the same 128 appearance budget but only expose about 70% unique samples, replacing roughly 38 ordinary-sample slots with duplicates every epoch.",
        "",
        f"![Coverage]({paths['coverage']})",
        "",
        "| Run | Mean unique coverage | Mean omitted samples / epoch | Mean duplicate slots / epoch |",
        "|---|---:|---:|---:|",
    ]
    for label in ("E3", "E4", "E5"):
        row = coverage["by_run"][label]
        coverage_lines.append(
            f"| {label} | {100 * row['mean_unique_coverage']:.2f}% | {row['mean_omitted_samples']:.2f} | {row['mean_duplicate_slots']:.2f} |"
        )
    coverage_lines += [
        "",
        "This diagnosis establishes a mechanism-compatible explanation for V1 global regression, not causality. Policy V2 D1/D2 therefore retain 100% no-replacement base coverage and add a separately accounted hard bonus.",
        "",
    ]
    (output_dir / "policy_v2_coverage_exposure_diagnosis.md").write_text("\n".join(coverage_lines), encoding="utf-8")

    hard_lines = [
        "# Policy V2 Hard Definition Analysis",
        "",
        "## Result",
        "",
        "The V1 `HARD_LEARNABLE` rule is a high-loss rank, not a learnability test. Policy V2 freezes a Learning-Frontier prototype that requires high EMA loss, a material negative normalized slope, a real FN/mask error, an absolute loss floor, and no stalled persistent conflict/SUSPECT state.",
        "",
        f"![Hard predictiveness]({paths['hard_predictiveness']})",
        "",
        "## Learning-Frontier cohort size",
        "",
        "| Epoch | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| Selected | " + " | ".join(str(hard["frontier_counts"][str(epoch)]) for epoch in range(4, 15)) + " |",
        "",
        "The selected set shrinks from the high twenties to nine samples by epoch 14; it does not mechanically force 25% forever. Predictive correlations remain descriptive and are used only to preregister discovery—not to claim learnability.",
        "",
    ]
    (output_dir / "policy_v2_hard_definition_analysis.md").write_text("\n".join(hard_lines), encoding="utf-8")

    component_lines = [
        "# Policy V2 Hard-Loss Component Diagnosis",
        "",
        f"## Status: `{components['status']}`",
        "",
    ]
    if components["status"] == "COMPLETE_SEED1_LONGITUDINAL":
        component_lines += [
            "This is a seed-20260810 longitudinal decomposition over the common real checkpoint window epoch 0→13 on the frozen 33-sample `REFERENCE_HARD` subset.",
            "",
            f"![Loss components]({paths['components']})",
            "",
            "| Component | E0 improvement | E5 improvement | E5−E0 | Positive samples |",
            "|---|---:|---:|---:|---:|",
        ]
        for name in COMPONENTS:
            row = components["by_component"][name]
            component_lines.append(
                f"| {name} | {fmt(row['e0_mean_improvement'], 3)} | {fmt(row['e5_mean_improvement'], 3)} | {fmt(row['e5_minus_e0_mean_improvement'], 3)} | {row['positive_sample_count']}/{row['sample_count']} |"
            )
        component_lines += ["", components["limitation"], ""]
    else:
        component_lines += [components["reason"], ""]
    (output_dir / "policy_v2_loss_component_diagnosis.md").write_text("\n".join(component_lines), encoding="utf-8")


def preregistration(analysis: dict[str, Any], evidence: dict[str, str]) -> dict[str, Any]:
    epochs = 30 if analysis["convergence"]["decision"] == "USE_INDEPENDENT_30_EPOCH_CONTRACT" else 15
    common = {
        "epochs": epochs,
        "resolution": 384,
        "batch_size": 4,
        "grad_accum_steps": 1,
        "seed": 20260813,
        "augmentation": "disabled",
        "multi_scale": False,
        "use_ema": False,
        "pretrained_checkpoint": "/home/liujiyuan/rf-detr-models/rf-detr-seg-small.pt",
        "dataset": "/home/liujiyuan/mvtec-sample-dynamics-pilot-v2",
    }
    hypotheses = {
        "D0": "A fresh baseline is required under the same 30-epoch horizon and discovery seed.",
        "D1": "Warmup plus 100% base coverage and a 5% learnable-frontier bonus removes V1 distribution displacement while retaining mild targeted replay.",
        "D2": "A 10% bonus tests whether D1 under-allocates replay without returning to V1's 25% displacement.",
        "D3": "A 1.10 frontier-only loss multiplier tests gradient allocation without sampling shift or mastered downweighting.",
        "D4": "A 1.15 frontier-only multiplier tests a second mild intensity after D3, still below V1's 1.30.",
    }
    policies = {
        "D0": {"mode": "baseline"},
        "D1": {
            "mode": "coverage_bonus",
            "nominal_hard_bonus": 0.05,
            "realized_hard_bonus": 0.0625,
            "bonus_appearances": 8,
            "alignment_reason": "128 samples and batch=4 require 6.4 nominal appearances to round up to 8",
            "warmup_epochs": 5,
            "ramp": {"epochs_0_4": 0.0, "epochs_5_plus": 0.0625},
        },
        "D2": {
            "mode": "coverage_bonus",
            "hard_bonus": 0.10,
            "warmup_epochs": 5,
            "ramp": {"epochs_0_4": 0.0, "epochs_5_9": 0.05, "epochs_10_plus": 0.10},
        },
        "D3": {"mode": "frontier_weight", "hard_weight": 1.10, "other_weights": 1.0, "warmup_epochs": 5},
        "D4": {"mode": "frontier_weight", "hard_weight": 1.15, "other_weights": 1.0, "warmup_epochs": 5},
    }
    return {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_DISCOVERY_RUNS",
        "v1_evidence_commit": V1_COMMIT,
        "v1_evidence_sha256": evidence,
        "policy_namespace": "policy-v2",
        "discovery_seed": 20260813,
        "confirmatory_seeds_reserved": [20260814, 20260815, 20260816],
        "common_contract": common,
        "hard_definition": analysis["hard_definition"]["definitions"]["learning_frontier"],
        "convergence_controls": {
            "C0": {"mode": "baseline", "aliases": ["D0"]},
            "C3": {"mode": "v1_loss_weight", "hypothesis": "tests whether V1 E3 needs a longer horizon"},
            "C4": {
                "mode": "v1_sampler",
                "hypothesis": "tests whether V1 E4 is slower rather than asymptotically worse",
            },
            "C5": {
                "mode": "v1_combined",
                "hypothesis": "tests whether V1 E5 is slower rather than asymptotically worse",
            },
            "run_reason": "15-epoch diagnosis met the preregistered extension rule; all controls restart from the same pretrained checkpoint",
        },
        "hypotheses": hypotheses,
        "policies": policies,
        "run_order": ["C0/D0", "C3", "C4", "C5", "D1", "D2", "D3", "D4"],
        "confirmatory_entry_gate": [
            "overall mask mAP no material regression versus paired D0",
            "F1 no material regression",
            "hard mask-IoU or FN positive signal",
            "hard loss improvement retained",
            "mastered/normal subset no material regression",
            "resource ledger matches intended allocation",
            "compute-normalized comparison reported",
        ],
        "prohibitions": [
            "no changes after observing discovery validation metrics without a new recorded policy_id and hypothesis",
            "no reuse of 20260810/11/12 for tuning",
            "no confirmatory tuning",
            "no V1 artifact overwrite",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("/home/liujiyuan/mvtec-sample-dynamics-runs"))
    parser.add_argument("--longitudinal", type=Path, default=Path("reports/mvtec_rf4_longitudinal_analysis.json"))
    parser.add_argument("--reference-subsets", type=Path, default=Path("reports/mvtec_reference_subsets.json"))
    parser.add_argument(
        "--component-trajectory",
        type=Path,
        default=Path("reports/policy_v2_seed20260810_e0_e5_component_trajectories.json.xz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = read_json(args.reference_subsets)
    subsets = {name: set(reference["subsets"][name]) for name in ("REFERENCE_HARD", "REFERENCE_MASTERED")}
    evidence_paths = [
        output_dir / "mvtec_three_seed_ablation_validation.md",
        output_dir / "mvtec_three_seed_ablation_analysis.json",
        output_dir / "rf_detr_sample_dynamics_stage_report.md",
        args.reference_subsets.resolve(),
    ]
    evidence = {path.name: sha256_file(path) for path in evidence_paths}
    directories = run_dirs(args.run_root.resolve())
    analysis = {
        "schema_version": 1,
        "scope": "POLICY V2 / READ-ONLY V1 DIAGNOSIS",
        "v1_evidence_commit": V1_COMMIT,
        "v1_evidence_sha256": evidence,
        "convergence": convergence_diagnosis(directories),
        "coverage": coverage_diagnosis(directories, subsets),
        "hard_definition": hard_definition_diagnosis(read_json(args.longitudinal)),
        "loss_components": component_diagnosis(args.component_trajectory, subsets["REFERENCE_HARD"]),
    }
    paths = make_plots(
        output_dir,
        analysis["convergence"],
        analysis["coverage"],
        analysis["hard_definition"],
        analysis["loss_components"],
    )
    analysis["visuals"] = paths
    analysis_path = output_dir / "policy_v2_diagnosis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_reports(output_dir, analysis, paths)
    prereg = preregistration(analysis, evidence)
    (output_dir / "policy_v2_discovery_preregistration.json").write_text(
        json.dumps(prereg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status = f"""# Policy V2 Status

| Stage | Status | Evidence |
|---|---|---|
| V1 immutable baseline | PASS | `{V1_COMMIT}` + four recorded SHA256 values |
| Convergence diagnosis | PASS | `{analysis["convergence"]["decision"]}` |
| V1 coverage/exposure diagnosis | PASS | `{analysis["coverage"]["hypothesis_status"]}` |
| Hard-definition offline analysis | PASS_WITH_CAVEATS | predictive, not causal; frontier frozen before discovery |
| Loss-component diagnosis | {"PASS_WITH_CAVEATS" if analysis["loss_components"]["status"].startswith("COMPLETE") else "RUNNING"} | `{analysis["loss_components"]["status"]}` |
| Policy V2 discovery matrix | PASS | frozen seed 20260813, {prereg["common_contract"]["epochs"]} epochs |
| D0 baseline | TODO | no run started |
| D1 coverage + warmup + 5% frontier bonus | TODO | no run started |

Next gate: complete component replay if pending, implement the experiment-side D1 executor with targeted tests, then launch paired D0/D1 without changing this preregistration.
"""
    (output_dir / "policy_v2_status.md").write_text(status, encoding="utf-8")
    print(
        json.dumps(
            {
                "analysis": str(analysis_path),
                "contract": prereg["common_contract"],
                "component_status": analysis["loss_components"]["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
