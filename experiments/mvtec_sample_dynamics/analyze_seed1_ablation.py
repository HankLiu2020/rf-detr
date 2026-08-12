# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

#!/usr/bin/env python3
"""Build the frozen-contract Seed1 E0/E3/E4/E5 validation package.

The analysis is experiment-side and read-only. It combines validation metrics,
offline checkpoint trajectories, the E2-frozen reference subsets, and actual
resource ledgers. StatePolicy parameters are neither loaded nor modified here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from analyze_e5_combined import (
    actual_checkpoint_epochs,
    mean,
    merged_metrics,
    num,
    pct,
    probe_metric,
    read_json,
    resource_summary,
    subset_epoch_stats,
    trajectory_by_epoch,
)


RUNS = ("E0", "E3", "E4", "E5")
IDENTITY_FIELDS = (
    "dataset_manifest_sha256",
    "initial_model_parameter_sha256",
    "pretrain_weights_sha256",
    "seed",
)
CONTRACT_FIELDS = (
    "resolution",
    "batch_size",
    "grad_accum_steps",
    "epochs",
    "seed",
    "multi_scale",
    "scale_jitter",
    "aug_config",
    "augmentation_backend",
    "use_ema",
)
TRAIN_CONFIG_EXCLUSIONS = {
    "notes",
    "output_dir",
    "sample_dynamics_enabled",
    "sample_dynamics_mode",
    "sample_dynamics_output_dir",
    "sample_dynamics_probe_interval",
}


def load_trajectories(paths: list[Path]) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for path in paths:
        for label, payload in read_json(path)["runs"].items():
            if label in runs:
                raise ValueError(f"duplicate trajectory run {label!r}")
            runs[label] = payload
    missing = set(RUNS).difference(runs)
    if missing:
        raise ValueError(f"missing trajectory runs: {sorted(missing)}")
    return runs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_artifact_hashes(run_dir: Path) -> dict[str, str]:
    relative_paths = [
        "fairness_snapshot.json",
        "metrics.csv",
        "pilot_run_summary.json",
        "sample_order_hash.json",
        "training_config.json",
        "sample_dynamics/resource_history.json",
        "sample_dynamics/sample_state.json",
    ]
    relative_paths.extend(path.name for path in sorted(run_dir.glob("checkpoint_*.ckpt")))
    relative_paths.extend(path.name for path in sorted(run_dir.glob("checkpoint_best_*.pth")))
    return {
        relative: sha256_file(run_dir / relative)
        for relative in relative_paths
        if (run_dir / relative).is_file()
    }


def contract_check(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reference = runs["E0"]["fairness_snapshot"]
    identities = {
        field: {
            "equal": all(runs[label]["fairness_snapshot"].get(field) == reference.get(field) for label in RUNS),
            "value": reference.get(field),
        }
        for field in IDENTITY_FIELDS
    }
    locked = {
        field: {
            "equal": all(
                runs[label]["fairness_snapshot"]["locked_contract"].get(field)
                == reference["locked_contract"].get(field)
                for label in RUNS
            ),
            "value": reference["locked_contract"].get(field),
        }
        for field in CONTRACT_FIELDS
    }
    checkpoint_counts = {label: int(runs[label]["checkpoint_count_evaluated"]) for label in RUNS}
    reference_model = dict(reference["model_config"])
    reference_train = {
        key: value for key, value in reference["train_config"].items() if key not in TRAIN_CONFIG_EXCLUSIONS
    }
    full_model_equal = {
        label: dict(runs[label]["fairness_snapshot"]["model_config"]) == reference_model for label in RUNS
    }
    full_non_intervention_train_equal = {
        label: {
            key: value
            for key, value in runs[label]["fairness_snapshot"]["train_config"].items()
            if key not in TRAIN_CONFIG_EXCLUSIONS
        }
        == reference_train
        for label in RUNS
    }
    return {
        "identity": identities,
        "locked_contract": locked,
        "checkpoint_counts": checkpoint_counts,
        "full_model_config_equal": full_model_equal,
        "full_non_intervention_train_config_equal": full_non_intervention_train_equal,
        "all_equal": all(row["equal"] for row in identities.values())
        and all(row["equal"] for row in locked.values())
        and all(full_model_equal.values())
        and all(full_non_intervention_train_equal.values()),
        "all_15_trajectories": all(value == 15 for value in checkpoint_counts.values()),
    }


def metric_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch, values in sorted(merged_metrics(run_dir / "metrics.csv").items()):
        rows.append(
            {
                "epoch": epoch,
                "val_box_mAP_50_95": values.get("val/mAP_50_95"),
                "val_mask_mAP_50_95": values.get("val/segm_mAP_50_95"),
                "val_F1": values.get("val/F1"),
                "val_precision": values.get("val/precision"),
                "val_recall": values.get("val/recall"),
                "val_loss": values.get("val/loss"),
                "train_loss": values.get("train/loss"),
            }
        )
    return rows


def recovery_summary(run: dict[str, Any], sample_ids: set[str], start_epoch: int, end_epoch: int) -> dict[str, Any]:
    by_epoch = trajectory_by_epoch(run)
    first = by_epoch[start_epoch]
    last = by_epoch[end_epoch]
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(sample_ids):
        if sample_id not in first.get("loss", {}) or sample_id not in last.get("loss", {}):
            continue
        initial_loss = float(first["loss"][sample_id])
        final_loss = float(last["loss"][sample_id])
        initial_iou = probe_metric(first, sample_id, "matched_mask_iou")
        final_iou = probe_metric(last, sample_id, "matched_mask_iou")
        loss_improvement = initial_loss - final_loss
        iou_improvement = None if initial_iou is None or final_iou is None else final_iou - initial_iou
        rows.append(
            {
                "sample_id": sample_id,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "loss_improvement": loss_improvement,
                "initial_mask_iou": initial_iou,
                "final_mask_iou": final_iou,
                "mask_iou_improvement": iou_improvement,
            }
        )
    loss_recovered = sum(row["loss_improvement"] > 0 for row in rows)
    iou_recovered = sum(
        row["mask_iou_improvement"] is not None and row["mask_iou_improvement"] > 0 for row in rows
    )
    joint = sum(
        row["loss_improvement"] > 0
        and row["mask_iou_improvement"] is not None
        and row["mask_iou_improvement"] >= 0
        for row in rows
    )
    count = len(rows)
    return {
        "sample_count": count,
        "loss_decrease_count": loss_recovered,
        "loss_decrease_rate": loss_recovered / count if count else None,
        "mask_iou_increase_count": iou_recovered,
        "mask_iou_increase_rate": iou_recovered / count if count else None,
        "loss_decrease_and_mask_non_decrease_count": joint,
        "loss_decrease_and_mask_non_decrease_rate": joint / count if count else None,
        "mean_loss_improvement": mean(row["loss_improvement"] for row in rows),
        "mean_mask_iou_improvement": mean(row["mask_iou_improvement"] for row in rows),
        "samples": rows,
    }


def ratio_value(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def resource_gate(label: str, summary: dict[str, Any]) -> dict[str, Any]:
    hard = summary["REFERENCE_HARD"]
    mastered = summary["REFERENCE_MASTERED"]
    result = {
        "hard_to_mastered_appearance_ratio": ratio_value(
            hard.get("appearances_per_sample_per_epoch"), mastered.get("appearances_per_sample_per_epoch")
        ),
        "hard_to_mastered_applied_weight_ratio": ratio_value(
            hard.get("mean_applied_loss_weight_seen"), mastered.get("mean_applied_loss_weight_seen")
        ),
        "hard_to_mastered_effective_contribution_ratio": ratio_value(
            hard.get("effective_contribution_per_sample_per_epoch"),
            mastered.get("effective_contribution_per_sample_per_epoch"),
        ),
    }
    if label == "E3":
        exposure_isolated = (
            result["hard_to_mastered_appearance_ratio"] is not None
            and abs(result["hard_to_mastered_appearance_ratio"] - 1.0) <= 1e-9
        )
        no_cap_hits = hard.get("cap_hit_count", 0) == 0 and mastered.get("cap_hit_count", 0) == 0
        result["mechanism_pass"] = (
            result["hard_to_mastered_applied_weight_ratio"] is not None
            and result["hard_to_mastered_applied_weight_ratio"] > 1.0
            and exposure_isolated
            and no_cap_hits
        )
        result["isolation"] = {"uniform_exposure": exposure_isolated, "zero_cap_hits": no_cap_hits}
        result["criterion"] = "hard weight > mastered; hard/mastered exposure equal; no cap hits"
    elif label == "E4":
        loss_weight_disabled = all(
            row.get("mean_applied_loss_weight_seen") is None
            and row.get("effective_contribution_per_sample_per_epoch") is None
            for row in (hard, mastered, summary["ALL_TRAIN"])
        )
        result["mechanism_pass"] = (
            result["hard_to_mastered_appearance_ratio"] is not None
            and result["hard_to_mastered_appearance_ratio"] > 1.0
            and loss_weight_disabled
        )
        result["isolation"] = {"loss_weight_disabled": loss_weight_disabled}
        result["criterion"] = "hard appearances > mastered; loss weighting absent"
    else:
        exposure_increased = (
            result["hard_to_mastered_appearance_ratio"] is not None
            and result["hard_to_mastered_appearance_ratio"] > 1.0
        )
        cap_hits_reported = hard.get("cap_hit_count") is not None and mastered.get("cap_hit_count") is not None
        result["isolation"] = {
            "exposure_increased": exposure_increased,
            "cap_hits_reported": cap_hits_reported,
        }
        result["mechanism_pass"] = (
            exposure_increased
            and result["hard_to_mastered_effective_contribution_ratio"] is not None
            and result["hard_to_mastered_effective_contribution_ratio"] > 1.0
            and cap_hits_reported
        )
        result["criterion"] = "hard exposure and effective contribution > mastered; cap hits reported"
    return result


def make_plots(output_dir: Path, analysis: dict[str, Any]) -> list[str]:
    import matplotlib.pyplot as plt

    assets = output_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    colors = {"E0": "#4c78a8", "E3": "#59a14f", "E4": "#e15759", "E5": "#f28e2b"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for label in RUNS:
        rows = analysis["overall_metrics"][label]
        epochs = [row["epoch"] for row in rows]
        axes[0].plot(epochs, [row["val_mask_mAP_50_95"] for row in rows], marker="o", ms=3, label=label, color=colors[label])
        axes[1].plot(epochs, [row["val_box_mAP_50_95"] for row in rows], marker="o", ms=3, label=label, color=colors[label])
    axes[0].set_title("Mask mAP50:95")
    axes[1].set_title("Box mAP50:95")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.set_ylabel("metric")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(assets / "seed1_ablation_metrics.png", dpi=170)
    plt.close(fig)

    labels = ("E3", "E4", "E5")
    weight_ratio = [analysis["resource_gate"][label]["hard_to_mastered_applied_weight_ratio"] or 0 for label in labels]
    exposure_ratio = [analysis["resource_gate"][label]["hard_to_mastered_appearance_ratio"] or 0 for label in labels]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    ax.bar([value - 0.18 for value in x], weight_ratio, 0.36, label="applied loss-weight ratio", color="#59a14f")
    ax.bar([value + 0.18 for value in x], exposure_ratio, 0.36, label="appearance ratio", color="#e15759")
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("REFERENCE_HARD / REFERENCE_MASTERED")
    ax.set_title("Realized resource allocation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(assets / "seed1_ablation_resources.png", dpi=170)
    plt.close(fig)
    return [str(path.relative_to(output_dir)) for path in sorted(assets.glob("seed1_ablation_*.png"))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--subsets", type=Path, default=Path("reports/mvtec_reference_subsets.json"))
    parser.add_argument("--e0-dir", type=Path, required=True)
    parser.add_argument("--e3-dir", type=Path, required=True)
    parser.add_argument("--e4-dir", type=Path, required=True)
    parser.add_argument("--e5-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    run_dirs = {"E0": args.e0_dir, "E3": args.e3_dir, "E4": args.e4_dir, "E5": args.e5_dir}
    trajectories = load_trajectories(args.trajectory)
    subset_payload = read_json(args.subsets)
    subsets = {name: set(values) for name, values in subset_payload["subsets"].items()}
    hard_ids = subsets["REFERENCE_HARD"]
    mastered_ids = subsets["REFERENCE_MASTERED"]
    contract = contract_check(trajectories)
    overall = {label: metric_rows(run_dirs[label]) for label in RUNS}
    common_actual = sorted(set.intersection(*(set(actual_checkpoint_epochs(trajectories[label])) for label in RUNS)))
    if not common_actual:
        raise ValueError("the four runs have no common real checkpoint epoch")
    comparison_epoch = common_actual[-1]
    recovery = {
        label: recovery_summary(trajectories[label], hard_ids, 0, comparison_epoch) for label in RUNS
    }
    subset_trajectories = {
        label: {
            "REFERENCE_HARD": subset_epoch_stats(trajectories[label], hard_ids),
            "REFERENCE_MASTERED": subset_epoch_stats(trajectories[label], mastered_ids),
        }
        for label in RUNS
    }
    resource_epoch: dict[str, list[dict[str, Any]]] = {}
    resource_cumulative: dict[str, dict[str, Any]] = {}
    gates: dict[str, dict[str, Any]] = {}
    for label in ("E3", "E4", "E5"):
        epoch_rows, cumulative = resource_summary(
            read_json(run_dirs[label] / "sample_dynamics/resource_history.json"),
            {"REFERENCE_HARD": hard_ids, "REFERENCE_MASTERED": mastered_ids},
        )
        resource_epoch[label] = epoch_rows
        resource_cumulative[label] = cumulative
        gates[label] = resource_gate(label, cumulative)

    headline: dict[str, Any] = {}
    e0_final = overall["E0"][-1]
    for label in RUNS:
        final = overall[label][-1]
        best_mask = max(overall[label], key=lambda row: row["val_mask_mAP_50_95"] or float("-inf"))
        headline[label] = {
            "final": final,
            "best_mask_epoch": best_mask["epoch"],
            "best_mask_mAP_50_95": best_mask["val_mask_mAP_50_95"],
            "delta_vs_e0_final_mask_mAP_50_95": final["val_mask_mAP_50_95"] - e0_final["val_mask_mAP_50_95"],
            "delta_vs_e0_final_box_mAP_50_95": final["val_box_mAP_50_95"] - e0_final["val_box_mAP_50_95"],
            "delta_vs_e0_final_F1": final["val_F1"] - e0_final["val_F1"],
        }

    analysis = {
        "schema_version": 1,
        "scope": "NON-BENCHMARK / SINGLE-SEED MECHANISM VALIDATION",
        "seed": 20260810,
        "contract": contract,
        "source_artifact_sha256": {label: run_artifact_hashes(run_dirs[label]) for label in RUNS},
        "trajectory_artifact_sha256": {
            str(path): sha256_file(path) for path in args.trajectory
        },
        "checkpoint_retention": {
            "E0": "epoch 0-13 plus best_regular fallback retained; epoch 14 was previously cleaned",
            "E3": "epoch 14 and best checkpoints retained; epoch 0-13 removed after trajectory extraction",
            "E4": "epoch 14 and best checkpoints retained; epoch 0-13 removed after trajectory extraction",
            "E5": "epoch 0-14 and best checkpoints retained",
        },
        "frozen_subset_counts": {name: len(values) for name, values in subsets.items()},
        "comparison_checkpoint_epoch": comparison_epoch,
        "comparison_checkpoint_reason": "last common real checkpoint; E0 epoch 14 was previously cleaned",
        "overall_metrics": overall,
        "headline": headline,
        "hard_recovery": recovery,
        "subset_trajectories": subset_trajectories,
        "resource_epoch": resource_epoch,
        "resource_cumulative": resource_cumulative,
        "resource_gate": gates,
        "gate": {
            "correctness": "PASS" if contract["all_equal"] and contract["all_15_trajectories"] else "FAIL",
            "resource_mechanisms": "PASS" if all(row["mechanism_pass"] for row in gates.values()) else "FAIL",
            "algorithm_signal": "MIXED_SINGLE_SEED",
            "replication_required": True,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = args.output_dir / "mvtec_seed1_ablation_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = args.output_dir / "mvtec_seed1_ablation_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "record_type", "run", "epoch", "subset", "sample_count", "seen_sample_count", "mean_loss", "median_loss", "mean_mask_iou",
            "mean_matched_iou", "mean_fn", "mean_fp", "mean_class_error", "mean_gt_recall",
            "mean_exposure_multiplier", "mean_state_policy_weight", "mean_applied_loss_weight_seen",
            "mean_effective_contribution_seen", "total_appearances", "total_exposure_count",
            "total_effective_contribution", "cap_hit_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for label in RUNS:
            for subset_name in ("REFERENCE_HARD", "REFERENCE_MASTERED"):
                for row in subset_trajectories[label][subset_name]:
                    writer.writerow({"record_type": "checkpoint_trajectory", "run": label, "subset": subset_name, **row})
        for label in ("E3", "E4", "E5"):
            for row in resource_epoch[label]:
                writer.writerow({"record_type": "resource_ledger", "run": label, **row})

    plot_paths = make_plots(args.output_dir, analysis)
    report_path = args.output_dir / "mvtec_seed1_ablation_validation.md"
    metric_table = ""
    for label in RUNS:
        row = headline[label]
        final = row["final"]
        metric_table += (
            f"| {label} | {num(final['val_mask_mAP_50_95'])} | {num(row['delta_vs_e0_final_mask_mAP_50_95'])} | "
            f"{num(final['val_box_mAP_50_95'])} | {num(row['delta_vs_e0_final_box_mAP_50_95'])} | "
            f"{num(final['val_F1'])} | {num(row['delta_vs_e0_final_F1'])} | "
            f"{num(row['best_mask_mAP_50_95'])} (e{row['best_mask_epoch']}) |\n"
        )
    resource_table = ""
    for label in ("E3", "E4", "E5"):
        hard = resource_cumulative[label]["REFERENCE_HARD"]
        mastered = resource_cumulative[label]["REFERENCE_MASTERED"]
        gate = gates[label]
        resource_table += (
            f"| {label} | {num(hard['mean_applied_loss_weight_seen'])} / {num(mastered['mean_applied_loss_weight_seen'])} | "
            f"{num(hard['appearances_per_sample_per_epoch'])} / {num(mastered['appearances_per_sample_per_epoch'])} | "
            f"{num(hard['effective_contribution_per_sample_per_epoch'])} / {num(mastered['effective_contribution_per_sample_per_epoch'])} | "
            f"{gate['mechanism_pass']} |\n"
        )
    recovery_table = ""
    for label in RUNS:
        row = recovery[label]
        recovery_table += (
            f"| {label} | {row['loss_decrease_count']}/{row['sample_count']} ({pct(row['loss_decrease_rate'])}) | "
            f"{row['mask_iou_increase_count']}/{row['sample_count']} ({pct(row['mask_iou_increase_rate'])}) | "
            f"{row['loss_decrease_and_mask_non_decrease_count']}/{row['sample_count']} "
            f"({pct(row['loss_decrease_and_mask_non_decrease_rate'])}) |\n"
        )
    report = f"""# RF-DETR Sample Dynamics — Seed1 完整消融验证

> 范围：**NON-BENCHMARK / SINGLE-SEED MECHANISM VALIDATION**。本报告用于验证机制是否真实生效，并观察单 seed 信号；不能据此宣称算法稳定提升。

## Gate 结论

**Seed1 里程碑：correctness `{analysis['gate']['correctness']}`，resource mechanism `{analysis['gate']['resource_mechanisms']}`，algorithm signal `MIXED_SINGLE_SEED`。**

E0、E3、E4、E5 均在 Pilot v2、RF-DETR Seg Small、seed=20260810、15 epoch 的冻结契约下完成。初始模型、预训练权重、manifest、seed、完整 model config 及除干预开关/输出路径以外的完整 train config 全部一致。四条轨迹各有 15 个离线观察点；E0 epoch 14 是 `best_regular` fallback，其余为真实 epoch checkpoint，所以逐图恢复统一比较到共同真实 epoch 13。没有调整 StatePolicy。

## 两部分实验设计

1. **样本 Loss 分析与分类**：逐图记录 per-image normalized loss，通过长期状态将样本分为 `LEARNING / HARD_LEARNABLE / MASTERED / SUSPECT`；本轮所有比较使用 E2 预先冻结的 33 个 `REFERENCE_HARD` 与 43 个 `REFERENCE_MASTERED`，不根据 intervention 结果重新定义。
2. **下一轮资源调度**：E3 只改变 loss 权重，E4 只改变下一轮采样曝光，E5 同时改变二者；资源账本记录实际 batch 权重、实际出现次数和 capped effective contribution，而不是只读取配置开关。

## 最终验证指标

| Run | final mask mAP | Δmask vs E0 | final box mAP | Δbox vs E0 | final F1 | ΔF1 vs E0 | best mask mAP |
|---|---:|---:|---:|---:|---:|---:|---:|
{metric_table}

![四组验证指标轨迹](assets/seed1_ablation_metrics.png)

单 seed 呈现明显的指标分化：E3 最终 mask mAP 比 E0 高，但 box mAP 与 F1 较低；E4 的 box mAP 较高，但 mask mAP 与 F1 较低；E5 在本 seed 的三个最终指标均未超过 E0。这里应报告为“混合信号”，不能选择性宣称 E3 已经有效。

## 调度资源是否真的改变

表中均为 `REFERENCE_HARD / REFERENCE_MASTERED` 的实际累计均值。

| Run | applied loss weight | appearances / sample / epoch | effective contribution / sample / epoch | mechanism gate |
|---|---:|---:|---:|---:|
{resource_table}

![实际资源比率](assets/seed1_ablation_resources.png)

- E3 同时要求困难组实际 loss weight 高于 mastered、hard/mastered 曝光比严格为 1、cap hit 为 0。
- E4 同时要求困难组实际 appearance 高于 mastered、applied loss weight 与 weighted effective contribution 均未记录。
- E5 同时要求 hard/mastered exposure 和 effective contribution 比率大于 1，并明确记录 cap 命中。

因此，“识别困难样本 → 下一轮增加优化资源”的工程闭环已经可观测且可审计。但资源变多不等于效果必然变好，本 seed 已直接说明二者必须分开验收。

## 冻结困难子集的后续改善

为避免 E0 已清理的 epoch 14 checkpoint fallback 造成模型状态错配，逐图恢复统一比较 checkpoint epoch 0→{comparison_epoch}。

| Run | loss 下降 | mask IoU 上升 | loss 下降且 mask IoU 不降 |
|---|---:|---:|---:|
{recovery_table}

这些是描述性“后续改善”指标，并不证明 `HARD_LEARNABLE` 中每个样本都可学，也不能把自然恢复归因给调度。该状态名称在现阶段仍应解释为 **high-loss candidate**。

## 阶段结论与下一 Gate

1. Loss observer、四态分类、E3 权重路径、E4 采样路径、E5 组合路径均通过同一冻结契约下的真实训练与资源证据检查。
2. 当前没有稳定算法收益结论：E3/E4/E5 对不同指标作用方向不一致，E5 单 seed 为负信号。
3. 下一里程碑应固定完整矩阵、再跑 seed=20260811 和 20260812，输出 mean ± std；禁止只复现当前最有利的 E3 mask 指标。
4. Controlled corruption 仍只有 5 个 smoke 样本，不能升级为 RF8 precision/recall 结论；RF9 migration contract 也暂不冻结。

## 可审计证据

- [完整分析 JSON](mvtec_seed1_ablation_analysis.json)
- [逐 epoch/逐子集结果 CSV](mvtec_seed1_ablation_results.csv)
- [E0/E5 checkpoint trajectories](mvtec_e0_e5_checkpoint_trajectories.json)
- [E3 checkpoint trajectories](mvtec_e3_checkpoint_trajectories.json)
- [E4 checkpoint trajectories](mvtec_e4_checkpoint_trajectories.json)
- [E2 冻结子集](mvtec_reference_subsets.json)
- [指标图 PNG](assets/seed1_ablation_metrics.png)
- [资源图 PNG](assets/seed1_ablation_resources.png)

分析 JSON 保存了仍保留 checkpoint 和全部源账本的 SHA-256。E3/E4 epoch 0–13 checkpoint 已在逐图 trajectory 提取并校验后清理，故这些中间权重不可本地重放；其逐图 loss/probe 结果保留在 trajectory JSON 中。E3/E4 epoch 14、best checkpoints 以及 E0/E5 保留的 checkpoint 均已哈希。
"""
    report_path.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {"analysis": str(analysis_path), "csv": str(csv_path), "report": str(report_path), "plots": plot_paths},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
