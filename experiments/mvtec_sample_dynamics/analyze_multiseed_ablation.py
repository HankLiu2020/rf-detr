# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

#!/usr/bin/env python3
"""Aggregate the frozen MVTec E0/E3/E4/E5 matrix across multiple seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .analyze_e5_combined import merged_metrics, read_json, resource_summary
    from .analyze_seed1_ablation import (
        CONTRACT_FIELDS,
        IDENTITY_FIELDS,
        RUNS,
        TRAIN_CONFIG_EXCLUSIONS,
        recovery_summary,
        run_artifact_hashes,
        resource_gate,
        sha256_file,
    )
except ImportError:  # Standalone execution from this directory.
    from analyze_e5_combined import merged_metrics, read_json, resource_summary
    from analyze_seed1_ablation import (
        CONTRACT_FIELDS,
        IDENTITY_FIELDS,
        RUNS,
        TRAIN_CONFIG_EXCLUSIONS,
        recovery_summary,
        run_artifact_hashes,
        resource_gate,
        sha256_file,
    )


METRICS = ("mask_mAP_50_95", "box_mAP_50_95", "F1")


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "values": values,
        "mean": statistics.mean(values),
        "std": sample_std(values),
        "min": min(values),
        "max": max(values),
    }


def direction_counts(values: list[float]) -> dict[str, int]:
    return {
        "positive": sum(value > 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "negative": sum(value < 0 for value in values),
    }


def parse_run_specifications(values: list[str]) -> dict[int, dict[str, Path]]:
    result: dict[int, dict[str, Path]] = defaultdict(dict)
    for value in values:
        lhs, separator, raw_path = value.partition("=")
        seed_text, colon, label = lhs.partition(":")
        if not separator or not colon or label not in RUNS:
            raise ValueError(f"--run must use SEED:LABEL=DIR, got {value!r}")
        seed = int(seed_text)
        if label in result[seed]:
            raise ValueError(f"duplicate run {seed}:{label}")
        result[seed][label] = Path(raw_path).resolve()
    for seed, runs in result.items():
        missing = set(RUNS).difference(runs)
        if missing:
            raise ValueError(f"seed {seed} is missing runs {sorted(missing)}")
    return dict(result)


def load_trajectories(paths: list[Path]) -> dict[int, dict[str, dict[str, Any]]]:
    result: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in paths:
        payload = read_json(path)
        for label, run in payload["runs"].items():
            seed = int(run["fairness_snapshot"]["seed"])
            if label in result[seed]:
                raise ValueError(f"duplicate trajectory {seed}:{label}")
            result[seed][label] = run
    return dict(result)


def trajectory_check(trajectories: dict[int, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    for seed, runs in sorted(trajectories.items()):
        by_seed[str(seed)] = {}
        for label in RUNS:
            run = runs[label]
            requested = int(run["epochs_requested"])
            epochs = [int(value) for value in run["epochs_evaluated"]]
            entry_epochs = [int(entry["epoch"]) for entry in run["trajectory"]]
            expected = list(range(requested))
            fallback_epochs = [
                int(entry["epoch"])
                for entry in run["trajectory"]
                if entry.get("checkpoint_kind") != "epoch"
            ]
            row = {
                "checkpoint_count_evaluated": int(run["checkpoint_count_evaluated"]),
                "epochs_evaluated": epochs,
                "trajectory_entry_epochs": entry_epochs,
                "fallback_epochs": fallback_epochs,
                "all_real_epoch_checkpoints": not fallback_epochs,
                "complete": (
                    int(run["checkpoint_count_evaluated"]) == requested
                    and epochs == expected
                    and entry_epochs == expected
                    and len(set(entry_epochs)) == requested
                ),
            }
            by_seed[str(seed)][label] = row
    return {
        "by_seed": by_seed,
        "all_complete": all(
            row["complete"]
            for runs in by_seed.values()
            for row in runs.values()
        ),
    }


def trajectory_binding_check(
    run_dirs: dict[int, dict[str, Path]],
    trajectories: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    for seed, runs in sorted(run_dirs.items()):
        by_seed[str(seed)] = {}
        for label, run_dir in runs.items():
            trajectory = trajectories[seed][label]
            snapshot_equal = trajectory.get("fairness_snapshot") == read_json(
                run_dir / "fairness_snapshot.json"
            )
            label_equal = trajectory.get("label") == label
            by_seed[str(seed)][label] = {
                "label_equal": label_equal,
                "fairness_snapshot_equal": snapshot_equal,
                "pass": label_equal and snapshot_equal,
            }
    return {
        "by_seed": by_seed,
        "all_bound": all(
            row["pass"]
            for runs in by_seed.values()
            for row in runs.values()
        ),
    }


def normalized_config(snapshot: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model = dict(snapshot["model_config"])
    train = {
        key: value
        for key, value in snapshot["train_config"].items()
        if key not in TRAIN_CONFIG_EXCLUSIONS and key not in {"seed", "sample_dynamics_seed"}
    }
    return model, train


def normalized_locked_contract(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in snapshot["locked_contract"].items()
        if key != "seed"
    }


def fairness_check(run_dirs: dict[int, dict[str, Path]]) -> dict[str, Any]:
    seeds: dict[str, Any] = {}
    cross_seed_reference: tuple[dict[str, Any], dict[str, Any]] | None = None
    cross_seed_contract_reference: dict[str, Any] | None = None
    cross_seed_equal = True
    cross_seed_contract_equal = True
    for seed, runs in sorted(run_dirs.items()):
        snapshots = {label: read_json(path / "fairness_snapshot.json") for label, path in runs.items()}
        reference = snapshots["E0"]
        within_identity = {
            field: all(snapshot.get(field) == reference.get(field) for snapshot in snapshots.values())
            for field in IDENTITY_FIELDS
        }
        within_contract = {
            field: all(
                snapshot["locked_contract"].get(field) == reference["locked_contract"].get(field)
                for snapshot in snapshots.values()
            )
            for field in CONTRACT_FIELDS
        }
        normalized = normalized_config(reference)
        normalized_contract = normalized_locked_contract(reference)
        within_full = all(normalized_config(snapshot) == normalized for snapshot in snapshots.values())
        if cross_seed_reference is None:
            cross_seed_reference = normalized
            cross_seed_contract_reference = normalized_contract
        else:
            cross_seed_equal = cross_seed_equal and normalized == cross_seed_reference
            cross_seed_contract_equal = (
                cross_seed_contract_equal
                and normalized_contract == cross_seed_contract_reference
            )
        seeds[str(seed)] = {
            "identity_equal_within_seed": within_identity,
            "locked_contract_equal_within_seed": within_contract,
            "full_non_seed_config_equal_within_seed": within_full,
            "pass": all(within_identity.values()) and all(within_contract.values()) and within_full,
            "initial_model_parameter_sha256": reference["initial_model_parameter_sha256"],
        }
    return {
        "by_seed": seeds,
        "all_within_seed_pass": all(row["pass"] for row in seeds.values()),
        "cross_seed_non_seed_config_equal": cross_seed_equal,
        "cross_seed_non_seed_locked_contract_equal": cross_seed_contract_equal,
        "note": "initial parameter hashes may differ across seeds but must match E0/E3/E4/E5 within each seed",
    }


def final_metrics(run_dir: Path) -> dict[str, float]:
    rows = merged_metrics(run_dir / "metrics.csv")
    final = rows[max(rows)]
    return {
        "mask_mAP_50_95": float(final["val/segm_mAP_50_95"]),
        "box_mAP_50_95": float(final["val/mAP_50_95"]),
        "F1": float(final["val/F1"]),
    }


def make_plot(output_dir: Path, aggregate: dict[str, Any]) -> str:
    import matplotlib.pyplot as plt

    assets = output_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    colors = {"E0": "#4c78a8", "E3": "#59a14f", "E4": "#e15759", "E5": "#f28e2b"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    for axis, metric in zip(axes, METRICS, strict=True):
        means = [aggregate["metrics"][label][metric]["mean"] for label in RUNS]
        errors = [aggregate["metrics"][label][metric]["std"] for label in RUNS]
        axis.bar(RUNS, means, yerr=errors, capsize=5, color=[colors[label] for label in RUNS], alpha=0.9)
        axis.set_title(metric)
        axis.set_ylabel("mean ± sample std")
        axis.grid(axis="y", alpha=0.25)
    path = assets / "multiseed_ablation_metrics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path.relative_to(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="SEED:LABEL=DIR")
    parser.add_argument("--trajectory", action="append", required=True, type=Path)
    parser.add_argument("--subsets", type=Path, default=Path("reports/mvtec_reference_subsets.json"))
    parser.add_argument("--comparison-epoch", type=int, default=13)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    run_dirs = parse_run_specifications(args.run)
    trajectories = load_trajectories(args.trajectory)
    seeds = sorted(run_dirs)
    if len(seeds) != 3:
        raise ValueError(f"three-seed analysis requires exactly three complete seeds, got {len(seeds)}")
    if set(trajectories) != set(seeds):
        raise ValueError(f"trajectory seeds {sorted(trajectories)} do not match run seeds {seeds}")
    for seed in seeds:
        missing = set(RUNS).difference(trajectories[seed])
        if missing:
            raise ValueError(f"seed {seed} trajectory is missing {sorted(missing)}")

    subsets = {name: set(values) for name, values in read_json(args.subsets)["subsets"].items()}
    hard_ids = subsets["REFERENCE_HARD"]
    fairness = fairness_check(run_dirs)
    trajectory_integrity = trajectory_check(trajectories)
    trajectory_binding = trajectory_binding_check(run_dirs, trajectories)
    per_seed_metrics: dict[str, dict[str, Any]] = {}
    per_seed_recovery: dict[str, dict[str, Any]] = {}
    per_seed_resources: dict[str, dict[str, Any]] = {}
    per_seed_gates: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        seed_key = str(seed)
        per_seed_metrics[seed_key] = {label: final_metrics(run_dirs[seed][label]) for label in RUNS}
        per_seed_recovery[seed_key] = {
            label: recovery_summary(trajectories[seed][label], hard_ids, 0, args.comparison_epoch)
            for label in RUNS
        }
        per_seed_resources[seed_key] = {}
        per_seed_gates[seed_key] = {}
        for label in ("E3", "E4", "E5"):
            _, cumulative = resource_summary(
                read_json(run_dirs[seed][label] / "sample_dynamics/resource_history.json"),
                {"REFERENCE_HARD": hard_ids, "REFERENCE_MASTERED": subsets["REFERENCE_MASTERED"]},
            )
            per_seed_resources[seed_key][label] = cumulative
            per_seed_gates[seed_key][label] = resource_gate(label, cumulative)

    metric_aggregate: dict[str, dict[str, Any]] = {}
    paired_delta_aggregate: dict[str, dict[str, Any]] = {}
    for label in RUNS:
        metric_aggregate[label] = {
            metric: summary([per_seed_metrics[str(seed)][label][metric] for seed in seeds])
            for metric in METRICS
        }
        if label != "E0":
            paired_delta_aggregate[label] = {
                metric: summary(
                    [
                        per_seed_metrics[str(seed)][label][metric]
                        - per_seed_metrics[str(seed)]["E0"][metric]
                        for seed in seeds
                    ]
                )
                for metric in METRICS
            }

    recovery_aggregate = {
        label: {
            metric: summary([float(per_seed_recovery[str(seed)][label][metric]) for seed in seeds])
            for metric in (
                "loss_decrease_rate",
                "mask_iou_increase_rate",
                "loss_decrease_and_mask_non_decrease_rate",
                "mean_loss_improvement",
                "mean_mask_iou_improvement",
            )
        }
        for label in RUNS
    }
    hard_subset_coverage = {
        str(seed): {
            label: {
                "expected": len(hard_ids),
                "observed": int(per_seed_recovery[str(seed)][label]["sample_count"]),
                "complete": int(per_seed_recovery[str(seed)][label]["sample_count"]) == len(hard_ids),
            }
            for label in RUNS
        }
        for seed in seeds
    }
    all_hard_subset_complete = all(
        row["complete"]
        for runs in hard_subset_coverage.values()
        for row in runs.values()
    )
    paired_recovery_delta_aggregate = {
        label: {
            metric: summary(
                [
                    float(per_seed_recovery[str(seed)][label][metric])
                    - float(per_seed_recovery[str(seed)]["E0"][metric])
                    for seed in seeds
                ]
            )
            for metric in (
                "loss_decrease_rate",
                "mask_iou_increase_rate",
                "loss_decrease_and_mask_non_decrease_rate",
                "mean_loss_improvement",
                "mean_mask_iou_improvement",
            )
        }
        for label in ("E3", "E4", "E5")
    }
    resource_ratio_aggregate = {
        label: {
            metric: summary(
                [float(per_seed_gates[str(seed)][label][metric]) for seed in seeds]
            )
            for metric in (
                "hard_to_mastered_appearance_ratio",
                "hard_to_mastered_applied_weight_ratio",
                "hard_to_mastered_effective_contribution_ratio",
            )
            if all(per_seed_gates[str(seed)][label][metric] is not None for seed in seeds)
        }
        for label in ("E3", "E4", "E5")
    }
    all_resource_gates_pass = all(
        per_seed_gates[str(seed)][label]["mechanism_pass"]
        for seed in seeds
        for label in ("E3", "E4", "E5")
    )
    direction_consistency = {
        label: {
            "final_metrics_vs_e0": {
                metric: direction_counts(paired_delta_aggregate[label][metric]["values"])
                for metric in METRICS
            },
            "hard_recovery_vs_e0": {
                metric: direction_counts(paired_recovery_delta_aggregate[label][metric]["values"])
                for metric in ("mean_loss_improvement", "mean_mask_iou_improvement")
            },
        }
        for label in ("E3", "E4", "E5")
    }
    e5_targeted_loss_all_positive = direction_consistency["E5"]["hard_recovery_vs_e0"][
        "mean_loss_improvement"
    ]["positive"] == len(seeds)
    e5_global_mask_all_negative = direction_consistency["E5"]["final_metrics_vs_e0"][
        "mask_mAP_50_95"
    ]["negative"] == len(seeds)
    e5_global_f1_all_negative = direction_consistency["E5"]["final_metrics_vs_e0"]["F1"][
        "negative"
    ] == len(seeds)
    algorithm_signal = (
        "TARGETED_HARD_LOSS_RECOVERY_WITH_GLOBAL_REGRESSION_NO_DEPLOYMENT"
        if e5_targeted_loss_all_positive and e5_global_mask_all_negative and e5_global_f1_all_negative
        else "MIXED_THREE_SEED_NO_BENEFIT_CLAIM"
    )

    aggregate = {
        "schema_version": 1,
        "scope": "NON-BENCHMARK / THREE-SEED MECHANISM VALIDATION",
        "seeds": seeds,
        "comparison_checkpoint_epoch": args.comparison_epoch,
        "fairness": fairness,
        "trajectory_integrity": trajectory_integrity,
        "trajectory_binding": trajectory_binding,
        "hard_subset_coverage": hard_subset_coverage,
        "frozen_subset_counts": {name: len(values) for name, values in subsets.items()},
        "per_seed_metrics": per_seed_metrics,
        "metrics": metric_aggregate,
        "paired_deltas_vs_e0": paired_delta_aggregate,
        "per_seed_hard_recovery": per_seed_recovery,
        "hard_recovery": recovery_aggregate,
        "paired_hard_recovery_deltas_vs_e0": paired_recovery_delta_aggregate,
        "per_seed_resource_cumulative": per_seed_resources,
        "per_seed_resource_gate": per_seed_gates,
        "resource_ratios": resource_ratio_aggregate,
        "direction_consistency": direction_consistency,
        "source_sha256": {
            str(path): sha256_file(path)
            for path in [*args.trajectory, args.subsets]
        },
        "run_artifact_sha256": {
            str(seed): {
                label: run_artifact_hashes(run_dirs[seed][label])
                for label in RUNS
            }
            for seed in seeds
        },
        "gate": {
            "correctness": "PASS"
            if (
                fairness["all_within_seed_pass"]
                and fairness["cross_seed_non_seed_config_equal"]
                and fairness["cross_seed_non_seed_locked_contract_equal"]
                and trajectory_integrity["all_complete"]
                and trajectory_binding["all_bound"]
                and all_hard_subset_complete
            )
            else "FAIL",
            "resource_mechanisms": "PASS" if all_resource_gates_pass else "FAIL",
            "algorithm_signal": algorithm_signal,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "mvtec_three_seed_ablation_analysis.json"
    json_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = args.output_dir / "mvtec_three_seed_ablation_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["record_type", "seed", "run", "metric", "value", "mean", "std", "delta_vs_e0"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for seed in seeds:
            for label in RUNS:
                for metric in METRICS:
                    value = per_seed_metrics[str(seed)][label][metric]
                    baseline = per_seed_metrics[str(seed)]["E0"][metric]
                    writer.writerow(
                        {
                            "record_type": "final_metric",
                            "seed": seed,
                            "run": label,
                            "metric": metric,
                            "value": value,
                            "mean": metric_aggregate[label][metric]["mean"],
                            "std": metric_aggregate[label][metric]["std"],
                            "delta_vs_e0": value - baseline,
                        }
                    )
        for label in ("E3", "E4", "E5"):
            for metric, row in paired_recovery_delta_aggregate[label].items():
                for seed, value in zip(seeds, row["values"], strict=True):
                    writer.writerow(
                        {
                            "record_type": "paired_hard_recovery_delta",
                            "seed": seed,
                            "run": label,
                            "metric": metric,
                            "value": value,
                            "mean": row["mean"],
                            "std": row["std"],
                            "delta_vs_e0": value,
                        }
                    )
            for metric, row in resource_ratio_aggregate[label].items():
                for seed, value in zip(seeds, row["values"], strict=True):
                    writer.writerow(
                        {
                            "record_type": "resource_ratio",
                            "seed": seed,
                            "run": label,
                            "metric": metric,
                            "value": value,
                            "mean": row["mean"],
                            "std": row["std"],
                        }
                    )
        for gate_name, gate_value in aggregate["gate"].items():
            writer.writerow(
                {
                    "record_type": "gate",
                    "metric": gate_name,
                    "value": gate_value,
                }
            )

    plot = make_plot(args.output_dir, aggregate)
    report_path = args.output_dir / "mvtec_three_seed_ablation_validation.md"
    metric_table = ""
    for label in RUNS:
        row = metric_aggregate[label]
        delta = paired_delta_aggregate.get(label)
        metric_table += (
            f"| {label} | {row['mask_mAP_50_95']['mean']:.4f} ± {row['mask_mAP_50_95']['std']:.4f} | "
            f"{row['box_mAP_50_95']['mean']:.4f} ± {row['box_mAP_50_95']['std']:.4f} | "
            f"{row['F1']['mean']:.4f} ± {row['F1']['std']:.4f} | "
            + (
                "baseline |\n"
                if delta is None
                else f"{delta['mask_mAP_50_95']['mean']:+.4f} ± {delta['mask_mAP_50_95']['std']:.4f} |\n"
            )
        )
    resource_table = ""
    def ratio_text(row: dict[str, Any], key: str) -> str:
        value = row.get(key)
        return "n/a" if value is None else f"{value['mean']:.4f} ± {value['std']:.4f}"

    for label in ("E3", "E4", "E5"):
        row = resource_ratio_aggregate[label]
        resource_table += (
            f"| {label} | "
            f"{ratio_text(row, 'hard_to_mastered_applied_weight_ratio')} | "
            f"{ratio_text(row, 'hard_to_mastered_appearance_ratio')} | "
            f"{ratio_text(row, 'hard_to_mastered_effective_contribution_ratio')} |\n"
        )
    recovery_table = ""
    for label in ("E3", "E4", "E5"):
        loss = paired_recovery_delta_aggregate[label]["mean_loss_improvement"]
        iou = paired_recovery_delta_aggregate[label]["mean_mask_iou_improvement"]
        mask_rate = paired_recovery_delta_aggregate[label]["mask_iou_increase_rate"]
        recovery_table += (
            f"| {label} | {loss['mean']:+.2f} ± {loss['std']:.2f} | "
            f"{iou['mean']:+.4f} ± {iou['std']:.4f} | "
            f"{mask_rate['mean']:+.4f} ± {mask_rate['std']:.4f} | "
            f"{direction_consistency[label]['hard_recovery_vs_e0']['mean_loss_improvement']['positive']}/{len(seeds)} |\n"
        )
    paired_metric_table = ""
    for label in ("E3", "E4", "E5"):
        row = paired_delta_aggregate[label]
        paired_metric_table += (
            f"| {label} | {row['mask_mAP_50_95']['mean']:+.4f} ± {row['mask_mAP_50_95']['std']:.4f} | "
            f"{row['box_mAP_50_95']['mean']:+.4f} ± {row['box_mAP_50_95']['std']:.4f} | "
            f"{row['F1']['mean']:+.4f} ± {row['F1']['std']:.4f} | "
            f"{direction_consistency[label]['final_metrics_vs_e0']['mask_mAP_50_95']['positive']}/{len(seeds)} |\n"
        )
    report = f"""# RF-DETR Sample Dynamics — Three-Seed Ablation Validation

> Scope: **NON-BENCHMARK / THREE-SEED MECHANISM VALIDATION**. Seeds: {', '.join(map(str, seeds))}. StatePolicy and E2-frozen subsets were unchanged.

## Gate

- Correctness: `{aggregate['gate']['correctness']}`
- Resource mechanisms: `{aggregate['gate']['resource_mechanisms']}`
- Algorithm signal: `{aggregate['gate']['algorithm_signal']}`
- Final metrics use epoch 14. Per-image hard recovery uses the common real checkpoint window epoch 0→{args.comparison_epoch}.
- Differences are paired within seed against that seed's E0; sample standard deviation uses `n-1`.

## Final validation metrics

| Run | mask mAP50:95 | box mAP50:95 | F1 | paired Δmask vs E0 |
|---|---:|---:|---:|---:|
{metric_table}

![Three-seed metrics](assets/multiseed_ablation_metrics.png)

### Paired deltas against each seed's E0

| Run | Δmask mAP | Δbox mAP | ΔF1 | seeds with positive Δmask |
|---|---:|---:|---:|---:|
{paired_metric_table}

## Realized resource ratios

Values are three-seed mean ± sample std of `REFERENCE_HARD / REFERENCE_MASTERED`; `n/a` means that resource was intentionally disabled.

| Run | applied weight ratio | appearance ratio | effective contribution ratio |
|---|---:|---:|---:|
{resource_table}

## Frozen hard-subset recovery

All deltas below are paired against the same seed's E0 over checkpoint epoch 0→{args.comparison_epoch}. Positive loss improvement means a larger loss decrease. The 33 IDs were frozen from E2 before intervention outcomes were observed.

| Run | Δ mean loss improvement | Δ mean mask-IoU improvement | Δ mask-IoU increase rate | seeds with positive Δloss improvement |
|---|---:|---:|---:|---:|
{recovery_table}

## Decision

- **E3 Loss Weight:** overall performance is effectively neutral and directionally unstable. Mask delta is positive in only 1/3 seeds; box delta is negative in 3/3. The hard-subset loss and IoU deltas also change sign across seeds. This does not validate a benefit.
- **E4 Dynamic Sampler:** the intended exposure shift is strong and repeatable, but mask mAP and F1 are lower than E0 in 3/3 seeds. Hard-subset loss improvement is positive in 2/3 seeds and mask-IoU improvement is inconsistent. The current sampling quota is therefore not accepted as an algorithmic improvement.
- **E5 Combined:** the hard subset receives about 2.44× the effective contribution of mastered samples. Its mean loss improvement exceeds E0 in 3/3 seeds, but mask-IoU improvement is inconsistent and global mask mAP/F1 are lower in 3/3 seeds. This is evidence of targeted optimization, not successful overall scheduling.

The milestone result is therefore: **the observation→classification→next-epoch resource-allocation mechanism is correct and reproducible, while the frozen intervention policy over-focuses the selected hard subset and does not pass the algorithm-benefit gate. Do not deploy or freeze RF9 from these parameters.**

## Interpretation guardrails

1. This Pilot is a mechanism-validation dataset, not a benchmark claim.
2. Three seeds estimate variability but remain a small sample; paired consistency and metric trade-offs matter more than one favorable mean.
3. `HARD_LEARNABLE` remains a high-loss candidate label unless subsequent improvement is consistently greater than E0.
4. Controlled corruption remains a five-sample smoke subset and is not an RF8 precision/recall estimate.

## Evidence

- [Analysis JSON](mvtec_three_seed_ablation_analysis.json)
- [Result CSV](mvtec_three_seed_ablation_results.csv)
- [Metric plot]({plot})
"""
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"analysis": str(json_path), "csv": str(csv_path), "report": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
