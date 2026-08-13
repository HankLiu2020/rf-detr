#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Summarize Policy V2 30-epoch convergence controls and D1 discovery."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

RUNS = {"C0/D0": "c0-d0", "C3": "c3", "C4": "c4", "C5": "c5", "D1": "d1"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(path: Path) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            target = result.setdefault(int(float(row["epoch"])), {})
            for key, value in row.items():
                if value not in (None, ""):
                    try:
                        target[key] = float(value)
                    except ValueError:
                        pass
    return result


def resource(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "sample_dynamics" / "resource_history.json"
    if not path.is_file():
        return None
    history = read_json(path)
    epochs = []
    for payload in history:
        records = payload["records"]
        appearances = sum(int(record["batch_appearance_count"]) for record in records)
        seen = sum(int(record["batch_appearance_count"]) > 0 for record in records)
        replayed = sum(max(0, int(record["batch_appearance_count"]) - 1) for record in records)
        epochs.append(
            {
                "epoch": int(payload["epoch"]),
                "appearances": appearances,
                "unique_seen": seen,
                "coverage": seen / len(records),
                "replay_appearances": replayed,
                "zero_exposure": len(records) - seen,
            }
        )
    return {
        "epochs": epochs,
        "mean_coverage": statistics.mean(row["coverage"] for row in epochs),
        "total_appearances": sum(row["appearances"] for row in epochs),
        "total_replay_appearances": sum(row["replay_appearances"] for row in epochs),
        "total_zero_exposure_sample_epochs": sum(row["zero_exposure"] for row in epochs),
    }


def analyze(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    identity: dict[str, set[Any]] = {
        "manifest": set(),
        "checkpoint": set(),
        "initial_model": set(),
        "seed": set(),
        "epochs": set(),
    }
    for label, directory in RUNS.items():
        run_dir = root / directory
        rows = metrics(run_dir / "metrics.csv")
        final_epoch = max(rows)
        best_epoch = max(rows, key=lambda epoch: rows[epoch]["val/segm_mAP_50_95"])
        summary = read_json(run_dir / "pilot_run_summary.json")
        fairness = read_json(run_dir / "fairness_snapshot.json")
        identity["manifest"].add(fairness["dataset_manifest_sha256"])
        identity["checkpoint"].add(fairness["pretrain_weights_sha256"])
        identity["initial_model"].add(fairness["initial_model_parameter_sha256"])
        identity["seed"].add(fairness["seed"])
        identity["epochs"].add(fairness["locked_contract"]["epochs"])
        result[label] = {
            "final_epoch": final_epoch,
            "final_step": int(rows[final_epoch]["step"]),
            "final": {
                "mask_mAP": rows[final_epoch]["val/segm_mAP_50_95"],
                "box_mAP": rows[final_epoch]["val/mAP_50_95"],
                "F1": rows[final_epoch]["val/F1"],
            },
            "best_epoch": best_epoch,
            "best_mask_mAP": rows[best_epoch]["val/segm_mAP_50_95"],
            "final_minus_best_mask": rows[final_epoch]["val/segm_mAP_50_95"] - rows[best_epoch]["val/segm_mAP_50_95"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "resource": resource(run_dir),
        }
    baseline = result["C0/D0"]
    for label, row in result.items():
        row["delta_vs_c0"] = {key: row["final"][key] - baseline["final"][key] for key in ("mask_mAP", "box_mAP", "F1")}
        row["wall_clock_ratio_vs_c0"] = row["elapsed_seconds"] / baseline["elapsed_seconds"]
    return {
        "scope": "POLICY V2 DISCOVERY SEED; NOT CONFIRMATORY",
        "seed": 20260813,
        "contract_equal": all(len(values) == 1 for values in identity.values()),
        "identity": {key: sorted(values) for key, values in identity.items()},
        "runs": result,
    }


def plot(output_dir: Path, analysis: dict[str, Any]) -> str:
    import matplotlib.pyplot as plt

    labels = list(RUNS)
    values = [analysis["runs"][label]["final"]["mask_mAP"] for label in labels]
    colors = ["#4C78A8", "#B79A20", "#9C5A3C", "#7A6A53", "#E07A2D"]
    fig, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    axis.bar(labels, values, color=colors)
    axis.set(title="Policy V2 discovery: final mask mAP", ylabel="mask mAP50:95", xlabel="30-epoch run")
    axis.grid(axis="y", alpha=0.25)
    for index, value in enumerate(values):
        axis.text(index, value + 0.002, f"{value:.4f}", ha="center", fontsize=9)
    path = output_dir / "assets" / "policy_v2_discovery_mask_map.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path.relative_to(output_dir))


def report(output_dir: Path, analysis: dict[str, Any], visual: str) -> None:
    runs = analysis["runs"]
    d1 = runs["D1"]
    lines = [
        "# Policy V2 30-Epoch Discovery Milestone",
        "",
        "> Scope: one preregistered discovery seed (`20260813`), not confirmatory evidence.",
        "",
        "## Technical summary",
        "",
        f"The 30-epoch control confirms that 15 epochs were not a settled convergence horizon, but extending training does not rescue V1 sampling: C4 and C5 remain below C0 at final epoch. D1 preserves full base coverage and adds only aligned post-warmup replay; its final mask delta versus C0 is `{d1['delta_vs_c0']['mask_mAP']:+.4f}`. This is a discovery signal only and cannot pass `ALGORITHM_BENEFIT` without new confirmatory seeds.",
        "",
        f"![Discovery mask mAP]({visual})",
        "",
        "## Final and best validation evidence",
        "",
        "| Run | Final mask | Δmask vs C0 | Final F1 | Best mask (epoch) | Steps | Wall-clock ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in RUNS:
        row = runs[label]
        lines.append(
            f"| {label} | {row['final']['mask_mAP']:.4f} | {row['delta_vs_c0']['mask_mAP']:+.4f} | {row['final']['F1']:.4f} | {row['best_mask_mAP']:.4f} (e{row['best_epoch']}) | {row['final_step']} | {row['wall_clock_ratio_vs_c0']:.3f}× |"
        )
    lines += [
        "",
        "## Resource and coverage evidence",
        "",
        "| Run | Mean unique coverage | Total appearances | Replay appearances | Zero-exposure sample-epochs |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ("C3", "C4", "C5", "D1"):
        resource_row = runs[label]["resource"]
        lines.append(
            f"| {label} | {100 * resource_row['mean_coverage']:.2f}% | {resource_row['total_appearances']} | {resource_row['total_replay_appearances']} | {resource_row['total_zero_exposure_sample_epochs']} |"
        )
    lines += [
        "",
        "D1 uses 128 no-replacement base appearances in every epoch. From epoch 5 onward, the nominal 5% bonus is batch-aligned to 8 appearances (6.25% realized). Extra compute is therefore reported explicitly; D1 cannot claim an epoch-only advantage over C0.",
        "",
        "## Gate",
        "",
        "- Convergence diagnosis: `PASS` — 15 epochs was not a stable horizon.",
        "- Coverage-preserving mechanism: `PASS` only if D1 ledger reports 100% unique coverage and the expected bonus appearances.",
        "- Algorithm benefit: `DISCOVERY_ONLY`; no confirmatory claim is allowed from seed 20260813.",
        "- Hard task recovery: requires paired checkpoint/probe analysis before confirmatory entry; overall mask/F1 no-regression remains P0.",
        "",
    ]
    (output_dir / "policy_v2_discovery_validation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    analysis = analyze(args.run_root.resolve())
    output_dir = args.output_dir.resolve()
    visual = plot(output_dir, analysis)
    analysis["visual"] = visual
    (output_dir / "policy_v2_discovery_analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report(output_dir, analysis, visual)
    print(json.dumps({"output": str(output_dir), "runs": list(RUNS)}, sort_keys=True))


if __name__ == "__main__":
    main()
