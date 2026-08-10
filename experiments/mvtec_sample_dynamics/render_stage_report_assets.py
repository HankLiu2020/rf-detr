"""Render static assets for the RF-DETR Sample Dynamics stage report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


STATE_ORDER = ("LEARNING", "MASTERED", "HARD_LEARNABLE", "SUSPECT")
STATE_STYLE = {
    "LEARNING": {"color": "#2F6BFF", "linestyle": "-", "marker": "o"},
    "MASTERED": {"color": "#7A8B3A", "linestyle": "--", "marker": "s"},
    "HARD_LEARNABLE": {"color": "#E07A2D", "linestyle": "-", "marker": "^"},
    "SUSPECT": {"color": "#B04A7A", "linestyle": ":", "marker": "D"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path("reports/mvtec_rf4_longitudinal_analysis.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/assets/rf4_sample_state_trajectory.svg"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/rf_detr_sample_dynamics_stage_report_data.json"),
    )
    return parser.parse_args()


def correlation_summary(analysis: dict) -> list[dict]:
    rows = []
    for signal_key, signal_block in analysis["correlations"].items():
        for target_key, target_block in signal_block["targets"].items():
            for horizon_key in ("horizon_1", "horizon_2"):
                horizon = target_block[horizon_key]
                rows.append(
                    {
                        "signal_key": signal_key,
                        "signal": signal_block["signal"],
                        "target_key": target_key,
                        "target": target_block["label"],
                        "horizon": int(horizon_key.rsplit("_", 1)[1]),
                        "defined_windows": horizon["pearson"]["defined_windows"],
                        "mean_pearson": horizon["pearson"]["mean"],
                        "mean_spearman": horizon["spearman"]["mean"],
                    }
                )
    return rows


def stability_summary(analysis: dict) -> dict:
    fields = (
        "epoch_count",
        "sample_count",
        "total_transitions",
        "mean_transitions_per_sample",
        "unchanged_sample_count",
        "unchanged_sample_fraction",
        "mean_unique_labels_per_sample",
    )
    return {
        name: {field: values[field] for field in fields}
        for name, values in analysis["stability"].items()
    }


def write_summary(analysis: dict, output: Path) -> None:
    hard = analysis["hard_learnable_natural_improvement"]
    summary = {
        "schema_version": 1,
        "title": "RF-DETR Sample Dynamics 阶段性实验汇报",
        "scope": "NON-BENCHMARK / RF4 MECHANISM VALIDATION",
        "gate_status": analysis["gate_status"],
        "source_analysis": "reports/mvtec_rf4_longitudinal_analysis.json",
        "contract": {
            "epoch_count": analysis["epoch_count"],
            "sample_count": analysis["sample_count"],
            "manifest_sha256": analysis["manifest_sha256"],
            "state_sha256": analysis["state_sha256"],
            **analysis["contract"],
        },
        "policy": analysis["policy"],
        "epoch_summary": analysis["epoch_summary"],
        "stability": stability_summary(analysis),
        "future_predictivity": correlation_summary(analysis),
        "hard_learnable_natural_improvement": {
            "interpretation": hard["interpretation"],
            "excluded_controlled_corruption_event_count": hard[
                "excluded_controlled_corruption_event_count"
            ],
            "hard_events_excluding_last_epoch": hard["hard_events_excluding_last_epoch"],
            "unique_samples_with_hard_event": hard["unique_samples_with_hard_event"],
            "events_with_next_loss_improvement": hard[
                "events_with_next_loss_improvement"
            ],
            "events_with_any_future_loss_improvement": hard[
                "events_with_any_future_loss_improvement"
            ],
            "events_with_next_mask_iou_improvement": hard[
                "events_with_next_mask_iou_improvement"
            ],
            "events_with_any_future_mask_iou_improvement": hard[
                "events_with_any_future_mask_iou_improvement"
            ],
            "first_entry_sample_count": hard["first_entry_sample_count"],
            "first_entry_natural_improvement_by_horizon": hard[
                "first_entry_natural_improvement_by_horizon"
            ],
            "final_epoch_hard_learnable_count_excluding_controlled_corruption": hard[
                "final_epoch_hard_learnable_count"
            ],
        },
        "empty_gt_frozen_conflict_observations": analysis[
            "empty_gt_frozen_conflict_observations"
        ],
        "controlled_corruption": {
            "sample_count": analysis["controlled_corruption"]["sample_count"],
            "rf8_precision_recall_claimed": analysis["controlled_corruption"][
                "rf8_precision_recall_claimed"
            ],
            "samples": analysis["controlled_corruption"]["samples"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    epochs = analysis["epoch_summary"]

    if len(epochs) != analysis["epoch_count"]:
        raise ValueError("epoch_summary length does not match epoch_count")

    write_summary(analysis, args.summary_output)

    mpl.rcParams["svg.hashsalt"] = "rfdetr-sample-dynamics-stage-report"
    fig, ax = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    for state in STATE_ORDER:
        counts = [row["dynamics_state_counts"].get(state, 0) for row in epochs]
        style = STATE_STYLE[state]
        ax.plot(
            [row["epoch"] for row in epochs],
            counts,
            label=state,
            linewidth=2.4,
            markersize=5,
            **style,
        )

    ax.set_title("RF4 E2 Observe-only: Sample State Trajectory", loc="left", weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Sample count")
    ax.set_xticks([row["epoch"] for row in epochs])
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D9DEE7", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncols=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.text(
        0.01,
        0.01,
        "128 fixed samples, 15 epochs, seed 20260810. Counts sum to 128 per epoch. "
        "Source: reports/mvtec_rf4_longitudinal_analysis.json",
        fontsize=8.5,
        color="#4B5563",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_format = args.output.suffix.lstrip(".").lower() or "svg"
    fig.savefig(
        args.output,
        format=output_format,
        facecolor="white",
        dpi=160,
        metadata={"Date": None},
    )
    plt.close(fig)
    if output_format == "svg":
        svg = args.output.read_text(encoding="utf-8")
        args.output.write_text(
            "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
