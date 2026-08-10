#!/usr/bin/env python3
"""Prepare a deterministic 50-sample RF4 manual-review packet.

Selection is frozen from the E2 analysis before any intervention.  The script
does not assign review labels; a reviewer records ``reasonable``,
``questionable`` or ``wrong`` in a separate decisions JSON and this script
merges those decisions into the durable CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


AUDIT_SIZE = 50
REVIEW_LABELS = {"reasonable", "questionable", "wrong"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_ids(analysis: dict[str, Any]) -> list[str]:
    """Select 5 corruption, 20 natural-hard, 15 mastered and 10 other IDs."""
    subsets = analysis["reference_subsets"]["subsets"]
    all_train = list(analysis["per_sample"])
    selected: list[str] = []

    def add(values: list[str], limit: int | None = None) -> None:
        added = 0
        for sample_id in values:
            if sample_id not in selected:
                selected.append(sample_id)
                added += 1
                if limit is not None and added >= limit:
                    break

    add(sorted(subsets["CONTROLLED_CORRUPTION"]))
    add(sorted(subsets["CLEAN_NATURAL_HARD"]), 20)
    add(sorted(subsets["REFERENCE_MASTERED"]), 15)
    add(sorted(all_train), AUDIT_SIZE - len(selected))
    if len(selected) != AUDIT_SIZE:
        raise ValueError(f"deterministic audit selection expected {AUDIT_SIZE}, got {len(selected)}")
    return selected


def _source_path(record: dict[str, Any], pilot_dir: Path, source_root: Path) -> Path:
    source = source_root / str(record["source_image"])
    if source.is_file():
        return source
    pilot = pilot_dir / "train" / str(record["file_name"])
    if pilot.is_file():
        return pilot
    raise FileNotFoundError(f"cannot resolve image for {record['stable_sample_id']}: {source} / {pilot}")


def _make_contact_sheets(rows: list[dict[str, Any]], output_dir: Path, pilot_dir: Path, source_root: Path) -> list[str]:
    """Render compact image/mask sheets for visual inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    sheet_paths: list[str] = []
    tile_width, tile_height, label_height = 260, 260, 42
    for sheet_index in range(0, len(rows), 10):
        chunk = rows[sheet_index : sheet_index + 10]
        sheet = Image.new("RGB", (tile_width * 5, (tile_height + label_height) * 2), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(chunk):
            image = Image.open(_source_path(row["manifest"], pilot_dir, source_root)).convert("RGB")
            image.thumbnail((tile_width - 8, tile_height - 8))
            x = (offset % 5) * tile_width + (tile_width - image.width) // 2
            y = (offset // 5) * (tile_height + label_height) + (tile_height - image.height) // 2
            sheet.paste(image, (x, y))
            label = f"{offset + sheet_index:02d} {row['category']}/{row['defect_type']}\n{row['final_state']} {row['corruption_type'] or 'clean'}"
            draw.multiline_text(
                ((offset % 5) * tile_width + 4, (offset // 5) * (tile_height + label_height) + tile_height + 2),
                label,
                fill="black",
                font=font,
            )
        path = output_dir / f"sheet_{sheet_index // 10:02d}.png"
        sheet.save(path)
        sheet_paths.append(str(path))
    return sheet_paths


def build(args: argparse.Namespace) -> dict[str, Any]:
    analysis = _read_json(args.analysis_json)
    manifest = _read_json(args.manifest_json)
    decisions = _read_json(args.decisions_json) if args.decisions_json and args.decisions_json.is_file() else {}
    records = {str(record["stable_sample_id"]): dict(record) for record in manifest["records"]}
    selected_ids = _select_ids(analysis)
    per_sample = analysis["per_sample"]
    corruption_ids = set(analysis["reference_subsets"]["subsets"]["CONTROLLED_CORRUPTION"])
    rows: list[dict[str, Any]] = []
    for sample_id in selected_ids:
        if sample_id not in records or sample_id not in per_sample:
            raise ValueError(f"selected audit sample is absent from manifest/analysis: {sample_id}")
        record = records[sample_id]
        trajectory = per_sample[sample_id]
        final_probe = trajectory["probe"][-1] if "probe" in trajectory else {}
        rows.append(
            {
                "sample_id": sample_id,
                "manifest": record,
                "category": record.get("category"),
                "defect_type": record.get("defect_type"),
                "source_image": record.get("source_image"),
                "kind": record.get("kind"),
                "label_status": record.get("label_status"),
                "corruption_type": record.get("corruption_type"),
                "subset": (
                    "CONTROLLED_CORRUPTION" if sample_id in corruption_ids
                    else "CLEAN_NATURAL_HARD" if sample_id in analysis["reference_subsets"]["subsets"]["CLEAN_NATURAL_HARD"]
                    else "REFERENCE_MASTERED" if sample_id in analysis["reference_subsets"]["subsets"]["REFERENCE_MASTERED"]
                    else "OTHER_TRAIN"
                ),
                "final_state": trajectory["dynamics_state"][-1],
                "final_instant_bucket": trajectory["instant_bucket"][-1],
                "hard_loss_epochs": sum(bucket == "HARD" for bucket in trajectory["instant_bucket"]),
                "hard_state_epochs": sum(state == "HARD_LEARNABLE" for state in trajectory["dynamics_state"]),
                "final_loss": trajectory["loss"][-1],
                "mean_loss": sum(trajectory["loss"]) / len(trajectory["loss"]),
                "final_mask_iou": trajectory["matched_mask_iou"][-1],
                "final_fn": final_probe.get("fn", trajectory["fn"][-1]),
                "final_fp": final_probe.get("fp", trajectory["fp"][-1]),
                "final_class_error": final_probe.get("class_error", trajectory["class_error"][-1]),
                "frozen_conflict_epochs": sum(bool(value) for value in trajectory["frozen_probe_conflict"]),
                "analysis_conflict_epochs": sum(bool(value) for value in trajectory["probe_conflict"]),
                "review_label": decisions.get(sample_id, {}).get("review_label", ""),
                "review_reason": decisions.get(sample_id, {}).get("review_reason", ""),
                "reviewer": decisions.get(sample_id, {}).get("reviewer", ""),
            }
        )
    invalid = [row["sample_id"] for row in rows if row["review_label"] not in {"", *REVIEW_LABELS}]
    if invalid:
        raise ValueError(f"invalid review labels: {invalid}")
    sheet_paths = _make_contact_sheets(rows, args.contact_sheet_dir, args.pilot_dir, args.source_root)
    fieldnames = [key for key in rows[0] if key != "manifest"]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    label_counts = Counter(row["review_label"] or "PENDING" for row in rows)
    summary = {
        "schema_version": 1,
        "selection_rule": "5 controlled corruption + 20 CLEAN_NATURAL_HARD + 15 REFERENCE_MASTERED + deterministic other train fill",
        "sample_count": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "pending_count": label_counts.get("PENDING", 0),
        "csv": str(args.output_csv),
        "contact_sheets": sheet_paths,
        "decisions_json": str(args.decisions_json) if args.decisions_json else None,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("/home/inspur1/data/MVTec-AD"))
    parser.add_argument("--decisions-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--contact-sheet-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = build(parse_args())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
