# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the experiment-only MVTec Sample Dynamics Pilot builder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.mvtec_sample_dynamics.prepare_pilot import (
    _apply_corruption,
    _mask_annotation,
    prepare_pilot,
)


def _write_image(path: Path, *, mask: np.ndarray | None = None) -> None:
    """Write one tiny RGB image or binary mask fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask is None:
        Image.new("RGB", (16, 12), color=(32, 64, 96)).save(path)
    else:
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def _build_source(root: Path) -> Path:
    """Create one miniature MVTec-compatible category."""
    category = root / "sample"
    for index in range(4):
        _write_image(category / "train" / "good" / f"{index:03d}.png")
    for index in range(3):
        _write_image(category / "test" / "good" / f"{index:03d}.png")
    for defect_type in ("crack", "poke"):
        for index in range(4):
            image = category / "test" / defect_type / f"{index:03d}.png"
            _write_image(image)
            mask = np.zeros((12, 16), dtype=bool)
            mask[2:5, 2:6] = True
            if index % 2:
                mask[8:10, 11:14] = True
            _write_image(category / "ground_truth" / defect_type / f"{index:03d}_mask.png", mask=mask)
    return root


def _decode_uncompressed_rle(segmentation: dict[str, object]) -> np.ndarray:
    """Decode the small JSON RLE fixture without requiring pycocotools."""
    height, width = segmentation["size"]  # type: ignore[misc]
    values: list[int] = []
    bit = 0
    for count in segmentation["counts"]:  # type: ignore[union-attr]
        values.extend([bit] * int(count))
        bit = 1 - bit
    return np.asarray(values, dtype=bool).reshape((int(height), int(width)), order="F")


def test_union_mask_annotation_round_trips_disconnected_regions() -> None:
    """One semantic MVTec mask stays one COCO annotation despite multiple components."""
    mask = np.zeros((8, 10), dtype=bool)
    mask[1:3, 2:5] = True
    mask[6:8, 8:10] = True

    annotation = _mask_annotation(mask)

    assert annotation is not None
    assert annotation["area"] == int(mask.sum())
    assert annotation["bbox"] == [2, 1, 8, 7]
    assert np.array_equal(_decode_uncompressed_rle(annotation["segmentation"]), mask)


def test_drop_component_removes_only_smallest_component() -> None:
    """Controlled component omission preserves the pristine input mask."""
    pristine = np.zeros((8, 10), dtype=bool)
    pristine[1:5, 1:5] = True
    pristine[6:8, 8:10] = True

    corrupted, parameters = _apply_corruption(
        pristine,
        corruption_type="drop_component",
        sample_key="sample/test/crack/000.png",
        seed=7,
    )

    assert pristine.sum() == 20
    assert corrupted.sum() == 16
    assert parameters["removed_area"] == 4


def test_prepare_pilot_is_deterministic_and_keeps_splits_disjoint(tmp_path: Path) -> None:
    """Fixed inputs produce identical manifests, stable IDs, symlinks, and corruption ground truth."""
    source = _build_source(tmp_path / "source")
    first = tmp_path / "first"
    second = tmp_path / "second"
    kwargs = {
        "categories": ["sample"],
        "seed": 123,
        "train_per_defect": 2,
        "val_per_defect": 1,
        "train_normal_per_category": 2,
        "val_normal_per_category": 1,
        "corruption_rate": 0.5,
    }

    first_summary = prepare_pilot(source, first, **kwargs)
    second_summary = prepare_pilot(source, second, **kwargs)

    assert first_summary == second_summary
    assert first_summary["split_counts"] == {"train": 6, "valid": 3}
    assert first_summary["corrupted_train_samples"] == 2
    assert (first / "train" / "sample" / "test" / "crack" / "000.png").is_symlink() or any(
        path.is_symlink() for path in (first / "train").rglob("*.png")
    )
    first_manifest = json.loads((first / "split_manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "split_manifest.json").read_text(encoding="utf-8"))
    assert first_manifest == second_manifest
    train_sources = {record["source_image"] for record in first_manifest["records"] if record["split"] == "train"}
    val_sources = {record["source_image"] for record in first_manifest["records"] if record["split"] == "valid"}
    assert train_sources.isdisjoint(val_sources)
    assert all(
        record["stable_sample_id"].startswith("val:")
        for record in first_manifest["records"]
        if record["split"] == "valid"
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_pilot(source, first, **kwargs)
