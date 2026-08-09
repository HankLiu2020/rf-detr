# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for stable sample identifiers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from rfdetr.datasets.coco import CocoDetection
from rfdetr.datasets.sample_id import make_sample_id


def _write_coco_fixture(root: Path) -> tuple[Path, Path]:
    """Create a one-image COCO fixture.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     image_dir, annotation_file = _write_coco_fixture(Path(directory))
        ...     image_dir.name, annotation_file.name
        ('train', '_annotations.coco.json')
    """
    image_dir = root / "train"
    image_dir.mkdir()
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(image_dir / "nested-image.jpg")
    annotation_file = image_dir / "_annotations.coco.json"
    annotation_file.write_text(
        json.dumps(
            {
                "images": [{"id": 7, "file_name": "nested-image.jpg", "width": 8, "height": 6}],
                "annotations": [
                    {"id": 1, "image_id": 7, "category_id": 1, "bbox": [1, 1, 3, 2], "area": 6, "iscrowd": 0}
                ],
                "categories": [{"id": 1, "name": "object"}],
            }
        ),
        encoding="utf-8",
    )
    return image_dir, annotation_file


def test_make_sample_id_normalizes_split_and_path() -> None:
    """Identifiers are independent of host path separators and suffixes."""
    assert make_sample_id("train_0", 7, "images\\object.jpg") == "train:7:images/object.jpg"


def test_coco_dataset_preserves_sample_id_without_transforms(tmp_path: Path) -> None:
    """COCO conversion adds a stable ID that survives target conversion."""
    image_dir, annotation_file = _write_coco_fixture(tmp_path)
    dataset = CocoDetection(image_dir, annotation_file, transforms=None, split="train", remap_category_ids=True)

    _, target = dataset[0]

    assert target["sample_id"] == "train:7:nested-image.jpg"
    assert isinstance(target["image_id"], torch.Tensor)


@pytest.mark.parametrize(
    ("split", "image_id", "relative_path", "expected"),
    [("train", 1, "a.jpg", "train:1:a.jpg"), ("valid", 2, "folder/b.jpg", "valid:2:folder/b.jpg")],
)
def test_sample_id_is_deterministic(split: str, image_id: int, relative_path: str, expected: str) -> None:
    """Repeated construction produces byte-for-byte identical IDs."""
    assert make_sample_id(split, image_id, relative_path) == expected
