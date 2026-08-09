# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License [see LICENSE for details]
# ------------------------------------------------------------------------
"""Stable identifiers shared by RF-DETR dataset implementations."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import SupportsInt


def make_sample_id(split: str, image_id: SupportsInt | str, relative_path: str) -> str:
    """Build a stable, human-readable identifier for one dataset image.

    Args:
        split: Dataset split, such as ``"train"`` or ``"val"``. A compound split
            name is reduced to its first component for consistency with the
            dataset builders.
        image_id: Dataset-provided image identifier.
        relative_path: Image path relative to the dataset split directory.

    Returns:
        An identifier in the form ``split:image_id:relative/path``.

    Examples:
        >>> make_sample_id("train", 123, "images\\\\cat.jpg")
        'train:123:images/cat.jpg'
        >>> make_sample_id("val_0", "abc", "cat.jpg")
        'val:abc:cat.jpg'
    """
    normalized_split = split.split("_", maxsplit=1)[0]
    normalized_path = PurePosixPath(relative_path.replace("\\", "/")).as_posix()
    return f"{normalized_split}:{image_id}:{normalized_path}"
