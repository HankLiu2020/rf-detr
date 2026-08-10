#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Build a reproducible, non-benchmark MVTec-AD segmentation pilot.

The source MVTec tree is read-only.  Generated Roboflow-style COCO splits use
symlinks to source images and encode one semantic union-mask annotation per
abnormal image when the effective mask is non-empty. A ``drop_mask`` corruption
intentionally has no COCO annotation; connected components remain metadata and
are used only for the deterministic ``drop_component`` corruption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy import ndimage

DEFAULT_CATEGORIES = ("pill", "capsule", "grid")
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
CORRUPTION_TYPES = ("drop_mask", "mask_shift", "drop_component")


def _stable_digest(*parts: object) -> bytes:
    """Return a process-independent digest for deterministic selections."""
    text = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).digest()


def _stable_shuffle(paths: Iterable[Path], *, seed: int, namespace: str) -> list[Path]:
    """Shuffle paths using a seed that does not depend on Python hash randomization."""
    values = sorted(paths)
    local_seed = int.from_bytes(_stable_digest(seed, namespace)[:8], byteorder="big")
    random.Random(local_seed).shuffle(values)
    return values


def _image_id(source_relative_path: str) -> int:
    """Build a stable positive 63-bit COCO image ID from the source path."""
    digest_value = int.from_bytes(_stable_digest("mvtec-image", source_relative_path)[:8], byteorder="big")
    return digest_value & 0x7FFF_FFFF_FFFF_FFFF


def _logical_mask_digest(mask: np.ndarray[Any, Any]) -> str:
    """Hash the logical mask independently of source PNG encoding."""
    return hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes(order="C")).hexdigest()


def _load_mask(path: Path) -> np.ndarray[Any, Any]:
    """Load one MVTec mask and validate its binary contract."""
    values = np.asarray(Image.open(path).convert("L"))
    unique = set(np.unique(values).tolist())
    if not unique.issubset({0, 255}):
        raise ValueError(f"mask {path} is not binary 0/255: {sorted(unique)}")
    return values > 0


def _components(mask: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], int]:
    """Return 8-connected component labels and count."""
    return ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))


def _uncompressed_rle(mask: np.ndarray[Any, Any]) -> dict[str, Any]:
    """Encode a binary mask as JSON-safe COCO uncompressed RLE."""
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    transitions = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate((np.asarray([0]), transitions, np.asarray([len(flat)])))
    counts = np.diff(boundaries).astype(int).tolist()
    if len(flat) and int(flat[0]) == 1:
        counts.insert(0, 0)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def _mask_annotation(mask: np.ndarray[Any, Any]) -> dict[str, Any] | None:
    """Convert one non-empty semantic anomaly union mask into COCO geometry."""
    y_coordinates, x_coordinates = np.nonzero(mask)
    if not len(x_coordinates):
        return None
    x_min = int(x_coordinates.min())
    x_max = int(x_coordinates.max())
    y_min = int(y_coordinates.min())
    y_max = int(y_coordinates.max())
    return {
        "bbox": [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1],
        "area": int(mask.sum()),
        "segmentation": _uncompressed_rle(mask),
    }


def _source_record(source: Path, image: Path, *, split: str, kind: str, mask: Path | None) -> dict[str, Any]:
    """Create one manifest record before corruption/rendering."""
    relative_image = image.relative_to(source).as_posix()
    category = relative_image.split("/", maxsplit=1)[0]
    defect_type = image.parent.name
    return {
        "split": split,
        "category": category,
        "kind": kind,
        "defect_type": defect_type,
        "source_image": relative_image,
        "source_mask": mask.relative_to(source).as_posix() if mask is not None else None,
        "file_name": relative_image,
        "image_id": _image_id(relative_image),
        "label_status": "clean",
        "corruption_type": None,
        "corruption_params": {},
    }


def build_split_records(
    source: Path,
    *,
    categories: Iterable[str],
    seed: int,
    train_per_defect: int,
    val_per_defect: int,
    train_normal_per_category: int,
    val_normal_per_category: int,
) -> list[dict[str, Any]]:
    """Select mutually exclusive, category/defect-stratified Pilot records."""
    records: list[dict[str, Any]] = []
    for category_name in categories:
        category = source / category_name
        if not category.is_dir():
            raise FileNotFoundError(f"MVTec category does not exist: {category}")
        defect_dirs = sorted(path for path in (category / "test").iterdir() if path.is_dir() and path.name != "good")
        if not defect_dirs:
            raise ValueError(f"category {category_name!r} has no abnormal test directories")
        for defect_dir in defect_dirs:
            abnormal = _stable_shuffle(
                (path for path in defect_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
                seed=seed,
                namespace=f"{category_name}/{defect_dir.name}",
            )
            required = train_per_defect + val_per_defect
            if len(abnormal) < required:
                raise ValueError(
                    f"stratum {category_name}/{defect_dir.name} has {len(abnormal)} images, requires {required}"
                )
            val_images = abnormal[:val_per_defect]
            train_images = abnormal[val_per_defect:required]
            for split, images in (("train", train_images), ("valid", val_images)):
                for image in images:
                    mask = category / "ground_truth" / defect_dir.name / f"{image.stem}_mask.png"
                    if not mask.is_file():
                        raise FileNotFoundError(f"missing MVTec mask for {image}: {mask}")
                    records.append(_source_record(source, image, split=split, kind="abnormal", mask=mask))

        normal_sources = (
            (
                "train",
                category / "train" / "good",
                train_normal_per_category,
                f"{category_name}/normal-train",
            ),
            (
                "valid",
                category / "test" / "good",
                val_normal_per_category,
                f"{category_name}/normal-valid",
            ),
        )
        for split, normal_dir, limit, namespace in normal_sources:
            normal = _stable_shuffle(
                (path for path in normal_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
                seed=seed,
                namespace=namespace,
            )
            if len(normal) < limit:
                raise ValueError(f"normal stratum {namespace} has {len(normal)} images, requires {limit}")
            records.extend(
                _source_record(source, image, split=split, kind="normal", mask=None) for image in normal[:limit]
            )

    image_ids = [int(record["image_id"]) for record in records]
    if len(image_ids) != len(set(image_ids)):
        raise RuntimeError("stable MVTec image ID collision detected")
    source_images = [str(record["source_image"]) for record in records]
    if len(source_images) != len(set(source_images)):
        raise RuntimeError("one source image was selected into multiple Pilot splits")
    return sorted(
        records,
        key=lambda record: (
            str(record["split"]),
            str(record["category"]),
            str(record["defect_type"]),
            str(record["source_image"]),
        ),
    )


def assign_controlled_corruptions(records: list[dict[str, Any]], *, source: Path, rate: float, seed: int) -> None:
    """Mark a deterministic subset of train anomalies as known SUSPECT ground truth."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError("corruption rate must be in [0, 1]")
    candidates = [record for record in records if record["split"] == "train" and record["kind"] == "abnormal"]
    target_count = int(round(len(candidates) * rate))
    ranked = sorted(candidates, key=lambda record: _stable_digest(seed, "corrupt", record["source_image"]))
    remaining = list(ranked)
    for index in range(target_count):
        requested_type = CORRUPTION_TYPES[index % len(CORRUPTION_TYPES)]
        if requested_type == "drop_component":
            selected_index = next(
                (
                    candidate_index
                    for candidate_index, candidate in enumerate(remaining)
                    if _components(_load_mask(source / str(candidate["source_mask"])))[1] >= 2
                ),
                None,
            )
            if selected_index is None:
                requested_type = "mask_shift"
                selected_index = 0
        else:
            selected_index = 0
        record = remaining.pop(selected_index)
        record["label_status"] = "corrupted"
        record["corruption_type"] = requested_type


def _apply_corruption(
    pristine: np.ndarray[Any, Any],
    *,
    corruption_type: str | None,
    sample_key: str,
    seed: int,
) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
    """Apply one deterministic corruption and return its explicit parameters."""
    if corruption_type is None:
        return pristine.copy(), {}
    if corruption_type == "drop_mask":
        return np.zeros_like(pristine), {"removed_all_annotations": True}
    if corruption_type == "mask_shift":
        height, width = pristine.shape
        digest = _stable_digest(seed, "shift", sample_key)
        x_shift = max(1, int(round(width * 0.04))) * (1 if digest[0] % 2 else -1)
        y_shift = max(1, int(round(height * 0.04))) * (1 if digest[1] % 2 else -1)
        shifted = np.zeros_like(pristine)
        source_x_start = max(0, -x_shift)
        source_x_end = min(width, width - x_shift)
        source_y_start = max(0, -y_shift)
        source_y_end = min(height, height - y_shift)
        target_x_start = source_x_start + x_shift
        target_x_end = source_x_end + x_shift
        target_y_start = source_y_start + y_shift
        target_y_end = source_y_end + y_shift
        shifted[target_y_start:target_y_end, target_x_start:target_x_end] = pristine[
            source_y_start:source_y_end,
            source_x_start:source_x_end,
        ]
        return shifted, {"x_shift": x_shift, "y_shift": y_shift}
    if corruption_type == "drop_component":
        labels, component_count = _components(pristine)
        if component_count < 2:
            raise ValueError(f"drop_component requires at least two components: {sample_key}")
        component_areas = [
            (component_index, int((labels == component_index).sum()))
            for component_index in range(1, component_count + 1)
        ]
        removed_component, removed_area = min(component_areas, key=lambda item: (item[1], item[0]))
        corrupted = pristine.copy()
        corrupted[labels == removed_component] = False
        return corrupted, {"removed_component": removed_component, "removed_area": removed_area}
    raise ValueError(f"unknown corruption type: {corruption_type}")


def _render_dataset(
    records: list[dict[str, Any]],
    *,
    source: Path,
    output: Path,
    seed: int,
) -> list[dict[str, Any]]:
    """Render symlinked COCO splits and return enriched manifest records."""
    enriched: list[dict[str, Any]] = []
    for split in ("train", "valid"):
        split_dir = output / split
        split_dir.mkdir(parents=True, exist_ok=False)
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        annotation_id = 1
        for record in (item for item in records if item["split"] == split):
            source_image = source / str(record["source_image"])
            destination = split_dir / str(record["file_name"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(source_image.resolve(), destination)
            with Image.open(source_image) as image:
                width, height = image.size
            image_id = int(record["image_id"])
            images.append(
                {
                    "id": image_id,
                    "file_name": str(record["file_name"]),
                    "width": width,
                    "height": height,
                }
            )
            rendered = dict(record)
            rendered["stable_sample_id"] = f"{'val' if split == 'valid' else split}:{image_id}:{record['file_name']}"
            rendered["width"] = width
            rendered["height"] = height
            rendered["pristine_mask_sha256"] = None
            rendered["effective_mask_sha256"] = None
            rendered["pristine_component_count"] = 0
            rendered["effective_component_count"] = 0
            rendered["pristine_area_ratio"] = 0.0
            rendered["effective_area_ratio"] = 0.0
            if record["kind"] == "abnormal":
                mask_path = source / str(record["source_mask"])
                pristine = _load_mask(mask_path)
                if pristine.shape != (height, width):
                    raise ValueError(f"image/mask size mismatch: {source_image} vs {mask_path}")
                effective, parameters = _apply_corruption(
                    pristine,
                    corruption_type=record["corruption_type"],
                    sample_key=str(record["source_image"]),
                    seed=seed,
                )
                rendered["corruption_params"] = parameters
                _, pristine_components = _components(pristine)
                _, effective_components = _components(effective)
                rendered["pristine_mask_sha256"] = _logical_mask_digest(pristine)
                rendered["effective_mask_sha256"] = _logical_mask_digest(effective)
                rendered["pristine_component_count"] = pristine_components
                rendered["effective_component_count"] = effective_components
                rendered["pristine_area_ratio"] = float(pristine.mean())
                rendered["effective_area_ratio"] = float(effective.mean())
                mask_annotation = _mask_annotation(effective)
                if mask_annotation is not None:
                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": mask_annotation["bbox"],
                            "area": mask_annotation["area"],
                            "segmentation": mask_annotation["segmentation"],
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1
            enriched.append(rendered)
        payload = {
            "info": {
                "description": "MVTec-AD derived Sample Dynamics mechanism-validation split; not a benchmark",
                "version": "1.0",
            },
            "licenses": [],
            "categories": [{"id": 1, "name": "anomaly", "supercategory": "anomaly"}],
            "images": images,
            "annotations": annotations,
        }
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return enriched


def prepare_pilot(
    source: Path,
    output: Path,
    *,
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    seed: int = 20260810,
    train_per_defect: int = 4,
    val_per_defect: int = 2,
    train_normal_per_category: int = 20,
    val_normal_per_category: int = 8,
    corruption_rate: float = 0.08,
) -> dict[str, Any]:
    """Build one fixed Pilot dataset and return its summary."""
    source = source.resolve()
    output = output.resolve()
    category_names = tuple(categories)
    if not source.is_dir():
        raise FileNotFoundError(f"MVTec source does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing Pilot output: {output}")
    records = build_split_records(
        source,
        categories=category_names,
        seed=seed,
        train_per_defect=train_per_defect,
        val_per_defect=val_per_defect,
        train_normal_per_category=train_normal_per_category,
        val_normal_per_category=val_normal_per_category,
    )
    assign_controlled_corruptions(records, source=source, rate=corruption_rate, seed=seed)
    output.mkdir(parents=True)
    enriched = _render_dataset(records, source=source, output=output, seed=seed)

    split_counts = Counter(str(record["split"]) for record in enriched)
    kind_counts = Counter(f"{record['split']}:{record['kind']}" for record in enriched)
    corruption_counts = Counter(
        str(record["corruption_type"])
        for record in enriched
        if record["label_status"] == "corrupted"
    )
    config = {
        "source": str(source),
        "scope": "NON-BENCHMARK / MECHANISM VALIDATION",
        "categories": list(category_names),
        "seed": seed,
        "train_per_defect": train_per_defect,
        "val_per_defect": val_per_defect,
        "train_normal_per_category": train_normal_per_category,
        "val_normal_per_category": val_normal_per_category,
        "corruption_rate": corruption_rate,
    }
    manifest = {"config": config, "records": enriched}
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "split_manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in enriched),
        encoding="utf-8",
    )
    corruption_ground_truth = [
        {
            "stable_sample_id": record["stable_sample_id"],
            "source_image": record["source_image"],
            "corruption_type": record["corruption_type"],
            "corruption_params": record["corruption_params"],
        }
        for record in enriched
        if record["label_status"] == "corrupted"
    ]
    (output / "corruption_ground_truth.json").write_text(
        json.dumps(corruption_ground_truth, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "config": config,
        "split_counts": dict(sorted(split_counts.items())),
        "kind_counts": dict(sorted(kind_counts.items())),
        "corruption_counts": dict(sorted(corruption_counts.items())),
        "corrupted_train_samples": len(corruption_ground_truth),
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    """Parse the deterministic Pilot preparation CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/home/inspur1/data/MVTec-AD"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--train-per-defect", type=int, default=4)
    parser.add_argument("--val-per-defect", type=int, default=2)
    parser.add_argument("--train-normal-per-category", type=int, default=20)
    parser.add_argument("--val-normal-per-category", type=int, default=8)
    parser.add_argument("--corruption-rate", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    """Build the Pilot and print a compact machine-readable summary."""
    args = parse_args()
    summary = prepare_pilot(
        args.source,
        args.output,
        categories=args.categories,
        seed=args.seed,
        train_per_defect=args.train_per_defect,
        val_per_defect=args.val_per_defect,
        train_normal_per_category=args.train_normal_per_category,
        val_normal_per_category=args.val_normal_per_category,
        corruption_rate=args.corruption_rate,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
