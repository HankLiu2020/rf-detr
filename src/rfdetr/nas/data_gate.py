# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------
"""Strict target split-manifest validation for the formal NAS data gate."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


REQUIRED_SPLITS = ("train", "nas_val", "final_test")
SAMPLE_ID_KEYS = ("sample_id", "id", "image_id", "path", "file_name")
PATH_KEYS = ("path", "file_name", "image_path")
ANNOTATION_PATH_KEYS = ("annotation_path", "mask_path", "label_path", "annotation")


@dataclass
class TargetSplitManifestValidation:
    """Auditable schema, leakage, and path checks for one target manifest."""

    status: str
    schema_valid: bool
    leakage_free: bool
    paths_valid: bool
    manifest_path: str
    dataset_id: str | None
    dataset_version: str | None
    dataset_hash: str | None
    domain: str | None
    adapter: str | None
    adapter_version: str | None
    split_counts: dict[str, int]
    errors: list[str]
    warnings: list[str]

    @property
    def ready(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["ready"] = self.ready
        return value


def _load_manifest(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import yaml

        return yaml.safe_load(text)


def _metadata_value(payload: Mapping[str, Any], metadata: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if payload.get(key) not in (None, ""):
            return payload[key]
        if metadata.get(key) not in (None, ""):
            return metadata[key]
    return None


def _sample_id(sample: Any) -> str | None:
    if isinstance(sample, Mapping):
        for key in SAMPLE_ID_KEYS:
            if sample.get(key) not in (None, ""):
                return str(sample[key])
        return None
    if sample in (None, ""):
        return None
    return str(sample)


def _resolve_sample_path(value: Any, root: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def validate_target_split_manifest(
    manifest_path: str | Path,
    *,
    expected_dataset_id: str | None = None,
    expected_dataset_version: str | None = None,
    expected_dataset_hash: str | None = None,
    expected_domain: str | None = None,
    expected_adapter: str | None = None,
    expected_adapter_version: str | None = None,
    sample_root: str | Path | None = None,
    require_sample_paths: bool = True,
) -> TargetSplitManifestValidation:
    """Validate required splits, metadata, leakage, unique IDs, and paths.

    The manifest is deliberately independent of COCO/VOC/XML details.  An
    adapter may add domain-specific annotation checks later; this gate verifies
    that every declared sample has a unique identity and a readable image path,
    and rejects any explicit unreadable annotation path.
    """

    path = Path(manifest_path).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    split_counts = {split: 0 for split in REQUIRED_SPLITS}
    if not path.is_file():
        return TargetSplitManifestValidation(
            status="BLOCKED",
            schema_valid=False,
            leakage_free=False,
            paths_valid=False,
            manifest_path=str(path),
            dataset_id=None,
            dataset_version=None,
            dataset_hash=None,
            domain=None,
            adapter=None,
            adapter_version=None,
            split_counts=split_counts,
            errors=["split manifest does not exist or is not a file"],
            warnings=[],
        )
    try:
        payload = _load_manifest(path)
    except Exception as exc:
        return TargetSplitManifestValidation(
            status="BLOCKED",
            schema_valid=False,
            leakage_free=False,
            paths_valid=False,
            manifest_path=str(path),
            dataset_id=None,
            dataset_version=None,
            dataset_hash=None,
            domain=None,
            adapter=None,
            adapter_version=None,
            split_counts=split_counts,
            errors=[f"could not parse split manifest: {exc}"],
            warnings=[],
        )
    if not isinstance(payload, Mapping):
        errors.append("manifest root must be a mapping")
        payload = {}
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        errors.append("manifest metadata must be a mapping")
        metadata = {}

    dataset_id = _metadata_value(payload, metadata, "dataset_id")
    dataset_version = _metadata_value(payload, metadata, "dataset_version", "version")
    dataset_hash = _metadata_value(payload, metadata, "dataset_hash", "sha256", "hash")
    domain = _metadata_value(payload, metadata, "domain")
    adapter = _metadata_value(payload, metadata, "adapter")
    adapter_version = _metadata_value(payload, metadata, "adapter_version")
    for name, value in (
        ("dataset_id", dataset_id),
        ("domain", domain),
        ("adapter", adapter),
    ):
        if value in (None, ""):
            errors.append(f"manifest metadata missing {name}")
    if dataset_version in (None, "") and dataset_hash in (None, ""):
        errors.append("manifest metadata requires dataset_version or dataset_hash")
    for name, expected, actual in (
        ("dataset_id", expected_dataset_id, dataset_id),
        ("dataset_version", expected_dataset_version, dataset_version),
        ("dataset_hash", expected_dataset_hash, dataset_hash),
        ("domain", expected_domain, domain),
        ("adapter", expected_adapter, adapter),
        ("adapter_version", expected_adapter_version, adapter_version),
    ):
        if expected not in (None, "") and actual != expected:
            errors.append(f"manifest {name}={actual!r} differs from configured {expected!r}")

    raw_splits = payload.get("splits")
    if raw_splits is None:
        raw_splits = {split: payload.get(split) for split in REQUIRED_SPLITS}
    if not isinstance(raw_splits, Mapping):
        errors.append("manifest splits must be a mapping")
        raw_splits = {}
    root_value = sample_root or metadata.get("sample_root") or payload.get("sample_root") or path.parent
    root = Path(str(root_value)).expanduser()
    if not root.is_absolute():
        root = (path.parent / root).resolve()

    split_ids: dict[str, set[str]] = {}
    paths_valid = True
    for split in REQUIRED_SPLITS:
        samples = raw_splits.get(split)
        if not isinstance(samples, list):
            errors.append(f"split {split} must be a list")
            continue
        split_counts[split] = len(samples)
        if not samples:
            errors.append(f"split {split} must not be empty")
        ids: set[str] = set()
        for index, sample in enumerate(samples):
            identifier = _sample_id(sample)
            if identifier is None:
                errors.append(f"split {split}[{index}] has no sample id")
                continue
            if identifier in ids:
                errors.append(f"duplicate sample id {identifier!r} within split {split}")
            ids.add(identifier)
            if isinstance(sample, Mapping):
                image_value = next((sample.get(key) for key in PATH_KEYS if sample.get(key) not in (None, "")), None)
                if image_value is None:
                    if require_sample_paths:
                        errors.append(f"split {split}[{index}] has no readable image path field")
                else:
                    image_path = _resolve_sample_path(image_value, root)
                    if image_path is None or not image_path.is_file():
                        errors.append(f"split {split}[{index}] image path is not readable: {image_value!r}")
                        paths_valid = False
                for annotation_key in ANNOTATION_PATH_KEYS:
                    if sample.get(annotation_key) not in (None, ""):
                        annotation_path = _resolve_sample_path(sample[annotation_key], root)
                        if annotation_path is None or not annotation_path.is_file():
                            errors.append(
                                f"split {split}[{index}] annotation path is not readable: "
                                f"{sample[annotation_key]!r}"
                            )
                            paths_valid = False
                            break
            elif require_sample_paths:
                errors.append(f"split {split}[{index}] uses an id without a readable path field")
                paths_valid = False
        split_ids[split] = ids

    leakage_free = True
    for left_index, left in enumerate(REQUIRED_SPLITS):
        for right in REQUIRED_SPLITS[left_index + 1 :]:
            overlap = sorted(split_ids.get(left, set()) & split_ids.get(right, set()))
            if overlap:
                leakage_free = False
                errors.append(f"split leakage {left}∩{right}: {overlap[:5]}")

    schema_valid = not errors and all(split_counts[split] > 0 for split in REQUIRED_SPLITS)
    status = "PASS" if schema_valid and leakage_free and paths_valid else "BLOCKED"
    return TargetSplitManifestValidation(
        status=status,
        schema_valid=schema_valid,
        leakage_free=leakage_free,
        paths_valid=paths_valid,
        manifest_path=str(path),
        dataset_id=str(dataset_id) if dataset_id not in (None, "") else None,
        dataset_version=str(dataset_version) if dataset_version not in (None, "") else None,
        dataset_hash=str(dataset_hash) if dataset_hash not in (None, "") else None,
        domain=str(domain) if domain not in (None, "") else None,
        adapter=str(adapter) if adapter not in (None, "") else None,
        adapter_version=str(adapter_version) if adapter_version not in (None, "") else None,
        split_counts=split_counts,
        errors=errors,
        warnings=warnings,
    )
