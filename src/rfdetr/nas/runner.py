# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Safety-locked formal-search runner dry-run and four-lock data gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PLANNED_STAGES = (
    "train_elastic_supernet",
    "evaluate_inherited_subnets",
    "proxy_validation",
    "hardware_latency_benchmark",
    "pareto_filtering",
    "full_pareto_validation",
    "select_fast_balanced_accurate",
    "static_export",
)


def _resolve_marker(repo_root: Path, filename: Any) -> Path:
    path = Path(str(filename))
    return path if path.is_absolute() else repo_root / path


def _target_data_manifest_path(repo_root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def dry_run_manifest(
    config_path: str | Path,
    repo_root: str | Path,
    *,
    cli_confirmed: bool = False,
) -> dict[str, Any]:
    """Validate that a formal search remains disabled and return its manifest."""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for NAS runner config dry-run") from exc
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    full_search = config.get("full_search", {}) or {}
    safety = config.get("safety", {}) or {}
    target_data = config.get("target_data", {}) or {}
    enabled = bool(full_search.get("enabled", False))
    resolved_repo_root = Path(repo_root).expanduser().resolve()
    approval_filename = safety.get(
        "approval_filename",
        full_search.get("approval_marker", "NAS_FULL_SEARCH_APPROVED"),
    )
    gpu_approval_filename = safety.get("gpu_approval_filename", "GPU_FULL_SEARCH_APPROVED")
    target_data_ready_filename = safety.get(
        "target_data_ready_filename",
        target_data.get("ready_marker", "TARGET_DATASET_READY"),
    )
    approval_marker = _resolve_marker(resolved_repo_root, approval_filename)
    gpu_approval_marker = _resolve_marker(resolved_repo_root, gpu_approval_filename)
    target_data_ready_marker = _resolve_marker(resolved_repo_root, target_data_ready_filename)
    split_manifest = _target_data_manifest_path(resolved_repo_root, target_data.get("split_manifest"))
    target_data_fields_present = all(
        target_data.get(field) not in (None, "")
        for field in ("dataset_id", "split_manifest", "domain", "adapter")
    )
    target_data_ready = bool(
        target_data.get("enabled", False)
        and target_data_fields_present
        and split_manifest is not None
        and split_manifest.exists()
        and target_data_ready_marker.exists()
    )
    require_gpu_approval = bool(safety.get("require_gpu_approval", True))
    require_target_data = bool(safety.get("require_target_dataset_ready", True))
    gpu_approval_present = gpu_approval_marker.exists()
    gpu_lock_satisfied = gpu_approval_present if require_gpu_approval else True
    target_data_lock_satisfied = target_data_ready if require_target_data else True
    legacy_three_lock_satisfied = bool(enabled and cli_confirmed and approval_marker.exists())
    formal_search_allowed = bool(legacy_three_lock_satisfied and gpu_lock_satisfied and target_data_lock_satisfied)
    blocked_reasons = []
    if not enabled:
        blocked_reasons.append("full_search_disabled")
    if enabled and not cli_confirmed:
        blocked_reasons.append("cli_confirmation_missing")
    if enabled and not approval_marker.exists():
        blocked_reasons.append("approval_file_missing")
    if enabled and require_gpu_approval and not gpu_approval_present:
        blocked_reasons.append("gpu_approval_missing")
    if enabled and require_target_data and not target_data_ready:
        blocked_reasons.append("target_dataset_not_ready")
    manifest = {
        "config": str(config_path),
        "repo_root": str(resolved_repo_root),
        "full_search_enabled": enabled,
        "cli_confirmation_required": "--confirm-full-nas",
        "cli_confirmation_received": bool(cli_confirmed),
        "approval_marker": str(approval_marker),
        "approval_marker_exists": approval_marker.exists(),
        "gpu_approval_marker": str(gpu_approval_marker),
        "gpu_approval_marker_exists": gpu_approval_present,
        "target_data": {
            "enabled": bool(target_data.get("enabled", False)),
            "dataset_id": target_data.get("dataset_id"),
            "split_manifest": target_data.get("split_manifest"),
            "split_manifest_resolved": str(split_manifest) if split_manifest is not None else None,
            "split_manifest_exists": bool(split_manifest is not None and split_manifest.exists()),
            "domain": target_data.get("domain"),
            "adapter": target_data.get("adapter", "external"),
            "ready_marker": str(target_data_ready_marker),
            "ready_marker_exists": target_data_ready_marker.exists(),
            "ready": target_data_ready,
        },
        "three_lock_state": {
            "config_enabled": enabled,
            "cli_confirmed": bool(cli_confirmed),
            "approval_file_present": approval_marker.exists(),
            "formal_search_allowed": legacy_three_lock_satisfied,
        },
        "four_lock_state": {
            "config_enabled": enabled,
            "cli_confirmed": bool(cli_confirmed),
            "approval_file_present": approval_marker.exists(),
            "gpu_approval_present": gpu_approval_present,
            "target_dataset_ready": target_data_ready,
            "require_gpu_approval": require_gpu_approval,
            "require_target_dataset_ready": require_target_data,
            "formal_search_allowed": formal_search_allowed,
        },
        "planned_stages": [{"name": stage, "executed": False} for stage in PLANNED_STAGES],
        "formal_search_executed": False,
        "pareto_search_executed": False,
        "full_subnet_sweep_executed": False,
        "formal_search_allowed": formal_search_allowed,
        "blocked_reasons": blocked_reasons,
        "dry_run_status": "PASS" if not enabled and not approval_marker.exists() else "LOCKED",
        "action": "no model training/search performed",
    }
    return manifest


def write_dry_run_manifest(
    config_path: str | Path,
    repo_root: str | Path,
    output: str | Path,
    *,
    cli_confirmed: bool = False,
) -> dict[str, Any]:
    """Write the formal-search dry-run manifest."""

    manifest = dry_run_manifest(config_path, repo_root, cli_confirmed=cli_confirmed)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
