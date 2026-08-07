# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Safety-locked formal-search runner dry-run."""

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
    full_search = config.get("full_search", {})
    enabled = bool(full_search.get("enabled", False))
    resolved_repo_root = Path(repo_root).expanduser().resolve()
    approval_marker = resolved_repo_root / "NAS_FULL_SEARCH_APPROVED"
    manifest = {
        "config": str(config_path),
        "repo_root": str(resolved_repo_root),
        "full_search_enabled": enabled,
        "cli_confirmation_required": "--confirm-full-nas",
        "cli_confirmation_received": bool(cli_confirmed),
        "approval_marker": str(approval_marker),
        "approval_marker_exists": approval_marker.exists(),
        "three_lock_state": {
            "config_enabled": enabled,
            "cli_confirmed": bool(cli_confirmed),
            "approval_file_present": approval_marker.exists(),
            "formal_search_allowed": bool(enabled and cli_confirmed and approval_marker.exists()),
        },
        "planned_stages": [{"name": stage, "executed": False} for stage in PLANNED_STAGES],
        "formal_search_executed": False,
        "pareto_search_executed": False,
        "full_subnet_sweep_executed": False,
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
