# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Assert the formal NAS runner cannot execute from the preparation config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rfdetr.nas.runner import dry_run_manifest


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=repo_root / "configs/seg_small_nas_full.yaml", type=Path)
    parser.add_argument("--repo-root", default=repo_root, type=Path)
    parser.add_argument("--output", default=repo_root / "nas_artifacts/safety_lock_test.json", type=Path)
    args = parser.parse_args()
    manifest = dry_run_manifest(args.config, args.repo_root)
    passed = (
        manifest["dry_run_status"] == "PASS"
        and manifest["full_search_enabled"] is False
        and manifest["approval_marker_exists"] is False
        and manifest["formal_search_allowed"] is False
        and manifest["target_data"]["ready"] is False
        and manifest["formal_search_gate"]["formal_search_allowed"] is False
        and manifest["formal_search_executed"] is False
        and manifest["pareto_search_executed"] is False
        and manifest["full_subnet_sweep_executed"] is False
    )
    report = {"status": "PASS" if passed else "FAIL", "manifest": manifest}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
