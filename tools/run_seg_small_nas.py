# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Run only the safety-locked formal-search dry-run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rfdetr.nas.runner import write_dry_run_manifest


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=repo_root / "configs/seg_small_nas_full.yaml", type=Path)
    parser.add_argument("--repo-root", default=repo_root, type=Path)
    parser.add_argument("--output", default=repo_root / "nas_artifacts/FULL_SEARCH_MANIFEST.json", type=Path)
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--confirm-full-nas", action="store_true")
    args = parser.parse_args()
    manifest = write_dry_run_manifest(
        args.config,
        args.repo_root,
        args.output,
        cli_confirmed=args.confirm_full_nas,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.confirm_full_nas:
        print("FULL SEARCH NOT EXECUTED: formal_search_gate and preparation-only plan are still enforced")
    return 0 if manifest["dry_run_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
