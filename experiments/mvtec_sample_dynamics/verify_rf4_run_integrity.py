#!/usr/bin/env python3
"""Verify persisted run-level integrity for the RF4 MVTec longitudinal Gate.

This is experiment-side evidence checking.  It never loads a checkpoint or
reruns inference; it checks the contract, trajectory cardinality, sample-order
provenance, metric finiteness, and optimizer-step horizon emitted by the E2
runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    from experiments.mvtec_sample_dynamics.analyze_rf4_longitudinal import (
        EXPECTED_BATCH_SIZE,
        EXPECTED_CHECKPOINT_SHA256,
        EXPECTED_EPOCHS,
        EXPECTED_MANIFEST_SHA256,
        EXPECTED_SAMPLE_COUNT,
        EXPECTED_SEED,
        FROZEN_STATE_POLICY,
    )
except ModuleNotFoundError:  # direct ``python experiments/.../verify_rf4_run_integrity.py`` execution
    from analyze_rf4_longitudinal import (  # type: ignore[no-redef]
        EXPECTED_BATCH_SIZE,
        EXPECTED_CHECKPOINT_SHA256,
        EXPECTED_EPOCHS,
        EXPECTED_MANIFEST_SHA256,
        EXPECTED_SAMPLE_COUNT,
        EXPECTED_SEED,
        FROZEN_STATE_POLICY,
    )

EXPECTED_MANIFEST_RECORD_COUNT = 186
EXPECTED_VALID_SAMPLE_COUNT = 58
EXPECTED_RESOLUTION = 384
EXPECTED_OPTIMIZER_STEPS = EXPECTED_EPOCHS * (EXPECTED_SAMPLE_COUNT // EXPECTED_BATCH_SIZE)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _check(checks: dict[str, bool], failures: list[str], name: str, passed: bool, detail: str = "") -> None:
    checks[name] = bool(passed)
    if not passed:
        failures.append(f"{name}: {detail}" if detail else name)


def verify(run_dir: Path, dataset: Path, weights: Path) -> dict[str, Any]:
    required = (
        "fairness_snapshot.json",
        "training_config.json",
        "pilot_run_summary.json",
        "metrics.csv",
        "sample_order_hash.json",
        "sample_dynamics/sample_state.json",
    )
    checks: dict[str, bool] = {}
    failures: list[str] = []
    missing = [relative for relative in required if not (run_dir / relative).is_file()]
    _check(checks, failures, "required_artifacts", not missing, ", ".join(missing))
    if missing:
        return {"status": "FAIL", "run_dir": str(run_dir), "checks": checks, "failures": failures}

    fairness = _read_json(run_dir / "fairness_snapshot.json")
    training_config = _read_json(run_dir / "training_config.json")
    train_config = dict(training_config.get("train_config", training_config))
    summary = _read_json(run_dir / "pilot_run_summary.json")
    sample_order = _read_json(run_dir / "sample_order_hash.json")
    state_payload = _read_json(run_dir / "sample_dynamics/sample_state.json")
    manifest = _read_json(dataset / "split_manifest.json")

    _check(checks, failures, "manifest_hash", _sha256(dataset / "split_manifest.json") == EXPECTED_MANIFEST_SHA256)
    _check(checks, failures, "checkpoint_hash", _sha256(weights) == EXPECTED_CHECKPOINT_SHA256)
    records = list(manifest.get("records", []))
    train_records = [record for record in records if record.get("split") == "train"]
    valid_records = [record for record in records if record.get("split") == "valid"]
    _check(checks, failures, "manifest_record_count", len(records) == EXPECTED_MANIFEST_RECORD_COUNT, str(len(records)))
    _check(checks, failures, "manifest_train_count", len(train_records) == EXPECTED_SAMPLE_COUNT, str(len(train_records)))
    _check(checks, failures, "manifest_valid_count", len(valid_records) == EXPECTED_VALID_SAMPLE_COUNT, str(len(valid_records)))

    locked = dict(fairness.get("locked_contract", {}))
    expected_locked = {
        "resolution": EXPECTED_RESOLUTION,
        "batch_size": EXPECTED_BATCH_SIZE,
        "grad_accum_steps": 1,
        "epochs": EXPECTED_EPOCHS,
        "multi_scale": False,
        "scale_jitter": False,
        "aug_config": {},
        "augmentation_backend": "torchvision",
        "use_ema": False,
        "seed": EXPECTED_SEED,
    }
    _check(checks, failures, "fairness_experiment", fairness.get("experiment") == "E2 observe-only")
    _check(checks, failures, "fairness_locked_contract", locked == expected_locked, json.dumps(locked, sort_keys=True))
    _check(checks, failures, "fairness_initial_parameter_hash", bool(fairness.get("initial_model_parameter_sha256")))
    _check(checks, failures, "model_num_classes", fairness.get("model_config", {}).get("num_classes") == 1)
    _check(checks, failures, "model_resolution", fairness.get("model_config", {}).get("resolution") == EXPECTED_RESOLUTION)
    for key, expected in {
        "sample_dynamics_enabled": True,
        "sample_dynamics_mode": "observe",
        "sample_dynamics_probe_interval": 1,
        "sample_dynamics_probe_iou_threshold": 0.5,
        "sample_dynamics_probe_score_threshold": 0.05,
        "batch_size": EXPECTED_BATCH_SIZE,
        "epochs": EXPECTED_EPOCHS,
        "seed": EXPECTED_SEED,
        "multi_scale": False,
        "scale_jitter": False,
        "aug_config": {},
        "augmentation_backend": "torchvision",
        "use_ema": False,
        "grad_accum_steps": 1,
        "resume": None,
    }.items():
        _check(checks, failures, f"training_config_{key}", train_config.get(key) == expected, repr(train_config.get(key)))
    _check(checks, failures, "frozen_policy", state_payload.get("policy") == FROZEN_STATE_POLICY)

    state_records = dict(state_payload.get("states", {}))
    train_ids = {str(record["stable_sample_id"]) for record in train_records}
    state_ids = {str(sample_id) for sample_id in state_records}
    _check(checks, failures, "state_sample_count", len(state_records) == EXPECTED_SAMPLE_COUNT, str(len(state_records)))
    _check(checks, failures, "state_ids_match_manifest", state_ids == train_ids)
    trajectory_lengths = {
        len(record.get("history", [])) for record in state_records.values()
    }
    probe_lengths = {
        len(record.get("probe_history", [])) for record in state_records.values()
    }
    _check(checks, failures, "state_history_lengths", trajectory_lengths == {EXPECTED_EPOCHS}, repr(trajectory_lengths))
    _check(checks, failures, "state_probe_lengths", probe_lengths == {EXPECTED_EPOCHS}, repr(probe_lengths))
    state_numeric_ok = True
    probe_alignment_ok = True
    for sample_id, record in state_records.items():
        state_numeric_ok &= int(record.get("last_epoch", -1)) == EXPECTED_EPOCHS - 1
        state_numeric_ok &= all(_finite(value) for value in record.get("history", []))
        for probe in record.get("probe_history", []):
            probe_alignment_ok &= str(probe.get("sample_id")) == str(sample_id)
            for key in ("fn", "fp", "class_error", "gt_count", "matched_count", "gt_recall"):
                probe_alignment_ok &= _finite(probe.get(key))
            if probe.get("matched_mask_iou") is not None:
                probe_alignment_ok &= _finite(probe.get("matched_mask_iou"))
    _check(checks, failures, "state_numeric_finite", state_numeric_ok)
    _check(checks, failures, "probe_sample_id_alignment", probe_alignment_ok)

    epochs = list(sample_order.get("epochs", []))
    epoch_numbers = [int(record.get("epoch", -1)) for record in epochs]
    _check(checks, failures, "sample_order_epoch_count", len(epochs) == EXPECTED_EPOCHS, str(len(epochs)))
    _check(checks, failures, "sample_order_epoch_numbers", epoch_numbers == list(range(EXPECTED_EPOCHS)), repr(epoch_numbers))
    order_ok = True
    for record in epochs:
        order_ok &= int(record.get("sample_count", -1)) == EXPECTED_SAMPLE_COUNT
        order_ok &= _finite(record.get("epoch_seconds")) and float(record.get("epoch_seconds")) > 0.0
        order_ok &= int(record.get("peak_vram_bytes", 0)) > 0
        order_ok &= bool(SHA256_RE.fullmatch(str(record.get("sha256", ""))))
    _check(checks, failures, "sample_order_provenance", order_ok)

    metric_rows: list[dict[str, str]] = []
    metrics_ok = True
    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            metric_rows.append(row)
            for key, value in row.items():
                if value in (None, "") or key in {"epoch", "step"}:
                    continue
                metrics_ok &= _finite(value)
    _check(checks, failures, "metrics_all_finite", metrics_ok)
    metric_epochs = {int(float(row["epoch"])) for row in metric_rows if row.get("epoch") not in (None, "")}
    _check(checks, failures, "metrics_epoch_coverage", metric_epochs == set(range(EXPECTED_EPOCHS)), repr(sorted(metric_epochs)))
    validation_rows = [row for row in metric_rows if row.get("val/segm_mAP_50_95") not in (None, "")]
    train_rows = [row for row in metric_rows if row.get("train/loss") not in (None, "")]
    _check(checks, failures, "validation_epoch_coverage", {int(float(row["epoch"])) for row in validation_rows} == set(range(EXPECTED_EPOCHS)))
    _check(checks, failures, "train_loss_epoch_coverage", {int(float(row["epoch"])) for row in train_rows} == set(range(EXPECTED_EPOCHS)))
    steps = [int(float(row["step"])) for row in metric_rows if row.get("step") not in (None, "")]
    _check(checks, failures, "optimizer_step_horizon", bool(steps) and max(steps) == EXPECTED_OPTIMIZER_STEPS - 1, str(max(steps) if steps else None))

    checkpoint_ok = all((run_dir / f"checkpoint_{epoch}.ckpt").is_file() for epoch in range(EXPECTED_EPOCHS))
    checkpoint_ok &= (run_dir / "checkpoint_best_regular.pth").is_file()
    checkpoint_ok &= (run_dir / "checkpoint_best_total.pth").is_file()
    _check(checks, failures, "checkpoint_artifacts", checkpoint_ok)
    _check(checks, failures, "summary_contract", summary.get("epochs") == EXPECTED_EPOCHS and summary.get("batch_size") == EXPECTED_BATCH_SIZE)
    _check(checks, failures, "summary_elapsed_finite", _finite(summary.get("elapsed_seconds")) and float(summary["elapsed_seconds"]) > 0.0)
    _check(checks, failures, "summary_peak_vram", int(summary.get("peak_vram_bytes", 0)) > 0)

    return {
        "status": "PASS" if not failures else "FAIL",
        "run_dir": str(run_dir),
        "expected_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        "checks": checks,
        "failures": failures,
    }


def _write_report(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# RF4 Run Integrity",
        "",
        f"- Status: **{result['status']}**",
        f"- Run: `{result['run_dir']}`",
        f"- Expected optimizer steps: `{result['expected_optimizer_steps']}`",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for name, passed in result["checks"].items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    if result["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in result["failures"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify(args.run_dir.resolve(), args.dataset.resolve(), args.weights.resolve())
    args.output_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_json.resolve().write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(result, args.output_report.resolve())
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
