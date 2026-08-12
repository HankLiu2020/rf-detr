from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "experiments" / "mvtec_sample_dynamics" / "analyze_multiseed_ablation.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("analyze_multiseed_ablation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _snapshot(seed: int, initial_hash: str, *, lr: float = 1e-5) -> dict[str, object]:
    return {
        "dataset_manifest_sha256": "manifest",
        "initial_model_parameter_sha256": initial_hash,
        "pretrain_weights_sha256": "weights",
        "seed": seed,
        "model_config": {"num_classes": 1, "resolution": 384},
        "train_config": {
            "lr": lr,
            "seed": seed,
            "sample_dynamics_seed": seed,
            "sample_dynamics_mode": "observe",
            "sample_dynamics_enabled": False,
            "sample_dynamics_probe_interval": 0,
            "output_dir": "/tmp/run",
            "notes": {"experiment": "fixture"},
        },
        "locked_contract": {
            "resolution": 384,
            "batch_size": 4,
            "grad_accum_steps": 1,
            "epochs": 15,
            "seed": seed,
            "multi_scale": False,
            "scale_jitter": False,
            "aug_config": {},
            "augmentation_backend": "torchvision",
            "use_ema": False,
        },
    }


def _write_seed(tmp_path: Path, seed: int, initial_hash: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for label in MODULE.RUNS:
        path = tmp_path / str(seed) / label
        path.mkdir(parents=True)
        snapshot = _snapshot(seed, initial_hash)
        snapshot["train_config"] = {
            **snapshot["train_config"],
            "sample_dynamics_mode": {
                "E0": "observe",
                "E3": "loss_weight",
                "E4": "sampler",
                "E5": "combined",
            }[label],
            "sample_dynamics_enabled": label != "E0",
            "sample_dynamics_probe_interval": 0 if label == "E0" else 1,
            "output_dir": str(path),
        }
        (path / "fairness_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
        result[label] = path
    return result


def test_summary_uses_sample_standard_deviation() -> None:
    result = MODULE.summary([1.0, 2.0, 3.0])
    assert result["mean"] == 2.0
    assert result["std"] == 1.0
    assert result["n"] == 3
    assert MODULE.direction_counts([-1.0, 0.0, 2.0, 3.0]) == {
        "positive": 2,
        "zero": 1,
        "negative": 1,
    }


def test_fairness_allows_seed_specific_initialization_but_requires_within_seed_match(tmp_path: Path) -> None:
    run_dirs = {
        20260810: _write_seed(tmp_path, 20260810, "initial-a"),
        20260811: _write_seed(tmp_path, 20260811, "initial-b"),
    }
    result = MODULE.fairness_check(run_dirs)
    assert result["all_within_seed_pass"] is True
    assert result["cross_seed_non_seed_config_equal"] is True
    assert result["cross_seed_non_seed_locked_contract_equal"] is True

    changed = json.loads((run_dirs[20260811]["E4"] / "fairness_snapshot.json").read_text())
    changed["initial_model_parameter_sha256"] = "wrong"
    (run_dirs[20260811]["E4"] / "fairness_snapshot.json").write_text(json.dumps(changed))
    assert MODULE.fairness_check(run_dirs)["all_within_seed_pass"] is False


def test_fairness_rejects_cross_seed_non_seed_config_change(tmp_path: Path) -> None:
    run_dirs = {
        20260810: _write_seed(tmp_path, 20260810, "initial-a"),
        20260811: _write_seed(tmp_path, 20260811, "initial-b"),
    }
    for path in run_dirs[20260811].values():
        snapshot = json.loads((path / "fairness_snapshot.json").read_text())
        snapshot["train_config"]["lr"] = 2e-5
        (path / "fairness_snapshot.json").write_text(json.dumps(snapshot))
    assert MODULE.fairness_check(run_dirs)["cross_seed_non_seed_config_equal"] is False


def test_fairness_rejects_cross_seed_locked_contract_change(tmp_path: Path) -> None:
    run_dirs = {
        20260810: _write_seed(tmp_path, 20260810, "initial-a"),
        20260811: _write_seed(tmp_path, 20260811, "initial-b"),
    }
    for path in run_dirs[20260811].values():
        snapshot = json.loads((path / "fairness_snapshot.json").read_text())
        snapshot["locked_contract"]["batch_size"] = 8
        (path / "fairness_snapshot.json").write_text(json.dumps(snapshot))
    result = MODULE.fairness_check(run_dirs)
    assert result["all_within_seed_pass"] is True
    assert result["cross_seed_non_seed_locked_contract_equal"] is False


def test_trajectory_check_requires_every_requested_epoch() -> None:
    def run(epochs: list[int]) -> dict[str, object]:
        return {
            "epochs_requested": 3,
            "checkpoint_count_evaluated": len(epochs),
            "epochs_evaluated": epochs,
            "trajectory": [
                {"epoch": epoch, "checkpoint_kind": "epoch"}
                for epoch in epochs
            ],
        }

    trajectories = {
        1: {label: run([0, 1, 2]) for label in MODULE.RUNS},
        2: {label: run([0, 1, 2]) for label in MODULE.RUNS},
    }
    assert MODULE.trajectory_check(trajectories)["all_complete"] is True
    trajectories[2]["E5"] = run([0, 2])
    assert MODULE.trajectory_check(trajectories)["all_complete"] is False


def test_trajectory_check_reports_fallback_without_calling_it_a_real_epoch() -> None:
    run = {
        "epochs_requested": 2,
        "checkpoint_count_evaluated": 2,
        "epochs_evaluated": [0, 1],
        "trajectory": [
            {"epoch": 0, "checkpoint_kind": "epoch"},
            {"epoch": 1, "checkpoint_kind": "best_regular_fallback"},
        ],
    }
    result = MODULE.trajectory_check({1: {label: run for label in MODULE.RUNS}})
    assert result["all_complete"] is True
    assert result["by_seed"]["1"]["E0"]["all_real_epoch_checkpoints"] is False
    assert result["by_seed"]["1"]["E0"]["fallback_epochs"] == [1]


def test_trajectory_binding_requires_matching_label_and_snapshot(tmp_path: Path) -> None:
    run_dirs = {1: _write_seed(tmp_path, 1, "initial")}
    trajectories = {
        1: {
            label: {
                "label": label,
                "fairness_snapshot": json.loads((path / "fairness_snapshot.json").read_text()),
            }
            for label, path in run_dirs[1].items()
        }
    }
    assert MODULE.trajectory_binding_check(run_dirs, trajectories)["all_bound"] is True
    trajectories[1]["E5"]["label"] = "E4"
    assert MODULE.trajectory_binding_check(run_dirs, trajectories)["all_bound"] is False
