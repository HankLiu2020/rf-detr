from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "experiments" / "mvtec_sample_dynamics" / "analyze_seed1_ablation.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("analyze_seed1_ablation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_resource_gate_separates_loss_weight_and_sampler_semantics() -> None:
    e3 = {
        "REFERENCE_HARD": {
            "appearances_per_sample_per_epoch": 1.0,
            "mean_applied_loss_weight_seen": 1.2,
            "effective_contribution_per_sample_per_epoch": 1.2,
        },
        "REFERENCE_MASTERED": {
            "appearances_per_sample_per_epoch": 1.0,
            "mean_applied_loss_weight_seen": 0.8,
            "effective_contribution_per_sample_per_epoch": 0.8,
        },
    }
    e4 = {
        "REFERENCE_HARD": {
            "appearances_per_sample_per_epoch": 1.5,
            "mean_applied_loss_weight_seen": None,
            "effective_contribution_per_sample_per_epoch": None,
        },
        "REFERENCE_MASTERED": {
            "appearances_per_sample_per_epoch": 0.8,
            "mean_applied_loss_weight_seen": None,
            "effective_contribution_per_sample_per_epoch": None,
        },
        "ALL_TRAIN": {
            "appearances_per_sample_per_epoch": 1.0,
            "mean_applied_loss_weight_seen": None,
            "effective_contribution_per_sample_per_epoch": None,
        },
    }

    assert MODULE.resource_gate("E3", e3)["mechanism_pass"] is True
    e4_gate = MODULE.resource_gate("E4", e4)
    assert e4_gate["mechanism_pass"] is True
    assert e4_gate["hard_to_mastered_applied_weight_ratio"] is None

    e3["REFERENCE_HARD"]["appearances_per_sample_per_epoch"] = 1.2
    assert MODULE.resource_gate("E3", e3)["mechanism_pass"] is False
    e4["REFERENCE_HARD"]["mean_applied_loss_weight_seen"] = 1.1
    assert MODULE.resource_gate("E4", e4)["mechanism_pass"] is False


def test_contract_check_requires_all_four_runs_to_match() -> None:
    identity = {
        "dataset_manifest_sha256": "manifest",
        "initial_model_parameter_sha256": "initial",
        "pretrain_weights_sha256": "weights",
        "seed": 20260810,
        "model_config": {"num_classes": 1},
        "train_config": {"lr": 1e-5},
        "locked_contract": {
            "resolution": 384,
            "batch_size": 4,
            "grad_accum_steps": 1,
            "epochs": 15,
            "seed": 20260810,
            "multi_scale": False,
            "scale_jitter": False,
            "aug_config": {},
            "augmentation_backend": "torchvision",
            "use_ema": False,
        },
    }
    runs = {
        label: {"fairness_snapshot": {**identity}, "checkpoint_count_evaluated": 15}
        for label in MODULE.RUNS
    }
    assert MODULE.contract_check(runs)["all_equal"] is True
    runs["E4"]["fairness_snapshot"] = {**identity, "dataset_manifest_sha256": "different"}
    assert MODULE.contract_check(runs)["all_equal"] is False


def test_contract_check_rejects_non_intervention_train_config_difference() -> None:
    identity = {
        "dataset_manifest_sha256": "manifest",
        "initial_model_parameter_sha256": "initial",
        "pretrain_weights_sha256": "weights",
        "seed": 20260810,
        "model_config": {"num_classes": 1},
        "train_config": {"lr": 1e-5, "sample_dynamics_mode": "observe", "output_dir": "/e0"},
        "locked_contract": {field: None for field in MODULE.CONTRACT_FIELDS},
    }
    runs = {
        label: {"fairness_snapshot": {**identity}, "checkpoint_count_evaluated": 15}
        for label in MODULE.RUNS
    }
    runs["E3"]["fairness_snapshot"] = {
        **identity,
        "train_config": {"lr": 2e-5, "sample_dynamics_mode": "loss_weight", "output_dir": "/e3"},
    }
    assert MODULE.contract_check(runs)["all_equal"] is False
