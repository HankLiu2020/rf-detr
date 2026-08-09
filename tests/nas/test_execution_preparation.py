# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------
"""Tiny/mock tests for the formal NAS execution preparation layer."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from rfdetr.nas.architecture import ArchitectureSpec, NativeArchitecture
from rfdetr.nas.data_gate import validate_target_split_manifest
from rfdetr.nas.evaluation import evaluate_subnet, evaluate_subnet_pool
from rfdetr.nas.orchestration import build_formal_execution_plan, run_formal_execution_preparation
from rfdetr.nas.pareto import compute_pareto_front, select_fast_balanced_accurate
from rfdetr.nas.ranking import (
    run_proxy_search,
    run_ranking_gate,
    run_short_finetune_ranking,
    select_representative_architectures,
)
from rfdetr.nas.runner import dry_run_manifest
from rfdetr.nas.training import SupernetTrainConfig, train_elastic_supernet


def _native() -> NativeArchitecture:
    return NativeArchitecture(
        encoder="dinov2_windowed_small",
        resolution=384,
        patch_size=12,
        num_windows=2,
        decoder_layers=4,
        num_queries=100,
        num_select=100,
        group_detr=13,
        two_stage=True,
        bbox_reparam=True,
        lite_refpoint_refine=True,
        dec_pred_bbox_embed_share=False,
        hidden_dim=256,
        segmentation_head=True,
        mask_downsample_ratio=4,
        positional_encoding_size=32,
        projector_scale=("P4",),
    )


def _architecture(
    native: NativeArchitecture,
    *,
    resolution: int,
    patch_size: int,
    window: int,
) -> ArchitectureSpec:
    return ArchitectureSpec(
        resolution=resolution,
        patch_size=patch_size,
        num_windows=window,
        decoder_layers=native.decoder_layers,
        num_queries=native.num_queries,
        group_detr=native.group_detr,
        encoder=native.encoder,
    )


class _Controller:
    def __init__(self, native: NativeArchitecture) -> None:
        self.native = native
        self.active: ArchitectureSpec | None = None

    def activate(self, architecture: ArchitectureSpec) -> None:
        architecture.validate(self.native)
        self.active = architecture

    def validate_active_architecture(self) -> None:
        assert self.active is not None

    def reset_to_native(self) -> None:
        self.active = None


class _Accumulator:
    def __init__(self) -> None:
        self.updates = 0

    def reset(self) -> None:
        self.updates = 0

    def update(self, _outputs, _targets, _architecture) -> None:
        self.updates += 1

    def compute(self):
        return {"segm_ap": float(self.updates), "bbox_ap": 0.4}


def test_tiny_supernet_trainer_checkpoint_resume_and_ema(tmp_path) -> None:
    assert SupernetTrainConfig().checkpoint_interval_steps == 1000
    assert SupernetTrainConfig().debug_gradient_snapshot_interval == 0
    native = _native()
    controller = _Controller(native)
    architectures = (
        _architecture(native, resolution=384, patch_size=12, window=2),
        _architecture(native, resolution=480, patch_size=20, window=2),
    )
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = [object()]

    def loss_fn(current_model, _batch, architecture):
        return (current_model.weight.square().sum() + architecture.patch_size * 0.0), {"probe": 1.0}

    result = train_elastic_supernet(
        model,
        loader,
        native,
        architectures,
        {
            "max_optimizer_steps": 2,
            "seed": 7,
            "gradient_accumulation_steps": 2,
            "checkpoint_interval": 1,
        },
        tmp_path,
        controller=controller,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )
    assert result.status == "PASS"
    assert result.optimizer_step == 2
    assert result.ema_parameter_count == 2
    assert controller.active is None
    assert (tmp_path / "supernet_last.pt").exists()
    assert len(result.history) == 2

    resumed_model = nn.Linear(1, 1)
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1)
    resumed_controller = _Controller(native)
    resumed = train_elastic_supernet(
        resumed_model,
        loader,
        native,
        architectures,
        {"max_optimizer_steps": 3, "seed": 7},
        tmp_path,
        controller=resumed_controller,
        optimizer=resumed_optimizer,
        loss_fn=loss_fn,
        resume_from=tmp_path / "supernet_last.pt",
        resume=True,
    )
    assert resumed.optimizer_step == 3
    assert len(resumed.history) == 3
    assert resumed_controller.active is None


def test_inherited_pool_is_bounded_and_returns_metric_placeholders() -> None:
    native = _native()
    architectures = [
        _architecture(native, resolution=384, patch_size=12, window=2),
        _architecture(native, resolution=480, patch_size=20, window=2),
    ]
    model = nn.Linear(4, 4)
    loader = [(torch.zeros(1, 3, 4, 4), [{"labels": torch.zeros(0, dtype=torch.int64)}])]
    controller = _Controller(native)
    evaluator = _Accumulator()
    active_latency_architectures = []

    def forward_fn(current_model, _images, _targets, _architecture):
        return {"probe": current_model(torch.ones(1, 4))}

    def latency_fn(current_model, architecture):
        assert controller.active is not None
        active_latency_architectures.append(controller.active)
        return current_model(torch.ones(1, 4))

    evaluations = evaluate_subnet_pool(
        model,
        architectures,
        loader,
        controller=controller,
        forward_fn=forward_fn,
        evaluator=evaluator,
        latency_fn=latency_fn,
        max_batches=1,
        latency_warmup_runs=1,
        latency_runs=2,
    )
    assert len(evaluations) == 2
    assert evaluations[0].metrics["segm_ap"] == 1.0
    assert evaluations[0].metrics["ap50"] is None
    assert all(value is not None for value in active_latency_architectures)
    assert all(evaluation.latency_ms is not None for evaluation in evaluations)
    aggregated = evaluate_subnet(
        model,
        architectures[0],
        loader * 2,
        controller=controller,
        forward_fn=forward_fn,
        evaluator=evaluator,
        max_batches=2,
    )
    assert aggregated.metrics["segm_ap"] == 2.0
    with pytest.raises(TypeError, match="multi-batch evaluation"):
        evaluate_subnet(
            model,
            architectures[0],
            loader * 2,
            forward_fn=forward_fn,
            evaluator=lambda _outputs, _targets, _architecture: {"segm_ap": 0.5},
            max_batches=2,
        )
    with pytest.raises(RuntimeError, match="preparation cap"):
        evaluate_subnet_pool(
            model,
            architectures + [architectures[0], architectures[1]],
            loader,
            max_architectures=3,
        )


def test_ranking_gate_proxy_and_pareto_helpers(tmp_path) -> None:
    native = _native()
    architectures = [
        _architecture(native, resolution=384, patch_size=12, window=2),
        _architecture(native, resolution=480, patch_size=20, window=2),
        _architecture(native, resolution=576, patch_size=12, window=1),
    ]
    assert len(select_representative_architectures(architectures, max_count=2)) == 2
    gate = run_ranking_gate(
        [{"segm_ap": 0.60}, {"segm_ap": 0.70}, {"segm_ap": 0.80}],
        [{"segm_ap": 0.61}, {"segm_ap": 0.69}, {"segm_ap": 0.81}],
        minimum_spearman=0.9,
        minimum_pairs=3,
    )
    assert gate.passed is True
    proxy = run_proxy_search(architectures[:2], gate, lambda architecture: {"architecture": architecture.to_dict()})
    assert proxy.status == "PASS"
    assert len(proxy.evaluations) == 2

    front = compute_pareto_front(
        [
            {"id": "fast", "accuracy": 0.70, "latency": 5.0},
            {"id": "balanced", "accuracy": 0.80, "latency": 8.0},
            {"id": "accurate", "accuracy": 0.90, "latency": 12.0},
            {"id": "dominated", "accuracy": 0.60, "latency": 20.0},
        ]
    )
    assert {record["id"] for record in front} == {"fast", "balanced", "accurate"}
    representatives = select_fast_balanced_accurate(front)
    assert set(representatives) == {"Fast", "Balanced", "Accurate"}
    assert representatives["Fast"]["id"] == "fast"
    assert representatives["Accurate"]["id"] == "accurate"
    blocked_variance = run_ranking_gate(
        [{"segm_ap": 0.7}] * 8,
        [{"segm_ap": 0.71}] * 8,
    )
    assert blocked_variance.passed is False
    assert blocked_variance.reason == "insufficient_rank_variance"

    checkpoint = tmp_path / "short-ft-test.pt"
    torch.save({"model": nn.Linear(1, 1).state_dict()}, checkpoint)
    short_results = run_short_finetune_ranking(
        architectures[:2],
        checkpoint_path=checkpoint,
        model_factory=lambda: nn.Linear(1, 1),
        optimizer_factory=lambda model: torch.optim.SGD(model.parameters(), lr=0.01),
        controller_factory=lambda _model: _Controller(native),
        train_loader_factory=lambda: [object()],
        loss_fn=lambda model, _batch, _architecture: model.weight.square().sum(),
        evaluate_fn=lambda _model, _architecture: {"segm_ap": 0.5},
        steps=1,
        max_architectures=2,
    )
    assert len(short_results) == 2
    assert all(result.optimizer_steps == 1 for result in short_results)


def test_runner_has_target_data_and_gpu_locks(tmp_path) -> None:
    config_path = tmp_path / "formal.yaml"
    config_path.write_text(
        """
full_search:
  enabled: true
safety:
  require_gpu_approval: true
  require_target_dataset_ready: true
  approval_filename: APPROVED
  gpu_approval_filename: GPU_APPROVED
  target_data_ready_filename: TARGET_READY
target_data:
  enabled: true
  dataset_id: industrial-v1
  split_manifest: splits.json
  domain: internal
  adapter: external
  adapter_version: v1
  manifest_root: .
""",
        encoding="utf-8",
    )
    blocked = dry_run_manifest(config_path, tmp_path, cli_confirmed=True)
    assert blocked["formal_search_allowed"] is False
    assert "gpu_approval_missing" in blocked["blocked_reasons"]
    assert "target_dataset_not_ready" in blocked["blocked_reasons"]

    for filename in ("train.png", "nas_val.png", "final_test.png"):
        (tmp_path / filename).write_bytes(b"sample")
    (tmp_path / "splits.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "dataset_id": "industrial-v1",
                    "dataset_version": "2026.08",
                    "domain": "internal",
                    "adapter": "external",
                    "adapter_version": "v1",
                },
                "splits": {
                    "train": [{"id": "train-0", "path": "train.png"}],
                    "nas_val": [{"id": "nas-0", "path": "nas_val.png"}],
                    "final_test": [{"id": "test-0", "path": "final_test.png"}],
                },
            }
        ),
        encoding="utf-8",
    )
    for marker in ("APPROVED", "GPU_APPROVED", "TARGET_READY"):
        (tmp_path / marker).touch()
    unlocked_manifest = dry_run_manifest(config_path, tmp_path, cli_confirmed=True)
    assert unlocked_manifest["target_data"]["ready"] is True
    assert unlocked_manifest["target_data"]["split_validation"]["leakage_free"] is True
    assert unlocked_manifest["formal_search_gate"]["formal_search_allowed"] is True
    assert unlocked_manifest["formal_search_executed"] is False


def test_formal_execution_plan_is_blocked_without_gate(tmp_path) -> None:
    config_path = tmp_path / "formal.yaml"
    config_path.write_text("full_search:\n  enabled: false\n", encoding="utf-8")
    manifest = dry_run_manifest(config_path, tmp_path)
    plan = build_formal_execution_plan(manifest)
    assert plan.status == "BLOCKED"
    assert all(stage["executed"] is False for stage in plan.stages)
    called = []
    result = run_formal_execution_preparation(
        manifest,
        stage_callbacks={"train_elastic_supernet": lambda: called.append(True)},
    )
    assert result.status == "BLOCKED"
    assert called == []


def test_formal_preparation_orchestration_stops_at_failed_ranking_gate() -> None:
    calls = []
    result = run_formal_execution_preparation(
        {"formal_search_gate": {"formal_search_allowed": True}, "blocked_reasons": []},
        stage_callbacks={
            "train_elastic_supernet": lambda: calls.append("train") or {"status": "PASS"},
            "ranking_gate": lambda: calls.append("ranking") or {"passed": False},
            "proxy_validation": lambda: calls.append("proxy") or {"status": "PASS"},
        },
    )
    assert result.status == "BLOCKED"
    assert calls == ["train", "ranking"]
    assert result.blocked_reasons == ["ranking_gate_failed"]


def test_target_split_manifest_rejects_leakage_and_empty_split(tmp_path) -> None:
    manifest_path = tmp_path / "splits.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "industrial-v1",
                "dataset_hash": "sha256:abc",
                "domain": "internal",
                "adapter": "external",
                "splits": {
                    "train": [{"id": "same", "path": "missing.png"}],
                    "nas_val": [{"id": "same", "path": "missing.png"}],
                    "final_test": [],
                },
            }
        ),
        encoding="utf-8",
    )
    result = validate_target_split_manifest(manifest_path)
    assert result.ready is False
    assert result.leakage_free is False
    assert result.schema_valid is False
    assert any("split leakage" in error for error in result.errors)
