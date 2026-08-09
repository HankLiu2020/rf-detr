# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------
"""Ordered formal-NAS execution plan, kept behind the runner safety gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping


EXECUTION_STAGES = (
    "data_gate",
    "gpu_approval_gate",
    "train_elastic_supernet",
    "evaluate_representative_subnets",
    "ranking_gate",
    "proxy_validation",
    "hardware_latency_benchmark",
    "pareto_filtering",
    "topk_full_validation",
    "select_fast_balanced_accurate",
    "static_export",
)


@dataclass(frozen=True)
class ExecutionLimits:
    """Preparation limits; formal values must be supplied in a future phase."""

    max_optimizer_steps: int = 2
    max_architectures: int = 3
    max_batches: int = 2


@dataclass
class FormalExecutionResult:
    """Ordered stage result without implying that formal search was executed."""

    status: str
    preparation_only: bool
    formal_search_allowed: bool
    blocked_reasons: list[str]
    stages: list[dict[str, Any]]
    limits: dict[str, int]
    formal_search_executed: bool = False
    full_subnet_sweep_executed: bool = False
    pareto_search_executed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_formal_execution_plan(
    gate_manifest: Mapping[str, Any],
    *,
    preparation_only: bool = True,
    limits: ExecutionLimits | None = None,
) -> FormalExecutionResult:
    """Build the stage DAG from the canonical ``formal_search_gate`` only."""

    limits = limits or ExecutionLimits()
    if limits.max_optimizer_steps <= 0 or limits.max_architectures <= 0 or limits.max_batches <= 0:
        raise ValueError("execution preparation limits must be positive")
    gate = gate_manifest.get("formal_search_gate", {})
    formal_search_allowed = bool(gate.get("formal_search_allowed", False))
    blocked_reasons = list(gate_manifest.get("blocked_reasons", []))
    if not formal_search_allowed and not blocked_reasons:
        blocked_reasons = ["formal_search_gate_not_satisfied"]
    status = "BLOCKED" if not formal_search_allowed else ("PREPARATION_ONLY" if preparation_only else "READY")
    stages = [
        {
            "name": name,
            "executed": False,
            "allowed": formal_search_allowed,
        }
        for name in EXECUTION_STAGES
    ]
    return FormalExecutionResult(
        status=status,
        preparation_only=preparation_only,
        formal_search_allowed=formal_search_allowed,
        blocked_reasons=blocked_reasons if not formal_search_allowed else [],
        stages=stages,
        limits=asdict(limits),
    )


def run_formal_execution_preparation(
    gate_manifest: Mapping[str, Any],
    *,
    stage_callbacks: Mapping[str, Callable[[], Any]] | None = None,
    limits: ExecutionLimits | None = None,
) -> FormalExecutionResult:
    """Run only explicitly injected tiny callbacks after a satisfied gate.

    The repository runner calls this in preparation mode only after a future
    caller supplies a target dataset and approvals.  With the current default
    manifest the function returns ``BLOCKED`` and invokes no callback.
    """

    result = build_formal_execution_plan(gate_manifest, preparation_only=True, limits=limits)
    if result.status == "BLOCKED":
        return result
    callbacks = stage_callbacks or {}
    stage_results = []
    for stage in result.stages:
        name = stage["name"]
        callback = callbacks.get(name)
        if callback is None or name in {"data_gate", "gpu_approval_gate"}:
            stage_results.append(stage)
            continue
        value = callback()
        stage = dict(stage)
        stage["executed"] = True
        stage["result"] = value
        stage_results.append(stage)
        if name == "ranking_gate" and isinstance(value, Mapping) and not bool(value.get("passed", False)):
            result.status = "BLOCKED"
            result.blocked_reasons = ["ranking_gate_failed"]
            break
    result.stages = stage_results
    return result
