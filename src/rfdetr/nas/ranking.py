# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------
"""Inherited-subnet ranking gate and proxy-search preparation helpers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from rfdetr.nas.architecture import ArchitectureSpec
from rfdetr.nas.evaluation import SubnetEvaluation


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = average
        position = end
    return ranks


def spearman_rank_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute tie-aware Spearman correlation without a scipy dependency."""

    if len(left) != len(right):
        raise ValueError("Spearman inputs must have equal length")
    if len(left) < 2:
        raise ValueError("Spearman correlation needs at least two observations")
    left_rank, right_rank = _rank(left), _rank(right)
    left_mean = sum(left_rank) / len(left_rank)
    right_mean = sum(right_rank) / len(right_rank)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_rank, right_rank))
    left_scale = math.sqrt(sum((a - left_mean) ** 2 for a in left_rank))
    right_scale = math.sqrt(sum((b - right_mean) ** 2 for b in right_rank))
    if left_scale == 0.0 or right_scale == 0.0:
        return 1.0 if left_rank == right_rank else 0.0
    return numerator / (left_scale * right_scale)


def _score(value: Any, score_key: str) -> float:
    if isinstance(value, SubnetEvaluation):
        candidate = value.metrics.get(score_key)
    elif isinstance(value, Mapping):
        metrics = value.get("metrics")
        candidate = metrics.get(score_key) if isinstance(metrics, Mapping) else value.get(score_key)
    else:
        candidate = value
    if candidate is None or not isinstance(candidate, (int, float)):
        raise ValueError(f"ranking result has no numeric {score_key!r} score")
    return float(candidate)


@dataclass(frozen=True)
class RankingGateResult:
    """Decision record for inherited-vs-short-finetune ranking reliability."""

    status: str
    passed: bool
    spearman: float | None
    threshold: float
    pairs: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_representative_architectures(
    architectures: Iterable[ArchitectureSpec],
    *,
    max_count: int = 3,
) -> tuple[ArchitectureSpec, ...]:
    """Select a deterministic bounded representative set for the ranking gate."""

    values = list(architectures)
    if max_count <= 0:
        raise ValueError("max_count must be positive")
    if not values:
        return ()
    if max_count == 1:
        return (values[0],)
    if len(values) <= max_count:
        return tuple(values)
    positions = [round(index * (len(values) - 1) / (max_count - 1)) for index in range(max_count)]
    return tuple(values[position] for position in positions)


def run_ranking_gate(
    inherited_results: Sequence[Any],
    short_finetune_results: Sequence[Any],
    *,
    score_key: str = "segm_ap",
    minimum_spearman: float = 0.8,
) -> RankingGateResult:
    """Compare inherited and optional short-finetune scores before proxy search."""

    if not 0.0 <= minimum_spearman <= 1.0:
        raise ValueError("minimum_spearman must be in [0, 1]")
    if len(inherited_results) != len(short_finetune_results):
        raise ValueError("ranking gate result sets must have equal length")
    if len(inherited_results) < 2:
        return RankingGateResult(
            status="BLOCKED",
            passed=False,
            spearman=None,
            threshold=minimum_spearman,
            pairs=len(inherited_results),
            reason="at least two representative subnets are required",
        )
    inherited = [_score(value, score_key) for value in inherited_results]
    fine_tuned = [_score(value, score_key) for value in short_finetune_results]
    correlation = spearman_rank_correlation(inherited, fine_tuned)
    passed = correlation >= minimum_spearman
    return RankingGateResult(
        status="PASS" if passed else "BLOCKED",
        passed=passed,
        spearman=correlation,
        threshold=minimum_spearman,
        pairs=len(inherited),
        reason="ranking gate passed" if passed else "inherited ranking correlation is below threshold",
    )


@dataclass
class ProxySearchResult:
    """Bounded proxy-search result or an explicit gate-blocked decision."""

    status: str
    gate: RankingGateResult
    evaluations: list[Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        serialized = []
        for value in self.evaluations:
            serialized.append(value.to_dict() if hasattr(value, "to_dict") else value)
        return {
            "status": self.status,
            "gate": self.gate.to_dict(),
            "evaluations": serialized,
            "reason": self.reason,
        }


def run_proxy_search(
    architectures: Iterable[ArchitectureSpec],
    gate: RankingGateResult,
    evaluator: Callable[[ArchitectureSpec], Any],
    *,
    max_architectures: int = 3,
) -> ProxySearchResult:
    """Run only a caller-authorized tiny proxy set after a passing gate."""

    values = list(architectures)
    if not gate.passed:
        return ProxySearchResult(
            status="BLOCKED",
            gate=gate,
            evaluations=[],
            reason="ranking gate failed; proxy search was not executed",
        )
    if len(values) > max_architectures:
        raise RuntimeError(
            f"proxy candidate count {len(values)} exceeds preparation cap {max_architectures}; "
            "formal proxy sweep is safety-locked"
        )
    evaluations = [evaluator(architecture) for architecture in values]
    return ProxySearchResult(
        status="PASS",
        gate=gate,
        evaluations=evaluations,
        reason="bounded proxy set evaluated",
    )
