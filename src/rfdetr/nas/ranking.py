# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------
"""Inherited-subnet ranking gate and proxy-search preparation helpers."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

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


def spearman_rank_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
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
        return None
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
    minimum_pairs: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_representative_architectures(
    architectures: Iterable[ArchitectureSpec],
    *,
    max_count: int = 10,
) -> tuple[ArchitectureSpec, ...]:
    """Select representatives by structural coverage, then fill by diversity.

    The priority probes cover native/high-token/coarse cases, every available
    patch and window value, decoder corners, and query corners.  This avoids
    making the representative set depend only on the incidental list order.
    """

    values = list(architectures)
    if max_count <= 0:
        raise ValueError("max_count must be positive")
    if not values:
        return ()
    if max_count == 1:
        return (values[0],)

    selected: list[ArchitectureSpec] = []

    def append(value: ArchitectureSpec | None) -> None:
        if value is not None and value not in selected and len(selected) < max_count:
            selected.append(value)

    def first(predicate: Callable[[ArchitectureSpec], bool]) -> ArchitectureSpec | None:
        return next((value for value in values if predicate(value)), None)

    append(first(lambda value: value.native))
    append(
        max(
            values,
            key=lambda value: (
                value.resolution // value.patch_size,
                -value.num_windows,
                value.decoder_layers,
                value.num_queries,
            ),
        )
    )
    append(max(values, key=lambda value: (value.patch_size, -value.resolution, value.num_windows)))
    for patch_size in sorted({value.patch_size for value in values}):
        append(first(lambda value, patch_size=patch_size: value.patch_size == patch_size))
    for num_windows in sorted({value.num_windows for value in values}):
        append(first(lambda value, num_windows=num_windows: value.num_windows == num_windows))
    decoder_values = sorted({value.decoder_layers for value in values})
    for decoder_layers in (decoder_values[0], 1, 2, decoder_values[-1]):
        append(first(lambda value, decoder_layers=decoder_layers: value.decoder_layers == decoder_layers))
    query_values = sorted({value.num_queries for value in values})
    for num_queries in (query_values[0], query_values[-1]):
        append(first(lambda value, num_queries=num_queries: value.num_queries == num_queries))

    while len(selected) < min(max_count, len(values)):
        seen_patches = {value.patch_size for value in selected}
        seen_windows = {value.num_windows for value in selected}
        seen_decoders = {value.decoder_layers for value in selected}
        seen_queries = {value.num_queries for value in selected}
        remaining = [value for value in values if value not in selected]
        candidate = max(
            remaining,
            key=lambda value: (
                int(value.patch_size not in seen_patches)
                + int(value.num_windows not in seen_windows)
                + int(value.decoder_layers not in seen_decoders)
                + int(value.num_queries not in seen_queries),
                value.resolution,
                -value.patch_size,
            ),
        )
        selected.append(candidate)
    return tuple(selected)


def run_ranking_gate(
    inherited_results: Sequence[Any],
    short_finetune_results: Sequence[Any],
    *,
    score_key: str = "segm_ap",
    minimum_spearman: float = 0.8,
    minimum_pairs: int = 8,
) -> RankingGateResult:
    """Compare inherited and optional short-finetune scores before proxy search."""

    if not 0.0 <= minimum_spearman <= 1.0:
        raise ValueError("minimum_spearman must be in [0, 1]")
    if minimum_pairs < 2:
        raise ValueError("minimum_pairs must be at least 2")
    if len(inherited_results) != len(short_finetune_results):
        raise ValueError("ranking gate result sets must have equal length")
    if len(inherited_results) < minimum_pairs:
        return RankingGateResult(
            status="BLOCKED",
            passed=False,
            spearman=None,
            threshold=minimum_spearman,
            pairs=len(inherited_results),
            minimum_pairs=minimum_pairs,
            reason=f"at least {minimum_pairs} representative subnets are required",
        )
    inherited = [_score(value, score_key) for value in inherited_results]
    fine_tuned = [_score(value, score_key) for value in short_finetune_results]
    correlation = spearman_rank_correlation(inherited, fine_tuned)
    if correlation is None:
        return RankingGateResult(
            status="BLOCKED",
            passed=False,
            spearman=None,
            threshold=minimum_spearman,
            pairs=len(inherited),
            minimum_pairs=minimum_pairs,
            reason="insufficient_rank_variance",
        )
    passed = correlation >= minimum_spearman
    return RankingGateResult(
        status="PASS" if passed else "BLOCKED",
        passed=passed,
        spearman=correlation,
        threshold=minimum_spearman,
        pairs=len(inherited),
        minimum_pairs=minimum_pairs,
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


@dataclass
class ShortFinetuneResult:
    """One equal-budget short-finetune ranking result."""

    architecture: dict[str, Any]
    metrics: dict[str, Any]
    optimizer_steps: int
    source_checkpoint: str
    status: str = "PASS"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _short_loss(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value.mean() if value.ndim else value
    if isinstance(value, tuple) and value and isinstance(value[0], Tensor):
        return value[0].mean() if value[0].ndim else value[0]
    raise TypeError("short_finetune loss_fn must return a Tensor or (Tensor, metrics)")


def _metric_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key, child in value.items():
        if isinstance(child, Tensor):
            if child.numel() != 1:
                raise ValueError(f"short-finetune metric {key!r} must be scalar")
            result[str(key)] = float(child.detach().cpu())
        else:
            result[str(key)] = child
    return result


def run_short_finetune_ranking(
    architectures: Iterable[ArchitectureSpec],
    *,
    checkpoint_path: str | Path,
    model_factory: Callable[[], nn.Module],
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    controller_factory: Callable[[nn.Module], Any],
    train_loader_factory: Callable[[], Iterable[Any]],
    loss_fn: Callable[[nn.Module, Any, ArchitectureSpec], Any],
    evaluate_fn: Callable[[nn.Module, ArchitectureSpec], Mapping[str, Any]],
    steps: int = 1,
    seed: int = 0,
    max_architectures: int = 12,
) -> list[ShortFinetuneResult]:
    """Short-finetune each subnet from the same inherited checkpoint state.

    A fresh model, optimizer, controller, data iterator, and RNG seed are used
    for every architecture.  This prevents subnet A's updates from becoming
    subnet B's starting point.  The function is an execution primitive; the
    safety-locked runner still decides whether it may be called formally.
    """

    values = list(architectures)
    if not values:
        return []
    if len(values) > max_architectures:
        raise RuntimeError(
            f"short-finetune candidate count {len(values)} exceeds preparation cap {max_architectures}"
        )
    if steps <= 0:
        raise ValueError("short-finetune steps must be positive")
    source = Path(checkpoint_path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state, Mapping):
        raise ValueError("short-finetune checkpoint must contain a model state mapping")

    results = []
    for architecture in values:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = model_factory()
        model.load_state_dict(state, strict=True)
        controller = controller_factory(model)
        optimizer = optimizer_factory(model)
        data_iterator = iter(train_loader_factory())
        model.train()
        for _ in range(steps):
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_loader_factory())
                batch = next(data_iterator)
            try:
                controller.activate(architecture)
                controller.validate_active_architecture()
                optimizer.zero_grad(set_to_none=True)
                loss = _short_loss(loss_fn(model, batch, architecture))
                loss.backward()
                optimizer.step()
            finally:
                controller.reset_to_native()
        try:
            controller.activate(architecture)
            controller.validate_active_architecture()
            model.eval()
            metrics = _metric_mapping(evaluate_fn(model, architecture))
        finally:
            controller.reset_to_native()
        results.append(
            ShortFinetuneResult(
                architecture=architecture.to_dict(),
                metrics=metrics,
                optimizer_steps=steps,
                source_checkpoint=str(source),
            )
        )
    return results
