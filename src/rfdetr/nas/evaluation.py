# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------
"""Inherited-subnet evaluation APIs with an explicit tiny-pool guard."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Protocol

import torch
from torch import Tensor, nn

from rfdetr.nas.architecture import ArchitectureSpec
from rfdetr.nas.schedule import resize_batch_to_architecture
from rfdetr.nas.training import _default_batch_adapter
from rfdetr.nas.validation import assert_finite
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list


DEFAULT_METRICS = ("segm_ap", "bbox_ap", "ap50", "ap75")
LegacyEvaluatorFn = Callable[[Any, list[dict[str, Any]], ArchitectureSpec], Mapping[str, Any]]
ForwardFn = Callable[[nn.Module, Tensor, list[dict[str, Any]], ArchitectureSpec], Any]
LatencyFn = Callable[[nn.Module, ArchitectureSpec], Any]


class SubnetEvaluator(Protocol):
    """Lifecycle contract for dataset-level AP aggregation."""

    def reset(self) -> None:
        """Clear state before one architecture evaluation."""

    def update(self, outputs: Any, targets: list[dict[str, Any]], architecture: ArchitectureSpec) -> None:
        """Accumulate predictions and targets from one batch."""

    def compute(self) -> Mapping[str, Any]:
        """Return dataset-level metrics after all batches."""


@dataclass
class SubnetEvaluation:
    """Metrics and resource placeholders for one inherited subnet."""

    architecture: dict[str, Any]
    metrics: dict[str, float | None]
    latency_ms: float | None
    peak_vram_bytes: int | None
    batches_evaluated: int
    status: str = "PASS"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(self.metrics)
        return value


def _metric_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Tensor):
        value = value.detach().cpu()
        if value.numel() != 1:
            raise ValueError("evaluator metrics must be scalar")
        value = value.item()
    if not isinstance(value, (int, float)):
        raise TypeError(f"evaluator metric must be numeric or None, got {type(value).__name__}")
    return float(value)


def _forward_default(
    model: nn.Module,
    images: Tensor,
    targets: list[dict[str, Any]],
    architecture: ArchitectureSpec,
) -> Any:
    resized_images, resized_targets = resize_batch_to_architecture(images, targets, architecture)
    del resized_targets
    return model(nested_tensor_from_tensor_list([image for image in resized_images]))


def _cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def benchmark_active_subnet_latency(
    model: nn.Module,
    architecture: ArchitectureSpec,
    latency_fn: LatencyFn,
    *,
    device: torch.device,
    warmup_runs: int = 5,
    timed_runs: int = 20,
) -> float:
    """Time one active-subnet forward callback with CUDA-safe synchronization.

    ``latency_fn`` must execute exactly one representative forward for the
    supplied model.  The caller keeps the controller active for this entire
    function, so the measured architecture cannot silently fall back to
    native state.
    """

    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if timed_runs <= 0:
        raise ValueError("timed_runs must be positive")
    with torch.inference_mode():
        for _ in range(warmup_runs):
            latency_fn(model, architecture)
        _cuda_synchronize(device)
        start = perf_counter()
        for _ in range(timed_runs):
            latency_fn(model, architecture)
        _cuda_synchronize(device)
    return (perf_counter() - start) * 1000.0 / timed_runs


def evaluate_subnet(
    supernet: nn.Module,
    architecture: ArchitectureSpec,
    val_loader: Iterable[Any],
    *,
    controller: Any | None = None,
    evaluator: SubnetEvaluator | LegacyEvaluatorFn | None = None,
    batch_adapter: Callable[[Any, torch.device], tuple[Tensor, list[dict[str, Any]]]] | None = None,
    forward_fn: ForwardFn | None = None,
    latency_fn: LatencyFn | None = None,
    device: str | torch.device | None = None,
    max_batches: int = 1,
    latency_warmup_runs: int = 5,
    latency_runs: int = 20,
) -> SubnetEvaluation:
    """Evaluate one inherited subnet without making latency a hidden search input.

    The default metric values are explicit ``None`` placeholders.  A real
    dataset adapter supplies ``evaluator`` later; the execution layer itself
    does not know whether the source data is COCO, VOC, or an internal format.
    """

    if max_batches <= 0:
        raise ValueError("max_batches must be positive")
    if controller is not None:
        architecture.validate(controller.native)
    lifecycle_evaluator = evaluator is not None and all(
        callable(getattr(evaluator, method, None)) for method in ("reset", "update", "compute")
    )
    if evaluator is not None and not lifecycle_evaluator and max_batches != 1:
        raise TypeError(
            "multi-batch evaluation requires a SubnetEvaluator with reset/update/compute; "
            "a per-batch callback is accepted only for one tiny batch"
        )
    model_device = torch.device(device) if device is not None else next(supernet.parameters()).device
    was_training = supernet.training
    supernet.eval()
    if model_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(model_device)

    metrics: dict[str, float | None] = {key: None for key in DEFAULT_METRICS}
    batches_evaluated = 0
    adapter = batch_adapter or _default_batch_adapter
    forward = forward_fn or _forward_default
    legacy_metrics: Mapping[str, Any] | None = None
    latency_ms: float | None = None
    peak_vram: int | None = None
    try:
        if controller is not None:
            controller.activate(architecture)
            controller.validate_active_architecture()
        if lifecycle_evaluator:
            evaluator.reset()  # type: ignore[union-attr]
        with torch.no_grad():
            for batch in val_loader:
                images, targets = adapter(batch, model_device)
                outputs = forward(supernet, images, targets, architecture)
                assert_finite(outputs, "subnet.outputs")
                if evaluator is not None:
                    if lifecycle_evaluator:
                        evaluator.update(outputs, targets, architecture)  # type: ignore[union-attr]
                    else:
                        legacy_metrics = evaluator(outputs, targets, architecture)  # type: ignore[operator]
                batches_evaluated += 1
                if batches_evaluated >= max_batches:
                    break
        if batches_evaluated == 0:
            raise ValueError("validation loader produced no batches")
        if lifecycle_evaluator:
            measured = evaluator.compute()  # type: ignore[union-attr]
            if not isinstance(measured, Mapping):
                raise TypeError("SubnetEvaluator.compute() must return a metric mapping")
            for key, value in measured.items():
                metrics[str(key)] = _metric_value(value)
        elif legacy_metrics is not None:
            for key, value in legacy_metrics.items():
                metrics[str(key)] = _metric_value(value)
        if latency_fn is not None:
            latency_ms = benchmark_active_subnet_latency(
                supernet,
                architecture,
                latency_fn,
                device=model_device,
                warmup_runs=latency_warmup_runs,
                timed_runs=latency_runs,
            )
        if model_device.type == "cuda" and torch.cuda.is_available():
            peak_vram = int(torch.cuda.max_memory_allocated(model_device))
    finally:
        if controller is not None:
            controller.reset_to_native()
        supernet.train(was_training)
    return SubnetEvaluation(
        architecture=architecture.to_dict(),
        metrics=metrics,
        latency_ms=latency_ms,
        peak_vram_bytes=peak_vram,
        batches_evaluated=batches_evaluated,
    )


def evaluate_subnet_pool(
    supernet: nn.Module,
    architectures: Iterable[ArchitectureSpec],
    val_loader: Iterable[Any],
    *,
    controller: Any | None = None,
    evaluator: SubnetEvaluator | LegacyEvaluatorFn | None = None,
    batch_adapter: Callable[[Any, torch.device], tuple[Tensor, list[dict[str, Any]]]] | None = None,
    forward_fn: ForwardFn | None = None,
    latency_fn: LatencyFn | None = None,
    device: str | torch.device | None = None,
    max_batches: int = 1,
    max_architectures: int = 3,
    latency_warmup_runs: int = 5,
    latency_runs: int = 20,
) -> list[SubnetEvaluation]:
    """Evaluate a preparation pool and refuse an accidental formal sweep."""

    values = list(architectures)
    if max_architectures <= 0:
        raise ValueError("max_architectures must be positive")
    if len(values) > max_architectures:
        raise RuntimeError(
            f"subnet pool contains {len(values)} architectures; preparation cap is {max_architectures}; "
            "formal full sweep is safety-locked"
        )
    return [
        evaluate_subnet(
            supernet,
            architecture,
            val_loader,
            controller=controller,
            evaluator=evaluator,
            batch_adapter=batch_adapter,
            forward_fn=forward_fn,
            latency_fn=latency_fn,
            device=device,
            max_batches=max_batches,
            latency_warmup_runs=latency_warmup_runs,
            latency_runs=latency_runs,
        )
        for architecture in values
    ]
