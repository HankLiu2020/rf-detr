# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------
"""Reusable, safety-neutral elastic supernet training primitives.

The functions in this module are an execution-layer API.  They do not enable
formal search by themselves; the safety-locked runner remains the authority
that decides whether a caller may run anything beyond a tiny preparation
probe.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from rfdetr.nas.architecture import ArchitectureSpec, NativeArchitecture
from rfdetr.nas.checkpoint import load_nas_checkpoint, save_nas_checkpoint
from rfdetr.nas.schedule import architecture_for_step, resize_batch_to_architecture
from rfdetr.nas.validation import assert_finite, gradient_snapshot
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list


@dataclass(frozen=True)
class SupernetTrainConfig:
    """Bounded knobs shared by tiny preparation probes and future training."""

    max_optimizer_steps: int = 1
    seed: int = 0
    sampling_policy: str = "balanced_patch"
    gradient_accumulation_steps: int = 1
    checkpoint_interval_steps: int = 1000
    debug_gradient_snapshot_interval: int = 0
    ema_decay: float | None = 0.999
    ddp: bool = False

    @classmethod
    def from_value(cls, value: "SupernetTrainConfig | Mapping[str, Any] | Any") -> "SupernetTrainConfig":
        """Normalize a dataclass, mapping, or train-config namespace."""

        if isinstance(value, cls):
            return value
        fields = {
            field: cls.__dataclass_fields__[field].default
            for field in cls.__dataclass_fields__
        }
        if isinstance(value, Mapping):
            fields.update({key: value[key] for key in fields if key in value})
            if "checkpoint_interval" in value and "checkpoint_interval_steps" not in value:
                fields["checkpoint_interval_steps"] = value["checkpoint_interval"]
        else:
            fields.update({key: getattr(value, key) for key in fields if hasattr(value, key)})
        return cls(**fields)


@dataclass
class SupernetTrainResult:
    """Auditable result of one bounded or caller-authorized train invocation."""

    status: str
    optimizer_step: int
    history: list[dict[str, Any]]
    checkpoint_path: str | None
    ema_parameter_count: int
    ddp: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LossFn = Callable[[nn.Module, Any, ArchitectureSpec], Tensor | tuple[Tensor, Mapping[str, Any]]]
BatchAdapter = Callable[[Any, torch.device], Any]


def _search_space_values(search_space: Any) -> tuple[ArchitectureSpec, ...]:
    values = getattr(search_space, "syntactically_valid", search_space)
    result = tuple(values)
    if not all(isinstance(value, ArchitectureSpec) for value in result):
        raise TypeError("search_space must contain ArchitectureSpec values")
    return result


def _native_spec(native: NativeArchitecture) -> ArchitectureSpec:
    return ArchitectureSpec(
        resolution=native.resolution,
        patch_size=native.patch_size,
        num_windows=native.num_windows,
        decoder_layers=native.decoder_layers,
        num_queries=native.num_queries,
        group_detr=native.group_detr,
        encoder=native.encoder,
        native=True,
        metadata={"native_identity": True},
    )


def _supernet_candidates(
    native: NativeArchitecture,
    search_space: Any,
) -> tuple[ArchitectureSpec, ...]:
    """Keep encoder elasticity while fixing decoder and query capacity to native."""

    candidates = []
    for architecture in _search_space_values(search_space):
        if architecture.invalid_reason is not None:
            continue
        if architecture.decoder_layers != native.decoder_layers or architecture.num_queries != native.num_queries:
            continue
        architecture.validate(native)
        candidates.append(architecture)
    if not candidates:
        raise ValueError("search_space has no valid native decoder/query supernet candidates")
    return tuple(candidates)


def _move_target(target: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in target.items()
    }


def _default_batch_adapter(batch: Any, device: torch.device) -> tuple[Tensor, list[dict[str, Any]]]:
    if isinstance(batch, Mapping):
        images, targets = batch.get("images"), batch.get("targets")
    elif isinstance(batch, (tuple, list)) and len(batch) == 2:
        images, targets = batch
    else:
        raise TypeError("train/validation batches must be (images, targets) or {'images', 'targets'}")
    if isinstance(images, Tensor):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError(f"images must have shape [B,C,H,W], got {tuple(images.shape)}")
        moved_images = images.to(device)
    else:
        image_values = [image.to(device) for image in images]
        if not image_values:
            raise ValueError("batch contains no images")
        moved_images = torch.stack(image_values, dim=0)
    if not isinstance(targets, (tuple, list)):
        raise TypeError("targets must be a list or tuple of target mappings")
    moved_targets = [_move_target(target, device) for target in targets]
    return moved_images, moved_targets


def _default_loss(
    model: nn.Module,
    batch: Any,
    architecture: ArchitectureSpec,
    criterion: Any,
    device: torch.device,
    batch_adapter: BatchAdapter | None,
) -> tuple[Tensor, dict[str, Any]]:
    adapter = batch_adapter or _default_batch_adapter
    images, targets = adapter(batch, device)
    images, targets = resize_batch_to_architecture(images, targets, architecture)
    outputs = model(nested_tensor_from_tensor_list([image for image in images]), targets=targets)
    assert_finite(outputs, "supernet.outputs")
    losses = criterion(outputs, targets)
    weighted = sum(
        value * criterion.weight_dict[key]
        for key, value in losses.items()
        if key in criterion.weight_dict
    )
    if not isinstance(weighted, Tensor):
        raise TypeError("criterion must produce at least one weighted tensor loss")
    assert_finite(losses, "supernet.losses")
    return weighted, {"losses": {key: float(value.detach()) for key, value in losses.items()}}


def _normalize_loss_result(value: Any) -> tuple[Tensor, dict[str, Any]]:
    if isinstance(value, Tensor):
        return value, {}
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], Tensor):
        metrics = dict(value[1]) if isinstance(value[1], Mapping) else {"callback_metric": value[1]}
        return value[0], metrics
    if isinstance(value, Mapping) and isinstance(value.get("loss"), Tensor):
        return value["loss"], {key: child for key, child in value.items() if key != "loss"}
    raise TypeError("loss_fn must return a Tensor, (Tensor, metrics), or {'loss': Tensor, ...}")


def _metric_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return float(value.detach().cpu())
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _move_ema_to_model_device(model: nn.Module, ema_state: dict[str, Tensor]) -> None:
    """Move a resumed EMA shadow to the parameter device once, not every step."""

    parameters = dict(model.named_parameters())
    for name, value in list(ema_state.items()):
        parameter = parameters.get(name)
        if parameter is not None and value.device != parameter.device:
            ema_state[name] = value.to(parameter.device)


def _update_ema(model: nn.Module, ema_state: dict[str, Tensor], decay: float) -> None:
    """Update EMA shadows on the same device as the live model parameters."""

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            value = parameter.detach()
            if name not in ema_state:
                ema_state[name] = value.clone()
            else:
                ema_state[name].mul_(decay).add_(value, alpha=1.0 - decay)


def _ema_cpu_state(ema_state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Make the only CPU EMA copy at checkpoint serialization time."""

    return {name: value.detach().cpu() for name, value in ema_state.items()}


def _gradient_parameter_count(model: nn.Module) -> int:
    """Count gradient-bearing parameters without synchronizing gradient values."""

    return sum(parameter.grad is not None for parameter in model.parameters())


def _distributed_status(model: nn.Module, controller: Any, requested: bool) -> dict[str, Any]:
    """Validate DDP/controller ownership and return rank metadata."""

    distributed = torch.distributed
    initialized = bool(distributed.is_available() and distributed.is_initialized())
    rank = int(distributed.get_rank()) if initialized else 0
    world_size = int(distributed.get_world_size()) if initialized else 1
    if requested and not initialized:
        raise RuntimeError("ddp=True requires an initialized torch.distributed process group")
    if initialized and requested:
        if not isinstance(model, DistributedDataParallel):
            raise TypeError("ddp=True requires the forward model to be DistributedDataParallel")
        if getattr(controller, "model", None) is not model.module:
            raise ValueError("Controller.model must be ddp_model.module; forward must use the DDP wrapper")
        ownership = "ddp.module"
    elif isinstance(model, DistributedDataParallel):
        raise ValueError("a DistributedDataParallel model requires ddp=True")
    else:
        ownership = "direct"
    return {
        "requested": requested,
        "initialized": initialized,
        "rank": rank,
        "world_size": world_size,
        "rank0_artifacts_only": initialized,
        "controller_owner": ownership,
    }


def _next_batch(iterator: Any, loader: Iterable[Any]) -> tuple[Any, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        try:
            return next(iterator), iterator
        except StopIteration as exc:
            raise ValueError("train_loader is empty") from exc


def _space_summary(search_space: Any) -> dict[str, Any]:
    summary = getattr(search_space, "to_summary", None)
    if callable(summary):
        return dict(summary())
    return {"candidate_count": len(_search_space_values(search_space))}


def train_elastic_supernet(
    model: nn.Module,
    train_loader: Iterable[Any],
    native: NativeArchitecture,
    search_space: Any,
    train_config: SupernetTrainConfig | Mapping[str, Any] | Any,
    output_dir: str | Path,
    *,
    controller: Any,
    optimizer: torch.optim.Optimizer,
    criterion: Any | None = None,
    loss_fn: LossFn | None = None,
    batch_adapter: BatchAdapter | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    checkpoint_path: str | Path | None = None,
    resume_from: str | Path | None = None,
    resume: bool = False,
) -> SupernetTrainResult:
    """Train a native-decoder/native-query elastic supernet through a callback API.

    ``loss_fn`` is the preferred integration point for an internal dataset
    adapter.  When omitted, the RF-DETR ``(images, targets)`` contract and the
    supplied criterion are used.  DDP is intentionally attach-only: the
    caller initializes DDP, wraps the model, and passes the controller; the
    controller's architecture consistency guard then runs on every rank.
    """

    config = SupernetTrainConfig.from_value(train_config)
    if config.max_optimizer_steps <= 0:
        raise ValueError("max_optimizer_steps must be positive")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.checkpoint_interval_steps <= 0:
        raise ValueError("checkpoint_interval_steps must be positive")
    if config.debug_gradient_snapshot_interval < 0:
        raise ValueError("debug_gradient_snapshot_interval must be non-negative")
    if loss_fn is None and criterion is None:
        raise ValueError("criterion is required when loss_fn is not supplied")
    if controller is None:
        raise ValueError("NativeBoundedElasticController is required for elastic training")
    ddp_status = _distributed_status(model, controller, config.ddp)

    candidates = _supernet_candidates(native, search_space)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    saved_checkpoint = Path(checkpoint_path) if checkpoint_path is not None else destination / "supernet_last.pt"
    resume_path = Path(resume_from) if resume_from is not None else (saved_checkpoint if resume else None)
    if resume and resume_path is None:
        raise ValueError("resume=True requires checkpoint_path or resume_from")

    history: list[dict[str, Any]] = []
    ema_state: dict[str, Tensor] = {}
    start_step = 0
    if resume_path is not None:
        restored = load_nas_checkpoint(model, optimizer, resume_path, scheduler=scheduler)
        history = list(restored.get("history", []))
        start_step = int(restored["optimizer_step"])
        ema_state = dict(restored.get("ema") or {})

    if start_step > config.max_optimizer_steps:
        raise ValueError(
            f"checkpoint optimizer_step={start_step} exceeds max_optimizer_steps={config.max_optimizer_steps}"
        )
    model_device = next(model.parameters()).device
    _move_ema_to_model_device(model, ema_state)
    data_iterator = iter(train_loader)
    model.train()

    for optimizer_step in range(start_step, config.max_optimizer_steps):
        architecture = architecture_for_step(
            candidates,
            seed=config.seed,
            optimizer_step=optimizer_step,
            policy=config.sampling_policy,
        )
        metric_accumulator: dict[str, list[float]] = {}
        weighted_loss = 0.0
        try:
            controller.activate(architecture)
            controller.validate_active_architecture()
            optimizer.zero_grad(set_to_none=True)
            for _ in range(config.gradient_accumulation_steps):
                batch, data_iterator = _next_batch(data_iterator, train_loader)
                callback_batch = batch_adapter(batch, model_device) if loss_fn is not None and batch_adapter else batch
                if loss_fn is None:
                    loss, metrics = _default_loss(
                        model,
                        callback_batch,
                        architecture,
                        criterion,
                        model_device,
                        batch_adapter if loss_fn is None else None,
                    )
                else:
                    loss, metrics = _normalize_loss_result(loss_fn(model, callback_batch, architecture))
                if loss.ndim != 0:
                    loss = loss.mean()
                assert_finite(loss, "supernet.loss")
                weighted_loss += float(loss.detach().cpu())
                for key, value in metrics.items():
                    numeric = _metric_value(value)
                    if isinstance(numeric, (int, float)):
                        metric_accumulator.setdefault(str(key), []).append(float(numeric))
                (loss / config.gradient_accumulation_steps).backward()

            gradient_parameter_count = _gradient_parameter_count(model)
            if gradient_parameter_count == 0:
                raise RuntimeError("supernet step produced no gradients")
            debug_snapshot = None
            if (
                config.debug_gradient_snapshot_interval > 0
                and (optimizer_step + 1) % config.debug_gradient_snapshot_interval == 0
            ):
                debug_snapshot = gradient_snapshot(model)
                if not debug_snapshot:
                    raise RuntimeError("supernet debug gradient snapshot found no gradients")
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if config.ema_decay is not None:
                _update_ema(model, ema_state, float(config.ema_decay))
        finally:
            controller.reset_to_native()

        record: dict[str, Any] = {
            "optimizer_step": optimizer_step,
            "architecture": architecture.to_dict(),
            "weighted_loss": weighted_loss / config.gradient_accumulation_steps,
            "gradient_parameter_count": gradient_parameter_count,
            "gradient_debug_snapshot": debug_snapshot is not None,
            "lr": [float(group["lr"]) for group in optimizer.param_groups],
        }
        if debug_snapshot is not None:
            record["gradient_debug_parameter_count"] = len(debug_snapshot)
        record.update(
            {
                key: sum(values) / len(values)
                for key, values in metric_accumulator.items()
                if values
            }
        )
        history.append(record)

        should_save = (
            (optimizer_step + 1) % config.checkpoint_interval_steps == 0
            or optimizer_step + 1 == config.max_optimizer_steps
        )
        if should_save and ddp_status["rank"] == 0:
            future_preview = [
                architecture_for_step(
                    candidates,
                    seed=config.seed,
                    optimizer_step=future_step,
                    policy=config.sampling_policy,
                ).to_dict()
                for future_step in range(optimizer_step + 1, optimizer_step + 6)
            ]
            save_nas_checkpoint(
                model,
                optimizer,
                saved_checkpoint,
                optimizer_step=optimizer_step + 1,
                architecture=architecture,
                history=history,
                scheduler=scheduler,
                ema_state=_ema_cpu_state(ema_state),
                search_space_config=_space_summary(search_space),
                sampling_policy=config.sampling_policy,
                seed=config.seed,
                schedule_preview=future_preview,
                native_architecture=native.to_dict(),
            )
        if should_save and ddp_status["initialized"]:
            torch.distributed.barrier()

    if ddp_status["rank"] == 0:
        (destination / "supernet_metrics.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in history),
            encoding="utf-8",
        )
    if ddp_status["initialized"]:
        torch.distributed.barrier()
    return SupernetTrainResult(
        status="PASS",
        optimizer_step=config.max_optimizer_steps,
        history=history,
        checkpoint_path=str(saved_checkpoint),
        ema_parameter_count=len(ema_state),
        ddp=ddp_status,
    )
