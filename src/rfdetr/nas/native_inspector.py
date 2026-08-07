# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Inspect the architecture actually constructed from a checkpoint."""

from __future__ import annotations

import json
import importlib.util
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn

from rfdetr.nas.architecture import NativeArchitecture
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list


def _as_int_tuple(shape: torch.Size | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in shape)


def _context_parts(value: Any) -> tuple[nn.Module, Any, Any]:
    """Resolve RFDETR, ModelContext, or LWDETR input to module/config/args."""

    if hasattr(value, "model") and isinstance(value.model, nn.Module):
        return value.model, getattr(value, "model_config", None), getattr(value, "args", None)
    context = getattr(value, "model", None)
    if context is not None and hasattr(context, "model") and isinstance(context.model, nn.Module):
        args = getattr(context, "args", None)
        config = getattr(value, "model_config", None)
        return context.model, config, args
    if isinstance(value, nn.Module):
        return value, getattr(value, "model_config", None), getattr(value, "args", None)
    raise TypeError("expected RFDETR, ModelContext, or LWDETR module")


def _config_value(config: Any, args: Any, name: str, default: Any = None) -> Any:
    value = getattr(config, name, None) if config is not None else None
    if value is None and args is not None:
        value = getattr(args, name, None)
    return default if value is None else value


def _find_patch_projection(module: nn.Module) -> tuple[str, nn.Conv2d]:
    for name, child in module.named_modules():
        if name.endswith("patch_embeddings.projection") and isinstance(child, nn.Conv2d):
            return name, child
    raise RuntimeError("could not locate DINOv2 patch projection")


def _find_embeddings(module: nn.Module) -> tuple[str, nn.Module]:
    for name, child in module.named_modules():
        if name.endswith("embeddings") and hasattr(child, "position_embeddings"):
            return name, child
    raise RuntimeError("could not locate DINOv2 embeddings module")


def _parameter_name(module: nn.Module, parameter: nn.Parameter) -> str:
    for name, candidate in module.named_parameters():
        if candidate is parameter:
            return name
    raise RuntimeError("parameter is not present in the model parameter tree")


def _norm_counts(module: nn.Module) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._NormBase):
            counts[type(child).__name__] += 1
        elif isinstance(child, nn.LayerNorm) or type(child).__name__ == "LayerNorm":
            counts["LayerNorm"] += 1
        elif isinstance(child, nn.GroupNorm):
            counts["GroupNorm"] += 1
        elif isinstance(child, nn.InstanceNorm1d | nn.InstanceNorm2d | nn.InstanceNorm3d):
            counts[type(child).__name__] += 1
    return dict(sorted(counts.items()))


def _feature_shapes(module: nn.Module, resolution: int) -> tuple[tuple[int, ...], ...]:
    """Probe backbone feature shapes without retaining autograd state."""

    backbone = getattr(module, "backbone", None)
    if backbone is None:
        return ()
    device = next(module.parameters()).device
    dummy = torch.zeros(3, resolution, resolution, device=device)
    with torch.no_grad():
        features, _, _ = backbone(nested_tensor_from_tensor_list([dummy]))
    return tuple(_as_int_tuple(feature.tensors.shape) for feature in features)


def _checkpoint_args(path: str | Path | None) -> tuple[set[str], set[str]]:
    if path is None:
        return set(), set()
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, pickle.UnpicklingError):
        return set(), set()
    args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    if isinstance(args, dict):
        present = set(args)
    elif hasattr(args, "__dict__"):
        present = set(vars(args))
    else:
        present = set()
    expected = {"num_queries", "group_detr"}
    return present, expected - present


def _load_param_group_builder() -> Any:
    """Load param_groups.py without importing training.__init__ on CPU hosts."""

    source = Path(__file__).resolve().parents[1] / "training" / "param_groups.py"
    spec = importlib.util.spec_from_file_location("rfdetr_nas_param_groups", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load optimizer grouping source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_param_dict


def _optimizer_metadata(
    module: nn.Module,
    config: Any,
    train_config: Any | None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Capture the real RF-DETR parameter-group policy when a train config is supplied."""

    if train_config is None or config is None:
        return (), {"status": "UNAVAILABLE", "reason": "train_config_not_supplied_to_native_inspector"}
    try:
        from rfdetr._namespace import _namespace_from_configs

        namespace = _namespace_from_configs(config, train_config)
        raw_groups = _load_param_group_builder()(namespace, module)
        names = {id(parameter): name for name, parameter in module.named_parameters()}
        default_weight_decay = float(namespace.weight_decay)
        records = []
        for index, group in enumerate(raw_groups):
            parameter = group["params"]
            records.append(
                {
                    "index": index,
                    "name": names.get(id(parameter), "<unnamed>"),
                    "lr": float(group["lr"]),
                    "weight_decay": float(group.get("weight_decay", default_weight_decay)),
                }
            )
        policy = {
            "status": "PASS",
            "source": "rfdetr.training.param_groups.get_param_dict",
            "lr_encoder": float(namespace.lr_encoder),
            "lr_vit_layer_decay": float(namespace.lr_vit_layer_decay),
            "lr_component_decay": float(namespace.lr_component_decay),
            "weight_decay": default_weight_decay,
            "group_count": len(records),
        }
        return tuple(records), policy
    except Exception as exc:  # The dump must remain usable for module-only callers.
        return (), {"status": "UNAVAILABLE", "reason": f"optimizer_group_probe_failed: {exc}"}


def inspect_native_architecture(
    value: Any,
    checkpoint_path: str | Path | None = None,
    *,
    train_config: Any | None = None,
) -> NativeArchitecture:
    """Inspect a loaded RF-DETR-Seg-Small model and return measured native limits."""

    module, config, args = _context_parts(value)
    transformer = module.transformer
    decoder = transformer.decoder
    patch_name, projection = _find_patch_projection(module)
    _, embeddings = _find_embeddings(module)
    query_shape = _as_int_tuple(module.query_feat.weight.shape)
    ref_shape = _as_int_tuple(module.refpoint_embed.weight.shape)
    group_detr = int(getattr(module, "group_detr"))
    num_queries = int(getattr(module, "num_queries"))
    decoder_layers = int(getattr(transformer, "dec_layers"))
    dec_share = getattr(decoder, "bbox_embed", None) is not None
    checkpoint_path = checkpoint_path or _config_value(config, args, "pretrain_weights")
    present, missing = _checkpoint_args(checkpoint_path)

    discrepancies: list[str] = []
    expected_query_capacity = num_queries * group_detr
    if query_shape[0] != expected_query_capacity or ref_shape[0] != expected_query_capacity:
        discrepancies.append(
            f"query capacity mismatch: expected {expected_query_capacity}, query={query_shape}, refpoint={ref_shape}"
        )
    if "num_queries" in missing or "group_detr" in missing:
        discrepancies.append(
            "checkpoint args omit num_queries/group_detr; runtime config and embedding shapes are authoritative"
        )

    resolution = int(_config_value(config, args, "resolution"))
    norm_counts = _norm_counts(module)
    feature_shapes = _feature_shapes(module, resolution)
    encoder_proposals = sum(
        int(shape[-2]) * int(shape[-1]) for shape in feature_shapes if len(shape) >= 4
    )
    segmentation_head = getattr(module, "segmentation_head", None)
    segmentation_head_type = type(segmentation_head).__name__ if segmentation_head is not None else ""
    mask_feature_resolution = (
        max(1, resolution // int(_config_value(config, args, "mask_downsample_ratio", 4))),
        max(1, resolution // int(_config_value(config, args, "mask_downsample_ratio", 4))),
    )
    optimizer_groups, layer_decay_policy = _optimizer_metadata(module, config, train_config)
    paper_expected = {
        "resolution": 384,
        "patch_size": 12,
        "num_windows": 2,
        "decoder_layers": 4,
        "num_queries": 100,
        "backbone": "DINOv2-S",
    }
    paper_discrepancies = []
    for field_name in ("resolution", "patch_size", "num_windows", "decoder_layers", "num_queries"):
        runtime_value = {
            "resolution": resolution,
            "patch_size": int(_config_value(config, args, "patch_size")),
            "num_windows": int(_config_value(config, args, "num_windows")),
            "decoder_layers": decoder_layers,
            "num_queries": num_queries,
        }[field_name]
        if runtime_value != paper_expected[field_name]:
            paper_discrepancies.append(
                f"{field_name}: runtime={runtime_value} paper_expected={paper_expected[field_name]}"
            )
    model_variant = type(value).__name__
    if model_variant in {"ModelContext", "LWDETR"}:
        model_variant = "RFDETRSegSmall"
    return NativeArchitecture(
        encoder=str(_config_value(config, args, "encoder")),
        resolution=resolution,
        patch_size=int(_config_value(config, args, "patch_size")),
        num_windows=int(_config_value(config, args, "num_windows")),
        decoder_layers=decoder_layers,
        num_queries=num_queries,
        num_select=int(_config_value(config, args, "num_select", num_queries)),
        group_detr=group_detr,
        two_stage=bool(getattr(module, "two_stage")),
        bbox_reparam=bool(getattr(module, "bbox_reparam")),
        lite_refpoint_refine=bool(getattr(module, "lite_refpoint_refine")),
        dec_pred_bbox_embed_share=bool(dec_share),
        hidden_dim=int(getattr(transformer, "d_model")),
        segmentation_head=getattr(module, "segmentation_head", None) is not None,
        mask_downsample_ratio=int(_config_value(config, args, "mask_downsample_ratio", 4)),
        positional_encoding_size=int(_config_value(config, args, "positional_encoding_size")),
        out_feature_indexes=tuple(int(v) for v in (_config_value(config, args, "out_feature_indexes", []) or [])),
        projector_scale=tuple(str(v) for v in (_config_value(config, args, "projector_scale", []) or [])),
        query_capacity=int(query_shape[0]),
        query_embedding_shape=query_shape,
        refpoint_embedding_shape=ref_shape,
        patch_projection_shape=_as_int_tuple(projection.weight.shape),
        position_embedding_shape=_as_int_tuple(embeddings.position_embeddings.shape),
        encoder_feature_shapes=feature_shapes,
        norm_counts=norm_counts,
        requires_norm_recalibration=any(
            isinstance(child, nn.modules.batchnorm._NormBase) and child.track_running_stats
            for child in module.modules()
        ),
        checkpoint_args_present=tuple(sorted(present)),
        checkpoint_args_missing=tuple(sorted(missing)),
        checkpoint_discrepancies=tuple(discrepancies),
        model_variant=model_variant,
        backbone=str(_config_value(config, args, "encoder")),
        patch_embed_module=patch_name,
        patch_embed_parameter_name=_parameter_name(module, projection.weight),
        patch_embed_weight_shape=_as_int_tuple(projection.weight.shape),
        query_feat_shape=query_shape,
        refpoint_embed_shape=ref_shape,
        encoder_proposal_count_at_native_resolution=encoder_proposals,
        segmentation_head_type=segmentation_head_type,
        mask_head_per_layer=False if segmentation_head is not None else None,
        mask_feature_resolution=mask_feature_resolution,
        norm_types=norm_counts,
        batchnorm_module_count=sum(
            count for name, count in norm_counts.items() if "BatchNorm" in name or "SyncBatchNorm" in name
        ),
        optimizer_param_groups=optimizer_groups,
        layer_decay_policy=layer_decay_policy,
        paper_expected=paper_expected,
        paper_runtime_discrepancies=tuple(paper_discrepancies),
    )


def write_native_dump(
    value: Any,
    output: str | Path,
    checkpoint_path: str | Path | None = None,
    *,
    train_config: Any | None = None,
) -> NativeArchitecture:
    """Inspect and write a native architecture JSON file."""

    architecture = inspect_native_architecture(value, checkpoint_path=checkpoint_path, train_config=train_config)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(architecture.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return architecture
