# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Distributed synchronization for sample-dynamics epoch boundaries."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast

import torch.distributed as distributed

from rfdetr.sample_dynamics.observation import aggregate_observations
from rfdetr.sample_dynamics.state import SampleStateStore
from rfdetr.sample_dynamics.weighting import SampleWeightPolicy

ProbeFactory = Callable[[], Iterable[Mapping[str, Any]]]


def distributed_is_initialized() -> bool:
    """Return whether a multi-rank torch process group is active."""
    return distributed.is_available() and distributed.is_initialized() and distributed.get_world_size() > 1


def is_global_zero() -> bool:
    """Return whether this process owns global sample-dynamics updates."""
    return not distributed_is_initialized() or distributed.get_rank() == 0


def gather_observation_records(
    local_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]] | None:
    """Gather rank-local records and aggregate them once per sample on rank zero."""
    materialized = [dict(record) for record in local_records]
    if not distributed_is_initialized():
        return aggregate_observations(materialized)
    gathered: list[Any] | None = [None] * distributed.get_world_size() if distributed.get_rank() == 0 else None
    distributed.gather_object(materialized, gathered, dst=0)
    if gathered is None:
        return None
    flattened: list[Mapping[str, Any]] = []
    for rank_records in gathered:
        if not isinstance(rank_records, list):
            raise TypeError("gathered sample-dynamics observations must be lists")
        flattened.extend(cast(list[Mapping[str, Any]], rank_records))
    return aggregate_observations(flattened)


def _broadcast_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Broadcast one JSON/checkpoint-safe payload from rank zero."""
    if not distributed_is_initialized():
        if payload is None:
            raise RuntimeError("rank-zero sample-dynamics payload is missing")
        return payload
    objects: list[Any] = [payload if distributed.get_rank() == 0 else None]
    distributed.broadcast_object_list(objects, src=0)
    received = objects[0]
    if not isinstance(received, dict):
        raise TypeError("broadcast sample-dynamics payload must be a dictionary")
    return cast(dict[str, Any], received)


def synchronize_epoch_state(
    state_store: SampleStateStore,
    weight_policy: SampleWeightPolicy | None,
    local_records: Iterable[Mapping[str, Any]],
    *,
    epoch: int,
    probe_factory: ProbeFactory | None = None,
) -> SampleWeightPolicy | None:
    """Create one global state/policy snapshot and install it on every rank.

    All ranks first gather their drained observations.  Rank zero optionally
    runs the deterministic probe, updates the only authoritative StateStore,
    and broadcasts a checkpoint-safe snapshot.  Probe failures are also
    broadcast so non-zero ranks do not wait forever in a later collective.
    """
    global_records = gather_observation_records(local_records)
    payload: dict[str, Any] | None = None
    if is_global_zero():
        try:
            probes = list(probe_factory()) if probe_factory is not None else []
            if global_records:
                state_store.update(global_records, probe_records=probes, epoch=epoch)
                if weight_policy is not None:
                    weight_policy = state_store.build_weight_policy(weight_policy)
            payload = {
                "error": None,
                "state": state_store.state_dict(),
                "policy": weight_policy.state_dict() if weight_policy is not None else None,
            }
        except Exception as error:  # noqa: BLE001 - error must be propagated to every rank
            payload = {"error": f"{type(error).__name__}: {error}"}
    payload = _broadcast_payload(payload)
    error_message = payload.get("error")
    if error_message is not None:
        raise RuntimeError(f"rank-zero sample-dynamics epoch update failed: {error_message}")
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise TypeError("sample-dynamics state payload is missing")
    state_store.load_state_dict(state)
    policy_state = payload.get("policy")
    if weight_policy is not None and isinstance(policy_state, Mapping):
        weight_policy.load_state_dict(policy_state)
    return weight_policy
