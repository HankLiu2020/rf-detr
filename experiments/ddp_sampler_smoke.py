#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Two-rank DDP smoke for global state, sampler-plan, and loss-weight contracts."""

from __future__ import annotations

import os

import torch
import torch.distributed as distributed

from rfdetr.sample_dynamics import (
    BucketQuotaSampler,
    SampleState,
    SampleStateStore,
    SampleWeightPolicy,
    StatePolicy,
    synchronize_epoch_state,
)


def main() -> None:
    """Initialize two ranks, compare global lists, and finish with a barrier."""
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    distributed.init_process_group(backend=backend)
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    collective_device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")

    sample_ids = [f"train:{index}:image-{index}.jpg" for index in range(17)]
    local_records = [
        {
            "sample_id": sample_ids[rank],
            "weighted_per_image_normalized_loss": 0.1,
            "per_image_normalized_losses": {"loss_ce": 0.1},
        },
        {
            "sample_id": sample_ids[rank + world_size],
            "weighted_per_image_normalized_loss": 1.0,
            "per_image_normalized_losses": {"loss_ce": 1.0},
        },
    ]
    store = SampleStateStore(StatePolicy(window_size=2, max_history=4, suspect_patience=1))

    def rank_zero_probe() -> list[dict[str, object]]:
        assert distributed.get_rank() == 0
        return [
            {
                "sample_id": sample_ids[world_size],
                "fn": 1,
                "fp": 0,
                "class_error": 0,
                "gt_recall": 0.0,
                "gt_count": 1,
            }
        ]

    weight_policy = synchronize_epoch_state(
        store,
        SampleWeightPolicy(),
        local_records,
        epoch=0,
        probe_factory=rank_zero_probe,
    )
    assert weight_policy is not None
    snapshots: list[object] = [None] * world_size
    distributed.all_gather_object(
        snapshots,
        {"state": store.state_dict(), "policy": weight_policy.state_dict()},
    )
    assert all(snapshot == snapshots[0] for snapshot in snapshots)

    # Rank-local batches intentionally contain only one state.  Local mean
    # normalization would collapse both to 1.0 and erase the policy ratio;
    # global normalization must preserve HARD_LEARNABLE > MASTERED.
    weight_policy.weights = {
        "hard-0": weight_policy.hard_learnable,
        "hard-1": weight_policy.hard_learnable,
        "mastered-0": weight_policy.mastered,
        "mastered-1": weight_policy.mastered,
    }
    local_weight_ids = ("hard-0", "hard-1") if rank == 0 else ("mastered-0", "mastered-1")
    local_weights = weight_policy.batch_weights(local_weight_ids, device=collective_device)
    gathered_weights = [torch.empty_like(local_weights) for _ in range(world_size)]
    distributed.all_gather(gathered_weights, local_weights)
    global_weights = torch.cat(gathered_weights)
    assert abs(torch.mean(global_weights).item() - 1.0) < 1e-6
    assert torch.min(gathered_weights[0]).item() > torch.max(gathered_weights[1]).item()

    local_exposures = {sample_id: (2.2 if rank == 0 else 0.5) for sample_id in local_weight_ids}
    capped_local_weights = weight_policy.batch_weights(
        local_weight_ids,
        device=collective_device,
        exposure_multipliers=local_exposures,
        effective_cap=2.5,
    )
    gathered_capped_weights = [torch.empty_like(capped_local_weights) for _ in range(world_size)]
    distributed.all_gather(gathered_capped_weights, capped_local_weights)
    global_capped_weights = torch.cat(gathered_capped_weights)
    assert abs(torch.mean(global_capped_weights).item() - 1.0) < 1e-6
    assert torch.max(gathered_capped_weights[0]).item() * 2.2 <= 2.5 + 1e-6

    # Deliberately corrupt one rank-local provider after synchronization.  The
    # sampler must still use rank zero's single broadcast global plan.
    if rank == 1:
        store.states[sample_ids[0]].state = SampleState.MASTERED
    sampler = BucketQuotaSampler(
        len(sample_ids),
        sample_ids,
        num_samples=12,
        state_provider=lambda: {sample_id: record.state for sample_id, record in store.states.items()},
        world_size=world_size,
        rank=rank,
        seed=20260809,
    )
    sampler.set_epoch(3)
    global_indices = torch.tensor(sampler.global_epoch_indices(), dtype=torch.int64, device=collective_device)
    gathered = [torch.empty_like(global_indices) for _ in range(world_size)]
    distributed.all_gather(gathered, global_indices)
    assert all(torch.equal(global_indices, other) for other in gathered)
    assert len(list(iter(sampler))) == len(sampler)
    if rank == 0:
        print(f"ddp_state_sampler_smoke=PASS world_size={world_size} pid={os.getpid()}")
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
