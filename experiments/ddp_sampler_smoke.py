#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Two-rank DDP smoke for the global-list-then-rank-slice sampler contract."""

from __future__ import annotations

import os

import torch
import torch.distributed as distributed

from rfdetr.sample_dynamics import BucketQuotaSampler


def main() -> None:
    """Initialize two ranks, compare global lists, and finish with a barrier."""
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    distributed.init_process_group(backend=backend)
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    sample_ids = [f"train:{index}:image-{index}.jpg" for index in range(17)]
    sampler = BucketQuotaSampler(
        len(sample_ids),
        sample_ids,
        num_samples=12,
        world_size=world_size,
        rank=rank,
        seed=20260809,
    )
    sampler.set_epoch(3)
    collective_device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    global_indices = torch.tensor(sampler.global_epoch_indices(), dtype=torch.int64, device=collective_device)
    gathered = [torch.empty_like(global_indices) for _ in range(world_size)]
    distributed.all_gather(gathered, global_indices)
    assert all(torch.equal(global_indices, other) for other in gathered)
    assert len(list(iter(sampler))) == len(sampler)
    if rank == 0:
        print(f"ddp_sampler_smoke=PASS world_size={world_size} pid={os.getpid()}")
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
