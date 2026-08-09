# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License 2.0 (see LICENSE for details)
# ------------------------------------------------------------------------
"""Two-process gloo regression for preparation-trainer DDP ownership/artifacts."""

from __future__ import annotations

import os

import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from rfdetr.nas.architecture import ArchitectureSpec, NativeArchitecture
from rfdetr.nas.training import train_elastic_supernet


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


class _Controller:
    def __init__(self, native: NativeArchitecture, model: nn.Module) -> None:
        self.native = native
        self.model = model
        self.active = None

    def activate(self, architecture: ArchitectureSpec) -> None:
        architecture.validate(self.native)
        self.active = architecture

    def validate_active_architecture(self) -> None:
        assert self.active is not None

    def reset_to_native(self) -> None:
        self.active = None


def _worker(rank: int, world_size: int, init_file: str, output_dir: str) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        native = _native()
        architecture = ArchitectureSpec(
            resolution=384,
            patch_size=12,
            num_windows=2,
            decoder_layers=4,
            num_queries=100,
            group_detr=13,
            encoder=native.encoder,
        )
        module = nn.Linear(1, 1)
        ddp_model = DistributedDataParallel(module)
        controller = _Controller(native, module)
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

        def loss_fn(model, _batch, _architecture):
            return model.module.weight.square().sum()

        result = train_elastic_supernet(
            ddp_model,
            [object()],
            native,
            [architecture],
            {
                "max_optimizer_steps": 1,
                "checkpoint_interval_steps": 1,
                "ddp": True,
            },
            output_dir,
            controller=controller,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )
        assert result.ddp["controller_owner"] == "ddp.module"
        assert result.ddp["world_size"] == 2
    finally:
        distributed.destroy_process_group()


def test_two_process_gloo_ddp_rank0_artifacts(tmp_path) -> None:
    init_file = tmp_path / "ddp-init"
    multiprocessing.spawn(
        _worker,
        args=(2, str(init_file), str(tmp_path / "output")),
        nprocs=2,
        join=True,
    )
    assert (tmp_path / "output" / "supernet_last.pt").is_file()
    assert (tmp_path / "output" / "supernet_metrics.jsonl").is_file()
