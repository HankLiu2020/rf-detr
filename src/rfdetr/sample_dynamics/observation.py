# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Loss-observation packets and their detached training buffer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor


@dataclass
class PerSampleLossPacket:
    """Per-image loss numerators produced alongside the official criterion loss.

    ``global_num_boxes`` is the batch/global denominator used by the official
    criterion.  It is intentionally kept separate from ``gt_count`` so callers
    do not mistake a batch-global normalizer for an image-local one.

    Args:
        sample_ids: Stable dataset identifiers in batch order.
        global_num_boxes: Official batch/global loss denominator.
        gt_count: Number of ground-truth instances in each image.
        matched_count: Number of Hungarian matches in each image.
        raw_numerators: Loss numerators before division by the global denominator.
        normalized_losses: Raw numerators divided by ``global_num_boxes``.
        per_image_normalized_losses: Raw numerators divided by ``max(gt_count, 1)``;
            this is a diagnostic view and is not the official training loss.

    Examples:
        >>> import torch
        >>> packet = PerSampleLossPacket.from_numerators(
        ...     ("train:1:a.jpg", "train:2:b.jpg"), torch.tensor(3.0),
        ...     torch.tensor([2, 1]), torch.tensor([2, 1]),
        ...     {"loss_ce": torch.tensor([2.0, 1.0])},
        ... )
        >>> packet.normalized_losses["loss_ce"].tolist()
        [0.6666666865348816, 0.3333333432674408]
    """

    sample_ids: tuple[str, ...]
    global_num_boxes: Tensor
    gt_count: Tensor
    matched_count: Tensor
    raw_numerators: dict[str, Tensor]
    normalized_losses: dict[str, Tensor]
    per_image_normalized_losses: dict[str, Tensor]

    @classmethod
    def from_numerators(
        cls,
        sample_ids: Iterable[str],
        global_num_boxes: Tensor,
        gt_count: Tensor,
        matched_count: Tensor,
        raw_numerators: dict[str, Tensor],
    ) -> "PerSampleLossPacket":
        """Build a packet and derive both diagnostic normalizer views.

        Args:
            sample_ids: Stable IDs in the same order as the batch dimension.
            global_num_boxes: Official criterion denominator.
            gt_count: Per-image ground-truth counts.
            matched_count: Per-image match counts.
            raw_numerators: Per-image loss numerators keyed by official loss name.

        Returns:
            A validated packet with detached tensors.

        Raises:
            ValueError: If a packet field does not have one value per sample.
        """
        ids = tuple(str(sample_id) for sample_id in sample_ids)
        batch_size = len(ids)
        gt = gt_count.detach()
        matched = matched_count.detach()
        denominator = global_num_boxes.detach().reshape(()).clamp_min(1.0)
        raw = {name: values.detach() for name, values in raw_numerators.items()}
        normalized = {name: values / denominator for name, values in raw.items()}
        image_denominator = gt.to(dtype=denominator.dtype).clamp_min(1.0)
        per_image = {name: values / image_denominator for name, values in raw.items()}
        if gt.numel() != batch_size or matched.numel() != batch_size:
            raise ValueError("gt_count and matched_count must contain one value per sample_id")
        if any(values.numel() != batch_size for values in raw.values()):
            raise ValueError("every per-sample loss numerator must contain one value per sample_id")
        return cls(ids, denominator, gt, matched, raw, normalized, per_image)

    @property
    def batch_size(self) -> int:
        """Return the number of images represented by this packet."""
        return len(self.sample_ids)

    @property
    def component_numerators(self) -> dict[str, Tensor]:
        """Alias used by downstream consumers that call components numerators."""
        return self.raw_numerators

    def weighted_normalized_loss(self, weight_dict: dict[str, float]) -> Tensor:
        """Return the official weighted normalized loss for every image.

        Losses without an entry in ``weight_dict`` are excluded, matching the reduction in
        ``RFDETRModelModule.training_step``.
        """
        result = torch.zeros(self.batch_size, dtype=self.global_num_boxes.dtype, device=self.global_num_boxes.device)
        for name, weight in weight_dict.items():
            values = self.normalized_losses.get(name)
            if values is not None:
                result = result + values.to(result) * float(weight)
        return result

    def detached_cpu(self) -> "PerSampleLossPacket":
        """Return a CPU copy that cannot retain an autograd graph."""
        return PerSampleLossPacket(
            self.sample_ids,
            self.global_num_boxes.detach().cpu(),
            self.gt_count.detach().cpu(),
            self.matched_count.detach().cpu(),
            {name: value.detach().cpu() for name, value in self.raw_numerators.items()},
            {name: value.detach().cpu() for name, value in self.normalized_losses.items()},
            {name: value.detach().cpu() for name, value in self.per_image_normalized_losses.items()},
        )


class SampleObservationBuffer:
    """Detached append-only buffer for RF2 observations.

    The buffer stores ordinary Python values rather than tensors or model references.  It is intentionally small and
    explicit so later state-policy code can consume the same records without coupling itself to Lightning.
    """

    def __init__(self, max_records: int | None = None) -> None:
        if max_records is not None and max_records < 1:
            raise ValueError(f"max_records must be >= 1, got {max_records}")
        self.max_records = max_records
        self.records: list[dict[str, Any]] = []

    def append(
        self,
        packet: PerSampleLossPacket,
        *,
        global_step: int,
        epoch: int,
        weight_dict: dict[str, float] | None = None,
    ) -> None:
        """Append one detached packet, one JSON-friendly record per image."""
        cpu_packet = packet.detached_cpu()
        weighted = cpu_packet.weighted_normalized_loss(weight_dict or {})
        for index, sample_id in enumerate(cpu_packet.sample_ids):
            record: dict[str, Any] = {
                "sample_id": sample_id,
                "global_step": int(global_step),
                "epoch": int(epoch),
                "global_num_boxes": float(cpu_packet.global_num_boxes.item()),
                "gt_count": int(cpu_packet.gt_count[index].item()),
                "matched_count": int(cpu_packet.matched_count[index].item()),
                "raw_numerators": {
                    name: float(values[index].item()) for name, values in cpu_packet.raw_numerators.items()
                },
                "normalized_losses": {
                    name: float(values[index].item()) for name, values in cpu_packet.normalized_losses.items()
                },
                "per_image_normalized_losses": {
                    name: float(values[index].item()) for name, values in cpu_packet.per_image_normalized_losses.items()
                },
                "weighted_normalized_loss": float(weighted[index].item()),
            }
            self.records.append(record)
        if self.max_records is not None and len(self.records) > self.max_records:
            del self.records[: len(self.records) - self.max_records]

    def clear(self) -> None:
        """Remove all buffered records."""
        self.records.clear()

    def export_jsonl(self, path: str | Path) -> None:
        """Write records as newline-delimited JSON, creating parent folders."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
