# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Small explicit MVTec-AD anomaly-mask adapter for NAS smoke tests."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import Tensor


def _to_tensor(image: Image.Image) -> Tensor:
    values = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return values.view(image.height, image.width, len(image.getbands())).permute(2, 0, 1).contiguous().float() / 255.0


def _find_mask(root: Path, category: str, stem: str) -> Path | None:
    candidates = sorted((root / "ground_truth").glob(f"*/{stem}_mask.*"))
    for candidate in candidates:
        if candidate.parent.name == category:
            return candidate
    return candidates[0] if candidates else None


class MVTecSegmentationDataset(torch.utils.data.Dataset[tuple[Tensor, dict[str, Tensor]]]):
    """Map MVTec images and pixel masks to one-class RF-DETR targets."""

    def __init__(self, root: str | Path, category: str = "pill", split: str = "test") -> None:
        self.root = Path(root)
        self.category = category
        self.split = split
        image_root = self.root / category / split
        self.images = sorted(image_root.glob("*/*.png"))
        if not self.images:
            raise FileNotFoundError(f"no MVTec PNG images found under {image_root}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        image_tensor = _to_tensor(image)
        category = image_path.parent.name
        mask_path = _find_mask(self.root / self.category, category, image_path.stem)
        if mask_path is None:
            mask = torch.zeros(image.height, image.width, dtype=torch.bool)
        else:
            mask = _to_tensor(Image.open(mask_path).convert("L"))[0] > 0

        if mask.any():
            ys, xs = torch.where(mask)
            x0, x1 = xs.min().float(), xs.max().float() + 1
            y0, y1 = ys.min().float(), ys.max().float() + 1
            height, width = float(image.height), float(image.width)
            boxes = torch.tensor(
                [[((x0 + x1) / 2) / width, ((y0 + y1) / 2) / height, (x1 - x0) / width, (y1 - y0) / height]],
                dtype=torch.float32,
            )
            labels = torch.zeros(1, dtype=torch.int64)
            masks = mask.unsqueeze(0)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, image.height, image.width), dtype=torch.bool)
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": boxes[:, 2] * boxes[:, 3] * image.width * image.height,
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
            "orig_size": torch.tensor([image.height, image.width], dtype=torch.int64),
            "size": torch.tensor([image.height, image.width], dtype=torch.int64),
            "path": str(image_path),
        }
        return image_tensor, target


def collate_segmentation_batch(batch: list[tuple[Tensor, dict[str, Tensor]]]) -> tuple[Tensor, list[dict[str, Tensor]]]:
    """Stack same-sized MVTec samples and keep variable-length targets."""

    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)
