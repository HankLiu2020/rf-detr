#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Run a real RF-DETR Seg Small segmentation-head GPU smoke without downloading weights."""

from __future__ import annotations

import argparse

import numpy as np

from rfdetr import RFDETRSegSmall


def main() -> None:
    """Instantiate Seg Small and execute one public inference call."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model = RFDETRSegSmall(
        pretrain_weights=None,
        device=args.device,
        num_classes=2,
        num_queries=10,
        group_detr=1,
    )
    image = np.zeros((384, 384, 3), dtype=np.uint8)
    detections = model.predict(image, threshold=0.0)
    has_masks = getattr(detections, "mask", None) is not None
    print(f"variant={model.size} detections={len(detections)} has_masks={has_masks}")


if __name__ == "__main__":
    main()
