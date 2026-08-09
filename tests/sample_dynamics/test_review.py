# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for RF8 SUSPECT review export."""

from __future__ import annotations

import json

from rfdetr.sample_dynamics import ReviewExporter, SampleState, SampleStateRecord


def test_review_exporter_selects_suspect_and_keeps_evidence(tmp_path) -> None:
    """Only SUSPECT records are exported, with state/probe evidence intact."""
    suspect = SampleStateRecord(
        sample_id="train:7:images/example.jpg",
        state=SampleState.SUSPECT,
        current_loss=1.2,
        slope=0.01,
        forgetting_count=1,
        probe_conflict_count=3,
        policy_version=4,
        history=[1.0, 1.2],
    )
    mastered = SampleStateRecord(sample_id="train:8:images/ok.jpg", state=SampleState.MASTERED)
    output = tmp_path / "review.jsonl"

    queue = ReviewExporter.export(
        [suspect, mastered],
        observations=[{"sample_id": suspect.sample_id, "relative_path": "images/example.jpg"}],
        probe_records=[
            {
                "sample_id": suspect.sample_id,
                "fn": 2,
                "fp": 1,
                "class_error": 1,
                "matched_iou": 0.4,
                "gt_recall": 0.5,
                "gt_count": 4,
                "pred_count": 3,
            }
        ],
        output_path=output,
    )

    assert [entry["sample_id"] for entry in queue] == [suspect.sample_id]
    assert queue[0]["path"] == "images/example.jpg"
    assert queue[0]["forgetting_count"] == 1
    assert queue[0]["fn"] == 2
    assert json.loads(output.read_text(encoding="utf-8"))["sample_id"] == suspect.sample_id
