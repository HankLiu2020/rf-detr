# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for rank-zero sample-dynamics epoch coordination."""

from __future__ import annotations

from rfdetr.sample_dynamics import SampleStateStore, SampleWeightPolicy, StatePolicy, synchronize_epoch_state


def _record(sample_id: str, value: float) -> dict[str, object]:
    """Return one image-local observer record."""
    return {
        "sample_id": sample_id,
        "weighted_per_image_normalized_loss": value,
        "per_image_normalized_losses": {"loss_ce": value},
    }


def test_synchronize_epoch_state_aggregates_and_connects_probe() -> None:
    """The non-DDP path uses the same authoritative epoch/probe lifecycle."""
    store = SampleStateStore(StatePolicy(window_size=2, max_history=4, suspect_patience=1))
    policy = SampleWeightPolicy()
    probe_calls = 0

    def probe() -> list[dict[str, object]]:
        nonlocal probe_calls
        probe_calls += 1
        return [
            {
                "sample_id": "hard",
                "fn": 1,
                "fp": 0,
                "class_error": 0,
                "gt_recall": 0.0,
                "gt_count": 1,
            }
        ]

    policy = synchronize_epoch_state(
        store,
        policy,
        [_record("easy", 0.1), _record("hard", 1.0), _record("hard", 1.0)],
        epoch=0,
        probe_factory=probe,
    )

    hard = store.get("hard")
    assert probe_calls == 1
    assert hard is not None
    assert len(hard.history) == 1
    assert hard.probe_conflict_count == 1
    assert policy is not None
    assert policy.version == 1
