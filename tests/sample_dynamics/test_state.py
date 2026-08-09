# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for RF4 state tracking."""

from __future__ import annotations

from rfdetr.sample_dynamics import SampleState, SampleStateStore, StatePolicy, window_slope


def _record(sample_id: str, value: float) -> dict[str, object]:
    """Build one detached observer-like record."""
    return {
        "sample_id": sample_id,
        "weighted_normalized_loss": value,
        "weighted_per_image_normalized_loss": value,
        "normalized_losses": {"loss_ce": value},
        "per_image_normalized_losses": {"loss_ce": value},
    }


def test_window_slope_and_instant_loss_baseline_are_deterministic() -> None:
    """The replaceable trend primitive and instant percentile baseline are stable."""
    records = [_record("a", 0.1), _record("b", 0.2), _record("c", 0.3)]
    store = SampleStateStore()

    assert window_slope([1.0, 2.0, 3.0]) == 1.0
    assert store.instant_loss_baseline(records) == {"a": 0.0, "b": 0.5, "c": 1.0}


def test_state_store_tracks_ema_hard_patience_probe_conflict_and_difficulty() -> None:
    """Repeated stalled probe conflicts produce SUSPECT without changing training."""
    policy = StatePolicy(window_size=3, max_history=8, suspect_patience=3)
    store = SampleStateStore(policy)
    probe = {"sample_id": "hard", "fn": 1, "fp": 0, "class_error": 0, "gt_recall": 0.0, "gt_count": 1}

    for epoch in range(3):
        store.update([_record("easy", 0.1), _record("hard", 1.0)], probe_records=[probe], epoch=epoch)

    state = store.get("hard")
    assert state is not None
    assert state.state is SampleState.SUSPECT
    assert state.consecutive_hard_count == 3
    assert state.probe_conflict_count == 3
    assert state.loss_ema is not None
    assert state.slope == 0.0
    assert state.difficulty > 0.0


def test_state_store_counts_forgetting_and_round_trips_checkpoint() -> None:
    """A mastered sample becoming hard increments forgetting and survives reload."""
    store = SampleStateStore(StatePolicy(window_size=2, max_history=4))
    store.update([_record("sample", 0.1), _record("other", 1.0)], epoch=0)
    assert store.get("sample").state is SampleState.MASTERED  # type: ignore[union-attr]

    store.update([_record("sample", 1.0), _record("other", 0.1)], epoch=1)
    assert store.get("sample").forgetting_count == 1  # type: ignore[union-attr]

    restored = SampleStateStore()
    restored.load_state_dict(store.state_dict())
    assert restored.snapshot() == store.snapshot()


def test_state_store_updates_once_per_sample_per_epoch() -> None:
    """Replacement and replay appearances do not advance patience repeatedly."""
    store = SampleStateStore(StatePolicy(window_size=2, max_history=4, suspect_patience=3))

    store.update([_record("easy", 0.1), _record("hard", 1.0), _record("hard", 1.2), _record("hard", 0.8)], epoch=0)

    hard = store.get("hard")
    assert hard is not None
    assert len(hard.history) == 1
    assert hard.consecutive_hard_count == 1
    assert hard.current_loss == 1.0


def test_state_store_ranks_image_local_loss_before_global_loss() -> None:
    """Target-rich images are not hard solely because their global numerator is larger."""
    target_rich = _record("many-targets", 10.0)
    target_rich["weighted_per_image_normalized_loss"] = 1.0
    target_sparse = _record("one-target", 2.0)
    target_sparse["weighted_per_image_normalized_loss"] = 2.0
    store = SampleStateStore()

    store.update([target_rich, target_sparse], epoch=0)

    assert store.get("many-targets").loss_percentile == 0.0  # type: ignore[union-attr]
    assert store.get("one-target").loss_percentile == 1.0  # type: ignore[union-attr]
