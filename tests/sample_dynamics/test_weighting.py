# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for RF5 previous-policy sample weights."""

from __future__ import annotations

import pytest
import torch

from rfdetr.sample_dynamics import SampleStateStore, SampleWeightPolicy, cap_effective_contribution


def test_policy_maps_states_and_normalizes_each_batch() -> None:
    """State weights are capped and batch-normalized without boosting SUSPECT."""
    policy = SampleWeightPolicy()
    policy.update_from_states(
        [
            {"sample_id": "mastered", "state": "MASTERED"},
            {"sample_id": "learning", "state": "LEARNING"},
            {"sample_id": "hard", "state": "HARD_LEARNABLE"},
            {"sample_id": "suspect", "state": "SUSPECT"},
        ]
    )
    weights = policy.batch_weights(["mastered", "learning", "hard", "suspect"], device="cpu")

    assert policy.version == 1
    assert weights.mean().item() == pytest.approx(1.0)
    assert weights[2].item() > weights[0].item()
    assert weights[3].item() <= weights[2].item()
    assert torch.all((weights >= 0.7) & (weights <= 1.3))


def test_state_store_builds_next_policy_version() -> None:
    """A policy is generated only after state observations are updated."""
    store = SampleStateStore()
    store.update(
        [
            {"sample_id": "easy", "weighted_normalized_loss": 0.1},
            {"sample_id": "hard", "weighted_normalized_loss": 1.0},
        ]
    )

    policy = store.build_weight_policy()

    assert policy.version == 1
    assert policy.weights["easy"] <= 1.0
    assert policy.weights["hard"] >= 1.0


def test_combined_policy_caps_weight_times_sampler_exposure() -> None:
    """RF7 keeps combined sampler and loss-weight contribution under 2.5."""
    policy = SampleWeightPolicy()
    policy.update_from_states(
        [
            {"sample_id": "hard", "state": "HARD_LEARNABLE"},
            {"sample_id": "learning", "state": "LEARNING"},
            {"sample_id": "mastered", "state": "MASTERED"},
        ]
    )
    exposure = {"hard": 2.0, "learning": 0.5, "mastered": 0.5}
    weights = policy.batch_weights(
        ["hard", "learning", "mastered"],
        device="cpu",
        exposure_multipliers=exposure,
        effective_cap=2.5,
    )

    assert weights.mean().item() == pytest.approx(1.0)
    assert all(
        float(weight) * exposure[sample_id] <= 2.5 + 1e-5
        for sample_id, weight in zip(("hard", "learning", "mastered"), weights)
    )
    assert cap_effective_contribution(1.3, 3.0) == pytest.approx(2.5)


def test_combined_policy_prioritizes_cap_for_overexposed_replay_samples() -> None:
    """An exposure above cap/min lowers only that sample's effective floor."""
    policy = SampleWeightPolicy()
    policy.update_from_states(
        [
            {"sample_id": "hard", "state": "HARD_LEARNABLE"},
            {"sample_id": "learning", "state": "LEARNING"},
            {"sample_id": "mastered", "state": "MASTERED"},
        ]
    )
    exposure = {"hard": 5.0, "learning": 0.5, "mastered": 0.5}
    weights = policy.batch_weights(
        ["hard", "learning", "mastered"],
        device="cpu",
        exposure_multipliers=exposure,
        effective_cap=2.5,
    )

    assert weights.mean().item() == pytest.approx(1.0)
    assert float(weights[0]) < policy.minimum
    assert float(weights[0]) * exposure["hard"] <= 2.5 + 1e-5
    assert torch.all(weights <= torch.tensor([0.5, 1.3, 1.3]) + 1e-5)


def test_combined_policy_keeps_cap_when_mean_one_is_infeasible() -> None:
    """A fully overexposed batch clips safely instead of raising."""
    policy = SampleWeightPolicy()
    policy.update_from_states(
        [
            {"sample_id": "a", "state": "HARD_LEARNABLE"},
            {"sample_id": "b", "state": "LEARNING"},
            {"sample_id": "c", "state": "MASTERED"},
        ]
    )
    weights = policy.batch_weights(
        ["a", "b", "c"],
        device="cpu",
        exposure_multipliers={"a": 5.0, "b": 5.0, "c": 5.0},
        effective_cap=2.5,
    )

    assert weights.mean().item() < 1.0
    assert torch.all(weights * 5.0 <= 2.5 + 1e-5)
