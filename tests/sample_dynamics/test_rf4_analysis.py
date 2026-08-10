"""Focused tests for the experiment-side RF4 longitudinal analyzer."""

import math

from experiments.mvtec_sample_dynamics.analyze_rf4_longitudinal import (
    _analysis_matched_mask_iou,
    _analysis_probe_conflict,
    _build_correlation_analysis,
    _correlation_entry,
    _distribution,
    _hard_candidate_analysis,
    _stability,
)
from rfdetr.sample_dynamics.state import StatePolicy


def test_mask_iou_without_match_is_undefined_but_true_zero_is_preserved() -> None:
    """Separate no-match serialization from a real zero-IoU match."""
    assert _analysis_matched_mask_iou({"gt_count": 1, "matched_count": 0, "matched_mask_iou": 0.0}) is None
    assert _analysis_matched_mask_iou({"gt_count": 0, "matched_count": 0, "matched_mask_iou": 0.0}) is None
    assert _analysis_matched_mask_iou({"gt_count": 1, "matched_count": 1, "matched_mask_iou": 0.0}) == 0.0


def test_empty_gt_recall_zero_is_not_analysis_conflict() -> None:
    """Keep frozen state replay separate from the analysis-side undefined recall rule."""
    assert not _analysis_probe_conflict(
        {"gt_count": 0, "matched_count": 0, "gt_recall": 0.0, "fn": 0, "class_error": 0},
        StatePolicy(),
    )


def test_constant_correlation_is_explicitly_undefined() -> None:
    """Do not manufacture a correlation when either trajectory is constant."""
    result = _correlation_entry([1.0, 1.0, 1.0], [0.0, 1.0, 2.0])
    assert result["pearson"] is None
    assert result["spearman"] is None


def test_stability_exposes_per_sample_distribution() -> None:
    """Longitudinal stability must retain both summary quantiles and sample IDs."""
    result = _stability({"a": ["EASY", "HARD", "HARD"], "b": ["EASY", "EASY", "EASY"]})
    assert result["transition_count_by_sample"] == {"a": 1, "b": 0}
    assert result["transition_count_distribution"]["median"] == 0.5
    assert result["transition_count_distribution"]["p90"] == 0.9


def test_distribution_is_deterministic_for_uneven_sample_counts() -> None:
    """P90 uses an explicit interpolated quantile rather than an opaque library default."""
    result = _distribution([0.0, 1.0, 2.0, 10.0])
    assert result["median"] == 1.5
    assert math.isclose(result["p90"], 7.6)


def test_future_correlation_includes_fn_reduction() -> None:
    """Future-predictability analysis must expose FN reduction as a target."""
    sample = {
        "instant_loss_percentile": [0.9, 0.7, 0.5],
        "difficulty": [0.8, 0.6, 0.4],
        "loss": [3.0, 2.0, 1.0],
        "matched_mask_iou": [0.2, 0.4, 0.6],
        "fn": [2, 1, 0],
    }
    result = _build_correlation_analysis(
        {"sample": sample},
        signal_key="difficulty",
        signal_label="Dynamics",
        epoch_count=3,
    )
    assert "fn_reduction" in result["targets"]


def test_hard_candidate_natural_recovery_excludes_corruption() -> None:
    """Controlled corruption must not inflate natural HARD_LEARNABLE recovery counts."""
    trajectory = {
        "dynamics_state": ["HARD_LEARNABLE", "HARD_LEARNABLE", "LEARNING"],
        "instant_bucket": ["HARD", "HARD", "MIDDLE"],
        "loss": [3.0, 2.0, 1.0],
        "matched_mask_iou": [0.1, 0.2, 0.3],
        "fn": [1, 1, 0],
    }
    result = _hard_candidate_analysis({"natural": trajectory, "corrupt": trajectory}, StatePolicy(), excluded_sample_ids={"corrupt"})
    assert result["unique_samples_with_hard_event"] == 1
    assert result["excluded_controlled_corruption_event_count"] == 2
