"""Focused tests for the experiment-side RF4 longitudinal analyzer."""

from experiments.mvtec_sample_dynamics.analyze_rf4_longitudinal import (
    _analysis_matched_mask_iou,
    _analysis_probe_conflict,
    _correlation_entry,
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
