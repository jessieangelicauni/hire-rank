from __future__ import annotations

import pytest

from candidate_ranking.evaluation.criteria_optimization import CriteriaEvalRecord, RequirementTriple, triple_key
from scripts.evaluate_criteria_comparison import compare_criteria


def _record(cand_id: str, score: float, confidence: float) -> CriteriaEvalRecord:
    return CriteriaEvalRecord(
        triple=RequirementTriple(job_description_id="jd-1", candidate_id=cand_id, requirement="Python"),
        score=score,
        confidence=confidence,
    )


def test_compare_criteria_reports_mean_confidence_and_accuracy_for_each_side():
    baseline = [_record("c-1", score=2.0, confidence=0.6), _record("c-2", score=1.0, confidence=0.4)]
    optimized = [_record("c-1", score=3.0, confidence=0.9), _record("c-2", score=3.0, confidence=0.85)]
    proxy_labels = {triple_key(baseline[0].triple): 3, triple_key(baseline[1].triple): 3}

    report = compare_criteria(baseline, optimized, proxy_labels)

    assert report["baseline"]["mean_confidence"] == pytest.approx(0.5)
    assert report["baseline"]["accuracy"] == pytest.approx(0.5)  # only c-1 (2 vs 3) is within 1
    assert report["optimized"]["mean_confidence"] == pytest.approx(0.875)
    assert report["optimized"]["accuracy"] == pytest.approx(1.0)  # both within 1 of label 3
    assert report["wilcoxon"] is not None
    assert "p_value" in report["wilcoxon"]


def test_compare_criteria_returns_none_wilcoxon_for_identical_confidences():
    baseline = [_record("c-1", score=2.0, confidence=0.5), _record("c-2", score=2.0, confidence=0.5)]
    optimized = [_record("c-1", score=2.0, confidence=0.5), _record("c-2", score=2.0, confidence=0.5)]
    proxy_labels = {triple_key(baseline[0].triple): 2, triple_key(baseline[1].triple): 2}

    report = compare_criteria(baseline, optimized, proxy_labels)

    assert report["wilcoxon"] is None  # scipy.wilcoxon raises on all-zero differences
