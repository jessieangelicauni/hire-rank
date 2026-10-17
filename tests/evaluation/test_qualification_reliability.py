from __future__ import annotations

import pytest

from candidate_ranking.models import Assessment, RequirementEvaluation
from candidate_ranking.evaluation.qualification_reliability import (
    cohens_kappa,
    build_reliability_report,
)


def _assessment(qualified: bool, evidence_strength: str = "strong") -> Assessment:
    return Assessment(
        job_profile_id="jd-1", candidate_id="cand-1", generated_by_model="test",
        must_have_evaluations=[
            RequirementEvaluation(
                requirement="Python", evidence="x",
                evidence_strength=evidence_strength,
            )
        ],
        qualified=qualified,
    )


def test_cohens_kappa_perfect_agreement_returns_one():
    labels = [True, True, False, False, True]
    assert cohens_kappa(labels, labels) == pytest.approx(1.0)


def test_cohens_kappa_no_data_returns_one():
    assert cohens_kappa([], []) == 1.0


def test_cohens_kappa_chance_level_agreement_is_near_zero():
    # Independent 50/50 label sequences with matching base rates and no real
    # correlation should land near 0, not score high just from raw overlap.
    a = [True, False, True, False, True, False, True, False]
    b = [True, True, False, False, True, True, False, False]
    kappa = cohens_kappa(a, b)
    assert -0.3 <= kappa <= 0.3


def test_cohens_kappa_systematic_disagreement_is_negative():
    a = [True, True, False, False]
    b = [False, False, True, True]
    assert cohens_kappa(a, b) < 0


def test_build_reliability_report_perfect_consistency():
    repeated = {
        "jd-1/cand-1": [_assessment(True), _assessment(True), _assessment(True)],
        "jd-1/cand-2": [_assessment(False), _assessment(False), _assessment(False)],
    }
    report = build_reliability_report(repeated)
    assert report.n_pairs == 2
    assert report.n_repeats == 3
    assert report.qualified_agreement_rate == pytest.approx(1.0)
    assert report.qualified_cohens_kappa == pytest.approx(1.0)
    assert report.evidence_strength_agreement_rate == pytest.approx(1.0)


def test_build_reliability_report_flags_inconsistent_evidence_strength():
    repeated = {
        "jd-1/cand-1": [
            _assessment(True, evidence_strength="strong"),
            _assessment(False, evidence_strength="none"),
        ],
    }
    report = build_reliability_report(repeated)
    assert report.qualified_agreement_rate == pytest.approx(0.0)
    assert report.evidence_strength_agreement_rate == pytest.approx(0.0)


def test_build_reliability_report_skips_pairs_with_fewer_than_two_repeats():
    repeated = {
        "jd-1/cand-1": [_assessment(True)],
        "jd-1/cand-2": [_assessment(True), _assessment(True)],
    }
    report = build_reliability_report(repeated)
    assert report.n_pairs == 1
    assert "jd-1/cand-1" not in report.per_pair_qualified


def test_build_reliability_report_raises_when_no_pairs_qualify():
    with pytest.raises(ValueError, match="No pairs"):
        build_reliability_report({"jd-1/cand-1": [_assessment(True)]})
