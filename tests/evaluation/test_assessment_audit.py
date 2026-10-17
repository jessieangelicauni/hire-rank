from __future__ import annotations

import pytest
from candidate_ranking.models import Assessment, RequirementEvaluation
from candidate_ranking.evaluation.assessment_audit import audit_assessments, AssessmentAuditReport


def _eval(requirement: str, strength: str, evidence: str = "") -> RequirementEvaluation:
    return RequirementEvaluation(
        requirement=requirement,
        evidence=evidence, evidence_strength=strength,
    )


def _assessment(
    jd_id: str,
    cid: str,
    evals: list[RequirementEvaluation],
    qualified: bool | None = None,
) -> Assessment:
    if qualified is None:
        qualified = all(ev.evidence_strength != "none" for ev in evals)
    return Assessment(
        job_profile_id=jd_id, candidate_id=cid,
        generated_by_model="jev/jev-latest",
        must_have_evaluations=evals,
        qualified=qualified,
    )


# --- evidence coverage ---

def test_evidence_coverage_rate_counts_non_empty_evidence():
    assessments = {
        "jd-1": {
            "a": _assessment("jd-1", "a", [
                _eval("Python", "strong", evidence="Built FastAPI at Tokopedia"),
                _eval("SQL", "none", evidence=""),
            ]),
        }
    }
    report = audit_assessments(assessments)
    assert report.must_have_evals_total == 2
    assert report.must_have_evals_with_evidence == 1
    assert report.evidence_coverage_rate == pytest.approx(0.5)


def test_evidence_coverage_rate_is_one_when_all_filled():
    assessments = {
        "jd-1": {
            "a": _assessment("jd-1", "a", [
                _eval("Python", "strong", evidence="did Python"),
            ]),
        }
    }
    report = audit_assessments(assessments)
    assert report.evidence_coverage_rate == pytest.approx(1.0)


# --- qualified flag integrity ---

def test_no_flag_errors_when_flag_matches_evidence_strength():
    assessments = {
        "jd-1": {
            "a": _assessment("jd-1", "a", [_eval("Python", "strong")], qualified=True),
            "b": _assessment("jd-1", "b", [_eval("Python", "none")], qualified=False),
        }
    }
    report = audit_assessments(assessments)
    assert report.qualified_flag_errors == 0


def test_flag_error_detected_when_qualified_wrong():
    assessments = {
        "jd-1": {
            "a": _assessment("jd-1", "a", [_eval("Python", "none")], qualified=True),
        }
    }
    report = audit_assessments(assessments)
    assert report.qualified_flag_errors == 1


# --- qualification funnel ---

def test_qualification_rate_per_jd():
    assessments = {
        "jd-1": {
            "a": _assessment("jd-1", "a", [_eval("Python", "strong")]),
            "b": _assessment("jd-1", "b", [_eval("Python", "none")]),
            "c": _assessment("jd-1", "c", [_eval("Python", "strong")]),
        }
    }
    report = audit_assessments(assessments)
    assert report.qualification_rate_per_jp["jd-1"] == pytest.approx(2 / 3)
    assert report.mean_qualification_rate == pytest.approx(2 / 3)


def test_zero_qualified_jd_flagged():
    assessments = {
        "jd-1": {
            "a": _assessment("jd-1", "a", [_eval("Python", "none")]),
        }
    }
    report = audit_assessments(assessments)
    assert "jd-1" in report.zero_qualified_jps
    assert "jd-1" not in report.fully_qualified_jps


def test_fully_qualified_jd_flagged():
    assessments = {
        "jd-1": {
            "a": _assessment("jd-1", "a", [_eval("Python", "strong")]),
            "b": _assessment("jd-1", "b", [_eval("Python", "adequate")]),
        }
    }
    report = audit_assessments(assessments)
    assert "jd-1" in report.fully_qualified_jps
    assert "jd-1" not in report.zero_qualified_jps


def test_empty_assessments_returns_zero_counts():
    report = audit_assessments({})
    assert report.n_jps == 0
    assert report.n_assessments == 0
    assert report.evidence_coverage_rate == pytest.approx(1.0)
    assert report.mean_qualification_rate == pytest.approx(0.0)


# --- evidence strength distribution ---

def test_evidence_strength_distribution_sums_to_one():
    assessments = {
        "jd-1": {
            "a": _assessment("jd-1", "a", [
                _eval("Python", "strong"),
                _eval("SQL", "none"),
            ]),
            "b": _assessment("jd-1", "b", [
                _eval("Python", "adequate"),
                _eval("SQL", "weak"),
            ]),
        }
    }
    report = audit_assessments(assessments)
    total = sum(report.evidence_strength_distribution.values())
    assert total == pytest.approx(1.0)
    assert report.evidence_strength_distribution["strong"] == pytest.approx(0.25)
    assert report.evidence_strength_distribution["none"] == pytest.approx(0.25)


# --- most common disqualifiers ---

def test_most_common_disqualifiers_ranked_by_count():
    assessments = {
        "jd-1": {
            "a": _assessment("jd-1", "a", [
                _eval("Python", "none"),
                _eval("SQL", "none"),
            ]),
            "b": _assessment("jd-1", "b", [
                _eval("Python", "none"),
            ]),
        }
    }
    report = audit_assessments(assessments)
    disqs = dict(report.most_common_disqualifiers)
    assert disqs["Python"] == 2
    assert disqs["SQL"] == 1
    assert report.most_common_disqualifiers[0][0] == "Python"  # most common first


def test_most_common_disqualifiers_empty_when_all_qualified():
    assessments = {
        "jd-1": {
            "a": _assessment("jd-1", "a", [_eval("Python", "strong")]),
        }
    }
    report = audit_assessments(assessments)
    assert report.most_common_disqualifiers == []
