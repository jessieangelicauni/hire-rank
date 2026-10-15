from __future__ import annotations
import pytest
from pydantic import ValidationError
from candidate_ranking.models import (
    Assessment, CareerTrajectory, JDRequirement, JDRequirements,
    JDSkills, JobProfile, RedFlag, RequirementEvaluation,
)

def _req_eval(verdict="MET") -> RequirementEvaluation:
    return RequirementEvaluation(
        requirement="Python", verdict=verdict, evidence="FastAPI at PT XYZ",
        evidence_strength="strong", recency="current", years_demonstrated=3,
    )

def _trajectory() -> CareerTrajectory:
    return CareerTrajectory(
        direction="ascending", evidence="Junior→Senior 2019-2023",
        tenure_pattern="stable", average_tenure_months=24,
        most_recent_role_duration_months=18,
        has_multinational_experience=False, multinational_evidence=None,
    )

def _assessment(qualified=True) -> Assessment:
    return Assessment(
        job_description_id="jd-1", candidate_id="cand-1",
        generated_by_model="qwen2.5:14b",
        must_have_evaluations=[_req_eval()],
        qualified=qualified,
    )

def test_assessment_qualified_flag():
    assert _assessment(qualified=True).qualified is True
    assert _assessment(qualified=False).qualified is False

def test_assessment_disqualification_reason_none_when_qualified():
    a = _assessment(qualified=True)
    assert a.disqualification_reason is None

def test_assessment_has_no_jev_fields():
    assert "requirement_scores" not in Assessment.model_fields
    assert "composite_fit_score" not in Assessment.model_fields
    assert "confidence" not in Assessment.model_fields
    assert "seniority_years_fit_score" not in Assessment.model_fields

def test_requirement_evaluation_verdict_literals():
    for v in ("MET", "PARTIALLY_MET", "NOT_MET"):
        ev = _req_eval(verdict=v)
        assert ev.verdict == v
    with pytest.raises(ValidationError):
        _req_eval(verdict="UNKNOWN")

def test_career_trajectory_roundtrips():
    t = _trajectory()
    assert t.direction == "ascending"
    assert t.has_multinational_experience is False

def test_red_flag_duration_optional():
    rf = RedFlag(type="employment_gap", detail="6-month gap", severity="minor")
    assert rf.duration_months is None

def test_jd_requirements_lists():
    jdr = JDRequirements(
        job_description_id="jd-1", generated_by_model="qwen2.5:14b",
        must_have=[JDRequirement(requirement="Python", context="Primary language")],
        nice_to_have=[JDRequirement(requirement="Docker", context="Preferred")],
    )
    assert len(jdr.must_have) == 1
    assert jdr.must_have[0].requirement == "Python"

def test_jd_requirements_defaults_empty():
    jdr = JDRequirements(job_description_id="jd-1", generated_by_model="qwen2.5:14b")
    assert jdr.must_have == []
    assert jdr.nice_to_have == []

def test_jd_skills_unchanged():
    jds = JDSkills(
        job_description_id="jd-1", generated_by_model="qwen2.5:14b",
        technical_skills=["Python"], must_have_skills=["Python"],
    )
    assert jds.must_have_skills == ["Python"]

def test_job_profile_defaults_empty():
    jp = JobProfile(job_description_id="jd-1", generated_by_model="qwen2.5:14b")
    assert jp.must_have_skills == []
    assert jp.nice_to_have_skills == []
    assert jp.education_requirement is None
    assert jp.seniority_requirement is None
    assert jp.other_requirements == []

def test_job_profile_fields_roundtrip():
    jp = JobProfile(
        job_description_id="jd-1", generated_by_model="qwen2.5:14b",
        must_have_skills=["Python"], nice_to_have_skills=["Docker"],
        education_requirement="Bachelor's degree", seniority_requirement="5+ years",
        other_requirements=["AWS Certified Solutions Architect"],
    )
    assert jp.must_have_skills == ["Python"]
    assert jp.nice_to_have_skills == ["Docker"]
    assert jp.education_requirement == "Bachelor's degree"
    assert jp.seniority_requirement == "5+ years"
    assert jp.other_requirements == ["AWS Certified Solutions Architect"]
