from __future__ import annotations
import pytest
from pydantic import ValidationError
from candidate_ranking.models import (
    AdaptiveRequirement, AdaptiveRequirementEvaluation, Assessment, JobProfile, RequirementEvaluation,
)

def _req_eval(evidence_strength="strong") -> RequirementEvaluation:
    return RequirementEvaluation(
        requirement="Python", evidence="FastAPI at PT XYZ",
        evidence_strength=evidence_strength,
    )

def _assessment(qualified=True) -> Assessment:
    return Assessment(
        job_profile_id="jd-1", candidate_id="cand-1",
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

def test_assessment_has_no_career_trajectory_field():
    assert "career_trajectory" not in Assessment.model_fields

def test_requirement_evaluation_evidence_strength_literals():
    for s in ("strong", "adequate", "weak", "none"):
        ev = _req_eval(evidence_strength=s)
        assert ev.evidence_strength == s
    with pytest.raises(ValidationError):
        _req_eval(evidence_strength="UNKNOWN")

def test_job_profile_defaults_empty():
    jp = JobProfile(id="jd-1", title="Backend", raw_text="req.", source_path="j.txt", generated_by_model="qwen2.5:14b")
    assert jp.must_have_skills == []
    assert jp.nice_to_have_skills == []
    assert jp.education_requirement is None
    assert jp.seniority_requirement is None
    assert jp.other_requirements == []

def test_job_profile_adaptive_fields_default_empty():
    jp = JobProfile(id="jd-1", title="Backend", raw_text="req.", source_path="j.txt", generated_by_model="qwen2.5:14b")
    assert jp.skill_criteria == []
    assert jp.total_years_criteria is None
    assert jp.adaptive_requirements == []

def test_job_profile_fields_roundtrip():
    jp = JobProfile(
        id="jd-1", title="Backend", raw_text="req.", source_path="j.txt", generated_by_model="qwen2.5:14b",
        must_have_skills=["Python"], nice_to_have_skills=["Docker"],
        education_requirement="Bachelor's degree", seniority_requirement="5+ years",
        other_requirements=["AWS Certified Solutions Architect"],
    )
    assert jp.must_have_skills == ["Python"]
    assert jp.nice_to_have_skills == ["Docker"]
    assert jp.education_requirement == "Bachelor's degree"
    assert jp.seniority_requirement == "5+ years"
    assert jp.other_requirements == ["AWS Certified Solutions Architect"]

def test_requirement_evaluation_accepts_none_evidence_strength():
    ev = RequirementEvaluation(
        requirement="Python",
        evidence="Not found", evidence_strength="none",
    )
    assert ev.evidence_strength == "none"

def test_adaptive_requirement_roundtrips():
    req = AdaptiveRequirement(
        source="education", requirement="Bachelor's degree in CS",
        criteria_meets="Has a CS or related bachelor's degree",
        criteria_partially_meets="Has a related but non-CS technical degree",
        criteria_does_not_meet="No degree mentioned",
    )
    assert req.source == "education"
    assert req.criteria_meets == "Has a CS or related bachelor's degree"

def test_adaptive_requirement_evaluation_roundtrips():
    ev = AdaptiveRequirementEvaluation(
        source="seniority", requirement="5+ years", evidence="6 years at PT XYZ",
        match_level="meets", confidence=0.9,
    )
    assert ev.match_level == "meets"
    assert ev.confidence == 0.9
    with pytest.raises(ValidationError):
        AdaptiveRequirementEvaluation(
            source="seniority", requirement="5+ years", evidence="x",
            match_level="UNKNOWN", confidence=0.9,
        )
