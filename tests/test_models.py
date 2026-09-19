from __future__ import annotations

import pytest
from pydantic import ValidationError

from candidate_ranking.models import Assessment, JDSkills


def test_assessment_accepts_new_score_based_fields():
    assessment = Assessment(
        job_description_id="jd-1",
        candidate_id="cand-1",
        generated_by_model="typesafe/jev",
        overall_fit_score=87.5,
        overall_recommendation="hire",
        meets_min_qualifications=True,
        requirement_scores={"python": 100.0, "sql": 50.0},
        confidence={"overall_fit_score": 0.9},
    )
    assert assessment.overall_fit_score == 87.5
    assert assessment.requirement_scores["python"] == 100.0


def test_assessment_rejects_score_out_of_range():
    with pytest.raises(ValidationError):
        Assessment(
            job_description_id="jd-1",
            candidate_id="cand-1",
            generated_by_model="typesafe/jev",
            overall_fit_score=150.0,
            overall_recommendation="hire",
            meets_min_qualifications=True,
        )


def test_assessment_rejects_unknown_recommendation():
    with pytest.raises(ValidationError):
        Assessment(
            job_description_id="jd-1",
            candidate_id="cand-1",
            generated_by_model="typesafe/jev",
            overall_fit_score=50.0,
            overall_recommendation="strongly_hire",
            meets_min_qualifications=True,
        )


def test_assessment_no_longer_has_free_text_fields():
    assert "strengths" not in Assessment.model_fields
    assert "weaknesses" not in Assessment.model_fields
    assert "reasoning" not in Assessment.model_fields
    assert "additional_skills" not in Assessment.model_fields


def test_assessment_accepts_certification_seniority_education_fields():
    assessment = Assessment(
        job_description_id="jd-1",
        candidate_id="cand-1",
        generated_by_model="typesafe/jev",
        overall_fit_score=87.5,
        overall_recommendation="hire",
        meets_min_qualifications=True,
        certification_results={"AWS Certified Solutions Architect": True, "PMP": False},
        seniority_fit_score=75.0,
        education_fit_score=None,
    )
    assert assessment.certification_results["AWS Certified Solutions Architect"] is True
    assert assessment.seniority_fit_score == 75.0
    assert assessment.education_fit_score is None


def test_assessment_certification_and_seniority_default_empty():
    assessment = Assessment(
        job_description_id="jd-1",
        candidate_id="cand-1",
        generated_by_model="typesafe/jev",
        overall_fit_score=50.0,
        overall_recommendation="maybe",
        meets_min_qualifications=False,
    )
    assert assessment.certification_results == {}
    assert assessment.seniority_fit_score is None
    assert assessment.education_fit_score is None


def test_jd_skills_accepts_certifications_seniority_education():
    jd_skills = JDSkills(
        job_description_id="jd-1",
        generated_by_model="qwen2.5:14b",
        technical_skills=["Python"],
        certifications=["AWS Certified Solutions Architect"],
        seniority_requirement="5+ years of backend development experience",
        education_requirement="Bachelor's degree in Computer Science or related field",
    )
    assert jd_skills.certifications == ["AWS Certified Solutions Architect"]
    assert jd_skills.seniority_requirement == "5+ years of backend development experience"


def test_jd_skills_certifications_seniority_education_default_empty():
    jd_skills = JDSkills(job_description_id="jd-1", generated_by_model="qwen2.5:14b", technical_skills=["Python"])
    assert jd_skills.certifications == []
    assert jd_skills.seniority_requirement is None
    assert jd_skills.education_requirement is None
