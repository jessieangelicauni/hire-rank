from __future__ import annotations

import pytest
from pydantic import ValidationError

from candidate_ranking.models import Assessment


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
