from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from candidate_ranking.models import Assessment, Candidate, JDSkills, JobDescription
from candidate_ranking.scoring.assessment import (
    AssessmentGenerationError,
    JEV_MODEL_NAME,
    _build_questions,
    generate_assessment,
    load_or_generate_assessment,
)
from candidate_ranking.scoring.jev_client import JevAnswer, JevClientError


def _jd() -> JobDescription:
    return JobDescription(id="jd-1", title="Backend Engineer", raw_text="Needs Python and SQL.", source_path="jd.pdf")


def _candidate() -> Candidate:
    return Candidate(
        id="cand-1", source_path="cv.pdf", raw_text="I know Python.", num_pages=1, char_count=20,
        parse_status="ok", skills=["Python"],
    )


def _jd_skills() -> JDSkills:
    return JDSkills(job_description_id="jd-1", generated_by_model="qwen2.5:14b", technical_skills=["Python", "SQL"])


def _jd_skills_full() -> JDSkills:
    return JDSkills(
        job_description_id="jd-1",
        generated_by_model="qwen2.5:14b",
        technical_skills=["Python"],
        certifications=["AWS Certified Solutions Architect"],
        seniority_requirement="5+ years of backend development experience",
        seniority_min_years=5.0,
        education_requirement="Bachelor's degree in Computer Science or related field",
    )


def _high_confidence_answers() -> list[JevAnswer]:
    return [
        JevAnswer(key="requirement::Python", kind="score", value=4.0, confidence=0.9),
        JevAnswer(key="requirement::SQL", kind="score", value=1.0, confidence=0.8),
    ]


def test_generate_assessment_maps_jev_answers_onto_assessment():
    jev_client = Mock()
    jev_client.evaluate.return_value = _high_confidence_answers()

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills())

    assert assessment.job_description_id == "jd-1"
    assert assessment.candidate_id == "cand-1"
    assert assessment.generated_by_model == JEV_MODEL_NAME
    assert assessment.requirement_scores == {"Python": 100.0, "SQL": 25.0}
    assert assessment.composite_fit_score == 62.5  # mean(100, 25)
    assert assessment.confidence["requirement::Python"] == 0.9
    jev_client.evaluate.assert_called_once()
    state, questions = jev_client.evaluate.call_args.args
    assert "Backend Engineer" in state
    assert "I know Python." in state
    question_keys = {q.key for q in questions}
    assert question_keys == {"requirement::Python", "requirement::SQL"}


def test_generate_assessment_skips_requirement_scores_without_jd_skills():
    jev_client = Mock()
    jev_client.evaluate.return_value = []

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, jd_skills=None)

    assert assessment.requirement_scores == {}
    assert assessment.composite_fit_score == 0.0
    _, questions = jev_client.evaluate.call_args.args
    assert questions == []


def test_generate_assessment_retries_once_on_low_confidence_then_accepts():
    jev_client = Mock()
    jd_skills = JDSkills(
        job_description_id="jd-1", generated_by_model="qwen2.5:14b", technical_skills=["Python"],
        seniority_requirement="5+ years", seniority_min_years=5.0,
    )
    low_confidence_answers = [
        JevAnswer(key="requirement::Python", kind="score", value=4.0, confidence=0.9),
        JevAnswer(key="seniority_years", kind="score", value=3.0, confidence=0.3),
    ]
    high_confidence_answers = [
        JevAnswer(key="requirement::Python", kind="score", value=4.0, confidence=0.9),
        JevAnswer(key="seniority_years", kind="score", value=3.0, confidence=0.9),
    ]
    jev_client.evaluate.side_effect = [low_confidence_answers, high_confidence_answers]

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, jd_skills)

    assert jev_client.evaluate.call_count == 2
    assert assessment.confidence["seniority_years"] == 0.9
    second_state, _ = jev_client.evaluate.call_args_list[1].args
    assert "low-confidence" in second_state


def test_generate_assessment_wraps_jev_client_error():
    jev_client = Mock()
    jev_client.evaluate.side_effect = JevClientError("network down")

    with pytest.raises(AssessmentGenerationError, match="network down"):
        generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills())


def test_generate_assessment_preserves_probability_distributions():
    jev_client = Mock()
    jev_client.evaluate.return_value = [
        JevAnswer(key="requirement::Python", kind="score", value=4.0, confidence=0.9,
                   probabilities={"0": 0, "1": 0, "2": 0, "3": 0.1, "4": 0.9}),
    ]

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills())

    assert assessment.requirement_probabilities["Python"] == {"0": 0, "1": 0, "2": 0, "3": 0.1, "4": 0.9}
    assert assessment.seniority_probabilities is None
    assert assessment.education_probabilities is None


def test_load_or_generate_assessment_uses_cache_on_second_call(tmp_path: Path):
    jev_client = Mock()
    jev_client.evaluate.return_value = _high_confidence_answers()

    first = load_or_generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, tmp_path, _jd_skills())
    second = load_or_generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, tmp_path, _jd_skills())

    assert first == second
    jev_client.evaluate.assert_called_once()
    cache_file = tmp_path / "assessments" / "jd-1" / "cand-1.json"
    assert cache_file.exists()
    cached = json.loads(cache_file.read_text(encoding="utf-8"))
    assert cached["assessment"]["composite_fit_score"] == 62.5


def test_load_or_generate_assessment_cache_key_depends_on_jd_skills(tmp_path: Path):
    jev_client = Mock()
    jev_client.evaluate.return_value = _high_confidence_answers()

    load_or_generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, tmp_path, _jd_skills())
    assert jev_client.evaluate.call_count == 1

    load_or_generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, tmp_path, jd_skills=None)
    assert jev_client.evaluate.call_count == 2


def test_generate_assessment_builds_seniority_education_questions():
    jev_client = Mock()
    jev_client.evaluate.return_value = [
        JevAnswer(key="requirement::Python", kind="score", value=4.0, confidence=0.9),
        JevAnswer(key="seniority_years", kind="score", value=3.0, confidence=0.85),
        JevAnswer(key="education", kind="score", value=1.0, confidence=0.8),
    ]

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills_full())

    _, questions = jev_client.evaluate.call_args.args
    question_keys = {q.key for q in questions}
    assert "seniority_years" in question_keys
    assert "education" in question_keys
    assert not any(k.startswith("certification::") for k in question_keys)
    assert assessment.seniority_years_fit_score == 75.0
    assert assessment.education_fit_score == 25.0


def test_generate_assessment_omits_seniority_education_questions_when_not_stated():
    jev_client = Mock()
    jev_client.evaluate.return_value = [
        JevAnswer(key="requirement::Python", kind="score", value=4.0, confidence=0.9),
        JevAnswer(key="requirement::SQL", kind="score", value=1.0, confidence=0.8),
    ]

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills())

    _, questions = jev_client.evaluate.call_args.args
    question_keys = {q.key for q in questions}
    assert "seniority_years" not in question_keys
    assert "education" not in question_keys
    assert not any(k.startswith("certification::") for k in question_keys)
    assert assessment.seniority_years_fit_score is None
    assert assessment.education_fit_score is None


def test_seniority_and_education_criteria_are_five_level_evidence_based_scores():
    jd_skills = JDSkills(
        job_description_id="jd-1", generated_by_model="qwen2.5:14b", technical_skills=[],
        seniority_requirement="5+ years of backend experience",
        seniority_min_years=5.0,
        education_requirement="Bachelor's degree in Computer Science",
    )
    questions = _build_questions(jd_skills)
    by_key = {q.key: q for q in questions}

    assert by_key["seniority_years"].kind == "score"
    assert by_key["education"].kind == "score"
    assert len(by_key["seniority_years"].criteria) == 5
    assert len(by_key["education"].criteria) == 5

    seniority_years_criteria = " ".join(by_key["seniority_years"].criteria).lower()
    education_criteria = " ".join(by_key["education"].criteria).lower()

    for banned_phrase in ("supports that the candidate", "does not support that the candidate"):
        assert banned_phrase not in seniority_years_criteria
        assert banned_phrase not in education_criteria

    assert any(term in seniority_years_criteria for term in ("total experience", "years"))
    assert any(term in education_criteria for term in ("degree", "field", "credential"))


def test_build_questions_never_includes_recommendation_qualifications_certification():
    questions = _build_questions(
        JDSkills(
            job_description_id="jd-1", generated_by_model="qwen2.5:14b", technical_skills=["Python"],
            certifications=["PMP"], seniority_requirement="5+ years", seniority_min_years=5.0,
            education_requirement="Bachelor's degree",
        )
    )
    question_keys = {q.key for q in questions}
    assert "overall_recommendation" not in question_keys
    assert "meets_min_qualifications" not in question_keys
    assert not any(k.startswith("certification::") for k in question_keys)
    assert all(q.kind == "score" for q in questions)


def test_assessment_model_has_no_recommendation_qualifications_certification_fields():
    removed_fields = {
        "overall_recommendation", "meets_min_qualifications",
        "certification_results", "recommendation_probabilities",
    }
    assert removed_fields.isdisjoint(Assessment.model_fields)
