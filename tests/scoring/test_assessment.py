from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from candidate_ranking.models import Candidate, JDSkills, JobDescription
from candidate_ranking.scoring.assessment import (
    AssessmentGenerationError,
    JEV_MODEL_NAME,
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


def _high_confidence_answers() -> list[JevAnswer]:
    return [
        JevAnswer(key="overall_fit_score", kind="score", value=3.0, confidence=0.9),
        JevAnswer(key="overall_recommendation", kind="choice", value="hire", confidence=0.85),
        JevAnswer(key="meets_min_qualifications", kind="noul", value=True, confidence=0.95),
        JevAnswer(key="requirement::Python", kind="score", value=4.0, confidence=0.9),
        JevAnswer(key="requirement::SQL", kind="score", value=1.0, confidence=0.8),
    ]


def test_generate_assessment_maps_jev_answers_onto_assessment():
    jev_client = Mock()
    jev_client.evaluate.return_value = _high_confidence_answers()

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills(), n_calls=1)

    assert assessment.job_description_id == "jd-1"
    assert assessment.candidate_id == "cand-1"
    assert assessment.generated_by_model == JEV_MODEL_NAME
    assert assessment.overall_fit_score == 75.0  # score 3 of 4 max -> 3 * (100/4)
    assert assessment.overall_recommendation == "hire"
    assert assessment.meets_min_qualifications is True
    assert assessment.requirement_scores == {"Python": 100.0, "SQL": 25.0}
    assert assessment.confidence["overall_fit_score"] == 0.9
    jev_client.evaluate.assert_called_once()
    state, questions = jev_client.evaluate.call_args.args
    assert "Backend Engineer" in state
    assert "I know Python." in state
    question_keys = {q.key for q in questions}
    assert question_keys == {
        "overall_fit_score", "overall_recommendation", "meets_min_qualifications",
        "requirement::Python", "requirement::SQL",
    }


def test_generate_assessment_skips_requirement_scores_without_jd_skills():
    jev_client = Mock()
    jev_client.evaluate.return_value = [
        JevAnswer(key="overall_fit_score", kind="score", value=2.0, confidence=0.9),
        JevAnswer(key="overall_recommendation", kind="choice", value="maybe", confidence=0.7),
        JevAnswer(key="meets_min_qualifications", kind="noul", value=False, confidence=0.8),
    ]

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, jd_skills=None, n_calls=1)

    assert assessment.requirement_scores == {}
    _, questions = jev_client.evaluate.call_args.args
    assert all(not q.key.startswith("requirement::") for q in questions)


def test_generate_assessment_retries_once_on_low_confidence_then_accepts():
    jev_client = Mock()
    low_confidence_answers = [
        JevAnswer(key="overall_fit_score", kind="score", value=3.0, confidence=0.3),
        JevAnswer(key="overall_recommendation", kind="choice", value="hire", confidence=0.85),
        JevAnswer(key="meets_min_qualifications", kind="noul", value=True, confidence=0.95),
    ]
    jev_client.evaluate.side_effect = [low_confidence_answers, _high_confidence_answers()]

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills(), n_calls=1)

    assert jev_client.evaluate.call_count == 2
    assert assessment.confidence["overall_fit_score"] == 0.9
    second_state, _ = jev_client.evaluate.call_args_list[1].args
    assert "low-confidence" in second_state


def test_generate_assessment_wraps_jev_client_error():
    jev_client = Mock()
    jev_client.evaluate.side_effect = JevClientError("network down")

    with pytest.raises(AssessmentGenerationError, match="network down"):
        generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills(), n_calls=1)


def test_generate_assessment_wraps_unusable_jev_response():
    missing_key_client = Mock()
    missing_key_client.evaluate.return_value = [
        JevAnswer(key="overall_fit_score", kind="score", value=3.0, confidence=0.9),
        JevAnswer(key="overall_recommendation", kind="choice", value="hire", confidence=0.85),
    ]
    with pytest.raises(AssessmentGenerationError, match="jd-1/cand-1"):
        generate_assessment(_jd(), _candidate(), missing_key_client, JEV_MODEL_NAME, _jd_skills(), n_calls=1)

    invalid_value_client = Mock()
    invalid_value_client.evaluate.return_value = [
        JevAnswer(key="overall_fit_score", kind="score", value=3.0, confidence=0.9),
        JevAnswer(key="overall_recommendation", kind="choice", value="strongly_hire", confidence=0.85),
        JevAnswer(key="meets_min_qualifications", kind="noul", value=True, confidence=0.95),
    ]
    with pytest.raises(AssessmentGenerationError, match="jd-1/cand-1"):
        generate_assessment(_jd(), _candidate(), invalid_value_client, JEV_MODEL_NAME, _jd_skills(), n_calls=1)


def test_generate_assessment_defaults_to_three_calls_and_averages():
    jev_client = Mock()
    jev_client.evaluate.side_effect = [
        [
            JevAnswer(key="overall_fit_score", kind="score", value=2.0, confidence=0.9),
            JevAnswer(key="overall_recommendation", kind="choice", value="hire", confidence=0.9),
            JevAnswer(key="meets_min_qualifications", kind="noul", value=True, confidence=0.9),
            JevAnswer(key="requirement::Python", kind="score", value=2.0, confidence=0.8),
            JevAnswer(key="requirement::SQL", kind="score", value=0.0, confidence=0.8),
        ],
        [
            JevAnswer(key="overall_fit_score", kind="score", value=4.0, confidence=0.9),
            JevAnswer(key="overall_recommendation", kind="choice", value="hire", confidence=0.9),
            JevAnswer(key="meets_min_qualifications", kind="noul", value=True, confidence=0.9),
            JevAnswer(key="requirement::Python", kind="score", value=4.0, confidence=0.8),
            JevAnswer(key="requirement::SQL", kind="score", value=2.0, confidence=0.8),
        ],
        [
            JevAnswer(key="overall_fit_score", kind="score", value=3.0, confidence=0.9),
            JevAnswer(key="overall_recommendation", kind="choice", value="maybe", confidence=0.9),
            JevAnswer(key="meets_min_qualifications", kind="noul", value=True, confidence=0.9),
            JevAnswer(key="requirement::Python", kind="score", value=3.0, confidence=0.8),
            JevAnswer(key="requirement::SQL", kind="score", value=1.0, confidence=0.8),
        ],
    ]

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills())

    assert jev_client.evaluate.call_count == 3
    assert assessment.overall_fit_score == pytest.approx((50.0 + 100.0 + 75.0) / 3)
    assert assessment.overall_recommendation == "hire"  # 2 of 3 votes
    assert assessment.meets_min_qualifications is True  # unanimous
    assert assessment.requirement_scores["Python"] == pytest.approx((50.0 + 100.0 + 75.0) / 3)
    assert assessment.requirement_scores["SQL"] == pytest.approx((0.0 + 50.0 + 25.0) / 3)


def test_generate_assessment_n_calls_one_skips_averaging():
    jev_client = Mock()
    jev_client.evaluate.return_value = _high_confidence_answers()

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills(), n_calls=1)

    jev_client.evaluate.assert_called_once()
    assert assessment.overall_fit_score == 75.0


def test_load_or_generate_assessment_uses_cache_on_second_call(tmp_path: Path):
    jev_client = Mock()
    jev_client.evaluate.return_value = _high_confidence_answers()

    first = load_or_generate_assessment(
        _jd(), _candidate(), jev_client, JEV_MODEL_NAME, tmp_path, _jd_skills(), n_calls=1
    )
    second = load_or_generate_assessment(
        _jd(), _candidate(), jev_client, JEV_MODEL_NAME, tmp_path, _jd_skills(), n_calls=1
    )

    assert first == second
    jev_client.evaluate.assert_called_once()
    cache_file = tmp_path / "assessments" / "jd-1" / "cand-1.json"
    assert cache_file.exists()
    cached = json.loads(cache_file.read_text(encoding="utf-8"))
    assert cached["assessment"]["overall_fit_score"] == 75.0


def test_load_or_generate_assessment_cache_key_depends_on_n_calls(tmp_path: Path):
    jev_client = Mock()
    jev_client.evaluate.return_value = _high_confidence_answers()

    load_or_generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, tmp_path, _jd_skills(), n_calls=1)
    assert jev_client.evaluate.call_count == 1

    load_or_generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, tmp_path, _jd_skills(), n_calls=3)
    assert jev_client.evaluate.call_count == 4
