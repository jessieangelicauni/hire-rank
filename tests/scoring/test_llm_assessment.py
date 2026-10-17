from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from candidate_ranking.models import (
    Assessment, Candidate, JobDescription, JobProfile, RequirementEvaluation,
)
from candidate_ranking.scoring.llm_assessment import (
    AssessmentGenerationError,
    filter_assessable_candidates,
    generate_assessment,
    load_or_generate_assessment,
)


def _jd() -> JobDescription:
    return JobDescription(id="jd-1", title="Backend Engineer",
                          raw_text="Python 3+ yrs required.", source_path="jd.pdf")


def _candidate(parse_status="ok") -> Candidate:
    return Candidate(id="cand-1", source_path="cv.pdf", raw_text="I built FastAPI services.",
                     num_pages=1, char_count=30, parse_status=parse_status)


def _job_profile() -> JobProfile:
    return JobProfile(
        job_description_id="jd-1", generated_by_model="qwen2.5:14b",
        must_have_skills=["Python"], nice_to_have_skills=["Docker"],
    )


def _assessment(qualified: bool = True) -> Assessment:
    return Assessment(
        job_description_id="jd-1",
        candidate_id="cand-1",
        generated_by_model="jev/jev-latest+qwen2.5:14b",
        must_have_evaluations=[
            RequirementEvaluation(
                requirement="Python",
                verdict="MET" if qualified else "NOT_MET",
                evidence="FastAPI at PT XYZ" if qualified else "",
                evidence_strength="strong" if qualified else "none",
                recency="unknown",
                years_demonstrated=0,
            )
        ],
        qualified=qualified,
        disqualification_reason=None if qualified else "Failed must-have requirements: Python",
    )


def _fake_answers(profile: JobProfile, must_have_strength: str = "strong") -> list:
    from candidate_ranking.scoring.jev_client import JevAnswer
    answers = []
    for i in range(len(profile.must_have_skills)):
        answers.append(JevAnswer(key=f"must_have_{i}_evidence_strength", kind="choice",
                                 value=must_have_strength, confidence=0.9))
    for i in range(len(profile.nice_to_have_skills)):
        answers.append(JevAnswer(key=f"nice_to_have_{i}_evidence_strength", kind="choice",
                                 value="adequate", confidence=0.7))
    if profile.education_requirement:
        answers.append(JevAnswer(key="education_requirement", kind="choice",
                                 value="meets", confidence=0.8))
    if profile.seniority_requirement:
        answers.append(JevAnswer(key="seniority_requirement", kind="choice",
                                 value="meets_or_exceeds", confidence=0.8))
    for i in range(len(profile.other_requirements)):
        answers.append(JevAnswer(key=f"other_req_{i}", kind="noul", value=True, confidence=0.8))
    answers += [
        JevAnswer(key="total_relevant_years", kind="score", value=2.0, confidence=0.8),
        JevAnswer(key="career_direction", kind="choice", value="ascending", confidence=0.8),
        JevAnswer(key="tenure_pattern", kind="choice", value="stable", confidence=0.8),
        JevAnswer(key="average_tenure_months", kind="score", value=2.0, confidence=0.8),
        JevAnswer(key="most_recent_role_duration_months", kind="score", value=2.0, confidence=0.8),
        JevAnswer(key="has_multinational_experience", kind="noul", value=False, confidence=0.8),
    ]
    return answers


def _fake_free_text(profile: JobProfile):
    from candidate_ranking.scoring.llm_assessment import _FreeTextOutput, _SkillEvidenceItem
    return _FreeTextOutput(
        must_have_evidence=[
            _SkillEvidenceItem(requirement=s, evidence=f"Evidence for {s}")
            for s in profile.must_have_skills
        ],
        nice_to_have_evidence=[
            _SkillEvidenceItem(requirement=s, evidence=f"Evidence for {s}")
            for s in profile.nice_to_have_skills
        ],
        strongest_achievements=["Led migration project"],
        career_trajectory_evidence="Promoted from junior to senior in 3 years",
    )


def test_generate_assessment_returns_qualified_assessment():
    profile = _job_profile()
    jev = MagicMock()
    jev.evaluate.return_value = _fake_answers(profile)
    chain = MagicMock()
    chain.invoke.return_value = _fake_free_text(profile)
    result = generate_assessment(_jd(), _candidate(), profile, jev, chain, "qwen2.5:14b")
    assert isinstance(result, Assessment)
    assert result.qualified is True
    assert result.must_have_evaluations[0].requirement == "Python"
    assert result.must_have_evaluations[0].verdict == "MET"
    assert result.must_have_evaluations[0].evidence == "Evidence for Python"


def test_generate_assessment_computes_qualified_from_verdicts():
    """qualified must be derived from must-have verdicts, never trusted from Jev."""
    profile = _job_profile()
    answers = _fake_answers(profile, must_have_strength="none")  # none → NOT_MET → disqualified
    jev = MagicMock()
    jev.evaluate.return_value = answers
    chain = MagicMock()
    chain.invoke.return_value = _fake_free_text(profile)
    result = generate_assessment(_jd(), _candidate(), profile, jev, chain, "qwen2.5:14b")
    assert result.qualified is False
    assert result.disqualification_reason is not None
    assert "Python" in result.disqualification_reason


def test_generate_assessment_raises_on_jev_error():
    from candidate_ranking.scoring.jev_client import JevClientError
    jev = MagicMock()
    jev.evaluate.side_effect = JevClientError("network failure")
    with pytest.raises(AssessmentGenerationError):
        generate_assessment(_jd(), _candidate(), _job_profile(), jev, MagicMock(), "qwen2.5:14b")


def test_generate_assessment_raises_on_free_text_error():
    from candidate_ranking.utils import GenerationError
    profile = _job_profile()
    jev = MagicMock()
    jev.evaluate.return_value = _fake_answers(profile)
    chain = MagicMock()
    chain.invoke.side_effect = GenerationError("schema mismatch")
    with pytest.raises(AssessmentGenerationError):
        generate_assessment(_jd(), _candidate(), profile, jev, chain, "qwen2.5:14b")


def test_load_or_generate_assessment_uses_cache(tmp_path: Path):
    profile = _job_profile()
    with patch("candidate_ranking.scoring.llm_assessment.generate_assessment",
               return_value=_assessment()) as mock_gen:
        first = load_or_generate_assessment(
            _jd(), _candidate(), profile, MagicMock(), MagicMock(), "qwen2.5:14b", tmp_path
        )
        second = load_or_generate_assessment(
            _jd(), _candidate(), profile, MagicMock(), MagicMock(), "qwen2.5:14b", tmp_path
        )
    assert first == second
    assert mock_gen.call_count == 1
    assert (tmp_path / "assessments" / "jd-1" / "cand-1.json").exists()


def test_load_or_generate_assessment_regenerates_when_profile_changes(tmp_path: Path):
    profile_v1 = _job_profile()
    profile_v2 = JobProfile(
        job_description_id="jd-1", generated_by_model="qwen2.5:14b",
        must_have_skills=["Go"],
    )
    with patch("candidate_ranking.scoring.llm_assessment.generate_assessment",
               return_value=_assessment()) as mock_gen:
        load_or_generate_assessment(
            _jd(), _candidate(), profile_v1, MagicMock(), MagicMock(), "qwen2.5:14b", tmp_path
        )
        load_or_generate_assessment(
            _jd(), _candidate(), profile_v2, MagicMock(), MagicMock(), "qwen2.5:14b", tmp_path
        )
    assert mock_gen.call_count == 2


def test_filter_assessable_candidates_excludes_failed_parse():
    candidates = [
        _candidate("ok"),
        Candidate(id="bad", source_path="x.pdf", raw_text="", num_pages=0,
                  char_count=0, parse_status="failed"),
    ]
    result = filter_assessable_candidates(candidates)
    assert len(result) == 1
    assert result[0].id == "cand-1"


def test_build_jev_questions_keys_for_minimal_profile():
    from candidate_ranking.scoring.llm_assessment import _build_jev_questions
    profile = JobProfile(
        job_description_id="jd-1", generated_by_model="m",
        must_have_skills=["Python", "PostgreSQL"],
        nice_to_have_skills=["Docker"],
    )
    questions = _build_jev_questions(profile)
    keys = [q.key for q in questions]
    assert "must_have_0_evidence_strength" in keys
    assert "must_have_1_evidence_strength" in keys
    assert "nice_to_have_0_evidence_strength" in keys
    assert "education_requirement" not in keys
    assert "seniority_requirement" not in keys
    assert "total_relevant_years" in keys
    assert "career_direction" in keys
    # 2 must + 1 nih + 1 total_years + 5 career = 9
    assert len(questions) == 9


def test_build_jev_questions_includes_edu_seniority_other_when_set():
    from candidate_ranking.scoring.llm_assessment import _build_jev_questions
    profile = JobProfile(
        job_description_id="jd-1", generated_by_model="m",
        must_have_skills=["Python"],
        education_requirement="Bachelor's degree in Computer Science or related",
        seniority_requirement="Senior (5+ years)",
        other_requirements=["Must have valid driver's license", "Willing to relocate"],
    )
    questions = _build_jev_questions(profile)
    keys = [q.key for q in questions]
    assert "education_requirement" in keys
    assert "seniority_requirement" in keys
    assert "other_req_0" in keys
    assert "other_req_1" in keys
    # 1 must + 1 edu + 1 seniority + 2 other + 1 total_years + 5 career = 11
    assert len(questions) == 11


def test_score_to_int_interpolates_correctly():
    from candidate_ranking.scoring.llm_assessment import _score_to_int
    ticks = [0, 3, 6, 10, 15, 20]
    assert _score_to_int(0.0, ticks) == 0
    assert _score_to_int(5.0, ticks) == 20
    assert _score_to_int(2.0, ticks) == 6
    assert _score_to_int(2.5, ticks) == 8


def test_score_to_int_clamps_out_of_range():
    from candidate_ranking.scoring.llm_assessment import _score_to_int
    ticks = [0, 3, 6, 10, 15, 20]
    assert _score_to_int(99.0, ticks) == 20
    assert _score_to_int(-1.0, ticks) == 0


def test_build_free_text_chain_returns_runnable():
    from candidate_ranking.scoring.llm_assessment import build_free_text_chain
    chain = build_free_text_chain(MagicMock())
    assert chain is not None


def test_free_text_output_model_validates():
    from candidate_ranking.scoring.llm_assessment import _FreeTextOutput, _SkillEvidenceItem
    out = _FreeTextOutput(
        must_have_evidence=[_SkillEvidenceItem(requirement="Python", evidence="Built FastAPI")],
        strongest_achievements=["Led migration to microservices"],
    )
    assert out.must_have_evidence[0].requirement == "Python"
    assert out.career_trajectory_evidence == ""
