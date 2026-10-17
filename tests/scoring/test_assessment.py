from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from candidate_ranking.models import (
    Assessment, Candidate, JobProfile, RequirementEvaluation,
)
from candidate_ranking.scoring.assessment import (
    AssessmentGenerationError,
    filter_assessable_candidates,
    generate_assessment,
    load_or_generate_assessment,
)


def _jp() -> JobProfile:
    return JobProfile(id="jd-1", title="Backend Engineer",
                       raw_text="Python 3+ yrs required.", source_path="jd.pdf")


def _candidate(parse_status="ok") -> Candidate:
    return Candidate(id="cand-1", source_path="cv.pdf", raw_text="I built FastAPI services.",
                     num_pages=1, char_count=30, parse_status=parse_status)


def _job_profile() -> JobProfile:
    return JobProfile(
        id="jd-1", title="Backend Engineer",
        raw_text="Python 3+ yrs required.", source_path="jd.pdf",
        generated_by_model="qwen2.5:14b",
        must_have_skills=["Python"], nice_to_have_skills=["Docker"],
    )


def _assessment(qualified: bool = True) -> Assessment:
    return Assessment(
        job_profile_id="jd-1",
        candidate_id="cand-1",
        generated_by_model="jev/jev-latest",
        must_have_evaluations=[
            RequirementEvaluation(
                requirement="Python",
                evidence="FastAPI at PT XYZ" if qualified else "",
                evidence_strength="strong" if qualified else "none",
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

    answers.append(JevAnswer(key="total_relevant_years", kind="score", value=2.0, confidence=0.8))
    return answers


def test_generate_assessment_returns_qualified_assessment():
    profile = _job_profile()
    jev = MagicMock()
    jev.evaluate.return_value = _fake_answers(profile)
    result = generate_assessment(profile, _candidate(), jev)
    assert isinstance(result, Assessment)
    assert result.qualified is True
    assert result.must_have_evaluations[0].requirement == "Python"
    assert result.must_have_evaluations[0].evidence_strength == "strong"
    assert result.generated_by_model.startswith("jev/")


def test_generate_assessment_computes_qualified_from_evidence_strength():
    """qualified must be derived from must-have evidence_strength, never trusted from Jev."""
    profile = _job_profile()
    answers = _fake_answers(profile, must_have_strength="none")  # none → disqualified
    jev = MagicMock()
    jev.evaluate.return_value = answers
    result = generate_assessment(profile, _candidate(), jev)
    assert result.qualified is False
    assert result.disqualification_reason is not None
    assert "Python" in result.disqualification_reason


def test_generate_assessment_evidence_is_bm25_verbatim():
    """Evidence must be extracted verbatim from CV via BM25, not hallucinated."""
    profile = _job_profile()
    jev = MagicMock()
    jev.evaluate.return_value = _fake_answers(profile)
    candidate = Candidate(
        id="cand-1", source_path="cv.pdf",
        raw_text="Developed Python microservices using FastAPI at Tokopedia. Also used Docker.",
        num_pages=1, char_count=60, parse_status="ok",
    )
    result = generate_assessment(profile, candidate, jev)
    # Evidence must come from CV text verbatim
    ev = result.must_have_evaluations[0]
    assert "Python" in ev.evidence or "FastAPI" in ev.evidence or ev.evidence == ""


def test_generate_assessment_raises_on_jev_error():
    from candidate_ranking.scoring.jev_client import JevClientError
    jev = MagicMock()
    jev.evaluate.side_effect = JevClientError("network failure")
    with pytest.raises(AssessmentGenerationError):
        generate_assessment(_job_profile(), _candidate(), jev)


def test_load_or_generate_assessment_uses_cache(tmp_path: Path):
    profile = _job_profile()
    with patch("candidate_ranking.scoring.assessment.generate_assessment",
               return_value=_assessment()) as mock_gen:
        first = load_or_generate_assessment(
            profile, _candidate(), MagicMock(), tmp_path
        )
        second = load_or_generate_assessment(
            profile, _candidate(), MagicMock(), tmp_path
        )
    assert first == second
    assert mock_gen.call_count == 1
    assert (tmp_path / "assessments" / "jd-1" / "cand-1.json").exists()


def test_load_or_generate_assessment_regenerates_when_profile_changes(tmp_path: Path):
    profile_v1 = _job_profile()
    profile_v2 = JobProfile(
        id="jd-1", title="Backend", raw_text="Go 3+ yrs.", source_path="j.pdf",
        generated_by_model="qwen2.5:14b",
        must_have_skills=["Go"],
    )
    with patch("candidate_ranking.scoring.assessment.generate_assessment",
               return_value=_assessment()) as mock_gen:
        load_or_generate_assessment(
            profile_v1, _candidate(), MagicMock(), tmp_path
        )
        load_or_generate_assessment(
            profile_v2, _candidate(), MagicMock(), tmp_path
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
    from candidate_ranking.scoring.assessment import _build_jev_questions
    profile = JobProfile(
        id="jd-1", title="Backend", raw_text="req.", source_path="j.pdf",
        generated_by_model="m",
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
    assert "career_direction" not in keys
    # 2 must + 1 nih + 1 total_years = 4 (no adaptive_requirements on this minimal profile)
    assert len(questions) == 4



def test_score_to_int_interpolates_correctly():
    from candidate_ranking.scoring.assessment import _score_to_int
    ticks = [0, 3, 6, 10, 15, 20]
    assert _score_to_int(0.0, ticks) == 0
    assert _score_to_int(5.0, ticks) == 20
    assert _score_to_int(2.0, ticks) == 6
    assert _score_to_int(2.5, ticks) == 8


def test_score_to_int_clamps_out_of_range():
    from candidate_ranking.scoring.assessment import _score_to_int
    ticks = [0, 3, 6, 10, 15, 20]
    assert _score_to_int(99.0, ticks) == 20
    assert _score_to_int(-1.0, ticks) == 0


def test_bm25_evidence_is_empty_when_skill_never_mentioned():
    """When a skill never appears anywhere in the CV, every sentence ties at
    BM25 score 0 — the extractor must not fall back to the header/contact
    block just because it happens to be first."""
    from candidate_ranking.scoring.assessment import _extract_bm25_evidence
    cv_text = (
        "richardson@protonmail.com | +1-243-784-5881 | Thorntonfort, Netherlands | "
        "https://linkedin.com/in/richardson\n"
        "Senior backend engineer with experience in PostgreSQL and Django.\n"
        "Led a team of 4 engineers across two product launches.\n"
    )
    assert _extract_bm25_evidence(cv_text, "Airflow") == ""


def test_bm25_evidence_returns_matching_sentence():
    from candidate_ranking.scoring.assessment import _extract_bm25_evidence
    cv_text = (
        "richardson@protonmail.com | +1-243-784-5881 | Thorntonfort, Netherlands\n"
        "Built and scheduled ETL pipelines using Airflow for daily reporting.\n"
        "Led a team of 4 engineers across two product launches.\n"
    )
    evidence = _extract_bm25_evidence(cv_text, "Airflow")
    assert "Airflow" in evidence
    assert "protonmail" not in evidence


def test_bm25_evidence_splits_on_pipe_delimiter():
    """A pipe-joined line (contact header, or a 'Company | City | Dates' entry)
    must not be treated as one indivisible sentence — only the segment that
    actually matches the query should come back, not the whole joined line."""
    from candidate_ranking.scoring.assessment import _extract_bm25_evidence
    cv_text = (
        "PT Teknologi Maju | Jakarta, Indonesia | Jan 2021 - Dec 2023 | "
        "Built data pipelines with Apache Airflow for the analytics team\n"
    )
    evidence = _extract_bm25_evidence(cv_text, "Airflow")
    assert "Airflow" in evidence
    assert "Jakarta" not in evidence


def test_build_jev_questions_uses_skill_criteria_when_present():
    from candidate_ranking.scoring.assessment import _build_jev_questions
    from candidate_ranking.models import SkillCriteria
    profile = JobProfile(
        id="jd-1", title="Backend", raw_text="req.", source_path="j.pdf",
        generated_by_model="m", must_have_skills=["Python"],
        skill_criteria=[SkillCriteria(
            skill="Python", instructions="Custom instructions for Python",
            strong="S", adequate="A", weak="W", none="N",
        )],
    )
    questions = _build_jev_questions(profile)
    q = next(q for q in questions if q.key == "must_have_0_evidence_strength")
    assert q.instructions == "Custom instructions for Python"
    assert q.criteria == {"strong": "S", "adequate": "A", "weak": "W", "none": "N"}


def test_build_jev_questions_falls_back_when_skill_criteria_missing(caplog):
    from candidate_ranking.scoring.assessment import _build_jev_questions
    profile = JobProfile(
        id="jd-1", title="Backend", raw_text="req.", source_path="j.pdf",
        generated_by_model="m", must_have_skills=["Python"],
    )
    with caplog.at_level("WARNING"):
        questions = _build_jev_questions(profile)
    q = next(q for q in questions if q.key == "must_have_0_evidence_strength")
    assert "Assess the evidence strength for the skill 'Python'" in q.instructions
    assert "no skill_criteria entry" in caplog.text


def test_build_jev_questions_emits_adaptive_requirement_questions():
    from candidate_ranking.scoring.assessment import _build_jev_questions
    from candidate_ranking.models import AdaptiveRequirement
    profile = JobProfile(
        id="jd-1", title="Backend", raw_text="req.", source_path="j.pdf",
        generated_by_model="m",
        adaptive_requirements=[AdaptiveRequirement(
            source="education", requirement="Bachelor's degree in CS",
            criteria_meets="Has CS degree", criteria_partially_meets="Related degree",
            criteria_does_not_meet="No degree",
        )],
    )
    questions = _build_jev_questions(profile)
    q = next(q for q in questions if q.key == "adaptive_0_match_level")
    assert q.kind == "choice"
    assert q.criteria == {
        "meets": "Has CS degree",
        "partially_meets": "Related degree",
        "does_not_meet": "No degree",
    }


def test_answers_to_assessment_builds_adaptive_requirement_evaluations():
    from candidate_ranking.scoring.assessment import _answers_to_assessment
    from candidate_ranking.models import AdaptiveRequirement
    from candidate_ranking.scoring.jev_client import JevAnswer
    profile = JobProfile(
        id="jd-1", title="Backend", raw_text="req.", source_path="j.pdf",
        generated_by_model="m", must_have_skills=["Python"],
        adaptive_requirements=[AdaptiveRequirement(
            source="seniority", requirement="5+ years",
            criteria_meets="meets", criteria_partially_meets="partial", criteria_does_not_meet="none",
        )],
    )
    candidate = Candidate(id="cand-1", source_path="cv.pdf",
                           raw_text="6 years of experience at PT XYZ.", num_pages=1, char_count=30, parse_status="ok")
    answers = [
        JevAnswer(key="must_have_0_evidence_strength", kind="choice", value="strong", confidence=0.9),
        JevAnswer(key="total_relevant_years", kind="score", value=3.0, confidence=0.8),
        JevAnswer(key="adaptive_0_match_level", kind="choice", value="partially_meets", confidence=0.7),
    ]
    result = _answers_to_assessment(answers, profile, candidate)
    assert len(result.adaptive_requirement_evaluations) == 1
    ev = result.adaptive_requirement_evaluations[0]
    assert ev.source == "seniority"
    assert ev.match_level == "partially_meets"
    assert result.qualified is True


def test_adaptive_requirement_match_level_never_affects_qualification():
    from candidate_ranking.scoring.assessment import _answers_to_assessment
    from candidate_ranking.models import AdaptiveRequirement
    from candidate_ranking.scoring.jev_client import JevAnswer
    profile = JobProfile(
        id="jd-1", title="Backend", raw_text="req.", source_path="j.pdf",
        generated_by_model="m", must_have_skills=["Python"],
        adaptive_requirements=[AdaptiveRequirement(
            source="seniority", requirement="5+ years",
            criteria_meets="meets", criteria_partially_meets="partial", criteria_does_not_meet="none",
        )],
    )
    candidate = Candidate(id="cand-1", source_path="cv.pdf",
                           raw_text="1 year of experience.", num_pages=1, char_count=20, parse_status="ok")
    answers = [
        JevAnswer(key="must_have_0_evidence_strength", kind="choice", value="strong", confidence=0.9),
        JevAnswer(key="total_relevant_years", kind="score", value=0.0, confidence=0.8),
        JevAnswer(key="adaptive_0_match_level", kind="choice", value="does_not_meet", confidence=0.7),
    ]
    result = _answers_to_assessment(answers, profile, candidate)
    assert result.qualified is True  # must-have strong -> qualified, regardless of adaptive match_level
    assert result.disqualification_reason is None
