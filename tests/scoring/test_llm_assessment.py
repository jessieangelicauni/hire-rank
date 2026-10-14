from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from candidate_ranking.models import (
    Assessment, Candidate, JDRequirement, JDRequirements, JobDescription,
    RequirementEvaluation,
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


def _jd_requirements() -> JDRequirements:
    return JDRequirements(
        job_description_id="jd-1", generated_by_model="qwen2.5:14b",
        must_have=[JDRequirement(requirement="Python", context="Primary language")],
        nice_to_have=[JDRequirement(requirement="Docker", context="Preferred")],
    )


def _assessment(qualified: bool = True) -> Assessment:
    """Build a minimal Assessment domain object — no private module names needed."""
    return Assessment(
        job_description_id="jd-1",
        candidate_id="cand-1",
        generated_by_model="qwen2.5:14b",
        must_have_evaluations=[
            RequirementEvaluation(
                requirement="Python",
                verdict="MET" if qualified else "NOT_MET",
                evidence="FastAPI at PT XYZ" if qualified else "Only listed in skills section",
                evidence_strength="strong" if qualified else "weak",
                recency="current",
                years_demonstrated=3 if qualified else 0,
            )
        ],
        qualified=qualified,
        disqualification_reason=None if qualified else "Python not demonstrated in experience",
    )


def _mock_chain(qualified: bool = True) -> MagicMock:
    """
    Return a mock chain whose .invoke() returns the correct private Pydantic type
    so that invoke_and_validate's isinstance check passes.  We import the private
    class here — inside the helper — so test bodies stay decoupled from internals.
    """
    from candidate_ranking.scoring.llm_assessment import (
        _QualificationOutput, _RequirementEvaluationOutput,
    )
    verdict = "MET" if qualified else "NOT_MET"
    output = _QualificationOutput(
        must_have_evaluations=[
            _RequirementEvaluationOutput(
                requirement="Python",
                verdict=verdict,
                evidence="FastAPI at PT XYZ" if qualified else "Only listed in skills section",
                evidence_strength="strong" if qualified else "weak",
                recency="current",
                years_demonstrated=3 if qualified else 0,
            )
        ],
        qualified=qualified,
        disqualification_reason=None if qualified else "Python not demonstrated in experience",
    )
    chain = MagicMock()
    chain.invoke.return_value = output
    return chain


def test_generate_assessment_returns_qualified_assessment():
    result = generate_assessment(_jd(), _candidate(), _jd_requirements(), _mock_chain(), "qwen2.5:14b")
    assert isinstance(result, Assessment)
    assert result.qualified is True
    assert result.must_have_evaluations[0].requirement == "Python"
    assert result.must_have_evaluations[0].verdict == "MET"


def test_generate_assessment_returns_disqualified_assessment():
    result = generate_assessment(_jd(), _candidate(), _jd_requirements(), _mock_chain(False), "qwen2.5:14b")
    assert result.qualified is False
    assert result.disqualification_reason is not None


def test_generate_assessment_raises_on_chain_error():
    """Any exception from the chain — including non-ValueError — must surface as AssessmentGenerationError."""
    chain = MagicMock()
    chain.invoke.side_effect = ValueError("schema violation")
    with pytest.raises(AssessmentGenerationError):
        generate_assessment(_jd(), _candidate(), _jd_requirements(), chain, "qwen2.5:14b")


def test_generate_assessment_raises_on_unexpected_exception():
    """Non-ValueError exceptions (e.g. network errors) must also be wrapped."""
    chain = MagicMock()
    chain.invoke.side_effect = RuntimeError("connection refused")
    with pytest.raises(AssessmentGenerationError):
        generate_assessment(_jd(), _candidate(), _jd_requirements(), chain, "qwen2.5:14b")


def test_load_or_generate_assessment_uses_cache(tmp_path: Path):
    """Second call with identical inputs must hit cache and not invoke the LLM again."""
    with patch(
        "candidate_ranking.scoring.llm_assessment.generate_assessment",
        return_value=_assessment(),
    ) as mock_gen:
        first = load_or_generate_assessment(_jd(), _candidate(), _jd_requirements(), MagicMock(), "qwen2.5:14b", tmp_path)
        second = load_or_generate_assessment(_jd(), _candidate(), _jd_requirements(), MagicMock(), "qwen2.5:14b", tmp_path)
    assert first == second
    assert mock_gen.call_count == 1
    assert (tmp_path / "assessments" / "jd-1" / "cand-1.json").exists()


def test_load_or_generate_assessment_regenerates_when_requirements_change(tmp_path: Path):
    """Changing requirements must invalidate the cache and trigger a new LLM call."""
    reqs_v1 = _jd_requirements()
    reqs_v2 = JDRequirements(
        job_description_id="jd-1", generated_by_model="qwen2.5:14b",
        must_have=[JDRequirement(requirement="Go", context="Changed")],
    )
    with patch(
        "candidate_ranking.scoring.llm_assessment.generate_assessment",
        return_value=_assessment(),
    ) as mock_gen:
        load_or_generate_assessment(_jd(), _candidate(), reqs_v1, MagicMock(), "qwen2.5:14b", tmp_path)
        load_or_generate_assessment(_jd(), _candidate(), reqs_v2, MagicMock(), "qwen2.5:14b", tmp_path)
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
