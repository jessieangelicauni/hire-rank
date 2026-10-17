from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from candidate_ranking.models import JobProfile
from candidate_ranking.scoring.job_profile import (
    JobProfileGenerationError,
    _JobProfileOutput,
    _RubricOutput,
    _SkillCriteriaOutput,
    _TotalYearsCriteriaOutput,
    _AdaptiveRequirementOutput,
    _cache_key,
    generate_job_profile,
    load_or_generate_job_profile,
)


def _jp() -> JobProfile:
    return JobProfile(id="jd-1", title="Backend Engineer",
                       raw_text="Requires Python 3+ years. Docker preferred.", source_path="jd.pdf")


def _mock_chain(**overrides) -> MagicMock:
    output = _JobProfileOutput(
        reasoning="analysis",
        must_have_skills=overrides.get("must_have_skills", ["Python"]),
        nice_to_have_skills=overrides.get("nice_to_have_skills", ["Docker"]),
        education_requirement=overrides.get("education_requirement"),
        seniority_requirement=overrides.get("seniority_requirement"),
        other_requirements=overrides.get("other_requirements", []),
    )
    chain = MagicMock()
    chain.invoke.return_value = output
    return chain


_UNSET = object()


def _mock_rubric_chain(
    skill_criteria: list[_SkillCriteriaOutput] | object = _UNSET,
    total_years_criteria: _TotalYearsCriteriaOutput | None | object = _UNSET,
    adaptive_requirements: list[_AdaptiveRequirementOutput] | object = _UNSET,
) -> MagicMock:
    output = _RubricOutput(
        reasoning="rubric analysis",
        skill_criteria=skill_criteria if skill_criteria is not _UNSET else [
            _SkillCriteriaOutput(
                skill="Python", instructions="Assess Python evidence",
                strong="S", adequate="A", weak="W", none="N",
            ),
            _SkillCriteriaOutput(
                skill="Docker", instructions="Assess Docker evidence",
                strong="S", adequate="A", weak="W", none="N",
            ),
        ],
        total_years_criteria=(
            total_years_criteria if total_years_criteria is not _UNSET
            else _TotalYearsCriteriaOutput(instructions="Calibrate relevant years for this role")
        ),
        adaptive_requirements=adaptive_requirements if adaptive_requirements is not _UNSET else [],
    )
    chain = MagicMock()
    chain.invoke.return_value = output
    return chain


def test_generate_job_profile_returns_job_profile():
    result = generate_job_profile(_jp(), _mock_chain(), _mock_rubric_chain(), "qwen2.5:14b")
    assert isinstance(result, JobProfile)
    assert result.id == "jd-1"
    assert result.generated_by_model == "qwen2.5:14b"
    assert result.must_have_skills == ["Python"]
    assert result.nice_to_have_skills == ["Docker"]


def test_generate_job_profile_carries_education_and_seniority():
    chain = _mock_chain(education_requirement="Bachelor's degree", seniority_requirement="3+ years")
    result = generate_job_profile(_jp(), chain, _mock_rubric_chain(), "qwen2.5:14b")
    assert result.education_requirement == "Bachelor's degree"
    assert result.seniority_requirement == "3+ years"


def test_generate_job_profile_raises_on_chain_error_without_retry():
    chain = MagicMock()
    chain.invoke.side_effect = ValueError("schema violation")
    with pytest.raises(JobProfileGenerationError):
        generate_job_profile(_jp(), chain, _mock_rubric_chain(), "qwen2.5:14b")
    assert chain.invoke.call_count == 1  # no retry loop


def test_load_or_generate_uses_cache_on_second_call(tmp_path: Path):
    chain = _mock_chain()
    rubric_chain = _mock_rubric_chain()
    cache_path = tmp_path / "job_profile.json"
    first = load_or_generate_job_profile(_jp(), chain, rubric_chain, "qwen2.5:14b", cache_path)
    second = load_or_generate_job_profile(_jp(), chain, rubric_chain, "qwen2.5:14b", cache_path)
    assert first == second
    assert chain.invoke.call_count == 1
    assert rubric_chain.invoke.call_count == 1


def test_load_or_generate_regenerates_on_model_change(tmp_path: Path):
    chain = _mock_chain()
    rubric_chain = _mock_rubric_chain()
    cache_path = tmp_path / "job_profile.json"
    load_or_generate_job_profile(_jp(), chain, rubric_chain, "model-a", cache_path)
    load_or_generate_job_profile(_jp(), chain, rubric_chain, "model-b", cache_path)
    assert chain.invoke.call_count == 2


def test_load_or_generate_discards_corrupt_cache(tmp_path: Path, caplog):
    chain = _mock_chain()
    rubric_chain = _mock_rubric_chain()
    cache_path = tmp_path / "job_profile.json"
    jp = _jp()
    real_key = _cache_key(jp, "qwen2.5:14b")
    cache_path.write_text(
        json.dumps({"jd-1": {"cache_key": real_key, "job_profile": {"bad": "data"}}}), encoding="utf-8"
    )
    with caplog.at_level("WARNING"):
        result = load_or_generate_job_profile(jp, chain, rubric_chain, "qwen2.5:14b", cache_path)
    assert isinstance(result, JobProfile)
    assert chain.invoke.call_count == 1
    assert "Discarding invalid cached job profile" in caplog.text


def test_generate_job_profile_builds_skill_criteria_matched_by_name():
    """Rubric skill order differs from must_have_skills order — must match by name, not position."""
    chain = _mock_chain(must_have_skills=["Python", "PostgreSQL"], nice_to_have_skills=[])
    rubric_chain = _mock_rubric_chain(skill_criteria=[
        _SkillCriteriaOutput(skill="PostgreSQL", instructions="PG instructions",
                              strong="S-pg", adequate="A-pg", weak="W-pg", none="N-pg"),
        _SkillCriteriaOutput(skill="Python", instructions="Py instructions",
                              strong="S-py", adequate="A-py", weak="W-py", none="N-py"),
    ])
    result = generate_job_profile(_jp(), chain, rubric_chain, "qwen2.5:14b")
    by_skill = {c.skill: c for c in result.skill_criteria}
    assert by_skill["Python"].instructions == "Py instructions"
    assert by_skill["PostgreSQL"].instructions == "PG instructions"


def test_generate_job_profile_falls_back_when_skill_criteria_missing_one_skill(caplog):
    chain = _mock_chain(must_have_skills=["Python", "Rust"], nice_to_have_skills=[])
    rubric_chain = _mock_rubric_chain(skill_criteria=[
        _SkillCriteriaOutput(skill="Python", instructions="Py instructions",
                              strong="S", adequate="A", weak="W", none="N"),
        # "Rust" missing — must fall back, not fail the whole JP
    ])
    with caplog.at_level("WARNING"):
        result = generate_job_profile(_jp(), chain, rubric_chain, "qwen2.5:14b")
    by_skill = {c.skill: c for c in result.skill_criteria}
    assert by_skill["Python"].instructions == "Py instructions"
    assert "Rust" in by_skill
    assert "no skill-criteria rubric returned for skill 'Rust'" in caplog.text


def test_generate_job_profile_falls_back_when_total_years_criteria_missing(caplog):
    chain = _mock_chain()
    rubric_chain = _mock_rubric_chain(total_years_criteria=None)
    with caplog.at_level("WARNING"):
        result = generate_job_profile(_jp(), chain, rubric_chain, "qwen2.5:14b")
    assert result.total_years_criteria is not None
    assert "no total_years_criteria returned" in caplog.text


def test_generate_job_profile_builds_adaptive_requirements_for_education_seniority_other():
    chain = _mock_chain(education_requirement="Bachelor's in CS", seniority_requirement="5+ years",
                         other_requirements=["AWS Certified"])
    rubric_chain = _mock_rubric_chain(adaptive_requirements=[
        _AdaptiveRequirementOutput(source="other", requirement="AWS Certified",
                                    criteria_meets="has cert", criteria_partially_meets="in progress",
                                    criteria_does_not_meet="no cert"),
        _AdaptiveRequirementOutput(source="education", requirement="Bachelor's in CS",
                                    criteria_meets="has CS degree", criteria_partially_meets="related degree",
                                    criteria_does_not_meet="no degree"),
        _AdaptiveRequirementOutput(source="seniority", requirement="5+ years",
                                    criteria_meets="5+ yrs", criteria_partially_meets="3-5 yrs",
                                    criteria_does_not_meet="<3 yrs"),
    ])
    result = generate_job_profile(_jp(), chain, rubric_chain, "qwen2.5:14b")
    by_source = {r.source: r for r in result.adaptive_requirements}
    assert set(by_source) == {"education", "seniority", "other"}
    assert by_source["other"].requirement == "AWS Certified"


def test_generate_job_profile_drops_adaptive_requirement_missing_from_rubric(caplog):
    chain = _mock_chain(education_requirement="Bachelor's in CS")
    rubric_chain = _mock_rubric_chain(adaptive_requirements=[])  # rubric returned nothing
    with caplog.at_level("WARNING"):
        result = generate_job_profile(_jp(), chain, rubric_chain, "qwen2.5:14b")
    assert result.adaptive_requirements == []
    assert "no adaptive-requirement rubric returned" in caplog.text


def test_generate_job_profile_falls_back_to_generic_rubric_when_rubric_chain_fails(caplog):
    """A rubric-call hiccup must not discard the whole JP — Call 1 already
    succeeded, so degrade every field to its generic fallback instead of
    raising, mirroring the per-field fallback philosophy below it."""
    chain = _mock_chain(must_have_skills=["Python"], nice_to_have_skills=[])
    rubric_chain = MagicMock()
    rubric_chain.invoke.side_effect = ValueError("rubric model timed out")
    with caplog.at_level("WARNING"):
        result = generate_job_profile(_jp(), chain, rubric_chain, "qwen2.5:14b")
    assert isinstance(result, JobProfile)
    assert result.must_have_skills == ["Python"]
    assert len(result.skill_criteria) == 1
    assert result.skill_criteria[0].skill == "Python"
    assert result.total_years_criteria is not None
    assert "Adaptive rubric generation for" in caplog.text


def test_generate_job_profile_skill_criteria_matches_despite_whitespace_case_drift():
    """The rubric echoes the skill name back verbatim in theory, but minor
    whitespace/case drift from the model must not fall through to the
    generic fallback when a real match exists."""
    chain = _mock_chain(must_have_skills=["PostgreSQL"], nice_to_have_skills=[])
    rubric_chain = _mock_rubric_chain(skill_criteria=[
        _SkillCriteriaOutput(skill=" postgresql ", instructions="PG instructions",
                              strong="S", adequate="A", weak="W", none="N"),
    ])
    result = generate_job_profile(_jp(), chain, rubric_chain, "qwen2.5:14b")
    assert result.skill_criteria[0].instructions == "PG instructions"
    assert result.skill_criteria[0].skill == "PostgreSQL"


def test_generate_job_profile_adaptive_requirement_matches_despite_whitespace_case_drift():
    chain = _mock_chain(education_requirement="Bachelor's in CS")
    rubric_chain = _mock_rubric_chain(adaptive_requirements=[
        _AdaptiveRequirementOutput(source="education", requirement=" Bachelor's In CS  ",
                                    criteria_meets="has CS degree", criteria_partially_meets="related degree",
                                    criteria_does_not_meet="no degree"),
    ])
    result = generate_job_profile(_jp(), chain, rubric_chain, "qwen2.5:14b")
    assert len(result.adaptive_requirements) == 1
    assert result.adaptive_requirements[0].criteria_meets == "has CS degree"
    assert result.adaptive_requirements[0].requirement == "Bachelor's in CS"
