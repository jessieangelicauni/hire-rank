from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from candidate_ranking.models import JobProfile
from candidate_ranking.scoring.job_profile import (
    JobProfileGenerationError,
    _JobProfileOutput,
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


def test_generate_job_profile_returns_job_profile():
    result = generate_job_profile(_jp(), _mock_chain(), "qwen2.5:14b")
    assert isinstance(result, JobProfile)
    assert result.id == "jd-1"
    assert result.generated_by_model == "qwen2.5:14b"
    assert result.must_have_skills == ["Python"]
    assert result.nice_to_have_skills == ["Docker"]


def test_generate_job_profile_carries_education_and_seniority():
    chain = _mock_chain(education_requirement="Bachelor's degree", seniority_requirement="3+ years")
    result = generate_job_profile(_jp(), chain, "qwen2.5:14b")
    assert result.education_requirement == "Bachelor's degree"
    assert result.seniority_requirement == "3+ years"


def test_generate_job_profile_raises_on_chain_error_without_retry():
    chain = MagicMock()
    chain.invoke.side_effect = ValueError("schema violation")
    with pytest.raises(JobProfileGenerationError):
        generate_job_profile(_jp(), chain, "qwen2.5:14b")
    assert chain.invoke.call_count == 1  # no retry loop


def test_load_or_generate_uses_cache_on_second_call(tmp_path: Path):
    chain = _mock_chain()
    cache_path = tmp_path / "job_profile.json"
    first = load_or_generate_job_profile(_jp(), chain, "qwen2.5:14b", cache_path)
    second = load_or_generate_job_profile(_jp(), chain, "qwen2.5:14b", cache_path)
    assert first == second
    assert chain.invoke.call_count == 1


def test_load_or_generate_regenerates_on_model_change(tmp_path: Path):
    chain = _mock_chain()
    cache_path = tmp_path / "job_profile.json"
    load_or_generate_job_profile(_jp(), chain, "model-a", cache_path)
    load_or_generate_job_profile(_jp(), chain, "model-b", cache_path)
    assert chain.invoke.call_count == 2


def test_load_or_generate_discards_corrupt_cache(tmp_path: Path, caplog):
    chain = _mock_chain()
    cache_path = tmp_path / "job_profile.json"
    jp = _jp()
    real_key = _cache_key(jp, "qwen2.5:14b")
    cache_path.write_text(
        json.dumps({"jd-1": {"cache_key": real_key, "job_profile": {"bad": "data"}}}), encoding="utf-8"
    )
    with caplog.at_level("WARNING"):
        result = load_or_generate_job_profile(jp, chain, "qwen2.5:14b", cache_path)
    assert isinstance(result, JobProfile)
    assert chain.invoke.call_count == 1
    assert "Discarding invalid cached job profile" in caplog.text
