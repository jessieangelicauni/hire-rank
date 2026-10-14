from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from candidate_ranking.models import JDRequirement, JDRequirements, JobDescription
from candidate_ranking.scoring.jd_requirements import (
    JDRequirementsGenerationError,
    _JDRequirementsOutput,
    _RequirementItem,
    generate_jd_requirements,
    load_or_generate_jd_requirements,
)


def _jd() -> JobDescription:
    return JobDescription(id="jd-1", title="Backend Engineer",
                          raw_text="Requires Python 3+ years. Docker preferred.", source_path="jd.pdf")


def _mock_chain(must_have=None, nice_to_have=None):
    output = _JDRequirementsOutput(
        must_have=must_have or [_RequirementItem(requirement="Python", context="Primary language")],
        nice_to_have=nice_to_have or [_RequirementItem(requirement="Docker", context="Preferred")],
    )
    chain = MagicMock()
    chain.invoke.return_value = output
    return chain


def test_generate_jd_requirements_returns_jd_requirements():
    result = generate_jd_requirements(_jd(), _mock_chain(), "qwen2.5:14b")
    assert isinstance(result, JDRequirements)
    assert result.job_description_id == "jd-1"
    assert result.generated_by_model == "qwen2.5:14b"
    assert result.must_have[0].requirement == "Python"
    assert result.nice_to_have[0].requirement == "Docker"


def test_generate_jd_requirements_raises_on_empty_must_have():
    chain = MagicMock()
    chain.invoke.return_value = _JDRequirementsOutput(must_have=[], nice_to_have=[])
    with pytest.raises(JDRequirementsGenerationError):
        generate_jd_requirements(_jd(), chain, "qwen2.5:14b")
    assert chain.invoke.call_count == 2  # retry loop makes 2 attempts


def test_load_or_generate_uses_cache_on_second_call(tmp_path: Path):
    chain = _mock_chain()
    cache_path = tmp_path / "jd_requirements.json"
    first = load_or_generate_jd_requirements(_jd(), chain, "qwen2.5:14b", cache_path)
    second = load_or_generate_jd_requirements(_jd(), chain, "qwen2.5:14b", cache_path)
    assert first == second
    assert chain.invoke.call_count == 1


def test_load_or_generate_regenerates_on_model_change(tmp_path: Path):
    chain = _mock_chain()
    cache_path = tmp_path / "jd_requirements.json"
    load_or_generate_jd_requirements(_jd(), chain, "model-a", cache_path)
    load_or_generate_jd_requirements(_jd(), chain, "model-b", cache_path)
    assert chain.invoke.call_count == 2


def test_load_or_generate_discards_corrupt_cache(tmp_path: Path):
    chain = _mock_chain()
    cache_path = tmp_path / "jd_requirements.json"
    # Write a corrupt cache entry for jd-1
    cache_path.write_text(json.dumps({"jd-1": {"cache_key": "wrong-key", "jd_requirements": {"bad": "data"}}}), encoding="utf-8")
    # Should regenerate despite cache file existing
    result = load_or_generate_jd_requirements(_jd(), chain, "qwen2.5:14b", cache_path)
    assert isinstance(result, JDRequirements)
    assert chain.invoke.call_count == 1
