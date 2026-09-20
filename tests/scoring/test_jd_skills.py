from __future__ import annotations

from unittest.mock import Mock

from candidate_ranking.models import JobDescription
from candidate_ranking.scoring.jd_skills import _GeneratedSkills, generate_jd_skills


def _jd() -> JobDescription:
    return JobDescription(
        id="jd-1", title="Backend Engineer",
        raw_text="Needs Python, 5+ years experience, AWS certification, Bachelor's degree.",
        source_path="jd.pdf",
    )


def test_generate_jd_skills_extracts_certifications_seniority_education():
    chain = Mock()
    chain.invoke.return_value = _GeneratedSkills(
        reasoning="analysis",
        technical_skills=["Python"],
        certifications=["AWS Certified Solutions Architect"],
        seniority_requirement="5+ years of experience",
        education_requirement="Bachelor's degree",
    )

    jd_skills = generate_jd_skills(_jd(), chain, "qwen2.5:14b")

    assert jd_skills.technical_skills == ["Python"]
    assert jd_skills.certifications == ["AWS Certified Solutions Architect"]
    assert jd_skills.seniority_requirement == "5+ years of experience"
    assert jd_skills.education_requirement == "Bachelor's degree"


def test_generate_jd_skills_handles_no_certifications_or_seniority():
    chain = Mock()
    chain.invoke.return_value = _GeneratedSkills(
        reasoning="analysis",
        technical_skills=["Python"],
        certifications=[],
        seniority_requirement=None,
        education_requirement=None,
    )

    jd_skills = generate_jd_skills(_jd(), chain, "qwen2.5:14b")

    assert jd_skills.certifications == []
    assert jd_skills.seniority_requirement is None
    assert jd_skills.education_requirement is None


def test_generate_jd_skills_extracts_must_have_skills():
    chain = Mock()
    chain.invoke.return_value = _GeneratedSkills(
        reasoning="analysis",
        technical_skills=["Python", "SQL"],
        must_have_skills=["Python"],
    )

    jd_skills = generate_jd_skills(_jd(), chain, "qwen2.5:14b")

    assert jd_skills.must_have_skills == ["Python"]


def test_generate_jd_skills_filters_must_have_not_in_technical_skills(caplog):
    chain = Mock()
    chain.invoke.return_value = _GeneratedSkills(
        reasoning="analysis",
        technical_skills=["Python", "SQL"],
        must_have_skills=["Python", "Rust"],
    )

    with caplog.at_level("WARNING"):
        jd_skills = generate_jd_skills(_jd(), chain, "qwen2.5:14b")

    assert jd_skills.must_have_skills == ["Python"]
    assert "Rust" in caplog.text


def test_generate_jd_skills_must_have_skills_defaults_empty():
    chain = Mock()
    chain.invoke.return_value = _GeneratedSkills(
        reasoning="analysis",
        technical_skills=["Python"],
    )

    jd_skills = generate_jd_skills(_jd(), chain, "qwen2.5:14b")

    assert jd_skills.must_have_skills == []
