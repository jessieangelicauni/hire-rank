
import pytest

from candidate_ranking.models import JobDescription
from candidate_ranking.scoring.jd_skills import JDSkillsGenerationError, _GeneratedSkills, generate_jd_skills

JD = JobDescription(
    id="devops-engineer", title="DevOps Engineer", raw_text="Build infra with Terraform.",
    source_path="/jd/devops.txt",
)


class _FailNTimesThenSucceedChain:
    def __init__(self, fail_count: int, exc_factory):
        self._fail_count = fail_count
        self._exc_factory = exc_factory
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        if self.calls <= self._fail_count:
            raise self._exc_factory()
        return _GeneratedSkills(reasoning="fake reasoning", technical_skills=["Terraform"])


def test_generate_jd_skills_retries_once_after_a_transient_failure():
    chain = _FailNTimesThenSucceedChain(fail_count=1, exc_factory=lambda: ValueError("transient glitch"))

    jd_skills = generate_jd_skills(JD, chain, "fake-model")

    assert jd_skills.technical_skills == ["Terraform"]
    assert chain.calls == 2


def test_generate_jd_skills_succeeds_on_first_attempt_without_retrying():
    chain = _FailNTimesThenSucceedChain(fail_count=0, exc_factory=lambda: ValueError("unused"))

    jd_skills = generate_jd_skills(JD, chain, "fake-model")

    assert jd_skills.technical_skills == ["Terraform"]
    assert chain.calls == 1


def test_generate_jd_skills_raises_after_exhausting_the_retry():
    chain = _FailNTimesThenSucceedChain(fail_count=2, exc_factory=lambda: ValueError("persistent failure"))

    with pytest.raises(JDSkillsGenerationError, match="persistent failure"):
        generate_jd_skills(JD, chain, "fake-model")

    assert chain.calls == 2
