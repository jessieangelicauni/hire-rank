
import numpy as np
import pytest

from candidate_ranking.scoring.assessment import (
    AssessmentGenerationError,
    _GeneratedAssessment,
    _assessment_cache_key,
    generate_assessment,
)
from candidate_ranking.models import Candidate, JDSkills, JobDescription

JD = JobDescription(
    id="cloud-engineer", title="Cloud Engineer", raw_text="Build cloud infra.", source_path="/jd/ce.txt"
)

CANDIDATE = Candidate(
    id="cv-001", source_path="/cv/1.pdf", raw_text="Jane: AWS, Python.", num_pages=1, char_count=18,
    parse_status="ok",
)

CANDIDATE_WITH_SKILLS = CANDIDATE.model_copy(update={"skills": ["AWS", "Python", "Docker"]})


class _FakeAssessmentChain:
    def __init__(self, result: _GeneratedAssessment):
        self._result = result
        self.calls = 0
        self.last_payload: dict | None = None

    def invoke(self, payload):
        self.calls += 1
        self.last_payload = payload
        return self._result


def test_generate_assessment_maps_strengths_and_weaknesses_through():
    chain = _FakeAssessmentChain(
        _GeneratedAssessment(
            reasoning="fake reasoning",
            strengths=["Built APIs with AWS Lambda."],
            weaknesses=["No Kubernetes experience noted."],
        )
    )

    assessment = generate_assessment(JD, CANDIDATE, chain, "fake-model")

    assert assessment.job_description_id == "cloud-engineer"
    assert assessment.candidate_id == "cv-001"
    assert assessment.generated_by_model == "fake-model"
    assert assessment.strengths == ["Built APIs with AWS Lambda."]
    assert assessment.weaknesses == ["No Kubernetes experience noted."]


def test_generate_assessment_maps_additional_skills_through():
    chain = _FakeAssessmentChain(
        _GeneratedAssessment(
            reasoning="fake reasoning",
            strengths=["Built APIs with AWS Lambda."],
            weaknesses=[],
            additional_skills=["Kafka", "Terraform"],
        )
    )

    assessment = generate_assessment(JD, CANDIDATE, chain, "fake-model")

    assert assessment.additional_skills == ["Kafka", "Terraform"]


def test_generate_assessment_defaults_additional_skills_to_empty():
    chain = _FakeAssessmentChain(
        _GeneratedAssessment(reasoning="fake reasoning", strengths=["Strong Python background."], weaknesses=[])
    )

    assessment = generate_assessment(JD, CANDIDATE, chain, "fake-model")

    assert assessment.additional_skills == []


def test_generate_assessment_allows_empty_weaknesses():
    chain = _FakeAssessmentChain(
        _GeneratedAssessment(reasoning="fake reasoning", strengths=["Strong Python background."], weaknesses=[])
    )

    assessment = generate_assessment(JD, CANDIDATE, chain, "fake-model")

    assert assessment.weaknesses == []


def test_generate_assessment_raises_on_schema_failure():
    class _BrokenChain:
        def invoke(self, payload):
            raise ValueError("model returned malformed output")

    with pytest.raises(AssessmentGenerationError):
        generate_assessment(JD, CANDIDATE, _BrokenChain(), "fake-model")


def test_assessment_cache_key_differs_when_candidate_cv_text_changes():
    other_candidate = CANDIDATE.model_copy(update={"raw_text": "Jane: GCP, Go."})

    key_one = _assessment_cache_key(JD, CANDIDATE, "fake-model")
    key_two = _assessment_cache_key(JD, other_candidate, "fake-model")

    assert key_one != key_two


def test_assessment_cache_key_differs_when_model_name_changes():
    key_one = _assessment_cache_key(JD, CANDIDATE, "model-a")
    key_two = _assessment_cache_key(JD, CANDIDATE, "model-b")

    assert key_one != key_two


def test_generate_assessment_passes_the_candidates_own_extracted_skills_to_the_chain():
    chain = _FakeAssessmentChain(
        _GeneratedAssessment(reasoning="fake reasoning", strengths=["Strong AWS background."], weaknesses=[])
    )

    generate_assessment(JD, CANDIDATE_WITH_SKILLS, chain, "fake-model")

    assert chain.last_payload is not None
    for skill in CANDIDATE_WITH_SKILLS.skills:
        assert skill in chain.last_payload["candidate_skills"]


def test_generate_assessment_handles_a_candidate_with_no_extracted_skills():
    chain = _FakeAssessmentChain(
        _GeneratedAssessment(reasoning="fake reasoning", strengths=["Strong AWS background."], weaknesses=[])
    )

    generate_assessment(JD, CANDIDATE, chain, "fake-model")

    assert chain.last_payload is not None


def test_assessment_cache_key_differs_when_candidate_skills_change():
    key_one = _assessment_cache_key(JD, CANDIDATE, "fake-model")
    key_two = _assessment_cache_key(JD, CANDIDATE_WITH_SKILLS, "fake-model")

    assert key_one != key_two


class _SequencedAssessmentChain:

    def __init__(self, results: list[_GeneratedAssessment]):
        self._results = results
        self.calls = 0
        self.payloads: list[dict] = []

    def invoke(self, payload):
        self.payloads.append(payload)
        result = self._results[min(self.calls, len(self._results) - 1)]
        self.calls += 1
        return result


def test_generate_assessment_retries_when_a_weakness_contradicts_candidates_own_skills():
    bad_result = _GeneratedAssessment(
        reasoning="fake reasoning", strengths=["Strong AWS background."],
        weaknesses=["Lacks experience with Docker."],
    )
    good_result = _GeneratedAssessment(
        reasoning="fake reasoning", strengths=["Strong AWS background."],
        weaknesses=["Lacks experience with Kubernetes."],
    )
    chain = _SequencedAssessmentChain([bad_result, good_result])

    assessment = generate_assessment(JD, CANDIDATE_WITH_SKILLS, chain, "fake-model")

    assert chain.calls == 2
    assert assessment.weaknesses == ["Lacks experience with Kubernetes."]
    assert "Docker" in chain.payloads[1]["retry_feedback"]


def test_generate_assessment_does_not_retry_when_no_contradiction():
    result = _GeneratedAssessment(
        reasoning="fake reasoning", strengths=["Strong AWS background."],
        weaknesses=["Lacks experience with Kubernetes."],
    )
    chain = _SequencedAssessmentChain([result])

    generate_assessment(JD, CANDIDATE_WITH_SKILLS, chain, "fake-model")

    assert chain.calls == 1
    assert chain.payloads[0]["retry_feedback"] == ""


def test_generate_assessment_drops_the_weakness_still_contradicted_after_max_attempts():
    bad_result = _GeneratedAssessment(
        reasoning="fake reasoning", strengths=["Strong AWS background."],
        weaknesses=["Lacks experience with Docker."],
    )
    chain = _SequencedAssessmentChain([bad_result, bad_result])

    assessment = generate_assessment(JD, CANDIDATE_WITH_SKILLS, chain, "fake-model")

    assert chain.calls == 2
    assert assessment.weaknesses == []


def test_generate_assessment_keeps_other_weaknesses_when_dropping_a_contradicting_one():
    bad_result = _GeneratedAssessment(
        reasoning="fake reasoning", strengths=["Strong AWS background."],
        weaknesses=["Lacks experience with Docker.", "No experience with Terraform noted."],
    )
    chain = _SequencedAssessmentChain([bad_result, bad_result])

    assessment = generate_assessment(JD, CANDIDATE_WITH_SKILLS, chain, "fake-model")

    assert assessment.weaknesses == ["No experience with Terraform noted."]


def _fake_embedder(strings: list[str]) -> np.ndarray:
    canonical_index = {"rest api design": 0, "restful api design": 0, "kubernetes": 1}
    dim = 3
    vectors = np.zeros((len(strings), dim))
    for row, s in enumerate(strings):
        vectors[row, canonical_index.get(s.lower(), dim - 1)] = 1.0
    return vectors


JD_SKILLS = JDSkills(
    job_description_id="cloud-engineer", generated_by_model="fake-model",
    technical_skills=["RESTful API design", "Kubernetes"],
)

CANDIDATE_WITH_REST_SKILL = CANDIDATE.model_copy(update={"skills": ["REST API Design"]})


def test_generate_assessment_retries_when_a_weakness_negates_a_jd_skill_matching_a_candidate_skill_under_different_wording():
    bad_result = _GeneratedAssessment(
        reasoning="fake reasoning", strengths=["Strong AWS background."],
        weaknesses=["No specific mention of RESTful API design."],
    )
    good_result = _GeneratedAssessment(
        reasoning="fake reasoning", strengths=["Strong AWS background."],
        weaknesses=["Lacks experience with Kubernetes."],
    )
    chain = _SequencedAssessmentChain([bad_result, good_result])

    assessment = generate_assessment(
        JD, CANDIDATE_WITH_REST_SKILL, chain, "fake-model", jd_skills=JD_SKILLS, skill_embedder=_fake_embedder,
    )

    assert chain.calls == 2
    assert assessment.weaknesses == ["Lacks experience with Kubernetes."]
    assert "REST API Design" in chain.payloads[1]["retry_feedback"]


def test_generate_assessment_drops_a_jd_vocab_weakness_still_contradicted_after_max_attempts():
    bad_result = _GeneratedAssessment(
        reasoning="fake reasoning", strengths=["Strong AWS background."],
        weaknesses=["No specific mention of RESTful API design."],
    )
    chain = _SequencedAssessmentChain([bad_result, bad_result])

    assessment = generate_assessment(
        JD, CANDIDATE_WITH_REST_SKILL, chain, "fake-model", jd_skills=JD_SKILLS, skill_embedder=_fake_embedder,
    )

    assert chain.calls == 2
    assert assessment.weaknesses == []


def test_generate_assessment_keeps_jd_vocab_weakness_unresolved_when_no_embedder_given():
    bad_result = _GeneratedAssessment(
        reasoning="fake reasoning", strengths=["Strong AWS background."],
        weaknesses=["No specific mention of RESTful API design."],
    )
    chain = _SequencedAssessmentChain([bad_result, bad_result])

    assessment = generate_assessment(JD, CANDIDATE_WITH_REST_SKILL, chain, "fake-model", jd_skills=JD_SKILLS)

    assert chain.calls == 1
    assert assessment.weaknesses == ["No specific mention of RESTful API design."]
