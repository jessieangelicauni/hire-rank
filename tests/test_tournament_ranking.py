
import re

import pytest

from candidate_ranking.models import Assessment, JobDescription
from candidate_ranking.ranking.tournament import (
    ListwiseRankingError,
    _ranking_result_model,
    _render_candidates_text,
    rank_subset,
    resolve_tournament_iterations,
    run_tournament_for_jd_repeat,
)

ASSESSMENTS = {
    f"cv-{i}": Assessment(
        job_description_id="jd-1",
        candidate_id=f"cv-{i}",
        generated_by_model="fake-model",
        strengths=["Strong Python background."],
        weaknesses=[],
    )
    for i in range(1, 6)
}
SUBSET = tuple(f"cv-{i}" for i in range(1, 6))


class _ScriptedRankingChain:

    def __init__(self, ranking_model, responses):
        self._ranking_model = ranking_model
        self._responses = list(responses)
        self.payloads = []

    def invoke(self, payload):
        self.payloads.append(payload)
        reasoning, ranking = self._responses.pop(0)
        return self._ranking_model(reasoning=reasoning, ranking=ranking)

    @property
    def calls(self) -> int:
        return len(self.payloads)


class _FakeChainFactory:

    def __init__(self, responses):
        self._responses = responses
        self.chain: _ScriptedRankingChain | None = None

    def __call__(self, ranking_model):
        self.chain = _ScriptedRankingChain(ranking_model, self._responses)
        return self.chain


def test_rank_subset_maps_ranking_to_real_candidate_ids():
    build_chain = _FakeChainFactory([("Candidate 4 has the strongest AWS and Terraform experience...", [4, 1, 3, 2, 5])])
    ranking = rank_subset("Job Description text", SUBSET, ASSESSMENTS, build_chain)
    assert ranking == ["cv-4", "cv-1", "cv-3", "cv-2", "cv-5"]


def test_rank_subset_retries_with_corrective_feedback_after_a_duplicate():
    build_chain = _FakeChainFactory(
        [
            ("fake reasoning", [1, 1, 2, 3, 5]),
            ("fake reasoning", [4, 1, 3, 2, 5]),
        ]
    )
    ranking = rank_subset("Job Description text", SUBSET, ASSESSMENTS, build_chain)

    assert ranking == ["cv-4", "cv-1", "cv-3", "cv-2", "cv-5"]
    chain = build_chain.chain
    assert chain.calls == 2
    assert chain.payloads[0]["retry_feedback"] == ""
    feedback = chain.payloads[1]["retry_feedback"]
    assert "not a permutation" in feedback
    assert "[1, 1, 2, 3, 5]" in feedback


def test_rank_subset_succeeds_on_first_attempt_without_retrying():
    build_chain = _FakeChainFactory([("fake reasoning", [4, 1, 3, 2, 5])])
    rank_subset("Job Description text", SUBSET, ASSESSMENTS, build_chain)
    assert build_chain.chain.calls == 1


def test_rank_subset_raises_after_exhausting_the_retry():
    build_chain = _FakeChainFactory(
        [
            ("fake reasoning", [1, 1, 2, 3, 5]),
            ("fake reasoning", [1, 1, 2, 3, 5]),
        ]
    )
    with pytest.raises(ListwiseRankingError, match="not a permutation"):
        rank_subset("Job Description text", SUBSET, ASSESSMENTS, build_chain)
    assert build_chain.chain.calls == 2


def test_ranking_result_model_forbids_wrong_length_arrays():
    model = _ranking_result_model(5)
    with pytest.raises(Exception):
        model(reasoning="x", ranking=[1, 2, 3, 4])
    with pytest.raises(Exception):
        model(reasoning="x", ranking=[1, 2, 3, 4, 5, 6])
    model(reasoning="x", ranking=[1, 2, 3, 4, 5])


def test_render_candidates_text_lists_strengths_and_weaknesses():
    assessment = Assessment(
        job_description_id="jd-1",
        candidate_id="cv-001",
        generated_by_model="fake-model",
        strengths=["Built APIs in Python."],
        weaknesses=["No Kubernetes experience noted."],
    )

    rendered = _render_candidates_text(("cv-001",), {"cv-001": assessment})

    assert "Built APIs in Python." in rendered
    assert "No Kubernetes experience noted." in rendered
    assert "Strengths:" in rendered
    assert "Weaknesses:" in rendered


def test_render_candidates_text_lists_additional_skills_separately_from_strengths():
    assessment = Assessment(
        job_description_id="jd-1",
        candidate_id="cv-001",
        generated_by_model="fake-model",
        strengths=["Built APIs in Python."],
        weaknesses=[],
        additional_skills=["Kafka", "Terraform"],
    )

    rendered = _render_candidates_text(("cv-001",), {"cv-001": assessment})

    assert "Additional skills mentioned" in rendered
    assert "Kafka" in rendered
    assert "Terraform" in rendered


def test_render_candidates_text_shows_placeholder_for_empty_weaknesses():
    assessment = Assessment(
        job_description_id="jd-1",
        candidate_id="cv-001",
        generated_by_model="fake-model",
        strengths=["Built APIs in Python."],
        weaknesses=[],
    )

    rendered = _render_candidates_text(("cv-001",), {"cv-001": assessment})

    assert "(none identified)" in rendered


def test_resolve_tournament_iterations_scales_with_pool_size():
    assert resolve_tournament_iterations(
        n_candidates=77, tournament_subset_size=5, target_appearances_per_candidate=8,
        tournament_iterations_min=10, tournament_iterations_max=60,
    ) == 60

    assert resolve_tournament_iterations(
        n_candidates=24, tournament_subset_size=5, target_appearances_per_candidate=8,
        tournament_iterations_min=10, tournament_iterations_max=60,
    ) == 39


def test_resolve_tournament_iterations_respects_the_floor_for_tiny_pools():
    assert resolve_tournament_iterations(
        n_candidates=3, tournament_subset_size=5, target_appearances_per_candidate=8,
        tournament_iterations_min=10, tournament_iterations_max=60,
    ) == 10


def test_resolve_tournament_iterations_respects_the_ceiling_for_huge_pools():
    assert resolve_tournament_iterations(
        n_candidates=1000, tournament_subset_size=5, target_appearances_per_candidate=8,
        tournament_iterations_min=10, tournament_iterations_max=60,
    ) == 60


class _IdentityRankingChain:

    def __init__(self, ranking_model):
        self._ranking_model = ranking_model

    def invoke(self, payload):
        size = len(re.findall(r"^\d+:", payload["candidates_text"], re.MULTILINE))
        return self._ranking_model(reasoning="fake reasoning", ranking=list(range(1, size + 1)))


def _identity_build_chain(ranking_model):
    return _IdentityRankingChain(ranking_model)


def _fake_pool_assessments(n: int) -> dict[str, Assessment]:
    return {
        f"cv-{i:03d}": Assessment(
            job_description_id="jd-1", candidate_id=f"cv-{i:03d}", generated_by_model="fake-model",
            strengths=["Strong background."], weaknesses=[],
        )
        for i in range(1, n + 1)
    }


def test_run_tournament_for_jd_repeat_runs_the_full_budget(tmp_path):
    jd = JobDescription(id="jd-1", title="Role", raw_text="Build things.", source_path="/jd/1.txt")

    result = run_tournament_for_jd_repeat(
        jd, _fake_pool_assessments(8), repeat_index=0,
        tournament_iterations=15, tournament_subset_size=5, num_subset_samples=10, num_mc_draws=10,
        pl_prior_variance=1.0, build_ranking_chain=_identity_build_chain,
        checkpoint_path=tmp_path / "state.json", rng_seed=0,
    )

    assert len(result.iteration_history) == 15
    assert result.stop_reason == "budget_exhausted"


def test_run_tournament_for_jd_repeat_reports_no_unseen_subset_stop_reason(tmp_path):
    jd = JobDescription(id="jd-1", title="Role", raw_text="Build things.", source_path="/jd/1.txt")

    result = run_tournament_for_jd_repeat(
        jd, _fake_pool_assessments(6), repeat_index=0,
        tournament_iterations=20, tournament_subset_size=5, num_subset_samples=10, num_mc_draws=10,
        pl_prior_variance=1.0, build_ranking_chain=_identity_build_chain,
        checkpoint_path=tmp_path / "state.json", rng_seed=0,
    )

    assert len(result.iteration_history) == 6
    assert result.stop_reason == "no_unseen_subset"
