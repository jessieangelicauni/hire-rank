# tests/scoring/test_sliding_window_ranking.py
from __future__ import annotations
import random
from typing import Callable
from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError
from candidate_ranking.models import Assessment, JobDescription, RequirementEvaluation
from candidate_ranking.scoring.sliding_window_ranking import (
    SLIDING_WINDOW_RANKING_PROMPT,
    _CandidateEvaluation,
    _sliding_window_model,
    build_sliding_window_chain_factory,
    _initial_sort,
    _rank_window,
    rank_sliding_window,
)


def test_sliding_window_model_accepts_correct_size():
    model = _sliding_window_model(3)
    instance = model(
        candidate_evaluations=[
            _CandidateEvaluation(id=1, key_strengths="s", key_gaps="g"),
            _CandidateEvaluation(id=2, key_strengths="s", key_gaps="g"),
            _CandidateEvaluation(id=3, key_strengths="s", key_gaps="g"),
        ],
        ranking=[2, 1, 3],
    )
    assert instance.ranking == [2, 1, 3]


def test_sliding_window_model_rejects_wrong_size():
    model = _sliding_window_model(3)
    with pytest.raises(ValidationError):
        model(
            candidate_evaluations=[
                _CandidateEvaluation(id=1, key_strengths="s", key_gaps="g"),
            ],
            ranking=[1],
        )


def test_build_sliding_window_chain_factory_builds_chain():
    llm = MagicMock()
    llm.with_structured_output.return_value = MagicMock()
    factory = build_sliding_window_chain_factory(llm)
    model = _sliding_window_model(3)
    chain = factory(model)
    assert chain is not None
    llm.with_structured_output.assert_called_once_with(model)


def test_sliding_window_prompt_has_required_input_variables():
    assert "jd_title" in SLIDING_WINDOW_RANKING_PROMPT.input_variables
    assert "jd_summary" in SLIDING_WINDOW_RANKING_PROMPT.input_variables
    assert "candidates_text" in SLIDING_WINDOW_RANKING_PROMPT.input_variables


def _jd() -> JobDescription:
    return JobDescription(id="jd-1", title="Backend Engineer",
                          raw_text="Python required.", source_path="jd.pdf")


def _assessment(candidate_id: str, strong_met: int = 1, years: int = 3) -> Assessment:
    evals = [
        RequirementEvaluation(
            requirement=f"req-{i}", verdict="MET", evidence="FastAPI",
            evidence_strength="strong", recency="current", years_demonstrated=years,
        )
        for i in range(strong_met)
    ]
    return Assessment(
        job_description_id="jd-1", candidate_id=candidate_id,
        generated_by_model="qwen2.5:14b", qualified=True,
        must_have_evaluations=evals,
        total_relevant_years=years,
    )


def test_initial_sort_orders_by_strong_met_count():
    assessments = {
        "cand-a": _assessment("cand-a", strong_met=0, years=5),
        "cand-b": _assessment("cand-b", strong_met=2, years=1),
        "cand-c": _assessment("cand-c", strong_met=1, years=3),
    }
    result = _initial_sort(assessments)
    assert result == ["cand-b", "cand-c", "cand-a"]


def test_initial_sort_breaks_ties_by_years():
    assessments = {
        "cand-a": _assessment("cand-a", strong_met=1, years=2),
        "cand-b": _assessment("cand-b", strong_met=1, years=5),
    }
    result = _initial_sort(assessments)
    assert result[0] == "cand-b"


def _rank_window_chain(ordering: list[int]) -> Callable:
    """Returns a chain factory that outputs the given fixed permutation."""
    def _build(model):
        chain = MagicMock()
        def _invoke(payload):
            size = len(ordering)
            evals = [
                _CandidateEvaluation(id=i, key_strengths="s", key_gaps="g")
                for i in range(1, size + 1)
            ]
            return model(candidate_evaluations=evals, ranking=ordering)
        chain.invoke.side_effect = _invoke
        return chain
    return _build


def test_rank_window_reorders_by_llm_output():
    window_ids = ["cand-a", "cand-b", "cand-c"]
    assessments = {cid: _assessment(cid) for cid in window_ids}
    result = _rank_window(_jd(), window_ids, assessments, _rank_window_chain([3, 1, 2]))
    assert result == ["cand-c", "cand-a", "cand-b"]


def test_rank_window_falls_back_on_invalid_permutation():
    window_ids = ["cand-a", "cand-b", "cand-c"]
    assessments = {cid: _assessment(cid) for cid in window_ids}

    def _build(model):
        chain = MagicMock()
        def _invoke(payload):
            evals = [_CandidateEvaluation(id=i, key_strengths="s", key_gaps="g") for i in range(1, 4)]
            return model(candidate_evaluations=evals, ranking=[1, 1, 2])  # duplicate — invalid permutation
        chain.invoke.side_effect = _invoke
        return chain

    result = _rank_window(_jd(), window_ids, assessments, _build)
    assert result == window_ids


def test_rank_window_falls_back_on_generation_error():
    window_ids = ["cand-a", "cand-b"]
    assessments = {cid: _assessment(cid) for cid in window_ids}

    def _build(model):
        chain = MagicMock()
        chain.invoke.side_effect = ValueError("LLM error")
        return chain

    result = _rank_window(_jd(), window_ids, assessments, _build)
    assert result == window_ids


def _identity_chain_factory() -> Callable:
    """Chain factory that always returns candidates in input order (no reranking)."""
    def _build(model):
        chain = MagicMock()
        def _invoke(payload):
            blocks = [b for b in payload["candidates_text"].split("\n\n") if b.strip()]
            size = len(blocks)
            evals = [
                _CandidateEvaluation(id=i, key_strengths="s", key_gaps="g")
                for i in range(1, size + 1)
            ]
            return model(candidate_evaluations=evals, ranking=list(range(1, size + 1)))
        chain.invoke.side_effect = _invoke
        return chain
    return _build


def test_rank_sliding_window_returns_dict_with_all_candidates():
    jd = _jd()
    assessments = {f"cand-{i}": _assessment(f"cand-{i}") for i in range(5)}
    result = rank_sliding_window(jd, assessments, _identity_chain_factory(), window_size=3, step=2)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(assessments.keys())
    assert all(isinstance(v, float) for v in result.values())


def test_rank_sliding_window_single_candidate_no_chain_call():
    jd = _jd()
    assessments = {"cand-a": _assessment("cand-a")}
    build_chain = MagicMock()
    result = rank_sliding_window(jd, assessments, build_chain, window_size=3, step=2)
    assert result == {"cand-a": 1.0}
    build_chain.assert_not_called()


def test_rank_sliding_window_utilities_reflect_initial_sort():
    jd = _jd()
    assessments = {
        "cand-a": _assessment("cand-a", strong_met=0, years=1),
        "cand-b": _assessment("cand-b", strong_met=2, years=5),
        "cand-c": _assessment("cand-c", strong_met=1, years=3),
    }
    # identity chain preserves order → initial sort determines final order
    # initial sort: cand-b (2 strong) > cand-c (1 strong) > cand-a (0 strong)
    result = rank_sliding_window(jd, assessments, _identity_chain_factory(), window_size=3, step=2)
    assert result["cand-b"] > result["cand-c"] > result["cand-a"]


def test_rank_sliding_window_rng_seed_same_seed_reproducible():
    jd = _jd()
    assessments = {f"cand-{i}": _assessment(f"cand-{i}", strong_met=1, years=i + 1) for i in range(5)}
    result_a = rank_sliding_window(jd, assessments, _identity_chain_factory(), window_size=3, step=2, rng_seed=42)
    result_b = rank_sliding_window(jd, assessments, _identity_chain_factory(), window_size=3, step=2, rng_seed=42)
    assert result_a == result_b


def test_rank_sliding_window_rng_seed_none_unchanged():
    jd = _jd()
    assessments = {f"cand-{i}": _assessment(f"cand-{i}", strong_met=1, years=i + 1) for i in range(5)}
    result_default = rank_sliding_window(jd, assessments, _identity_chain_factory(), window_size=3, step=2)
    result_none = rank_sliding_window(jd, assessments, _identity_chain_factory(), window_size=3, step=2, rng_seed=None)
    assert result_default == result_none
