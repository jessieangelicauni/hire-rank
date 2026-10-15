# tests/scoring/test_sliding_window_ranking.py
from __future__ import annotations
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
