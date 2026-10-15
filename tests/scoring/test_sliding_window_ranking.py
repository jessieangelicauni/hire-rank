# tests/scoring/test_sliding_window_ranking.py
from __future__ import annotations
from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError
from candidate_ranking.scoring.sliding_window_ranking import (
    SLIDING_WINDOW_RANKING_PROMPT,
    _CandidateEvaluation,
    _sliding_window_model,
    build_sliding_window_chain_factory,
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
