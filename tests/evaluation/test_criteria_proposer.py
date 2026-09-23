from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from candidate_ranking.evaluation.criteria_proposer import propose_criteria

_CURRENT = ["never", "rarely", "sometimes", "often", "always"]


@patch("candidate_ranking.evaluation.criteria_proposer.dspy.context")
@patch("candidate_ranking.evaluation.criteria_proposer.dspy.Predict")
def test_propose_criteria_parses_five_lines(mock_predict_cls, mock_context):
    mock_predict_instance = Mock(
        return_value=Mock(revised_criteria="Level A\nLevel B\nLevel C\nLevel D\nLevel E")
    )
    mock_predict_cls.return_value = mock_predict_instance
    mock_context.return_value.__enter__ = Mock(return_value=None)
    mock_context.return_value.__exit__ = Mock(return_value=False)

    result = propose_criteria(_CURRENT, [], lm=Mock())

    assert result == ["Level A", "Level B", "Level C", "Level D", "Level E"]


@patch("candidate_ranking.evaluation.criteria_proposer.dspy.context")
@patch("candidate_ranking.evaluation.criteria_proposer.dspy.Predict")
def test_propose_criteria_passes_current_criteria_and_hard_cases_to_predict(mock_predict_cls, mock_context):
    mock_predict_instance = Mock(
        return_value=Mock(revised_criteria="A\nB\nC\nD\nE")
    )
    mock_predict_cls.return_value = mock_predict_instance
    mock_context.return_value.__enter__ = Mock(return_value=None)
    mock_context.return_value.__exit__ = Mock(return_value=False)

    propose_criteria(_CURRENT, ["hard case 1"], lm=Mock())

    call_kwargs = mock_predict_instance.call_args.kwargs
    assert "never" in call_kwargs["current_criteria"]
    assert "hard case 1" in call_kwargs["hard_cases"]


@patch("candidate_ranking.evaluation.criteria_proposer.dspy.context")
@patch("candidate_ranking.evaluation.criteria_proposer.dspy.Predict")
def test_propose_criteria_rejects_wrong_line_count(mock_predict_cls, mock_context):
    mock_predict_instance = Mock(return_value=Mock(revised_criteria="Only\nThree\nLines"))
    mock_predict_cls.return_value = mock_predict_instance
    mock_context.return_value.__enter__ = Mock(return_value=None)
    mock_context.return_value.__exit__ = Mock(return_value=False)

    with pytest.raises(ValueError, match="exactly 5"):
        propose_criteria(_CURRENT, [], lm=Mock())
