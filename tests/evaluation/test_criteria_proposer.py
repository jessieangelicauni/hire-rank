from __future__ import annotations

from unittest.mock import Mock

import pytest

from candidate_ranking.evaluation.criteria_proposer import _ProposedCriteria, propose_criteria
from candidate_ranking.generation import GenerationError

_CURRENT = ["never", "rarely", "sometimes", "often", "always"]


def test_propose_criteria_returns_chain_levels():
    chain = Mock()
    chain.invoke.return_value = _ProposedCriteria(levels=["Level A", "Level B", "Level C", "Level D", "Level E"])

    result = propose_criteria(_CURRENT, [], chain)

    assert result == ["Level A", "Level B", "Level C", "Level D", "Level E"]


def test_propose_criteria_passes_current_criteria_and_hard_cases_to_chain():
    chain = Mock()
    chain.invoke.return_value = _ProposedCriteria(levels=["A", "B", "C", "D", "E"])

    propose_criteria(_CURRENT, ["hard case 1"], chain)

    call_payload = chain.invoke.call_args.args[0]
    assert "never" in call_payload["current_criteria"]
    assert "hard case 1" in call_payload["hard_cases"]


def test_propose_criteria_uses_none_yet_placeholder_when_no_hard_cases():
    chain = Mock()
    chain.invoke.return_value = _ProposedCriteria(levels=["A", "B", "C", "D", "E"])

    propose_criteria(_CURRENT, [], chain)

    call_payload = chain.invoke.call_args.args[0]
    assert call_payload["hard_cases"] == "(none yet)"


def test_propose_criteria_rejects_wrong_line_count():
    chain = Mock()
    chain.invoke.return_value = _ProposedCriteria(levels=["Only", "Three", "Lines"])

    with pytest.raises(GenerationError, match="exactly 5"):
        propose_criteria(_CURRENT, [], chain)
