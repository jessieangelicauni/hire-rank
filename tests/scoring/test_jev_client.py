from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from candidate_ranking.scoring.jev_client import JevAnswer, JevClient, JevClientError, JevQuestion


def _fake_response(answers: dict) -> Mock:
    response = Mock()
    response.raise_for_status = Mock()
    response.json = Mock(return_value={"model": "jev-1.13.0", "answers": answers, "usage": {}})
    return response


@patch("candidate_ranking.scoring.jev_client.requests.post")
def test_evaluate_parses_noul_choice_and_score_answers(mock_post):
    mock_post.return_value = _fake_response(
        {
            "meets_min_qualifications": {"type": "noul", "noul": 0.95},
            "overall_recommendation": {
                "type": "choice",
                "choice": "hire",
                "confidence": 0.8,
                "probabilities": {"hire": 0.8, "maybe": 0.15, "no": 0.05},
            },
            "seniority_years": {
                "type": "score",
                "score": 3.0,
                "confidence": 0.9,
                "legend": {"0": "0", "1": "25", "2": "50", "3": "75", "4": "100"},
                "probabilities": {"0": 0, "1": 0, "2": 0.1, "3": 0.9, "4": 0},
            },
        }
    )

    client = JevClient(api_token="token123")
    questions = [
        JevQuestion(key="meets_min_qualifications", kind="noul", instructions="...", criteria={"true": "...", "false": "..."}),
        JevQuestion(key="overall_recommendation", kind="choice", instructions="...", criteria={"hire": "...", "maybe": "...", "no": "..."}),
        JevQuestion(key="seniority_years", kind="score", instructions="...", criteria=["0", "25", "50", "75", "100"]),
    ]

    answers = client.evaluate("some state text", questions)

    by_key = {a.key: a for a in answers}
    assert by_key["meets_min_qualifications"] == JevAnswer(
        key="meets_min_qualifications", kind="noul", value=True, confidence=0.95
    )
    assert by_key["overall_recommendation"] == JevAnswer(
        key="overall_recommendation", kind="choice", value="hire", confidence=0.8
    )
    assert by_key["seniority_years"] == JevAnswer(
        key="seniority_years", kind="score", value=3.0, confidence=0.9
    )

    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["headers"]["Authorization"] == "Bearer token123"
    assert call_kwargs["json"]["model"] == "jev-latest"
    assert call_kwargs["json"]["state"] == "some state text"
    assert set(call_kwargs["json"]["questions"]) == {
        "meets_min_qualifications", "overall_recommendation", "seniority_years",
    }


@patch("candidate_ranking.scoring.jev_client.requests.post")
def test_evaluate_raises_jev_client_error_on_http_failure(mock_post):
    import requests

    mock_post.side_effect = requests.ConnectionError("boom")
    client = JevClient(api_token="token123")

    with pytest.raises(JevClientError, match="boom"):
        client.evaluate("state", [JevQuestion(key="k", kind="noul", instructions="i", criteria={"true": "t", "false": "f"})])


@patch("candidate_ranking.scoring.jev_client.requests.post")
def test_evaluate_raises_jev_client_error_on_missing_answers_key(mock_post):
    mock_post.return_value = _fake_response({})
    mock_post.return_value.json = Mock(return_value={"model": "jev-1.13.0"})
    client = JevClient(api_token="token123")

    with pytest.raises(JevClientError, match="answers"):
        client.evaluate("state", [JevQuestion(key="k", kind="noul", instructions="i", criteria={"true": "t", "false": "f"})])
