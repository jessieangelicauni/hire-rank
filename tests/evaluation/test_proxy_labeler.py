from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from candidate_ranking.evaluation.criteria_optimization import RequirementTriple, triple_key
from candidate_ranking.evaluation.proxy_labeler import ProxyLabelClient
from candidate_ranking.models import Candidate, JobDescription


def _jd() -> JobDescription:
    return JobDescription(id="jd-1", title="Backend Engineer", raw_text="Needs Python.", source_path="jd.pdf")


def _candidate() -> Candidate:
    return Candidate(
        id="cand-1", source_path="cv.pdf", raw_text="I know Python.", num_pages=1, char_count=20,
        parse_status="ok",
    )


def _anthropic_response(text: str) -> Mock:
    response = Mock()
    response.content = [Mock(text=text)]
    return response


def test_label_calls_anthropic_and_parses_level(tmp_path):
    client = Mock()
    client.messages.create.return_value = _anthropic_response("3")
    labeler = ProxyLabelClient(client, cache_path=tmp_path / "cache.json", model="claude-opus-5")
    triple = RequirementTriple(job_description_id="jd-1", candidate_id="cand-1", requirement="Python")

    level = labeler.label(triple, _jd(), _candidate())

    assert level == 3
    client.messages.create.assert_called_once()
    assert client.messages.create.call_args.kwargs["model"] == "claude-opus-5"


def test_label_uses_cache_on_second_call(tmp_path):
    client = Mock()
    client.messages.create.return_value = _anthropic_response("2")
    labeler = ProxyLabelClient(client, cache_path=tmp_path / "cache.json", model="claude-opus-5")
    triple = RequirementTriple(job_description_id="jd-1", candidate_id="cand-1", requirement="Python")

    first = labeler.label(triple, _jd(), _candidate())
    second = labeler.label(triple, _jd(), _candidate())

    assert first == second == 2
    client.messages.create.assert_called_once()


def test_label_reuses_cache_across_client_instances(tmp_path):
    cache_path = tmp_path / "cache.json"
    client = Mock()
    client.messages.create.return_value = _anthropic_response("2")
    labeler1 = ProxyLabelClient(client, cache_path=cache_path, model="claude-opus-5")
    triple = RequirementTriple(job_description_id="jd-1", candidate_id="cand-1", requirement="Python")
    labeler1.label(triple, _jd(), _candidate())

    labeler2 = ProxyLabelClient(client, cache_path=cache_path, model="claude-opus-5")
    result = labeler2.label(triple, _jd(), _candidate())

    assert result == 2
    client.messages.create.assert_called_once()


def test_label_persists_cache_to_disk(tmp_path):
    cache_path = tmp_path / "cache.json"
    client = Mock()
    client.messages.create.return_value = _anthropic_response("1")
    labeler = ProxyLabelClient(client, cache_path=cache_path, model="claude-opus-5")
    triple = RequirementTriple(job_description_id="jd-1", candidate_id="cand-1", requirement="Python")

    labeler.label(triple, _jd(), _candidate())

    saved = json.loads(cache_path.read_text(encoding="utf-8"))
    assert saved[triple_key(triple)] == 1


def test_label_raises_on_unparseable_response(tmp_path):
    client = Mock()
    client.messages.create.return_value = _anthropic_response("not a number")
    labeler = ProxyLabelClient(client, cache_path=tmp_path / "cache.json", model="claude-opus-5")
    triple = RequirementTriple(job_description_id="jd-1", candidate_id="cand-1", requirement="Python")

    with pytest.raises(ValueError, match="not parseable"):
        labeler.label(triple, _jd(), _candidate())


def test_label_raises_on_out_of_range_level(tmp_path):
    client = Mock()
    client.messages.create.return_value = _anthropic_response("9")
    labeler = ProxyLabelClient(client, cache_path=tmp_path / "cache.json", model="claude-opus-5")
    triple = RequirementTriple(job_description_id="jd-1", candidate_id="cand-1", requirement="Python")

    with pytest.raises(ValueError, match="out of range"):
        labeler.label(triple, _jd(), _candidate())


def test_label_raises_on_multi_digit_malformed_response(tmp_path):
    client = Mock()
    client.messages.create.return_value = _anthropic_response("10")
    labeler = ProxyLabelClient(client, cache_path=tmp_path / "cache.json", model="claude-opus-5")
    triple = RequirementTriple(job_description_id="jd-1", candidate_id="cand-1", requirement="Python")

    with pytest.raises(ValueError, match="out of range"):
        labeler.label(triple, _jd(), _candidate())
