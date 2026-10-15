from __future__ import annotations

from unittest.mock import MagicMock, patch

from candidate_ranking.models import Assessment, JobDescription, RequirementEvaluation
from candidate_ranking.scoring.active_listwise_ranking import rank_active_listwise

MODULE = "candidate_ranking.scoring.active_listwise_ranking"


def _jd() -> JobDescription:
    return JobDescription(
        id="jd-t", title="Engineer",
        raw_text="Python Django REST API",
        source_path="/tmp/jd.txt",
    )


def _assessment(cid: str, years: int = 3) -> Assessment:
    return Assessment(
        job_description_id="jd-t", candidate_id=cid, generated_by_model="test",
        must_have_evaluations=[RequirementEvaluation(
            requirement="Python", verdict="MET", evidence="used it",
            evidence_strength="strong", recency="current", years_demonstrated=years,
        )],
        total_relevant_years=years, qualified=True,
    )


def test_single_candidate_returns_immediately():
    result = rank_active_listwise(_jd(), {"a": _assessment("a")}, MagicMock())
    assert result == {"a": 1.0}


def test_small_pool_one_llm_call():
    assessments = {f"c{i}": _assessment(f"c{i}") for i in range(5)}
    call_count = 0

    def mock_rank(jd, subset_ids, asmt, llm):
        nonlocal call_count
        call_count += 1
        return list(subset_ids)

    with patch(f"{MODULE}._rank_subset", side_effect=mock_rank):
        result = rank_active_listwise(_jd(), assessments, MagicMock(), subset_size=8)

    assert call_count == 1
    assert set(result) == set(assessments)


def test_returns_float_utilities_for_all_candidates():
    assessments = {f"c{i}": _assessment(f"c{i}", years=i + 1) for i in range(10)}

    def mock_rank(jd, subset_ids, asmt, llm):
        return list(subset_ids)

    with patch(f"{MODULE}._rank_subset", side_effect=mock_rank):
        result = rank_active_listwise(_jd(), assessments, MagicMock(), subset_size=4)

    assert set(result) == set(assessments)
    assert all(isinstance(v, float) for v in result.values())


def test_llm_failure_does_not_crash():
    assessments = {f"c{i}": _assessment(f"c{i}") for i in range(10)}

    with patch(f"{MODULE}._rank_subset", return_value=None):
        result = rank_active_listwise(_jd(), assessments, MagicMock(), subset_size=4, max_iterations=3)

    assert set(result) == set(assessments)


def test_empty_evidence_does_not_crash_bm25_warm_start():
    assessments = {
        f"c{i}": Assessment(
            job_description_id="jd-t", candidate_id=f"c{i}", generated_by_model="test",
        )
        for i in range(5)
    }

    def mock_rank(jd, subset_ids, asmt, llm):
        return list(subset_ids)

    with patch(f"{MODULE}._rank_subset", side_effect=mock_rank):
        result = rank_active_listwise(_jd(), assessments, MagicMock(), subset_size=8)

    assert set(result) == set(assessments)


def test_consistent_ranking_orders_candidates():
    assessments = {f"c{i}": _assessment(f"c{i}", years=i + 1) for i in range(10)}

    def mock_rank(jd, subset_ids, asmt, llm):
        return sorted(subset_ids, key=lambda cid: asmt[cid].total_relevant_years, reverse=True)

    with patch(f"{MODULE}._rank_subset", side_effect=mock_rank):
        result = rank_active_listwise(
            _jd(), assessments, MagicMock(),
            subset_size=5, max_iterations=20, min_delta_u=0.01,
        )

    assert result["c9"] > result["c0"]
