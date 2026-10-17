from __future__ import annotations

import numpy as np
from unittest.mock import MagicMock, patch

from candidate_ranking.models import Assessment, JobProfile, RequirementEvaluation
from candidate_ranking.scoring.active_pairwise_ranking import rank_active_pairwise, _select_uncertain_subset

MODULE = "candidate_ranking.scoring.active_pairwise_ranking"


def _jp() -> JobProfile:
    return JobProfile(
        id="jd-t", title="Engineer",
        raw_text="Python Django REST API",
        source_path="/tmp/jd.txt",
    )


def _assessment(cid: str, years: int = 3, qualified: bool = True) -> Assessment:
    return Assessment(
        job_profile_id="jd-t", candidate_id=cid, generated_by_model="test",
        must_have_evaluations=[RequirementEvaluation(
            requirement="Python", evidence="used it",
            evidence_strength="strong",
        )],
        total_relevant_years=years, qualified=qualified,
    )


def test_single_candidate_returns_immediately():
    result = rank_active_pairwise(_jp(), {"a": _assessment("a")}, MagicMock())
    assert result == {"a": 1.0}


def test_small_pool_gets_more_than_one_independent_llm_ranking():
    assessments = {f"c{i}": _assessment(f"c{i}") for i in range(5)}
    call_count = 0

    def mock_rank(jp, subset_ids, asmt, llm):
        # A consistent, input-order-independent ranking — same opinion every
        # call, regardless of what order the subset happens to arrive in.
        nonlocal call_count
        call_count += 1
        return sorted(subset_ids, key=lambda cid: asmt[cid].total_relevant_years, reverse=True)

    with patch(f"{MODULE}._rank_subset", side_effect=mock_rank):
        result = rank_active_pairwise(_jp(), assessments, MagicMock(), subset_size=8, min_delta_u=0.01)

    # A small pool (n <= subset_size) must still get more than one independent
    # LLM ranking call — settling for N=1 was the bug (see active_pairwise_ranking.py).
    assert call_count > 1
    assert set(result) == set(assessments)


def test_returns_float_utilities_for_all_candidates():
    assessments = {f"c{i}": _assessment(f"c{i}", years=i + 1) for i in range(10)}

    def mock_rank(jp, subset_ids, asmt, llm):
        return list(subset_ids)

    with patch(f"{MODULE}._rank_subset", side_effect=mock_rank):
        result = rank_active_pairwise(_jp(), assessments, MagicMock(), subset_size=4)

    assert set(result) == set(assessments)
    assert all(isinstance(v, float) for v in result.values())


def test_llm_failure_does_not_crash():
    assessments = {f"c{i}": _assessment(f"c{i}") for i in range(10)}

    with patch(f"{MODULE}._rank_subset", return_value=None):
        result = rank_active_pairwise(_jp(), assessments, MagicMock(), subset_size=4, max_iterations=3)

    assert set(result) == set(assessments)


def test_empty_evidence_does_not_crash_bm25_warm_start():
    assessments = {
        f"c{i}": Assessment(
            job_profile_id="jd-t", candidate_id=f"c{i}", generated_by_model="test",
        )
        for i in range(5)
    }

    def mock_rank(jp, subset_ids, asmt, llm):
        return list(subset_ids)

    with patch(f"{MODULE}._rank_subset", side_effect=mock_rank):
        result = rank_active_pairwise(_jp(), assessments, MagicMock(), subset_size=8)

    assert set(result) == set(assessments)


def test_select_uncertain_subset_avoids_repeating_a_failed_combination():
    item_ids = ["a", "b", "c", "d", "e"]
    variances = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # top-3 by variance: a, b, c

    first_pick = _select_uncertain_subset(variances, item_ids, subset_size=3)
    assert set(first_pick) == {"a", "b", "c"}

    retried_pick = _select_uncertain_subset(
        variances, item_ids, subset_size=3, avoid_signature=frozenset({"a", "b", "c"})
    )
    assert set(retried_pick) != {"a", "b", "c"}
    assert set(retried_pick) == {"a", "b", "d"}  # lowest-variance member (c) swapped for next-best (d)


def test_select_uncertain_subset_unaffected_when_signature_differs():
    item_ids = ["a", "b", "c", "d"]
    variances = np.array([4.0, 3.0, 2.0, 1.0])
    pick = _select_uncertain_subset(variances, item_ids, subset_size=2, avoid_signature=frozenset({"x", "y"}))
    assert set(pick) == {"a", "b"}


def test_active_loop_does_not_get_stuck_on_one_unrankable_subset():
    assessments = {f"c{i}": _assessment(f"c{i}", years=i + 1) for i in range(10)}
    cursed = {"c9", "c8", "c7", "c6"}  # highest-variance subset at the start (most years)
    calls_on_cursed = 0

    def mock_rank(jp, subset_ids, asmt, llm):
        nonlocal calls_on_cursed
        if set(subset_ids) == cursed:
            calls_on_cursed += 1
            return None  # this exact combination always fails, like a real stuck LLM
        return list(subset_ids)

    with patch(f"{MODULE}._rank_subset", side_effect=mock_rank):
        result = rank_active_pairwise(
            _jp(), assessments, MagicMock(), subset_size=4, max_iterations=10,
        )

    assert set(result) == set(assessments)
    # the cursed combination should be hit once, then avoided — not retried every iteration
    assert calls_on_cursed <= 2


def test_disqualified_candidates_never_outrank_a_qualified_one():
    # A disqualified candidate the LLM would otherwise rate very highly must
    # still land below every qualified candidate, however poorly rated.
    assessments = {
        "weak-but-qualified": _assessment("weak-but-qualified", years=1, qualified=True),
        "strong-but-disqualified": _assessment("strong-but-disqualified", years=20, qualified=False),
    }

    def mock_rank(jp, subset_ids, asmt, llm):
        # LLM would rank the disqualified one first if given the chance.
        return sorted(subset_ids, key=lambda cid: asmt[cid].total_relevant_years, reverse=True)

    with patch(f"{MODULE}._rank_subset", side_effect=mock_rank):
        result = rank_active_pairwise(_jp(), assessments, MagicMock(), subset_size=8)

    assert result["weak-but-qualified"] > result["strong-but-disqualified"]


def test_disqualified_candidates_keep_a_relative_order_among_themselves():
    assessments = {
        "qualified-1": _assessment("qualified-1", qualified=True),
        "disqualified-a": _assessment("disqualified-a", qualified=False),
        "disqualified-b": _assessment("disqualified-b", qualified=False),
    }

    with patch(f"{MODULE}._rank_subset", side_effect=lambda jp, s, a, l: list(s)):
        result = rank_active_pairwise(_jp(), assessments, MagicMock(), subset_size=8)

    assert result["disqualified-a"] != result["disqualified-b"]
    assert result["qualified-1"] > result["disqualified-a"]
    assert result["qualified-1"] > result["disqualified-b"]


def test_all_disqualified_pool_still_returns_every_candidate():
    assessments = {f"c{i}": _assessment(f"c{i}", qualified=False) for i in range(5)}

    with patch(f"{MODULE}._rank_subset", side_effect=lambda jp, s, a, l: list(s)):
        result = rank_active_pairwise(_jp(), assessments, MagicMock(), subset_size=8)

    assert set(result) == set(assessments)


def test_consistent_ranking_orders_candidates():
    assessments = {f"c{i}": _assessment(f"c{i}", years=i + 1) for i in range(10)}

    def mock_rank(jp, subset_ids, asmt, llm):
        return sorted(subset_ids, key=lambda cid: asmt[cid].total_relevant_years, reverse=True)

    with patch(f"{MODULE}._rank_subset", side_effect=mock_rank):
        result = rank_active_pairwise(
            _jp(), assessments, MagicMock(),
            subset_size=5, max_iterations=20, min_delta_u=0.01,
        )

    assert result["c9"] > result["c0"]


def test_select_uncertain_subset_rotates_when_no_swap_candidate():
    """When n == subset_size (all in pool), avoid_signature must be escaped via rotation."""
    item_ids = ["a", "b", "c"]
    variances = np.array([1.0, 0.9, 0.8])
    # Top-3 == all 3, so there is no swap candidate.
    initial = _select_uncertain_subset(variances, item_ids, subset_size=3)
    avoid = frozenset(initial)
    # With iteration=1, the list must be rotated (not identical).
    result = _select_uncertain_subset(variances, item_ids, subset_size=3, avoid_signature=avoid, iteration=1)
    assert frozenset(result) == avoid          # same members — pool is the same
    assert result != initial                   # different ORDER — rotation happened


def test_select_uncertain_subset_iteration_zero_no_rotation():
    """iteration=0 (default) must not rotate when no swap candidate exists."""
    item_ids = ["a", "b", "c"]
    variances = np.array([1.0, 0.9, 0.8])
    initial = _select_uncertain_subset(variances, item_ids, subset_size=3)
    avoid = frozenset(initial)
    result = _select_uncertain_subset(variances, item_ids, subset_size=3, avoid_signature=avoid, iteration=0)
    assert result == initial  # iteration=0 → offset=0, no rotation


def test_select_uncertain_subset_rotation_varies_by_iteration():
    """Different iteration values must produce different rotations."""
    item_ids = ["a", "b", "c"]
    variances = np.array([1.0, 0.9, 0.8])
    initial = _select_uncertain_subset(variances, item_ids, subset_size=3)
    avoid = frozenset(initial)
    result1 = _select_uncertain_subset(variances, item_ids, subset_size=3, avoid_signature=avoid, iteration=1)
    result2 = _select_uncertain_subset(variances, item_ids, subset_size=3, avoid_signature=avoid, iteration=2)
    assert result1 != result2  # different iterations → different rotations


def test_rank_pair_jev_returns_winner_first():
    """Winner must be first regardless of which candidate was assigned A or B.

    A/B assignment is deterministically shuffled by hash to mitigate position
    bias, so the test recomputes the expected A/B assignment and verifies the
    returned order — not a hardcoded position.
    """
    import hashlib
    from candidate_ranking.scoring.jev_client import JevAnswer
    from candidate_ranking.scoring.active_pairwise_ranking import _rank_pair_jev
    jp = _jp()
    pair = ["alice", "bob"]
    assessments = {"alice": _assessment("alice", years=5), "bob": _assessment("bob", years=3)}

    # Mirror the hash-flip logic in _rank_pair_jev
    digest = hashlib.md5(f"{pair[0]}||{pair[1]}||{jp.id}".encode()).hexdigest()
    flip = int(digest, 16) % 2 == 1
    expected_a_id = pair[1] if flip else pair[0]
    expected_b_id = pair[0] if flip else pair[1]

    jev = MagicMock()
    jev.evaluate.return_value = [
        JevAnswer(key="stronger_fit", kind="choice", value="A", confidence=0.85)
    ]
    result = _rank_pair_jev(jp, pair, assessments, jev)
    # Jev says "A" wins — so the candidate assigned to A must come first
    assert result == [expected_a_id, expected_b_id]


def test_rank_pair_jev_position_assignment_is_deterministic():
    """Same inputs must produce the same A/B assignment on every call."""
    import hashlib
    from candidate_ranking.scoring.jev_client import JevAnswer
    from candidate_ranking.scoring.active_pairwise_ranking import _rank_pair_jev
    jp = _jp()
    assessments = {"x": _assessment("x"), "y": _assessment("y")}
    jev = MagicMock()
    jev.evaluate.return_value = [
        JevAnswer(key="stronger_fit", kind="choice", value="A", confidence=0.9)
    ]
    result1 = _rank_pair_jev(jp, ["x", "y"], assessments, jev)
    result2 = _rank_pair_jev(jp, ["x", "y"], assessments, jev)
    assert result1 == result2  # deterministic — no randomness


def test_rank_pair_jev_returns_none_on_jev_error():
    from candidate_ranking.scoring.jev_client import JevClientError
    from candidate_ranking.scoring.active_pairwise_ranking import _rank_pair_jev
    jev = MagicMock()
    jev.evaluate.side_effect = JevClientError("timeout")
    result = _rank_pair_jev(_jp(), ["a", "b"], {"a": _assessment("a"), "b": _assessment("b")}, jev)
    assert result is None


def test_rank_active_pairwise_uses_jev_pairwise_when_client_provided():
    """When jev_client is given, the LLM path must never be called."""
    from candidate_ranking.scoring.jev_client import JevAnswer
    assessments = {f"c{i}": _assessment(f"c{i}") for i in range(4)}
    jev = MagicMock()
    jev.evaluate.return_value = [
        JevAnswer(key="stronger_fit", kind="choice", value="A", confidence=0.9)
    ]
    with patch(f"{MODULE}._rank_subset") as mock_llm_rank:
        result = rank_active_pairwise(_jp(), assessments, jev_client=jev, subset_size=2, max_iterations=5)
    mock_llm_rank.assert_not_called()
    assert set(result) == set(assessments)


def test_active_ranking_prompt_has_no_career_trajectory_reference():
    from candidate_ranking.scoring.active_pairwise_ranking import ACTIVE_RANKING_PROMPT
    prompt_text = str(ACTIVE_RANKING_PROMPT)
    assert "career trajectory" not in prompt_text.lower()


def test_rank_pair_jev_instructions_have_no_career_trajectory_reference():
    from candidate_ranking.scoring.jev_client import JevAnswer
    from candidate_ranking.scoring.active_pairwise_ranking import _rank_pair_jev
    jp = _jp()
    jev = MagicMock()
    jev.evaluate.return_value = [JevAnswer(key="stronger_fit", kind="choice", value="A", confidence=0.9)]
    _rank_pair_jev(jp, ["a", "b"], {"a": _assessment("a"), "b": _assessment("b")}, jev)
    sent_question = jev.evaluate.call_args[0][1][0]
    assert "career trajectory" not in sent_question.instructions.lower()


def test_active_ranking_prompt_asks_jev_to_weigh_requirement_match():
    """The 'Other requirements' evidence block render.py sends is useless if
    the ranking prompt never tells Jev to weigh it."""
    from candidate_ranking.scoring.active_pairwise_ranking import ACTIVE_RANKING_PROMPT
    prompt_text = str(ACTIVE_RANKING_PROMPT).lower()
    assert "requirement match" in prompt_text


def test_rank_pair_jev_instructions_ask_jev_to_weigh_requirement_match():
    from candidate_ranking.scoring.jev_client import JevAnswer
    from candidate_ranking.scoring.active_pairwise_ranking import _rank_pair_jev
    jp = _jp()
    jev = MagicMock()
    jev.evaluate.return_value = [JevAnswer(key="stronger_fit", kind="choice", value="A", confidence=0.9)]
    _rank_pair_jev(jp, ["a", "b"], {"a": _assessment("a"), "b": _assessment("b")}, jev)
    sent_question = jev.evaluate.call_args[0][1][0]
    assert "requirement match" in sent_question.instructions.lower()
