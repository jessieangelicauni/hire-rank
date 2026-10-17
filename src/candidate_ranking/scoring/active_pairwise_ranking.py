from __future__ import annotations

import hashlib
import logging
import math
from typing import TYPE_CHECKING, Callable

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, create_model

from candidate_ranking.models import Assessment, JobProfile
from candidate_ranking.ranking.plackett_luce import fit_utilities, laplace_covariance
from candidate_ranking.scoring.render import (
    _extract_jd_summary,
    _render_candidate_evidence,
    _render_evidence_for_ranking,
)
from candidate_ranking.utils import GenerationError, invoke_and_validate

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from candidate_ranking.scoring.jev_client import JevClient

logger = logging.getLogger(__name__)

ACTIVE_RANKING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior hiring panel ranking candidates for a specific role.\n\n"
        "Rank ALL candidates from strongest to weakest fit for this role.\n\n"
        "RANKING CRITERIA (in priority order):\n"
        "1. Relevance — experience directly matching daily responsibilities\n"
        "2. Evidence quality — concrete achievements over vague responsibilities\n"
        "3. Career trajectory — ascending scope and seniority\n"
        "4. Recency — recent hands-on experience\n"
        "5. Nice-to-have coverage — breadth of preferred skills as tiebreaker\n\n"
        "Output ONLY the ranking permutation. No explanations. No ties. "
        "ranking must contain every candidate number exactly once.",
    ),
    ("human", "ROLE: {jd_title}\n\nJOB SUMMARY: {jd_summary}\n\nCANDIDATES:\n{candidates_text}"),
])


def _active_ranking_model(size: int) -> type[BaseModel]:
    return create_model(
        "_ActiveRankingResult",
        ranking=(list[int], Field(min_length=size, max_length=size)),
    )


def _bm25_sort(jp: JobProfile, assessments: dict[str, Assessment]) -> list[str]:
    from rank_bm25 import BM25Okapi

    cids = list(assessments)
    corpus = []
    for cid in cids:
        a = assessments[cid]
        parts = [f"{ev.requirement} {ev.evidence}" for ev in a.must_have_evaluations]
        parts += [f"{ev.requirement} {ev.evidence}" for ev in a.nice_to_have_evaluations if ev.evidence_strength != "none"]
        corpus.append(" ".join(parts).lower().split())

    if not any(corpus):
        # BM25Okapi divides by its vocabulary size while fitting; every candidate's
        # evidence tokenizing to nothing (e.g. a JD whose requirement extraction came
        # back empty) leaves that vocabulary empty and raises ZeroDivisionError.
        logger.warning("JD %s: no evidence text to warm-start BM25 — using arbitrary initial order", jp.id)
        return cids

    scores = BM25Okapi(corpus).get_scores(jp.raw_text.lower().split())
    return [cid for cid, _ in sorted(zip(cids, scores), key=lambda x: -x[1])]


def _select_uncertain_subset(
    variances: np.ndarray,
    item_ids: list[str],
    subset_size: int,
    avoid_signature: frozenset[str] | None = None,
    iteration: int = 0,
) -> list[str]:
    """Pick the subset_size items with highest utility variance.

    If that exact combination is `avoid_signature`, first try swapping the
    lowest-variance member for the next-highest-variance item outside the
    subset.  If no such item exists (n <= subset_size), rotate the list by
    `iteration` instead — ensuring each retry presents candidates in a
    different order even when the pool is exhausted.
    """
    order = np.argsort(variances)[::-1]
    selected = [item_ids[i] for i in order[:subset_size]]
    if avoid_signature is not None and frozenset(selected) == avoid_signature:
        swapped = False
        for i in order[subset_size:]:
            replacement = item_ids[i]
            if replacement not in selected:
                selected[-1] = replacement
                swapped = True
                break
        if not swapped and len(selected) > 1:
            # Single-item subsets can't be rotated meaningfully; skip.
            # No swap candidate (pool exhausted) — rotate by iteration
            offset = iteration % len(selected)
            selected = selected[offset:] + selected[:offset]
    return selected


def _rank_subset(
    jp: JobProfile,
    subset_ids: list[str],
    assessments: dict[str, Assessment],
    llm: "BaseChatModel",
) -> list[str] | None:
    size = len(subset_ids)
    token_to_id = {i + 1: cid for i, cid in enumerate(subset_ids)}
    model = _active_ranking_model(size)
    chain = ACTIVE_RANKING_PROMPT | llm.with_structured_output(model)

    def validate(result: BaseModel) -> str | None:
        r = result.ranking  # type: ignore[attr-defined]
        if set(r) != set(token_to_id) or len(r) != size:
            return f"ranking must be a permutation of {sorted(token_to_id)}, got {r}"
        return None

    try:
        result = invoke_and_validate(
            chain,
            {
                "jd_title": jp.title,
                "jd_summary": _extract_jd_summary(jp),
                "candidates_text": _render_evidence_for_ranking(subset_ids, assessments),
            },
            model, validate,
            log_context=f"Active ranking {jp.id} (size={size})",
        )
    except GenerationError as exc:
        logger.warning("Active ranking subset failed for %s: %s — skipping", jp.id, exc)
        return None

    return [token_to_id[t] for t in result.ranking]  # type: ignore[attr-defined]


def _rank_pair_jev(
    jp: JobProfile,
    pair_ids: list[str],
    assessments: dict[str, Assessment],
    jev_client: "JevClient",
) -> list[str] | None:
    """Ask Jev which of two candidates is the stronger fit. Returns [winner, loser] or None on error.

    A/B assignment is deterministically randomised per (jp, pair) using MD5 to
    mitigate position bias: the LLM's known tendency to favour the first-presented
    candidate (Zheng et al. 2023). Since _select_uncertain_subset always returns
    items in descending variance order, without this the highest-variance candidate
    would always occupy position A across all iterations, accumulating a systematic
    advantage unrelated to fit.
    """
    from candidate_ranking.scoring.jev_client import JevClientError, JevQuestion
    assert len(pair_ids) == 2, "pairwise comparator requires exactly 2 candidates"

    # Deterministic shuffle: stable across runs, varies by (jp, pair) combination.
    digest = hashlib.md5(f"{pair_ids[0]}||{pair_ids[1]}||{jp.id}".encode()).hexdigest()
    flip = int(digest, 16) % 2 == 1
    a_id, b_id = (pair_ids[1], pair_ids[0]) if flip else (pair_ids[0], pair_ids[1])

    state = (
        f"ROLE: {jp.title}\n\n"
        f"CANDIDATE A:\n{_render_candidate_evidence(a_id, assessments)}\n\n"
        f"CANDIDATE B:\n{_render_candidate_evidence(b_id, assessments)}"
    )
    question = JevQuestion(
        key="stronger_fit",
        kind="choice",
        instructions=(
            "Based on the evidence summaries above, which candidate is the stronger fit for this role?\n"
            "Consider: relevance of experience, evidence quality, career trajectory, recency, "
            "and nice-to-have coverage as a tiebreaker. Do not consider the candidate labels (A/B) "
            "as signals — evaluate purely on the evidence."
        ),
        criteria={
            "A": "Candidate A has stronger overall fit for this role",
            "B": "Candidate B has stronger overall fit for this role",
        },
    )
    try:
        answers = jev_client.evaluate(state, [question])
    except JevClientError as exc:
        logger.warning("Jev pairwise ranking failed for %s: %s — skipping", jp.id, exc)
        return None
    winner = answers[0].value
    winner_id = a_id if winner == "A" else b_id
    loser_id = b_id if winner == "A" else a_id
    return [winner_id, loser_id]


def rank_active_pairwise(
    jp: JobProfile,
    assessments: dict[str, Assessment],
    llm: "BaseChatModel | None" = None,
    jev_client: "JevClient | None" = None,
    subset_size: int = 8,
    max_iterations: int = 40,
    min_delta_u: float = 0.01,
    pl_prior_variance: float = 1.0,
) -> dict[str, float]:
    """Active listwise (or pairwise) ranking via variance-based subset selection + Plackett-Luce.

    Pass jev_client (with subset_size=2) for pairwise Jev mode.
    Pass llm for legacy Ollama listwise mode.

    Disqualified candidates are never ranked above a qualified one — the active
    loop only runs on qualified candidates; disqualified candidates are ordered
    by BM25 and placed strictly below the lowest qualified utility.
    """
    if jev_client is not None:
        def rank_fn(j, s, a): return _rank_pair_jev(j, s, a, jev_client)
    elif llm is not None:
        def rank_fn(j, s, a): return _rank_subset(j, s, a, llm)
    else:
        raise ValueError("Either llm or jev_client must be provided to rank_active_pairwise")

    qualified = {cid: a for cid, a in assessments.items() if a.qualified}
    disqualified = {cid: a for cid, a in assessments.items() if not a.qualified}

    if qualified and disqualified:
        qualified_utilities = _rank_active_pairwise_core(
            jp, qualified, rank_fn, subset_size, max_iterations, min_delta_u, pl_prior_variance
        )
        floor = min(qualified_utilities.values()) - 1.0
        disqualified_order = _bm25_sort(jp, disqualified)
        disqualified_utilities = {cid: floor - i for i, cid in enumerate(disqualified_order)}
        return {**qualified_utilities, **disqualified_utilities}

    return _rank_active_pairwise_core(
        jp, assessments, rank_fn, subset_size, max_iterations, min_delta_u, pl_prior_variance
    )


def _rank_active_pairwise_core(
    jp: JobProfile,
    assessments: dict[str, Assessment],
    rank_fn: "Callable[[JobProfile, list[str], dict[str, Assessment]], list[str] | None]",
    subset_size: int,
    max_iterations: int,
    min_delta_u: float,
    pl_prior_variance: float,
) -> dict[str, float]:
    """The algorithm itself, run over whatever pool it's given — see
    `rank_active_pairwise` for the qualified/disqualified split wrapped around it."""
    item_ids = list(assessments)
    n = len(item_ids)

    if n <= 1:
        return {cid: 1.0 for cid in item_ids}

    rankings: list[list[str]] = [_bm25_sort(jp, assessments)]
    u = fit_utilities(rankings, item_ids, pl_prior_variance)

    min_iter = math.ceil(n / subset_size)
    actual_max_iter = max(max_iterations, 2 * min_iter)

    last_failed_signature: frozenset[str] | None = None
    for iteration in range(1, actual_max_iter + 1):
        variances = np.diag(laplace_covariance(u, rankings, item_ids, pl_prior_variance))
        subset_ids = _select_uncertain_subset(variances, item_ids, subset_size, last_failed_signature, iteration)

        ordering = rank_fn(jp, subset_ids, assessments)
        if ordering is None:
            last_failed_signature = frozenset(subset_ids)
            continue
        last_failed_signature = None

        rankings.append(ordering)
        u_new = fit_utilities(rankings, item_ids, pl_prior_variance)
        delta_u = float(np.linalg.norm(u_new - u) / n)
        u = u_new

        if delta_u < min_delta_u and iteration >= min_iter:
            logger.info("Active ranking for %s converged at iteration %d (delta_u=%.4f)", jp.id, iteration, delta_u)
            break
    else:
        logger.warning("Active ranking for %s did not converge after %d iterations", jp.id, actual_max_iter)

    return {cid: float(u[i]) for i, cid in enumerate(item_ids)}
