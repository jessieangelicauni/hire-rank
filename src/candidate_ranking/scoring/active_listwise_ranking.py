from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, create_model

from candidate_ranking.models import Assessment, JobDescription
from candidate_ranking.ranking.plackett_luce import fit_utilities, laplace_covariance
from candidate_ranking.scoring.render import _extract_jd_summary, _render_evidence_for_ranking
from candidate_ranking.utils import GenerationError, invoke_and_validate

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

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


def _bm25_sort(jd: JobDescription, assessments: dict[str, Assessment]) -> list[str]:
    from rank_bm25 import BM25Okapi

    cids = list(assessments)
    corpus = []
    for cid in cids:
        a = assessments[cid]
        parts = [f"{ev.requirement} {ev.evidence}" for ev in a.must_have_evaluations]
        parts += [f"{ev.requirement} {ev.evidence}" for ev in a.nice_to_have_evaluations if ev.verdict in ("MET", "PARTIALLY_MET")]
        parts += a.strongest_achievements
        corpus.append(" ".join(parts).lower().split())
    scores = BM25Okapi(corpus).get_scores(jd.raw_text.lower().split())
    return [cid for cid, _ in sorted(zip(cids, scores), key=lambda x: -x[1])]


def _select_uncertain_subset(variances: np.ndarray, item_ids: list[str], subset_size: int) -> list[str]:
    return [item_ids[i] for i in np.argsort(variances)[::-1][:subset_size]]


def _rank_subset(
    jd: JobDescription,
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
                "jd_title": jd.title,
                "jd_summary": _extract_jd_summary(jd),
                "candidates_text": _render_evidence_for_ranking(subset_ids, assessments),
            },
            model, validate,
            log_context=f"Active ranking {jd.id} (size={size})",
        )
    except GenerationError as exc:
        logger.warning("Active ranking subset failed for %s: %s — skipping", jd.id, exc)
        return None

    return [token_to_id[t] for t in result.ranking]  # type: ignore[attr-defined]


def rank_active_listwise(
    jd: JobDescription,
    assessments: dict[str, Assessment],
    llm: "BaseChatModel",
    subset_size: int = 8,
    max_iterations: int = 40,
    min_delta_u: float = 0.01,
    pl_prior_variance: float = 1.0,
) -> dict[str, float]:
    """Active listwise ranking via variance-based subset selection + Plackett-Luce.

    Warm-starts from BM25 sort (0 LLM calls), then queries the LLM on the most
    uncertain candidate subsets until convergence or budget exhaustion.
    Higher utility = better fit.
    """
    item_ids = list(assessments)
    n = len(item_ids)

    if n <= 1:
        return {cid: 1.0 for cid in item_ids}

    rankings: list[list[str]] = [_bm25_sort(jd, assessments)]
    u = fit_utilities(rankings, item_ids, pl_prior_variance)

    if n <= subset_size:
        ordering = _rank_subset(jd, item_ids, assessments, llm)
        if ordering is not None:
            rankings.append(ordering)
            u = fit_utilities(rankings, item_ids, pl_prior_variance)
        return {cid: float(u[i]) for i, cid in enumerate(item_ids)}

    min_iter = math.ceil(n / subset_size)
    actual_max_iter = max(max_iterations, 2 * min_iter)

    for iteration in range(1, actual_max_iter + 1):
        variances = np.diag(laplace_covariance(u, rankings, item_ids, pl_prior_variance))
        subset_ids = _select_uncertain_subset(variances, item_ids, subset_size)

        ordering = _rank_subset(jd, subset_ids, assessments, llm)
        if ordering is None:
            continue

        rankings.append(ordering)
        u_new = fit_utilities(rankings, item_ids, pl_prior_variance)
        delta_u = float(np.linalg.norm(u_new - u) / n)
        u = u_new

        if delta_u < min_delta_u and iteration >= min_iter:
            logger.info("Active ranking for %s converged at iteration %d (delta_u=%.4f)", jd.id, iteration, delta_u)
            break
    else:
        logger.warning("Active ranking for %s did not converge after %d iterations", jd.id, actual_max_iter)

    return {cid: float(u[i]) for i, cid in enumerate(item_ids)}
