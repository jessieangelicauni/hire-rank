from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, create_model

from candidate_ranking.models import Assessment, JobProfile
from candidate_ranking.ranking.plackett_luce import fit_utilities, laplace_covariance
from candidate_ranking.scoring.criterion_score import criterion_score
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

JEV_PROBABILITY_NOISE_FLOOR = 0.05


@dataclass
class RankingDiagnostics:
    position_a_wins: int = 0
    position_b_wins: int = 0
    convergence_iteration: int | None = None
    tie_clusters: int = 0
    ensemble_disagreements: int = 0
    cluster_sizes: list[int] = field(default_factory=list)
    cluster_iterations: list[int | None] = field(default_factory=list)


ACTIVE_RANKING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior hiring panel ranking candidates for a specific role.\n\n"
        "Rank ALL candidates from strongest to weakest fit for this role.\n\n"
        "RANKING CRITERIA (in priority order):\n"
        "1. Relevance — experience directly matching daily responsibilities\n"
        "2. Evidence quality — concrete achievements over vague responsibilities\n"
        "3. Requirement match — education/seniority/other requirements met (see 'Other requirements' below)\n"
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
        parts = [f"{ev.requirement} {ev.evidence}" for ev in a.evaluations if ev.evidence_strength != "none"]
        corpus.append(" ".join(parts).lower().split())

    if not any(corpus):
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
                "jd_title": jp.id,
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


def _ask_jev_pairwise(
    jp: JobProfile,
    pair_ids: list[str],
    assessments: dict[str, Assessment],
    jev_client: "JevClient",
    diagnostics: RankingDiagnostics | None = None,
) -> tuple[str, str, float] | None:
    from candidate_ranking.scoring.jev_client import JevClientError, JevQuestion
    assert len(pair_ids) == 2, "pairwise comparator requires exactly 2 candidates"

    digest = hashlib.md5(f"{pair_ids[0]}||{pair_ids[1]}||{jp.id}".encode()).hexdigest()
    flip = int(digest, 16) % 2 == 1
    a_id, b_id = (pair_ids[1], pair_ids[0]) if flip else (pair_ids[0], pair_ids[1])

    state = (
        f"ROLE: {jp.id}\n\n"
        f"CANDIDATE A:\n{_render_candidate_evidence(a_id, assessments)}\n\n"
        f"CANDIDATE B:\n{_render_candidate_evidence(b_id, assessments)}"
    )
    question = JevQuestion(
        key="stronger_fit",
        kind="choice",
        instructions=(
            "Based on the evidence summaries above, which candidate is the stronger fit for this role?\n"
            "Consider: relevance of experience, evidence quality, requirement match (education/seniority/"
            "other requirements met), recency, and nice-to-have coverage as a tiebreaker. Do not consider "
            "the candidate labels (A/B) as signals — evaluate purely on the evidence. The order evidence "
            "items are listed in does not indicate importance or priority — weigh each requirement "
            "independently of its position in the list."
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
    confidence = answers[0].confidence
    if diagnostics is not None:
        if winner == "A":
            diagnostics.position_a_wins += 1
        else:
            diagnostics.position_b_wins += 1
    winner_id = a_id if winner == "A" else b_id
    loser_id = b_id if winner == "A" else a_id
    return winner_id, loser_id, confidence


def _rank_pair_jev(
    jp: JobProfile,
    pair_ids: list[str],
    assessments: dict[str, Assessment],
    jev_client: "JevClient",
    diagnostics: RankingDiagnostics | None = None,
) -> list[str] | None:
    result = _ask_jev_pairwise(jp, pair_ids, assessments, jev_client, diagnostics)
    if result is None:
        return None
    winner_id, loser_id, _confidence = result
    return [winner_id, loser_id]


def _rank_pair_jev_ensembled(
    jp: JobProfile,
    pair_ids: list[str],
    assessments: dict[str, Assessment],
    jev_client: "JevClient",
    repeats: int,
    diagnostics: RankingDiagnostics | None = None,
) -> list[str] | None:
    confidence_by_id: dict[str, float] = {pair_ids[0]: 0.0, pair_ids[1]: 0.0}
    winners: list[str] = []
    for _ in range(repeats):
        result = _ask_jev_pairwise(jp, pair_ids, assessments, jev_client, diagnostics)
        if result is None:
            continue
        winner_id, _loser_id, confidence = result
        confidence_by_id[winner_id] += confidence
        winners.append(winner_id)

    if not winners:
        return None
    if diagnostics is not None and len(set(winners)) > 1:
        diagnostics.ensemble_disagreements += 1

    winner_id = max(confidence_by_id, key=confidence_by_id.__getitem__)
    loser_id = pair_ids[1] if winner_id == pair_ids[0] else pair_ids[0]
    return [winner_id, loser_id]


def _tie_clusters(scores: dict[str, float], epsilon: float) -> list[list[str]]:
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)
    if not ordered:
        return []
    clusters: list[list[str]] = [[ordered[0]]]
    for cid in ordered[1:]:
        anchor_id = clusters[-1][0]
        if scores[anchor_id] - scores[cid] <= epsilon:
            clusters[-1].append(cid)
        else:
            clusters.append([cid])
    return clusters


def _rank_tiered_jev(
    jp: JobProfile,
    assessments: dict[str, Assessment],
    jev_client: "JevClient",
    subset_size: int,
    min_delta_u: float,
    pl_prior_variance: float,
    diagnostics: RankingDiagnostics | None,
    tie_epsilon: float | None,
    ensemble_repeats: int,
) -> dict[str, float]:
    item_ids = list(assessments)
    if len(item_ids) <= 1:
        if diagnostics is not None:
            diagnostics.convergence_iteration = 0
        return {cid: 1.0 for cid in item_ids}

    scores = {cid: criterion_score(assessments[cid]) for cid in item_ids}
    if tie_epsilon is None:
        tie_epsilon = max(1.0 / max(len(jp.evaluation_criteria), 1), JEV_PROBABILITY_NOISE_FLOOR)

    clusters = _tie_clusters(scores, tie_epsilon)
    if diagnostics is not None:
        diagnostics.tie_clusters = len(clusters)

    adjustment: dict[str, float] = {}
    for cluster in clusters:
        if len(cluster) == 1:
            adjustment[cluster[0]] = 0.0
            continue

        cluster_assessments = {cid: assessments[cid] for cid in cluster}

        def rank_fn(j: JobProfile, s: list[str], a: dict[str, Assessment]) -> list[str] | None:
            return _rank_pair_jev_ensembled(j, s, a, jev_client, ensemble_repeats, diagnostics)

        cluster_diagnostics = RankingDiagnostics() if diagnostics is not None else None
        cluster_utilities = _rank_active_pairwise_core(
            jp, cluster_assessments, rank_fn, subset_size,
            min_delta_u, pl_prior_variance, cluster_diagnostics,
        )
        if diagnostics is not None:
            diagnostics.cluster_sizes.append(len(cluster))
            diagnostics.cluster_iterations.append(cluster_diagnostics.convergence_iteration)
        values = list(cluster_utilities.values())
        lo, hi = min(values), max(values)
        spread = hi - lo
        for cid, u in cluster_utilities.items():
            normalized = 0.5 if spread == 0 else (u - lo) / spread
            adjustment[cid] = (normalized - 0.5) * (tie_epsilon / 2)

    return {cid: scores[cid] + adjustment[cid] for cid in item_ids}


def rank_active_pairwise(
    jp: JobProfile,
    assessments: dict[str, Assessment],
    llm: "BaseChatModel | None" = None,
    jev_client: "JevClient | None" = None,
    subset_size: int = 8,
    min_delta_u: float = 0.01,
    pl_prior_variance: float = 1.0,
    diagnostics: RankingDiagnostics | None = None,
    tie_epsilon: float | None = None,
    ensemble_repeats: int = 1,
) -> dict[str, float]:
    if jev_client is not None:
        return _rank_tiered_jev(
            jp, assessments, jev_client, subset_size, min_delta_u,
            pl_prior_variance, diagnostics, tie_epsilon, ensemble_repeats,
        )
    if llm is not None:
        def rank_fn(j, s, a): return _rank_subset(j, s, a, llm)
        return _rank_active_pairwise_core(
            jp, assessments, rank_fn, subset_size, min_delta_u, pl_prior_variance, diagnostics
        )
    raise ValueError("Either llm or jev_client must be provided to rank_active_pairwise")


def _rank_active_pairwise_core(
    jp: JobProfile,
    assessments: dict[str, Assessment],
    rank_fn: "Callable[[JobProfile, list[str], dict[str, Assessment]], list[str] | None]",
    subset_size: int,
    min_delta_u: float,
    pl_prior_variance: float,
    diagnostics: RankingDiagnostics | None = None,
) -> dict[str, float]:
    item_ids = list(assessments)
    n = len(item_ids)

    if n <= 1:
        if diagnostics is not None:
            diagnostics.convergence_iteration = 0
        return {cid: 1.0 for cid in item_ids}

    rankings: list[list[str]] = [_bm25_sort(jp, assessments)]
    u = fit_utilities(rankings, item_ids, pl_prior_variance)

    min_iter = math.ceil(n / subset_size)
    actual_max_iter = 2 * min_iter

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
            if diagnostics is not None:
                diagnostics.convergence_iteration = iteration
            break
    else:
        logger.warning("Active ranking for %s did not converge after %d iterations", jp.id, actual_max_iter)

    return {cid: float(u[i]) for i, cid in enumerate(item_ids)}
