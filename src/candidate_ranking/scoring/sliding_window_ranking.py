# src/candidate_ranking/scoring/sliding_window_ranking.py
from __future__ import annotations

import logging
import random
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, create_model

from candidate_ranking.utils import GenerationError, invoke_and_validate
from candidate_ranking.models import Assessment, JobDescription
from candidate_ranking.ranking.plackett_luce import fit_utilities
from candidate_ranking.scoring.render import _extract_jd_summary, _render_evidence_for_ranking

logger = logging.getLogger(__name__)

SLIDING_WINDOW_RANKING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior hiring panel ranking a group of pre-screened candidates.\n"
            "All candidates have already passed qualification screening.\n\n"
            "STEP 1 — For EACH candidate, write a brief evaluation:\n"
            "  key_strengths: what makes this candidate strong FOR THIS SPECIFIC ROLE\n"
            "  key_gaps: what is missing or weak for this role\n\n"
            "STEP 2 — Rank ALL candidates from strongest to weakest fit.\n\n"
            "RANKING CRITERIA (in priority order):\n"
            "1. Relevance — experience directly matching daily responsibilities\n"
            "2. Evidence quality — concrete achievements over vague responsibilities\n"
            "3. Career trajectory — ascending scope and seniority\n"
            "4. Recency — recent hands-on experience over older exposure\n"
            "5. Nice-to-have coverage — breadth of preferred skills as tiebreaker\n\n"
            "RULES:\n"
            "- Candidates are numbered 1..N. `ranking` must contain every number exactly once.\n"
            "- `candidate_evaluations` must contain one entry per candidate with the same id numbers.\n"
            "- Rank ALL candidates. No ties.\n"
            "- Judge only on information provided.",
        ),
        (
            "human",
            "ROLE: {jd_title}\n\nJOB SUMMARY: {jd_summary}\n\nCANDIDATES:\n{candidates_text}",
        ),
    ]
)


class _CandidateEvaluation(BaseModel):
    id: int = Field(ge=1)
    key_strengths: str = Field(min_length=1)
    key_gaps: str = Field(min_length=1)


def _sliding_window_model(size: int) -> type[BaseModel]:
    return create_model(
        "_SlidingWindowResult",
        candidate_evaluations=(
            list[_CandidateEvaluation],
            Field(min_length=size, max_length=size),
        ),
        ranking=(list[int], Field(min_length=size, max_length=size)),
    )


def build_sliding_window_chain_factory(
    llm: BaseChatModel,
) -> Callable[[type[BaseModel]], Runnable]:
    def _build(model: type[BaseModel]) -> Runnable:
        return SLIDING_WINDOW_RANKING_PROMPT | llm.with_structured_output(model)
    return _build


def _candidate_bm25_text(assessment: Assessment) -> str:
    parts = [
        f"{ev.requirement} {ev.evidence}"
        for ev in assessment.must_have_evaluations
    ]
    parts += [
        f"{ev.requirement} {ev.evidence}"
        for ev in assessment.nice_to_have_evaluations
        if ev.verdict in ("MET", "PARTIALLY_MET")
    ]
    parts += assessment.strongest_achievements
    return " ".join(parts)


def _initial_sort(jd: JobDescription, assessments: dict[str, Assessment]) -> list[str]:
    from rank_bm25 import BM25Okapi

    cids = list(assessments)
    corpus = [_candidate_bm25_text(assessments[cid]).lower().split() for cid in cids]
    query = jd.raw_text.lower().split()
    scores = BM25Okapi(corpus).get_scores(query)
    return [cid for cid, _ in sorted(zip(cids, scores), key=lambda x: -x[1])]


def _rank_window(
    jd: JobDescription,
    window_ids: list[str],
    assessments: dict[str, Assessment],
    build_chain: Callable[[type[BaseModel]], Runnable],
) -> list[str]:
    size = len(window_ids)
    token_to_id = {i + 1: cid for i, cid in enumerate(window_ids)}
    expected_tokens = set(token_to_id)
    model = _sliding_window_model(size)
    chain = build_chain(model)

    def validate(result: BaseModel) -> str | None:
        ranking = result.ranking  # type: ignore[attr-defined]
        if set(ranking) != expected_tokens or len(ranking) != size:
            return f"ranking must be a permutation of {sorted(expected_tokens)}, got {ranking}"
        return None

    try:
        result = invoke_and_validate(
            chain,
            {
                "jd_title": jd.title,
                "jd_summary": _extract_jd_summary(jd),
                "candidates_text": _render_evidence_for_ranking(window_ids, assessments),
            },
            model,
            validate,
            log_context=f"Sliding window for {jd.id} (size={size})",
        )
    except GenerationError as exc:
        logger.warning(
            "Sliding window failed for %s: %s — keeping original order", jd.id, exc
        )
        return list(window_ids)

    return [token_to_id[token] for token in result.ranking]  # type: ignore[attr-defined]


def rank_sliding_window(
    jd: JobDescription,
    assessments: dict[str, Assessment],
    build_chain: Callable[[type[BaseModel]], Runnable],
    window_size: int = 15,
    step: int = 7,
    rng_seed: int | None = None,
) -> dict[str, float]:
    """2-pass sliding window LLM ranking. Returns utility per candidate_id (higher = better)."""
    ordered = _initial_sort(jd, assessments)
    if rng_seed is not None:
        random.Random(rng_seed).shuffle(ordered)
    n = len(ordered)
    if n <= 1:
        return {cid: 1.0 for cid in ordered}

    effective_window = min(window_size, n)
    forward_starts = list(range(0, n - effective_window + 1, step)) or [0]

    for pass_starts in (forward_starts, list(reversed(forward_starts))):
        for start in pass_starts:
            end = min(start + effective_window, n)
            ordered[start:end] = _rank_window(jd, ordered[start:end], assessments, build_chain)

    return {cid: float(n - i) for i, cid in enumerate(ordered)}


def rank_sliding_window_ensemble(
    jd: JobDescription,
    assessments: dict[str, Assessment],
    build_chain: Callable[[type[BaseModel]], Runnable],
    window_size: int = 15,
    step: int = 7,
    seeds: tuple[int, ...] = (42,),
    pl_prior_variance: float = 1.0,
) -> dict[str, float]:
    """K-run ensemble with Plackett-Luce aggregation. Higher utility = better fit."""
    item_ids = list(assessments)
    if not seeds:
        raise ValueError("seeds must be non-empty")
    if pl_prior_variance <= 0.0:
        raise ValueError(f"pl_prior_variance must be positive, got {pl_prior_variance}")
    if len(item_ids) <= 1:
        return {cid: 1.0 for cid in item_ids}

    rankings: list[list[str]] = []
    for seed in seeds:
        utilities = rank_sliding_window(jd, assessments, build_chain, window_size, step, rng_seed=seed)
        ordering = sorted(item_ids, key=utilities.__getitem__, reverse=True)
        rankings.append(ordering)

    pl_utilities = fit_utilities(rankings, item_ids, pl_prior_variance)
    return {cid: float(pl_utilities[i]) for i, cid in enumerate(item_ids)}
