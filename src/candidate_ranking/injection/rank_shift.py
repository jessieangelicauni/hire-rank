from __future__ import annotations

from typing import Callable

from candidate_ranking.models import Assessment, JobDescription
from candidate_ranking.ranking.plackett_luce import fit_utilities
from candidate_ranking.ranking.tournament import rank_subset


def find_touched_subset_indices(history: list[dict], candidate_id: str, max_touched: int) -> list[int]:
    touched = [i for i, h in enumerate(history) if candidate_id in h["subset_candidate_ids"]]
    return touched[:max_touched]


def rank_of_utilities(utilities: dict[str, float], candidate_id: str) -> int:
    ordered = sorted(utilities.items(), key=lambda pair: pair[1], reverse=True)
    for position, (cid, _utility) in enumerate(ordered, start=1):
        if cid == candidate_id:
            return position
    raise ValueError(f"candidate_id {candidate_id!r} not found in utilities")


def compute_rank_shift(
    jd: JobDescription,
    pool_assessments: dict[str, Assessment],
    candidate_id: str,
    clean_rank: int,
    item_ids: list[str],
    combined_history: list[dict],
    modified_assessment: Assessment,
    build_ranking_chain: Callable,
    pl_prior_variance: float = 1.0,
    max_touched_subsets: int = 8,
) -> dict:
    jd_text = f"{jd.title}\n\n{jd.raw_text}"
    touched_indices = set(find_touched_subset_indices(combined_history, candidate_id, max_touched_subsets))
    untouched_rankings = [h["ranking"] for i, h in enumerate(combined_history) if i not in touched_indices]
    touched_subsets = [tuple(combined_history[i]["subset_candidate_ids"]) for i in sorted(touched_indices)]

    modified_pool = dict(pool_assessments)
    modified_pool[candidate_id] = modified_assessment

    reranked = [rank_subset(jd_text, subset, modified_pool, build_ranking_chain) for subset in touched_subsets]
    new_rankings = untouched_rankings + reranked

    utilities_array = fit_utilities(new_rankings, item_ids, pl_prior_variance)
    utilities = dict(zip(item_ids, utilities_array))
    modified_rank = rank_of_utilities(utilities, candidate_id)

    return {"clean_rank": clean_rank, "modified_rank": modified_rank, "rank_delta": clean_rank - modified_rank}
