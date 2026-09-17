from __future__ import annotations

from typing import Callable

from scipy.stats import wilcoxon

from candidate_ranking.models import Assessment, JobDescription
from candidate_ranking.ranking.plackett_luce import fit_utilities
from candidate_ranking.ranking.tournament import rank_subset

# --- Rank-shift measurement via selective subset recomputation ------------

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


# --- Significance analysis (Wilcoxon + Holm-Bonferroni) --------------------

def rank_biserial(first_deltas: list[float], second_deltas: list[float]) -> float:
    n_first_higher = sum(1 for first, second in zip(first_deltas, second_deltas) if first > second)
    n_second_higher = sum(1 for first, second in zip(first_deltas, second_deltas) if first < second)
    n = len(first_deltas)
    return (n_first_higher - n_second_higher) / n if n else 0.0


def holm_correct(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction, adjusted p-values returned in
    the same order as the input."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted_sorted = []
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (m - rank) * p_values[i])
        adjusted_sorted.append(min(running_max, 1.0))
    result = [0.0] * m
    for rank, i in enumerate(order):
        result[i] = adjusted_sorted[rank]
    return result


def wilcoxon_result(label: str, first_deltas: list[float], second_deltas: list[float]) -> dict | None:
    n_pairs = len(first_deltas)
    if n_pairs >= 1 and any(first != second for first, second in zip(first_deltas, second_deltas)):
        stat, p_value = wilcoxon(first_deltas, second_deltas)
        r = rank_biserial(first_deltas, second_deltas)
        return {
            "label": label, "n_pairs": n_pairs, "statistic": float(stat),
            "p_value": float(p_value), "rank_biserial_r": r,
        }
    return None


def _deltas_by_pair_key(results: list[dict], condition: str) -> dict[tuple, float]:
    return {(r["jd_id"], r["candidate_id"]): r["rank_delta"] for r in results if r["condition"] == condition}


def paired_deltas_vs_control(
    results: list[dict], attack_condition: str, control_condition: str = "control_no_injection",
) -> tuple[list[float], list[float]]:
    control_by_pair = _deltas_by_pair_key(results, control_condition)
    attack_deltas, control_deltas = [], []
    for r in results:
        if r["condition"] != attack_condition:
            continue
        key = (r["jd_id"], r["candidate_id"])
        if key in control_by_pair:
            attack_deltas.append(r["rank_delta"])
            control_deltas.append(control_by_pair[key])
    return attack_deltas, control_deltas


def paired_deltas_by_condition(
    results: list[dict], condition_a: str, condition_b: str,
) -> tuple[list[float], list[float]]:
    deltas_by_condition_at_key: dict[tuple, dict[str, float]] = {}
    for r in results:
        if r["condition"] in (condition_a, condition_b):
            key = (r["jd_id"], r["candidate_id"], r["variant_name"])
            deltas_by_condition_at_key.setdefault(key, {})[r["condition"]] = r["rank_delta"]
    deltas_a, deltas_b = [], []
    for conditions_seen in deltas_by_condition_at_key.values():
        if condition_a in conditions_seen and condition_b in conditions_seen:
            deltas_a.append(conditions_seen[condition_a])
            deltas_b.append(conditions_seen[condition_b])
    return deltas_a, deltas_b
