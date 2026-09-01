from __future__ import annotations

import numpy as np

from candidate_ranking.ranking.plackett_luce import index_map, ranking_hessian_contribution


def sample_candidate_subsets(
    candidate_ids: list[str],
    subset_size: int,
    num_samples: int,
    seen_subsets: set[frozenset[str]],
    rng: np.random.Generator,
) -> list[tuple[str, ...]]:
    size = min(subset_size, len(candidate_ids))
    sampled: list[tuple[str, ...]] = []
    sampled_keys: set[frozenset[str]] = set()
    max_attempts = num_samples * 20
    attempts = 0
    while len(sampled) < num_samples and attempts < max_attempts:
        attempts += 1
        choice = tuple(sorted(rng.choice(candidate_ids, size=size, replace=False).tolist()))
        key = frozenset(choice)
        if key in seen_subsets or key in sampled_keys:
            continue
        sampled_keys.add(key)
        sampled.append(choice)
    return sampled


def sample_pl_ranking(
    subset: tuple[str, ...], item_ids: list[str], utilities: np.ndarray, rng: np.random.Generator
) -> list[str]:
    id_to_index = index_map(item_ids)
    remaining = list(subset)
    ranking: list[str] = []
    while remaining:
        indices = [id_to_index[c] for c in remaining]
        u = utilities[indices]
        p = np.exp(u - np.max(u))
        p /= p.sum()
        chosen = rng.choice(len(remaining), p=p)
        ranking.append(remaining.pop(int(chosen)))
    return ranking


def _hypothetical_covariance(
    current_hessian: np.ndarray, draw: np.ndarray, item_ids: list[str], simulated_ranking: list[str]
) -> np.ndarray:
    id_to_index = index_map(item_ids)
    indices = [id_to_index[c] for c in simulated_ranking]
    hypothetical_hessian = current_hessian + ranking_hessian_contribution(draw, indices)
    return np.linalg.inv(hypothetical_hessian)


def score_subset_mc_kg(
    subset: tuple[str, ...],
    item_ids: list[str],
    utilities: np.ndarray,
    covariance: np.ndarray,
    num_mc_draws: int,
    rng: np.random.Generator,
) -> float:
    current_hessian = np.linalg.inv(covariance)
    current_mean_variance = float(np.mean(np.diag(covariance)))

    draws = rng.multivariate_normal(utilities, covariance, size=num_mc_draws)
    total_reduction = 0.0
    for draw in draws:
        simulated_ranking = sample_pl_ranking(subset, item_ids, draw, rng)
        try:
            hypothetical_covariance = _hypothetical_covariance(current_hessian, draw, item_ids, simulated_ranking)
        except np.linalg.LinAlgError:
            continue
        new_mean_variance = float(np.mean(np.diag(hypothetical_covariance)))
        total_reduction += current_mean_variance - new_mean_variance
    return total_reduction / num_mc_draws


def select_best_subset(
    candidate_ids: list[str],
    utilities: np.ndarray,
    covariance: np.ndarray,
    subset_size: int,
    num_subset_samples: int,
    num_mc_draws: int,
    seen_subsets: set[frozenset[str]],
    rng: np.random.Generator,
) -> tuple[str, ...] | None:
    candidates = sample_candidate_subsets(candidate_ids, subset_size, num_subset_samples, seen_subsets, rng)
    if not candidates:
        return None

    best_subset: tuple[str, ...] | None = None
    best_score = -np.inf
    for subset in candidates:
        score = score_subset_mc_kg(subset, candidate_ids, utilities, covariance, num_mc_draws, rng)
        if score > best_score:
            best_score = score
            best_subset = subset
    return best_subset
