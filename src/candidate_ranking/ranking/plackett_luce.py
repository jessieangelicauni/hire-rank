from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def index_map(item_ids: list[str]) -> dict[str, int]:
    return {item_id: i for i, item_id in enumerate(item_ids)}


def _negative_log_posterior(u: np.ndarray, ranking_indices: list[list[int]], prior_variance: float) -> float:
    nll = 0.0
    for indices in ranking_indices:
        for t in range(len(indices) - 1):
            remaining = indices[t:]
            nll += logsumexp(u[remaining]) - u[indices[t]]
    prior_term = float(np.sum(u**2)) / (2 * prior_variance)
    return nll + prior_term


def _negative_log_posterior_gradient(u: np.ndarray, ranking_indices: list[list[int]], prior_variance: float) -> np.ndarray:
    grad = u / prior_variance
    for indices in ranking_indices:
        for t in range(len(indices) - 1):
            remaining = indices[t:]
            remaining_u = u[remaining]
            weights = np.exp(remaining_u - np.max(remaining_u))
            weights /= weights.sum()
            for offset, idx in enumerate(remaining):
                grad[idx] += weights[offset]
            grad[indices[t]] -= 1.0
    return grad


def fit_utilities(rankings: list[list[str]], item_ids: list[str], prior_variance: float) -> np.ndarray:
    if not rankings:
        return np.zeros(len(item_ids))

    id_to_index = index_map(item_ids)
    ranking_indices = [[id_to_index[i] for i in ranking] for ranking in rankings]

    result = minimize(
        _negative_log_posterior,
        np.zeros(len(item_ids)),
        args=(ranking_indices, prior_variance),
        jac=_negative_log_posterior_gradient,
        method="L-BFGS-B",
    )
    return result.x


def ranking_hessian_contribution(utilities: np.ndarray, indices: list[int]) -> np.ndarray:
    n = len(utilities)
    h = np.zeros((n, n))
    for t in range(len(indices) - 1):
        remaining = indices[t:]
        remaining_u = utilities[remaining]
        p = np.exp(remaining_u - np.max(remaining_u))
        p /= p.sum()
        block = np.diag(p) - np.outer(p, p)
        for a, idx_a in enumerate(remaining):
            for b, idx_b in enumerate(remaining):
                h[idx_a, idx_b] += block[a, b]
    return h


def negative_log_posterior_hessian(utilities: np.ndarray, ranking_indices: list[list[int]], prior_variance: float) -> np.ndarray:
    n = len(utilities)
    hessian = np.eye(n) / prior_variance
    for indices in ranking_indices:
        hessian += ranking_hessian_contribution(utilities, indices)
    return hessian


def laplace_covariance(utilities: np.ndarray, rankings: list[list[str]], item_ids: list[str], prior_variance: float) -> np.ndarray:
    id_to_index = index_map(item_ids)
    ranking_indices = [[id_to_index[i] for i in ranking] for ranking in rankings]
    hessian = negative_log_posterior_hessian(utilities, ranking_indices, prior_variance)
    return np.linalg.inv(hessian)
