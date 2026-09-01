
import numpy as np

from candidate_ranking.ranking.mc_kg import select_best_subset

CANDIDATE_IDS = [f"cv-{i}" for i in range(8)]


def _fresh_state(n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(n), np.eye(n)


def test_select_best_subset_returns_a_valid_unseen_subset():
    utilities, covariance = _fresh_state()
    rng = np.random.default_rng(0)

    subset = select_best_subset(
        CANDIDATE_IDS, utilities, covariance, subset_size=3, num_subset_samples=10,
        num_mc_draws=5, seen_subsets=set(), rng=rng,
    )

    assert subset is not None
    assert len(subset) == 3
    assert set(subset).issubset(set(CANDIDATE_IDS))


def test_select_best_subset_never_returns_an_already_seen_subset():
    utilities, covariance = _fresh_state()
    rng = np.random.default_rng(0)
    seen = {frozenset(["cv-0", "cv-1", "cv-2"])}

    for _ in range(5):
        subset = select_best_subset(
            CANDIDATE_IDS, utilities, covariance, subset_size=3, num_subset_samples=10,
            num_mc_draws=5, seen_subsets=seen, rng=rng,
        )
        assert frozenset(subset) not in seen
        seen.add(frozenset(subset))


def test_select_best_subset_returns_none_once_every_subset_is_seen():
    utilities, covariance = _fresh_state(n=3)
    rng = np.random.default_rng(0)
    seen = {frozenset(["cv-0", "cv-1", "cv-2"])}

    subset = select_best_subset(
        ["cv-0", "cv-1", "cv-2"], utilities, covariance, subset_size=3, num_subset_samples=10,
        num_mc_draws=5, seen_subsets=seen, rng=rng,
    )

    assert subset is None


def test_select_best_subset_is_deterministic_given_the_same_rng_state():
    utilities, covariance = _fresh_state()
    rng_one = np.random.default_rng(0)
    rng_two = np.random.default_rng(0)

    subset_one = select_best_subset(
        CANDIDATE_IDS, utilities, covariance, subset_size=3, num_subset_samples=10,
        num_mc_draws=5, seen_subsets=set(), rng=rng_one,
    )
    subset_two = select_best_subset(
        CANDIDATE_IDS, utilities, covariance, subset_size=3, num_subset_samples=10,
        num_mc_draws=5, seen_subsets=set(), rng=rng_two,
    )

    assert subset_one == subset_two
