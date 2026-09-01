
import pytest

from candidate_ranking.evaluation.evaluation import build_ensemble_tournament_result, build_jd_evaluation
from candidate_ranking.models import TournamentIterationRecord, TournamentResult


def _iteration(iteration: int, subset: list[str], delta_u: float) -> TournamentIterationRecord:
    return TournamentIterationRecord(
        iteration=iteration, subset_candidate_ids=subset, ranking=subset, delta_u=delta_u
    )


def test_mean_abs_delta_u_is_normalized_by_sqrt_of_candidate_count():
    candidate_ids = ["cv-1", "cv-2", "cv-3", "cv-4"]
    repeat = TournamentResult(
        job_description_id="backend-engineer",
        repeat_index=0,
        candidate_ids=candidate_ids,
        final_utilities=dict.fromkeys(candidate_ids, 0.0),
        iteration_history=[
            _iteration(0, ["cv-1", "cv-2"], delta_u=2.0),
            _iteration(1, ["cv-2", "cv-3"], delta_u=4.0),
            _iteration(2, ["cv-3", "cv-4"], delta_u=6.0),
        ],
        status="ok",
    )

    evaluation = build_jd_evaluation("backend-engineer", repeats=[repeat], pl_prior_variance=1.0)

    assert evaluation.mean_abs_delta_u == pytest.approx(2.5)


def test_mean_abs_delta_u_scales_down_for_a_larger_candidate_pool():
    candidate_ids = [f"cv-{i}" for i in range(16)]
    repeat = TournamentResult(
        job_description_id="cloud-engineer",
        repeat_index=0,
        candidate_ids=candidate_ids,
        final_utilities=dict.fromkeys(candidate_ids, 0.0),
        iteration_history=[
            _iteration(0, ["cv-1", "cv-2"], delta_u=2.0),
            _iteration(1, ["cv-2", "cv-3"], delta_u=4.0),
            _iteration(2, ["cv-3", "cv-4"], delta_u=6.0),
        ],
        status="ok",
    )

    evaluation = build_jd_evaluation("cloud-engineer", repeats=[repeat], pl_prior_variance=1.0)

    assert evaluation.mean_abs_delta_u == pytest.approx(1.25)


def _repeat(repeat_index: int, candidate_ids: list[str], delta_us: list[float], status: str = "ok") -> TournamentResult:
    subsets = [candidate_ids[i % len(candidate_ids) : i % len(candidate_ids) + 2] for i in range(len(delta_us))]
    return TournamentResult(
        job_description_id="backend-engineer",
        repeat_index=repeat_index,
        candidate_ids=candidate_ids,
        final_utilities=dict.fromkeys(candidate_ids, 0.0),
        iteration_history=[
            _iteration(i, subset or candidate_ids[:2], delta_u=d) for i, (subset, d) in enumerate(zip(subsets, delta_us))
        ],
        status=status,
    )


def test_mean_kendall_tau_and_delta_u_average_across_all_ok_repeats():
    candidate_ids = ["cv-1", "cv-2", "cv-3", "cv-4"]
    repeats = [
        _repeat(0, candidate_ids, delta_us=[2.0, 4.0, 6.0]),
        _repeat(1, candidate_ids, delta_us=[2.0, 4.0, 10.0]),
    ]

    evaluation = build_jd_evaluation("backend-engineer", repeats=repeats, pl_prior_variance=1.0)

    assert evaluation.mean_abs_delta_u == pytest.approx(3.0)
    assert evaluation.abs_delta_u_std is not None
    assert evaluation.abs_delta_u_std > 0


def test_kendall_tau_and_delta_u_std_are_none_with_a_single_repeat():
    candidate_ids = ["cv-1", "cv-2", "cv-3", "cv-4"]
    repeats = [_repeat(0, candidate_ids, delta_us=[2.0, 4.0, 6.0])]

    evaluation = build_jd_evaluation("backend-engineer", repeats=repeats, pl_prior_variance=1.0)

    assert evaluation.kendall_tau_std is None
    assert evaluation.abs_delta_u_std is None


def test_failed_repeats_are_excluded_from_the_cross_repeat_average():
    candidate_ids = ["cv-1", "cv-2", "cv-3", "cv-4"]
    repeats = [
        _repeat(0, candidate_ids, delta_us=[2.0, 4.0, 6.0]),
        _repeat(1, candidate_ids, delta_us=[100.0, 200.0, 300.0], status="failed"),
    ]

    evaluation = build_jd_evaluation("backend-engineer", repeats=repeats, pl_prior_variance=1.0)

    assert evaluation.mean_abs_delta_u == pytest.approx(2.5)
    assert evaluation.abs_delta_u_std is None


def _tournament_result(
    repeat_index: int, utilities: dict[str, float], variances: dict[str, float] | None = None, status: str = "ok"
) -> TournamentResult:
    candidate_ids = list(utilities)
    return TournamentResult(
        job_description_id="backend-engineer",
        repeat_index=repeat_index,
        candidate_ids=candidate_ids,
        final_utilities=utilities,
        final_utility_variance=variances or {},
        iteration_history=[
            TournamentIterationRecord(iteration=0, subset_candidate_ids=candidate_ids, ranking=candidate_ids, delta_u=0.1)
        ],
        status=status,
    )


def test_ensemble_averages_utilities_across_ok_repeats():
    repeats = [
        _tournament_result(0, {"cv-1": 1.0, "cv-2": 0.0}),
        _tournament_result(1, {"cv-1": 2.0, "cv-2": 0.5}),
    ]

    ensemble = build_ensemble_tournament_result(repeats)

    assert ensemble.final_utilities == {"cv-1": pytest.approx(1.5), "cv-2": pytest.approx(0.25)}
    assert ensemble.status == "ok"
    assert ensemble.candidate_ids == ["cv-1", "cv-2"]


def test_ensemble_variance_combines_within_and_between_repeat_spread():
    repeats = [
        _tournament_result(0, {"cv-1": 1.0}, variances={"cv-1": 0.1}),
        _tournament_result(1, {"cv-1": 2.0}, variances={"cv-1": 0.3}),
    ]

    ensemble = build_ensemble_tournament_result(repeats)

    assert ensemble.final_utility_variance["cv-1"] == pytest.approx(0.7)


def test_ensemble_with_a_single_repeat_uses_only_its_own_within_repeat_variance():
    repeats = [_tournament_result(0, {"cv-1": 1.0}, variances={"cv-1": 0.1})]

    ensemble = build_ensemble_tournament_result(repeats)

    assert ensemble.final_utilities == {"cv-1": pytest.approx(1.0)}
    assert ensemble.final_utility_variance["cv-1"] == pytest.approx(0.1)


def test_ensemble_excludes_failed_repeats():
    repeats = [
        _tournament_result(0, {"cv-1": 1.0}),
        _tournament_result(1, {"cv-1": 999.0}, status="failed"),
    ]

    ensemble = build_ensemble_tournament_result(repeats)

    assert ensemble.final_utilities == {"cv-1": pytest.approx(1.0)}


def test_ensemble_raises_when_every_repeat_failed():
    repeats = [_tournament_result(0, {"cv-1": 1.0}, status="failed")]

    with pytest.raises(ValueError, match="no successful repeats"):
        build_ensemble_tournament_result(repeats)
