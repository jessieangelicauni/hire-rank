from __future__ import annotations

import json
import logging
import math
import re
import statistics
from pathlib import Path

from pydantic import ValidationError
from scipy.stats import kendalltau

from candidate_ranking.config import RunConfig
from candidate_ranking.models import (
    Assessment,
    ConvergencePoint,
    JDEvaluation,
    TournamentIterationRecord,
    TournamentResult,
)
from candidate_ranking.ranking.plackett_luce import fit_utilities
from candidate_ranking.ranking.tournament import load_tournament_checkpoint

logger = logging.getLogger(__name__)

_REPEAT_DIR_RE = re.compile(r"repeat_(\d+)")


def mean_or_none(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def stdev_or_none(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def compute_convergence_trace(
    iteration_history: list[TournamentIterationRecord], pl_prior_variance: float
) -> list[dict]:
    if len(iteration_history) < 2:
        return []

    touched_sets: list[list[str]] = []
    seen: set[str] = set()
    for record in iteration_history:
        seen.update(record.subset_candidate_ids)
        touched_sets.append(sorted(seen))

    utilities_at = []
    for i in range(len(iteration_history)):
        rankings_so_far = [r.ranking for r in iteration_history[: i + 1]]
        utilities_at.append(fit_utilities(rankings_so_far, touched_sets[i], pl_prior_variance))

    trace = []
    for i in range(1, len(iteration_history)):
        prev_ids, prev_u = touched_sets[i - 1], utilities_at[i - 1]
        curr_ids, curr_u = touched_sets[i], utilities_at[i]
        common = [cid for cid in prev_ids if cid in curr_ids]

        if len(common) < 2:
            tau = None
        else:
            prev_index = {cid: idx for idx, cid in enumerate(prev_ids)}
            curr_index = {cid: idx for idx, cid in enumerate(curr_ids)}
            prev_vals = [float(prev_u[prev_index[cid]]) for cid in common]
            curr_vals = [float(curr_u[curr_index[cid]]) for cid in common]
            result = kendalltau(prev_vals, curr_vals)
            tau = None if math.isnan(result.statistic) else float(result.statistic)

        trace.append(
            {"iteration": iteration_history[i].iteration, "kendall_tau": tau, "delta_u": iteration_history[i].delta_u}
        )
    return trace


K_FRACTIONS = {"10%": 0.10, "15%": 0.15, "20%": 0.20, "25%": 0.25}


def _repeat_convergence_means(
    repeat: TournamentResult, pl_prior_variance: float
) -> tuple[float | None, float | None]:
    trace = compute_convergence_trace(repeat.iteration_history, pl_prior_variance)
    taus = [point["kendall_tau"] for point in trace if point["kendall_tau"] is not None]
    n_candidates = len(repeat.candidate_ids)
    delta_us = [abs(point["delta_u"]) / math.sqrt(n_candidates) for point in trace] if n_candidates > 0 else []
    return mean_or_none(taus), mean_or_none(delta_us)


def build_ensemble_tournament_result(repeats: list[TournamentResult]) -> TournamentResult:
    ok_repeats = [r for r in repeats if r.status == "ok"]
    if not ok_repeats:
        raise ValueError("cannot build an ensemble result: no successful repeats")

    candidate_ids = ok_repeats[0].candidate_ids
    final_utilities: dict[str, float] = {}
    final_utility_variance: dict[str, float] = {}
    for cid in candidate_ids:
        point_estimates = [r.final_utilities[cid] for r in ok_repeats]
        final_utilities[cid] = statistics.mean(point_estimates)
        within_repeat_variance = mean_or_none([r.final_utility_variance.get(cid) for r in ok_repeats])
        between_repeat_variance = statistics.variance(point_estimates) if len(point_estimates) >= 2 else 0.0
        final_utility_variance[cid] = (within_repeat_variance or 0.0) + between_repeat_variance

    merged_history = [record for r in ok_repeats for record in r.iteration_history]

    return TournamentResult(
        job_description_id=ok_repeats[0].job_description_id,
        repeat_index=0,
        candidate_ids=candidate_ids,
        final_utilities=final_utilities,
        final_utility_variance=final_utility_variance,
        iteration_history=merged_history,
        status="ok",
    )


def build_jd_evaluation(
    jd_id: str,
    repeats: list[TournamentResult],
    pl_prior_variance: float,
) -> JDEvaluation:
    ok_repeats = [r for r in repeats if r.status == "ok"]
    primary = next((r for r in ok_repeats if r.repeat_index == 0), ok_repeats[0] if ok_repeats else None)
    trace = compute_convergence_trace(primary.iteration_history, pl_prior_variance) if primary is not None else []

    per_repeat_taus: list[float] = []
    per_repeat_delta_us: list[float] = []
    for repeat in ok_repeats:
        tau_mean, delta_u_mean = _repeat_convergence_means(repeat, pl_prior_variance)
        if tau_mean is not None:
            per_repeat_taus.append(tau_mean)
        if delta_u_mean is not None:
            per_repeat_delta_us.append(delta_u_mean)

    return JDEvaluation(
        job_description_id=jd_id,
        convergence_trace=[ConvergencePoint(**point) for point in trace],
        mean_kendall_tau=mean_or_none(per_repeat_taus),
        mean_abs_delta_u=mean_or_none(per_repeat_delta_us),
        kendall_tau_std=stdev_or_none(per_repeat_taus),
        abs_delta_u_std=stdev_or_none(per_repeat_delta_us),
    )


def write_jd_repeats(run_dir: Path, jd_id: str, results: list[TournamentResult]) -> None:
    jd_dir = run_dir / jd_id
    jd_dir.mkdir(parents=True, exist_ok=True)
    payload = [r.model_dump() for r in results]
    (jd_dir / "repeats.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_jd_repeats_from_tournament_checkpoints(run_dir: Path, jd_id: str) -> list[TournamentResult]:
    tournament_dir = run_dir / jd_id / "tournament"
    if not tournament_dir.exists():
        return []

    results = []
    for repeat_dir in sorted(tournament_dir.glob("repeat_*")):
        match = _REPEAT_DIR_RE.fullmatch(repeat_dir.name)
        if match is None:
            continue
        repeat_index = int(match.group(1))
        checkpoint = load_tournament_checkpoint(repeat_dir / "state.json")
        if checkpoint is None:
            continue
        try:
            candidate_ids = checkpoint["candidate_ids"]
            covariance = checkpoint["covariance"]
            results.append(
                TournamentResult(
                    job_description_id=jd_id,
                    repeat_index=repeat_index,
                    candidate_ids=candidate_ids,
                    final_utilities=checkpoint["utilities"],
                    final_utility_variance=dict(zip(candidate_ids, [covariance[i][i] for i in range(len(candidate_ids))])),
                    iteration_history=checkpoint["history"],
                    status=checkpoint["status"],
                )
            )
        except ValidationError as exc:
            logger.warning(
                "Skipping invalid tournament result reconstructed from checkpoint %s: %s",
                repeat_dir / "state.json", exc,
            )

    if results:
        logger.info(
            "No repeats.json found for %s/%s; reconstructed %d tournament result(s) from "
            "tournament checkpoint files under %s",
            jd_id, run_dir, len(results), tournament_dir,
        )
    return results


def load_jd_repeats(run_dir: Path, jd_id: str) -> list[TournamentResult]:
    path = run_dir / jd_id / "repeats.json"
    if not path.exists():
        return _load_jd_repeats_from_tournament_checkpoints(run_dir, jd_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Discarding unreadable repeats file %s: %s", path, exc)
        return []

    results = []
    for entry in data:
        try:
            results.append(TournamentResult.model_validate(entry))
        except ValidationError as exc:
            logger.warning("Skipping invalid tournament result in %s: %s", path, exc)
    return results


def load_assessment_results(run_dir: Path, jd_id: str) -> list[Assessment]:
    path = run_dir / jd_id / "assessments.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Skipping unreadable assessments file %s: %s", path, exc)
        return []

    results = []
    for candidate_id, entry in sorted(data.items()):
        try:
            results.append(Assessment.model_validate(entry))
        except ValidationError as exc:
            logger.warning("Skipping invalid assessment for %s/%s in %s: %s", jd_id, candidate_id, path, exc)
    return results


def load_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_run(cfg: RunConfig, run_id: str) -> list[JDEvaluation]:
    run_dir = cfg.runs_dir / run_id
    manifest = load_manifest(run_dir)

    evaluations = []
    for jd_id in manifest.get("jd_ids", []):
        repeats = load_jd_repeats(run_dir, jd_id)
        if not repeats:
            logger.warning(
                "No tournament repeats found for %s/%s; excluding from evaluation", run_id, jd_id
            )
            continue
        evaluations.append(build_jd_evaluation(jd_id, repeats, cfg.pl_prior_variance))
    return evaluations
