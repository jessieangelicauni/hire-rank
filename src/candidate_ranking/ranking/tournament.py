from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Callable, Literal, TypedDict

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, create_model

from candidate_ranking.generation import GenerationError, invoke_and_validate
from candidate_ranking.models import (
    Assessment,
    JobDescription,
    TournamentIterationRecord,
    TournamentResult,
)
from candidate_ranking.ranking.mc_kg import select_best_subset
from candidate_ranking.ranking.plackett_luce import fit_utilities, laplace_covariance

logger = logging.getLogger(__name__)

RANKING_PROMPT_VERSION = "must-have-evaluations-achievements-trajectory"

LISTWISE_RANKING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert hiring panel member ranking a small group of qualified candidates for a role.\n\n"
            "Task:\n"
            "- Read the job description and each candidate's qualification evidence.\n"
            "- Reason through each candidate in the `reasoning` field.\n"
            "- Compare candidates against each other on: relevance, evidence quality, trajectory, recency, nice-to-have coverage.\n"
            "- Order the candidates from strongest to weakest overall fit.\n"
            "- Put only the candidates' numbers, in that order, in the `ranking` field.\n\n"
            "Constraints:\n"
            "- Candidates are numbered starting at 1 in the list below.\n"
            "- `ranking` must contain every listed candidate number exactly once.\n"
            "- `ranking` must not contain any number that is not listed below.\n"
            "- Never substitute a candidate's name or any other text for their number in `ranking`.",
        ),
        ("human", "Job Description:\n{jd_text}\n\nCandidates:\n{candidates_text}{retry_feedback}"),
    ]
)


def _ranking_result_model(size: int) -> type[BaseModel]:
    return create_model(
        "_RankingResult",
        reasoning=(str, Field(min_length=1)),
        ranking=(list[int], Field(min_length=size, max_length=size)),
    )


class ListwiseRankingError(Exception):
    pass


def build_listwise_ranking_chain(llm: BaseChatModel) -> Callable[[type[BaseModel]], Runnable]:
    def _build(ranking_model: type[BaseModel]) -> Runnable:
        return LISTWISE_RANKING_PROMPT | llm.with_structured_output(ranking_model)

    return _build


def _render_candidates_text(subset: tuple[str, ...], assessments: dict[str, Assessment]) -> str:
    blocks = []
    for index, candidate_id in enumerate(subset, start=1):
        a = assessments[candidate_id]
        must_text = "\n".join(
            f"    [{ev.verdict}] {ev.requirement} ({ev.evidence_strength}, {ev.years_demonstrated}yr): {ev.evidence}"
            for ev in a.must_have_evaluations
        ) or "    (none)"
        met_nice = [ev for ev in a.nice_to_have_evaluations if ev.verdict in ("MET", "PARTIALLY_MET")]
        nice_text = ", ".join(f"{ev.requirement} [{ev.verdict}]" for ev in met_nice) or "(none)"
        ach_text = "\n".join(f"    - {ach}" for ach in a.strongest_achievements) or "    (none)"
        traj = a.career_trajectory
        traj_text = (
            f"{traj.direction}, avg {traj.average_tenure_months}mo/role"
            + (", multinational exp." if traj.has_multinational_experience else "")
        ) if traj else "unknown"
        blocks.append(
            f"{index}:\n"
            f"  Must-have evaluations:\n{must_text}\n"
            f"  Nice-to-have met: {nice_text}\n"
            f"  Achievements:\n{ach_text}\n"
            f"  Trajectory: {traj_text} | Relevant years: {a.total_relevant_years}"
        )
    return "\n\n".join(blocks)


def rank_subset(
    jd_text: str,
    subset: tuple[str, ...],
    assessments: dict[str, Assessment],
    build_chain: Callable[[type[BaseModel]], Runnable],
) -> list[str]:
    if len(subset) == 1:
        return list(subset)

    token_to_id = dict(enumerate(subset, start=1))
    expected_tokens = set(token_to_id)
    ranking_model = _ranking_result_model(len(subset))
    chain = build_chain(ranking_model)

    def validate(result: BaseModel) -> str | None:
        if set(result.ranking) != expected_tokens or len(result.ranking) != len(subset):
            return (
                f"ranking is not a permutation of the {len(subset)} listed candidate numbers: "
                f"expected {sorted(expected_tokens)} each exactly once, got {result.ranking}"
            )
        return None

    candidates_text = _render_candidates_text(subset, assessments)
    retry_feedback = ""
    last_error: GenerationError | None = None
    for attempt in range(2):
        try:
            result = invoke_and_validate(
                chain,
                {
                    "jd_text": jd_text,
                    "candidates_text": candidates_text,
                    "retry_feedback": retry_feedback,
                },
                ranking_model,
                validate,
                log_context=f"Listwise ranking (attempt {attempt + 1}/2)",
            )
            break
        except GenerationError as exc:
            last_error = exc
            retry_feedback = (
                "\n\nYour previous attempt was rejected: "
                f"{exc}\n"
                f"Respond with ONLY the {len(subset)} candidate numbers listed above, each used "
                "exactly once, ordered from strongest to weakest fit. Do not use candidate names "
                "or any other text -- numbers only."
            )
    else:
        raise ListwiseRankingError(str(last_error)) from last_error

    return [token_to_id[token] for token in result.ranking]


class TournamentCheckpointData(TypedDict):
    iteration: int
    candidate_ids: list[str]
    utilities: dict[str, float]
    covariance: list[list[float]]
    seen_subsets: list[list[str]]
    rng_state: dict
    history: list[dict]
    status: Literal["ok", "failed"]


def save_tournament_checkpoint(path: Path, data: TournamentCheckpointData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_tournament_checkpoint(path: Path) -> TournamentCheckpointData | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Discarding unreadable tournament checkpoint %s: %s", path, exc)
        return None


def _checkpoint_data(
    iteration: int,
    item_ids: list[str],
    utilities: np.ndarray,
    covariance: np.ndarray,
    seen_subsets: set[frozenset[str]],
    rng: np.random.Generator,
    history: list[TournamentIterationRecord],
) -> TournamentCheckpointData:
    return {
        "iteration": iteration,
        "candidate_ids": list(item_ids),
        "utilities": dict(zip(item_ids, utilities.tolist())),
        "covariance": covariance.tolist(),
        "seen_subsets": [sorted(s) for s in seen_subsets],
        "rng_state": rng.bit_generator.state,
        "history": [h.model_dump() for h in history],
        "status": "ok" if history else "failed",
    }


def resolve_tournament_iterations(
    n_candidates: int,
    tournament_subset_size: int,
    target_appearances_per_candidate: int,
    tournament_iterations_min: int,
    tournament_iterations_max: int,
) -> int:
    computed = math.ceil(target_appearances_per_candidate * n_candidates / tournament_subset_size)
    return max(tournament_iterations_min, min(computed, tournament_iterations_max))


def run_tournament_for_jd_repeat(
    jd: JobDescription,
    assessments: dict[str, Assessment],
    repeat_index: int,
    tournament_iterations: int,
    tournament_subset_size: int,
    num_subset_samples: int,
    num_mc_draws: int,
    pl_prior_variance: float,
    build_ranking_chain: Callable[[type[BaseModel]], Runnable],
    checkpoint_path: Path,
    rng_seed: int,
    target_appearances_per_candidate: int | None = None,
    tournament_iterations_min: int = 1,
) -> TournamentResult:
    item_ids = sorted(assessments.keys())
    jd_text = f"{jd.title}\n\n{jd.raw_text}"

    if target_appearances_per_candidate is not None:
        tournament_iterations = resolve_tournament_iterations(
            len(item_ids), tournament_subset_size, target_appearances_per_candidate,
            tournament_iterations_min, tournament_iterations,
        )

    checkpoint = load_tournament_checkpoint(checkpoint_path)
    if checkpoint is not None and checkpoint.get("candidate_ids") != item_ids:
        logger.warning(
            "Discarding stale tournament checkpoint %s for %s repeat %d: candidate set changed "
            "(checkpoint had %s, current run has %s)",
            checkpoint_path, jd.id, repeat_index, checkpoint.get("candidate_ids"), item_ids,
        )
        checkpoint = None

    if checkpoint is not None:
        iteration = checkpoint["iteration"]
        utilities = np.array([checkpoint["utilities"][cid] for cid in item_ids])
        covariance = np.array(checkpoint["covariance"])
        seen_subsets = {frozenset(s) for s in checkpoint["seen_subsets"]}
        rng = np.random.default_rng()
        rng.bit_generator.state = checkpoint["rng_state"]
        history = [TournamentIterationRecord.model_validate(h) for h in checkpoint["history"]]
    else:
        iteration = 0
        utilities = np.zeros(len(item_ids))
        covariance = np.eye(len(item_ids)) * pl_prior_variance
        seen_subsets = set()
        rng = np.random.default_rng(rng_seed)
        history = []

    rankings_so_far = [record.ranking for record in history]
    stop_reason: Literal["budget_exhausted", "no_unseen_subset"] = "budget_exhausted"

    while iteration < tournament_iterations:
        subset = select_best_subset(
            item_ids, utilities, covariance, tournament_subset_size, num_subset_samples, num_mc_draws,
            seen_subsets, rng,
        )
        if subset is None:
            stop_reason = "no_unseen_subset"
            logger.info(
                "Stopping %s repeat %d early at iteration %d: no unseen candidate subset left in a pool of "
                "%d candidate(s) (tournament_subset_size=%d)",
                jd.id, repeat_index, iteration, len(item_ids), tournament_subset_size,
            )
            break

        try:
            ranking = rank_subset(jd_text, subset, assessments, build_ranking_chain)
        except ListwiseRankingError as exc:
            logger.warning(
                "Listwise ranking failed for %s repeat %d iteration %d: %s", jd.id, repeat_index, iteration, exc
            )
            iteration += 1
            save_tournament_checkpoint(
                checkpoint_path, _checkpoint_data(iteration, item_ids, utilities, covariance, seen_subsets, rng, history)
            )
            continue

        seen_subsets.add(frozenset(subset))
        rankings_so_far.append(ranking)
        previous_utilities = utilities
        utilities = fit_utilities(rankings_so_far, item_ids, pl_prior_variance)
        try:
            covariance = laplace_covariance(utilities, rankings_so_far, item_ids, pl_prior_variance)
        except np.linalg.LinAlgError as exc:
            logger.warning(
                "Laplace covariance inversion failed for %s repeat %d iteration %d: %s; keeping previous covariance",
                jd.id, repeat_index, iteration, exc,
            )

        delta_u = float(np.linalg.norm(utilities - previous_utilities))
        history.append(
            TournamentIterationRecord(iteration=iteration, subset_candidate_ids=list(subset), ranking=ranking, delta_u=delta_u)
        )
        iteration += 1

        save_tournament_checkpoint(
            checkpoint_path, _checkpoint_data(iteration, item_ids, utilities, covariance, seen_subsets, rng, history)
        )

    status = "ok" if history else "failed"
    return TournamentResult(
        job_description_id=jd.id,
        repeat_index=repeat_index,
        candidate_ids=item_ids,
        final_utilities=dict(zip(item_ids, utilities.tolist())),
        final_utility_variance=dict(zip(item_ids, np.diag(covariance).tolist())),
        iteration_history=history,
        status=status,
        stop_reason=stop_reason,
    )
