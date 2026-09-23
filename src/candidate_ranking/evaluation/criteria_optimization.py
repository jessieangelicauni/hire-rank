from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from candidate_ranking.models import Candidate, JobDescription
from candidate_ranking.scoring.assessment import _build_state
from candidate_ranking.scoring.jev_client import JevClient, JevQuestion

REQUIREMENT_LEVEL_COUNT = 5
_REQUIREMENT_KEY_PREFIX = "requirement::"


class RequirementTriple(BaseModel):
    job_description_id: str
    candidate_id: str
    requirement: str


def triple_key(triple: RequirementTriple) -> str:
    return f"{triple.job_description_id}||{triple.candidate_id}||{triple.requirement}"


def validate_criteria_shape(criteria: list[str]) -> None:
    if len(criteria) != REQUIREMENT_LEVEL_COUNT:
        raise ValueError(f"criteria must have exactly {REQUIREMENT_LEVEL_COUNT} levels, got {len(criteria)}")
    for index, level in enumerate(criteria):
        if not level.strip():
            raise ValueError(f"criteria level {index} is empty")


def load_requirement_triples(
    assessments_by_jd: dict[str, dict], requirement_prefix: str = _REQUIREMENT_KEY_PREFIX
) -> list[RequirementTriple]:
    triples: list[RequirementTriple] = []
    for jd_id, candidates in assessments_by_jd.items():
        for candidate_id, assessment in candidates.items():
            for key in assessment.get("confidence", {}):
                if key.startswith(requirement_prefix):
                    triples.append(
                        RequirementTriple(
                            job_description_id=jd_id,
                            candidate_id=candidate_id,
                            requirement=key[len(requirement_prefix):],
                        )
                    )
    return triples


class CriteriaEvalRecord(BaseModel):
    triple: RequirementTriple
    score: float
    confidence: float


def build_requirement_question(requirement: str, criteria: list[str]) -> JevQuestion:
    validate_criteria_shape(criteria)
    return JevQuestion(
        key=f"{_REQUIREMENT_KEY_PREFIX}{requirement}",
        kind="score",
        instructions=f"How well does the candidate's CV support the requirement '{requirement}'?",
        criteria=criteria,
    )


def evaluate_criteria(
    criteria: list[str],
    triples: list[RequirementTriple],
    jds_by_id: dict[str, JobDescription],
    candidates_by_id: dict[str, Candidate],
    jev_client: JevClient,
) -> list[CriteriaEvalRecord]:
    records: list[CriteriaEvalRecord] = []
    for i, triple in enumerate(triples, start=1):
        jd = jds_by_id[triple.job_description_id]
        candidate = candidates_by_id[triple.candidate_id]
        state = _build_state(jd, candidate)
        question = build_requirement_question(triple.requirement, criteria)
        answers = jev_client.evaluate(state, [question])
        answer = answers[0]
        records.append(CriteriaEvalRecord(triple=triple, score=float(answer.value), confidence=answer.confidence))
        if i % 25 == 0 or i == len(triples):
            print(f"evaluate_criteria: {i}/{len(triples)} triple(s) done")
    return records


def compute_metric(records: list[CriteriaEvalRecord], proxy_labels: dict[str, int]) -> float:
    """Anti-Goodhart metric that credits Jev confidence only when Jev's picked level
    agrees with an independent Claude proxy label within 1 level. A confidently-wrong
    answer must NOT be rewarded."""
    if not records:
        return 0.0
    credited_confidences = []
    for record in records:
        label = proxy_labels[triple_key(record.triple)]
        agrees = abs(record.score - label) <= 1
        credited_confidences.append(record.confidence if agrees else 0.0)
    return statistics.mean(credited_confidences)


def select_hard_triples(records: list[CriteriaEvalRecord], k: int) -> list[RequirementTriple]:
    """Confidence-based hard-case mining: the K lowest-confidence triples under the
    current best criteria. A standard, well-established technique -- not an adversarial
    agent, and not claimed as novel on its own (see the design spec's Motivation)."""
    ordered = sorted(records, key=lambda record: record.confidence)
    return [record.triple for record in ordered[:k]]


def load_assessments_by_jd(run_dir: Path, jd_ids: list[str]) -> dict[str, dict]:
    assessments_by_jd: dict[str, dict] = {}
    for jd_id in jd_ids:
        path = run_dir / jd_id / "assessments.json"
        if not path.exists():
            continue
        assessments_by_jd[jd_id] = json.loads(path.read_text(encoding="utf-8"))
    return assessments_by_jd


class OptimizationRound(BaseModel):
    round_index: int
    criteria: list[str]
    metric: float


def run_optimization(
    seed_criteria: list[str],
    train_triples: list[RequirementTriple],
    validation_triples: list[RequirementTriple],
    jds_by_id: dict[str, JobDescription],
    candidates_by_id: dict[str, Candidate],
    jev_client: JevClient,
    proxy_labels: dict[str, int],
    propose_fn: Callable[[list[str], list[str]], list[str]],
    max_rounds: int,
    hard_case_count: int,
    patience: int,
    on_round: Callable[[list[str], list[OptimizationRound]], None] | None = None,
) -> tuple[list[str], list[OptimizationRound]]:
    history: list[OptimizationRound] = []
    best_metric: float | None = None
    best_criteria = seed_criteria
    hard_triples: list[RequirementTriple] = []
    rounds_without_improvement = 0

    for round_index in range(max_rounds):
        if round_index == 0:
            candidate_criteria = seed_criteria
        else:
            hard_case_summaries = [
                f"requirement={t.requirement}, jd={t.job_description_id}, candidate={t.candidate_id}"
                for t in hard_triples
            ]
            candidate_criteria = propose_fn(best_criteria, hard_case_summaries)

        train_records = evaluate_criteria(candidate_criteria, train_triples, jds_by_id, candidates_by_id, jev_client)
        round_metric = compute_metric(train_records, proxy_labels)
        history.append(OptimizationRound(round_index=round_index, criteria=candidate_criteria, metric=round_metric))

        if best_metric is None or round_metric > best_metric:
            best_metric = round_metric
            best_criteria = candidate_criteria
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1

        if on_round is not None:
            on_round(best_criteria, list(history))
        print(f"round {round_index}: metric={round_metric:.4f}, best={best_metric:.4f}")

        if rounds_without_improvement >= patience:
            break

        validation_records = evaluate_criteria(
            best_criteria, validation_triples, jds_by_id, candidates_by_id, jev_client
        )
        hard_triples = select_hard_triples(validation_records, hard_case_count)

    return best_criteria, history
