from __future__ import annotations

import json
import statistics
from pathlib import Path

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
    for triple in triples:
        jd = jds_by_id[triple.job_description_id]
        candidate = candidates_by_id[triple.candidate_id]
        state = _build_state(jd, candidate)
        question = build_requirement_question(triple.requirement, criteria)
        answers = jev_client.evaluate(state, [question])
        answer = answers[0]
        records.append(CriteriaEvalRecord(triple=triple, score=float(answer.value), confidence=answer.confidence))
    return records


def compute_metric(records: list[CriteriaEvalRecord], proxy_labels: dict[str, int]) -> float:
    """Anti-Goodhart metric that only credits confidence when Jev's score agrees with proxy label within 1 level."""
    if not records:
        return 0.0
    credited_confidences = []
    for record in records:
        label = proxy_labels[triple_key(record.triple)]
        agrees = abs(record.score - label) <= 1
        credited_confidences.append(record.confidence if agrees else 0.0)
    return statistics.mean(credited_confidences)


def select_hard_triples(records: list[CriteriaEvalRecord], k: int) -> list[RequirementTriple]:
    """Red-team critic: select the k lowest-confidence triples for the next optimization round."""
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
