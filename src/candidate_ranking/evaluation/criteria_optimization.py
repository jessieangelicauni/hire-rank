from __future__ import annotations

from pydantic import BaseModel

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
