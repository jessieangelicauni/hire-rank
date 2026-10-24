from __future__ import annotations

from candidate_ranking.models import Assessment, RequirementEvaluation

STRENGTH_WEIGHT: dict[str, float] = {"strong": 1.0, "partial": 0.5, "none": 0.0}


def expected_strength(evaluation: RequirementEvaluation) -> float:
    if evaluation.probabilities:
        return sum(
            STRENGTH_WEIGHT.get(level, 0.0) * p
            for level, p in evaluation.probabilities.items()
        )
    return STRENGTH_WEIGHT[evaluation.evidence_strength]


def criterion_score(assessment: Assessment) -> float:
    if not assessment.evaluations:
        return 0.0
    total = sum(expected_strength(e) for e in assessment.evaluations)
    return total / len(assessment.evaluations)


def criterion_score_breakdown(assessment: Assessment) -> list[dict]:
    return [
        {
            "requirement": e.requirement,
            "evidence_strength": e.evidence_strength,
            "weight": STRENGTH_WEIGHT[e.evidence_strength],
            "expected_strength": expected_strength(e),
            "probabilities": e.probabilities,
        }
        for e in assessment.evaluations
    ]
