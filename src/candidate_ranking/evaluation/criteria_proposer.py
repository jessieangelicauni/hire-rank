from __future__ import annotations

import dspy

from candidate_ranking.evaluation.criteria_optimization import validate_criteria_shape


class ProposeCriteria(dspy.Signature):
    """Propose a revised 5-level Score-question criteria list for judging how well a CV supports a job requirement."""

    current_criteria: str = dspy.InputField(
        desc="Current 5-level criteria, one per line, level 0 (weakest) to level 4 (strongest)."
    )
    hard_cases: str = dspy.InputField(
        desc="Real cases where the current criteria produced low-confidence, ambiguous judgments."
    )
    revised_criteria: str = dspy.OutputField(
        desc="Exactly 5 lines, one concrete situational description per level, weakest to strongest, no numbering."
    )


def propose_criteria(current_criteria: list[str], hard_case_summaries: list[str], lm: dspy.LM) -> list[str]:
    predict = dspy.Predict(ProposeCriteria)
    with dspy.context(lm=lm):
        result = predict(
            current_criteria="\n".join(current_criteria),
            hard_cases="\n".join(hard_case_summaries) if hard_case_summaries else "(none yet)",
        )
    proposed = [line.strip() for line in result.revised_criteria.strip().split("\n") if line.strip()]
    validate_criteria_shape(proposed)
    return proposed
