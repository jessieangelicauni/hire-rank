from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from candidate_ranking.evaluation.criteria_optimization import validate_criteria_shape
from candidate_ranking.generation import GenerationError, invoke_and_validate

PROPOSE_CRITERIA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are designing the criteria for a Score-type question that judges how well a CV supports a "
            "job requirement, for a model that matches evidence against concrete situational descriptions "
            "rather than bare labels.\n\n"
            "Propose a revised 5-level criteria list, one concrete situational description per level, "
            "weakest (level 0) to strongest (level 4). Each level must be a distinct, evidence-based "
            "situation the model can match against CV text -- not a bare label or number. List the 5 "
            "descriptions in the `levels` field, weakest first.",
        ),
        (
            "human",
            "Current 5-level criteria (level 0 to level 4):\n{current_criteria}\n\n"
            "Real cases where the current criteria produced low-confidence, ambiguous judgments:\n{hard_cases}\n\n"
            "Propose a revised 5-level criteria list.",
        ),
    ]
)


class _ProposedCriteria(BaseModel):
    levels: list[str] = Field(min_length=1)


def build_criteria_proposer_chain(llm: BaseChatModel) -> Runnable:
    return PROPOSE_CRITERIA_PROMPT | llm.with_structured_output(_ProposedCriteria)


def propose_criteria(current_criteria: list[str], hard_case_summaries: list[str], chain: Runnable) -> list[str]:
    def validate(result: _ProposedCriteria) -> str | None:
        try:
            validate_criteria_shape(result.levels)
        except ValueError as exc:
            return str(exc)
        return None

    result = invoke_and_validate(
        chain,
        {
            "current_criteria": "\n".join(current_criteria),
            "hard_cases": "\n".join(hard_case_summaries) if hard_case_summaries else "(none yet)",
        },
        _ProposedCriteria,
        validate,
        log_context="criteria proposal",
    )
    return result.levels
