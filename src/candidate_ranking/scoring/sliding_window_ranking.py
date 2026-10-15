# src/candidate_ranking/scoring/sliding_window_ranking.py
from __future__ import annotations

import logging
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, create_model

from candidate_ranking.models import Assessment, JobDescription

logger = logging.getLogger(__name__)

SLIDING_WINDOW_RANKING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior hiring panel ranking a group of pre-screened candidates.\n"
            "All candidates have already passed qualification screening.\n\n"
            "STEP 1 — For EACH candidate, write a brief evaluation:\n"
            "  key_strengths: what makes this candidate strong FOR THIS SPECIFIC ROLE\n"
            "  key_gaps: what is missing or weak for this role\n\n"
            "STEP 2 — Rank ALL candidates from strongest to weakest fit.\n\n"
            "RANKING CRITERIA (in priority order):\n"
            "1. Relevance — experience directly matching daily responsibilities\n"
            "2. Evidence quality — concrete achievements over vague responsibilities\n"
            "3. Career trajectory — ascending scope and seniority\n"
            "4. Recency — recent hands-on experience over older exposure\n"
            "5. Nice-to-have coverage — breadth of preferred skills as tiebreaker\n\n"
            "RULES:\n"
            "- Candidates are numbered 1..N. `ranking` must contain every number exactly once.\n"
            "- `candidate_evaluations` must contain one entry per candidate with the same id numbers.\n"
            "- Rank ALL candidates. No ties.\n"
            "- Judge only on information provided.",
        ),
        (
            "human",
            "ROLE: {jd_title}\n\nJOB SUMMARY: {jd_summary}\n\nCANDIDATES:\n{candidates_text}",
        ),
    ]
)


class _CandidateEvaluation(BaseModel):
    id: int = Field(ge=1)
    key_strengths: str = Field(min_length=1)
    key_gaps: str = Field(min_length=1)


def _sliding_window_model(size: int) -> type[BaseModel]:
    return create_model(
        "_SlidingWindowResult",
        candidate_evaluations=(
            list[_CandidateEvaluation],
            Field(min_length=size, max_length=size),
        ),
        ranking=(list[int], Field(min_length=size, max_length=size)),
    )


def build_sliding_window_chain_factory(
    llm: BaseChatModel,
) -> Callable[[type[BaseModel]], Runnable]:
    def _build(model: type[BaseModel]) -> Runnable:
        return SLIDING_WINDOW_RANKING_PROMPT | llm.with_structured_output(model)
    return _build
