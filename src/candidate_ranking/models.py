from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()


class JobDescription(BaseModel):
    id: str
    title: str
    raw_text: str
    source_path: str


class Candidate(BaseModel):
    id: str
    source_path: str
    raw_text: str
    num_pages: int
    char_count: int
    parse_status: Literal["ok", "failed"]
    error_message: str | None = None
    skills: list[str] = Field(default_factory=list)


class JDSkills(BaseModel):
    job_description_id: str
    generated_by_model: str
    technical_skills: list[str] = Field(default_factory=list)


class Assessment(BaseModel):
    job_description_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    generated_by_model: str = Field(min_length=1)
    strengths: list[str] = Field(min_length=1)
    weaknesses: list[str] = Field(default_factory=list)
    additional_skills: list[str] = Field(default_factory=list)
    reasoning: str | None = None


class TournamentIterationRecord(BaseModel):
    iteration: int = Field(ge=0)
    subset_candidate_ids: list[str] = Field(min_length=1)
    ranking: list[str] = Field(min_length=1)
    delta_u: float


class TournamentResult(BaseModel):
    job_description_id: str = Field(min_length=1)
    repeat_index: int = Field(ge=0)
    candidate_ids: list[str]
    final_utilities: dict[str, float]
    final_utility_variance: dict[str, float] = Field(default_factory=dict)
    iteration_history: list[TournamentIterationRecord]
    status: Literal["ok", "failed"]
    stop_reason: Literal["budget_exhausted", "no_unseen_subset"] | None = None


class ConvergencePoint(BaseModel):
    iteration: int = Field(ge=0)
    kendall_tau: float | None = Field(default=None, ge=-1.0, le=1.0)
    delta_u: float


class JDEvaluation(BaseModel):
    job_description_id: str = Field(min_length=1)
    convergence_trace: list[ConvergencePoint] = Field(default_factory=list)
    mean_kendall_tau: float | None = Field(default=None, ge=-1.0, le=1.0)
    mean_abs_delta_u: float | None = None
    kendall_tau_std: float | None = Field(default=None, ge=0.0)
    abs_delta_u_std: float | None = Field(default=None, ge=0.0)
