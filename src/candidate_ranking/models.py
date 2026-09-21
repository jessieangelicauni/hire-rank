from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, computed_field


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
    must_have_skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    seniority_requirement: str | None = None
    seniority_min_years: float | None = Field(default=None, ge=0)
    education_requirement: str | None = None


class Assessment(BaseModel):
    job_description_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    generated_by_model: str = Field(min_length=1)
    overall_recommendation: Literal["hire", "maybe", "no"]
    meets_min_qualifications: bool
    requirement_scores: dict[str, float] = Field(default_factory=dict)
    confidence: dict[str, float] = Field(default_factory=dict)
    certification_results: dict[str, bool] = Field(default_factory=dict)
    seniority_years_fit_score: float | None = Field(default=None, ge=0, le=100)
    education_fit_score: float | None = Field(default=None, ge=0, le=100)

    @computed_field
    @property
    def composite_fit_score(self) -> float:
        components = list(self.requirement_scores.values())
        if self.seniority_years_fit_score is not None:
            components.append(self.seniority_years_fit_score)
        if self.education_fit_score is not None:
            components.append(self.education_fit_score)
        return sum(components) / len(components) if components else 0.0


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
