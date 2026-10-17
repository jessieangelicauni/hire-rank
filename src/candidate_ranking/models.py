from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()


class JobProfile(BaseModel):
    id: str
    title: str
    raw_text: str
    source_path: str
    generated_by_model: str = ""
    must_have_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    education_requirement: str | None = None
    seniority_requirement: str | None = None
    other_requirements: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    id: str
    source_path: str
    raw_text: str
    num_pages: int
    char_count: int
    parse_status: Literal["ok", "failed"]
    error_message: str | None = None
    skills: list[str] = Field(default_factory=list)


class RequirementEvaluation(BaseModel):
    requirement: str
    verdict: Literal["MET", "PARTIALLY_MET", "NOT_MET"]
    evidence: str
    evidence_strength: Literal["strong", "adequate", "weak", "none"]
    recency: Literal["current", "recent", "dated", "unknown"]
    years_demonstrated: int = Field(ge=0, default=0)


class CareerTrajectory(BaseModel):
    direction: Literal["ascending", "lateral", "descending", "unclear"]
    tenure_pattern: Literal["stable", "variable", "concerning"]
    average_tenure_months: int = Field(ge=0, default=0)
    most_recent_role_duration_months: int = Field(ge=0, default=0)
    has_multinational_experience: bool


class Assessment(BaseModel):
    job_profile_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    generated_by_model: str = Field(min_length=1)
    must_have_evaluations: list[RequirementEvaluation] = Field(default_factory=list)
    nice_to_have_evaluations: list[RequirementEvaluation] = Field(default_factory=list)
    total_relevant_years: int = Field(ge=0, default=0)
    career_trajectory: CareerTrajectory | None = None
    qualified: bool = False
    disqualification_reason: str | None = None
