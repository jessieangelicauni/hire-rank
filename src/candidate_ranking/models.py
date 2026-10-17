from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()


class SkillCriteria(BaseModel):
    skill: str
    instructions: str
    strong: str
    adequate: str
    weak: str
    none: str


class TotalYearsCriteria(BaseModel):
    instructions: str


class AdaptiveRequirement(BaseModel):
    source: Literal["education", "seniority", "other"]
    requirement: str
    criteria_meets: str
    criteria_partially_meets: str
    criteria_does_not_meet: str


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
    skill_criteria: list[SkillCriteria] = Field(default_factory=list)
    total_years_criteria: TotalYearsCriteria | None = None
    adaptive_requirements: list[AdaptiveRequirement] = Field(default_factory=list)


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
    evidence: str
    evidence_strength: Literal["strong", "adequate", "weak", "none"]
    # Jev's own confidence in evidence_strength (0-1). Default of 1.0 only
    # applies to assessments cached before this field existed — it is a
    # placeholder, not a real recorded value, and evaluate-reliability
    # never reads from cache anyway (it bypasses it on every repeat).
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class AdaptiveRequirementEvaluation(BaseModel):
    source: Literal["education", "seniority", "other"]
    requirement: str
    evidence: str
    match_level: Literal["meets", "partially_meets", "does_not_meet"]
    confidence: float = Field(ge=0.0, le=1.0)


class Assessment(BaseModel):
    job_profile_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    generated_by_model: str = Field(min_length=1)
    must_have_evaluations: list[RequirementEvaluation] = Field(default_factory=list)
    nice_to_have_evaluations: list[RequirementEvaluation] = Field(default_factory=list)
    total_relevant_years: int = Field(ge=0, default=0)
    adaptive_requirement_evaluations: list[AdaptiveRequirementEvaluation] = Field(default_factory=list)
    qualified: bool = False
    disqualification_reason: str | None = None
