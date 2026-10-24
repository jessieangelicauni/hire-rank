from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()


class EvaluationCriterion(BaseModel):
    label: str
    instructions: str
    strong: str
    partial: str
    none: str


class JobProfile(BaseModel):
    id: str
    raw_text: str
    technical_skills: list[str] = Field(default_factory=list)
    evaluation_criteria: list[EvaluationCriterion] = Field(default_factory=list)


class SkillEntry(BaseModel):
    name: str
    context: str


class SkillGroup(BaseModel):
    names: list[str]
    context: str


class CandidateProfile(BaseModel):
    skills: list[SkillEntry] = Field(default_factory=list)

    def skills_grouped_by_context(self) -> list[SkillGroup]:
        order: list[str] = []
        names_by_context: dict[str, list[str]] = {}
        for skill in self.skills:
            if skill.context not in names_by_context:
                names_by_context[skill.context] = []
                order.append(skill.context)
            names_by_context[skill.context].append(skill.name)
        return [SkillGroup(names=names_by_context[ctx], context=ctx) for ctx in order]


class Candidate(BaseModel):
    id: str
    raw_text: str
    parse_status: Literal["ok", "failed"]
    profile: CandidateProfile | None = None


class RequirementEvaluation(BaseModel):
    requirement: str
    evidence: str
    evidence_strength: Literal["strong", "partial", "none"]
    probabilities: dict[str, float] | None = None


class Assessment(BaseModel):
    job_profile_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    evaluations: list[RequirementEvaluation] = Field(default_factory=list)
