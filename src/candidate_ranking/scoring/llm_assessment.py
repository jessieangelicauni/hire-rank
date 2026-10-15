from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, ValidationError

from candidate_ranking.utils import invoke_and_validate
from candidate_ranking.models import (
    Assessment, Candidate, CareerTrajectory, JDRequirements,
    JobDescription, RedFlag, RequirementEvaluation,
)
logger = logging.getLogger(__name__)

_cache_lock = threading.Lock()

# ── Prompt 2: Qualification Gate ─────────────────────────────────────────────

QUALIFICATION_GATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior technical recruiter screening a CV.\n\n"
            "EVIDENCE RULES — apply consistently:\n"
            "  STRONG   = skill demonstrated in work experience with specific system or project\n"
            "  ADEQUATE = skill mentioned in experience context but without project specifics\n"
            "  WEAK     = skill listed in skills/summary section ONLY, never shown in any experience or project\n\n"
            "Verdict mapping:\n"
            "  MET           = strong or adequate evidence\n"
            "  PARTIALLY_MET = weak evidence only\n"
            "  NOT_MET       = no evidence anywhere in CV\n\n"
            "NON-COMPENSATORY RULE:\n"
            "  If any MUST_HAVE verdict is NOT_MET → candidate is disqualified immediately.\n"
            "  Strong performance elsewhere does not compensate.\n\n"
            "EVIDENCE CITATION RULE:\n"
            "  For every verdict, quote or reference the exact section of the CV where the evidence appears.\n"
            "  If you cannot find specific text → verdict is PARTIALLY_MET (weak) or NOT_MET, never MET.\n\n"
            "MISSED EVIDENCE RULE:\n"
            "  Before concluding NOT_MET for any requirement, re-read ALL sections:\n"
            "  skills, experience, projects, education, certifications, and summary.\n"
            "  Skills are often implied — 'Django' implies Python, 'Spring Boot' implies Java.\n\n"
            "DURATION RULE:\n"
            "  Compute total_relevant_years by summing durations of roles where the candidate used\n"
            "  skills relevant to this JD. Use CV dates directly.\n"
            "  years_demonstrated per requirement = years that skill appears in active use.\n\n"
            "RED_FLAG RULE:\n"
            "  type must be exactly one of: EMPLOYMENT_GAP, SKILL_MISMATCH, REGRESSION, JOB_HOPPING, OTHER.\n"
            "  severity must be exactly one of: MINOR, MODERATE, MAJOR.\n\n"
            "Evaluate every MUST_HAVE requirement first. If candidate qualifies, also evaluate\n"
            "NICE_TO_HAVE requirements and fill all remaining fields.",
        ),
        (
            "human",
            "ROLE: {jd_title}\n\n"
            "MUST_HAVE REQUIREMENTS:\n{must_have_json}\n\n"
            "NICE_TO_HAVE REQUIREMENTS:\n{nice_to_have_json}\n\n"
            "CV:\n{cv_text}",
        ),
    ]
)


class _RequirementEvaluationOutput(BaseModel):
    requirement: str
    verdict: Literal["MET", "PARTIALLY_MET", "NOT_MET"]
    evidence: str = Field(min_length=1)
    evidence_strength: Literal["strong", "adequate", "weak"]
    recency: Literal["current", "recent", "dated", "unknown"] = "unknown"
    years_demonstrated: int = Field(ge=0, default=0)


class _CareerTrajectoryOutput(BaseModel):
    direction: Literal["ascending", "lateral", "descending", "unclear"]
    evidence: str
    tenure_pattern: Literal["stable", "variable", "concerning"]
    average_tenure_months: int = Field(ge=0, default=0)
    most_recent_role_duration_months: int = Field(ge=0, default=0)
    has_multinational_experience: bool
    multinational_evidence: str | None = None


class _RedFlagOutput(BaseModel):
    type: Literal["employment_gap", "skill_mismatch", "regression", "job_hopping", "other"]
    detail: str
    duration_months: int | None = None
    severity: Literal["minor", "moderate", "major"]


class _QualificationOutput(BaseModel):
    must_have_evaluations: list[_RequirementEvaluationOutput]
    nice_to_have_evaluations: list[_RequirementEvaluationOutput] = Field(default_factory=list)
    total_relevant_years: int = Field(ge=0, default=0)
    career_trajectory: _CareerTrajectoryOutput | None = None
    strongest_achievements: list[str] = Field(default_factory=list)
    notable_signals: str | None = None
    red_flags: list[_RedFlagOutput] = Field(default_factory=list)
    qualified: bool
    disqualification_reason: str | None = None


class AssessmentGenerationError(Exception):
    pass


def build_qualification_chain(llm: BaseChatModel) -> Runnable:
    return QUALIFICATION_GATE_PROMPT | llm.with_structured_output(_QualificationOutput)


def _output_to_assessment(
    jd: JobDescription,
    candidate: Candidate,
    model_name: str,
    output: _QualificationOutput,
) -> Assessment:
    must_have_evals = [
        RequirementEvaluation(
            requirement=ev.requirement,
            verdict=ev.verdict,
            evidence=ev.evidence,
            evidence_strength=ev.evidence_strength,
            recency=ev.recency,
            years_demonstrated=ev.years_demonstrated,
        )
        for ev in output.must_have_evaluations
    ]
    nice_to_have_evals = [
        RequirementEvaluation(
            requirement=ev.requirement,
            verdict=ev.verdict,
            evidence=ev.evidence,
            evidence_strength=ev.evidence_strength,
            recency=ev.recency,
            years_demonstrated=ev.years_demonstrated,
        )
        for ev in output.nice_to_have_evaluations
    ]
    traj = None
    if output.career_trajectory:
        t = output.career_trajectory
        traj = CareerTrajectory(
            direction=t.direction,
            evidence=t.evidence,
            tenure_pattern=t.tenure_pattern,
            average_tenure_months=t.average_tenure_months,
            most_recent_role_duration_months=t.most_recent_role_duration_months,
            has_multinational_experience=t.has_multinational_experience,
            multinational_evidence=t.multinational_evidence,
        )
    red_flags = [
        RedFlag(
            type=rf.type,
            detail=rf.detail,
            duration_months=rf.duration_months,
            severity=rf.severity,
        )
        for rf in output.red_flags
    ]
    return Assessment(
        job_description_id=jd.id,
        candidate_id=candidate.id,
        generated_by_model=model_name,
        must_have_evaluations=must_have_evals,
        nice_to_have_evaluations=nice_to_have_evals,
        total_relevant_years=output.total_relevant_years,
        career_trajectory=traj,
        strongest_achievements=output.strongest_achievements,
        notable_signals=output.notable_signals,
        red_flags=red_flags,
        qualified=output.qualified,
        disqualification_reason=output.disqualification_reason,
    )


def generate_assessment(
    jd: JobDescription,
    candidate: Candidate,
    jd_requirements: JDRequirements | None,
    chain: Runnable,
    model_name: str,
) -> Assessment:
    must_have_json = (
        json.dumps([{"requirement": r.requirement, "context": r.context} for r in jd_requirements.must_have], indent=2)
        if jd_requirements else "[]"
    )
    nice_to_have_json = (
        json.dumps([{"requirement": r.requirement, "context": r.context} for r in jd_requirements.nice_to_have], indent=2)
        if jd_requirements else "[]"
    )

    def validate(result: _QualificationOutput) -> str | None:
        # Only enforce non-empty evaluations when requirements were actually provided;
        # if jd_requirements is None the LLM received empty lists and may return no evals.
        if jd_requirements and not result.must_have_evaluations:
            return "must_have_evaluations is empty"
        return None

    log_context = f"Qualification gate for {jd.id}/{candidate.id}"
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            result = invoke_and_validate(
                chain,
                {
                    "jd_title": jd.title,
                    "must_have_json": must_have_json,
                    "nice_to_have_json": nice_to_have_json,
                    "cv_text": candidate.raw_text,
                },
                _QualificationOutput,
                validate,
                log_context=f"{log_context} (attempt {attempt + 1}/2)",
            )
            break
        except Exception as exc:  # noqa: BLE001 — translate ALL failures to domain error
            last_error = exc
    else:
        raise AssessmentGenerationError(str(last_error)) from last_error

    return _output_to_assessment(jd, candidate, model_name, result)


def _assessment_cache_path(cache_dir: Path, jd_id: str, candidate_id: str) -> Path:
    return cache_dir / "assessments" / jd_id / f"{candidate_id}.json"


def _assessment_cache_key(
    jd: JobDescription,
    candidate: Candidate,
    jd_requirements: JDRequirements | None,
    model_name: str,
) -> str:
    reqs_repr = jd_requirements.model_dump_json() if jd_requirements else "null"
    digest_input = f"{jd.id}||{jd.raw_text}||{candidate.id}||{candidate.raw_text}||{reqs_repr}||{model_name}"
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


def load_or_generate_assessment(
    jd: JobDescription,
    candidate: Candidate,
    jd_requirements: JDRequirements | None,
    chain: Runnable,
    model_name: str,
    cache_dir: Path,
) -> Assessment:
    path = _assessment_cache_path(cache_dir, jd.id, candidate.id)
    key = _assessment_cache_key(jd, candidate, jd_requirements, model_name)

    with _cache_lock:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("cache_key") == key:
                    return Assessment.model_validate(data["assessment"])
            except (json.JSONDecodeError, KeyError, ValidationError, OSError) as exc:
                logger.warning("Discarding unreadable assessment cache %s: %s", path, exc)

    # LLM call happens outside the lock to avoid blocking other threads.
    assessment = generate_assessment(jd, candidate, jd_requirements, chain, model_name)

    with _cache_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        # uuid suffix avoids collision when multiple processes write concurrently.
        tmp = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(
            json.dumps({"cache_key": key, "assessment": assessment.model_dump()}, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)
    return assessment


def filter_assessable_candidates(candidates: list[Candidate]) -> list[Candidate]:
    assessable = [c for c in candidates if c.parse_status == "ok"]
    excluded = len(candidates) - len(assessable)
    if excluded > 0:
        logger.warning("Excluded %d candidate(s) with failed CV parsing", excluded)
    return assessable
