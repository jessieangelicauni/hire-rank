from __future__ import annotations

import hashlib
import logging
import re
import threading
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, ValidationError

from candidate_ranking.utils import GenerationError, invoke_and_validate, load_json_cache, save_json_cache
from candidate_ranking.models import AdaptiveRequirement, JobProfile, SkillCriteria, TotalYearsCriteria
from candidate_ranking.scoring.assessment import fallback_skill_criteria, fallback_total_years_criteria

logger = logging.getLogger(__name__)

_cache_lock = threading.Lock()

JOB_PROFILE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior technical recruiter extracting a structured job profile "
            "from a job posting, for candidate evaluation and skill matching.\n\n"
            "In `reasoning` (max 3 sentences), note anything non-obvious before answering.\n\n"
            "Skills → must_have_skills and nice_to_have_skills hold ONLY concrete tools, "
            "software, languages, frameworks, libraries, or platforms by name (e.g. React, "
            "PostgreSQL, Terraform). Never a practice, methodology, architecture pattern, "
            "certification, or standard (e.g. \"microservices architecture\", \"user research\", "
            "\"CISSP\", \"SOC 2\") — those belong in `other_requirements` instead.\n"
            "- `must_have_skills`: if the JP has an explicit \"Must Have Skills\" section, use "
            "exactly those tools/technologies. Otherwise, use the core the role is built "
            "around — main language(s) plus 1-3 defining frameworks or platforms; infra/tooling "
            "(Docker, cloud, CI/CD) is not core unless it IS the role.\n"
            "- `nice_to_have_skills`: every other named tool/technology mentioned anywhere in "
            "the job posting (narrative included) that isn't already in must_have_skills.\n\n"
            "Non-skill requirements:\n"
            "- `education_requirement`: one sentence summary if the JP states one, else null.\n"
            "- `seniority_requirement`: one sentence summary of seniority/years-of-experience "
            "if stated, else null.\n"
            "- `other_requirements`: anything else the JP requires or prefers — practices, "
            "methodologies, certifications, compliance standards, domain knowledge, soft "
            "skills, languages (spoken), etc. — each as a short phrase.",
        ),
        ("human", "Job Title: {title}\n\nJob Posting:\n{description}"),
    ]
)


class _JobProfileOutput(BaseModel):
    reasoning: str = Field(min_length=1)
    must_have_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    education_requirement: str | None = None
    seniority_requirement: str | None = None
    other_requirements: list[str] = Field(default_factory=list)


class JobProfileGenerationError(Exception):
    pass


_MUST_HAVE_LINE_RE = re.compile(r"^Must Have Skills:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def _parse_explicit_must_have(raw_text: str) -> list[str] | None:
    """Parse a literal "Must Have Skills: A, B, C" line out of the job posting text.

    The LLM is unreliable at using this list verbatim (it tends to pad it
    back in with narrative bullets), so when the line is present we bypass
    the LLM for must_have_skills entirely and parse it deterministically.
    """
    match = _MUST_HAVE_LINE_RE.search(raw_text)
    if match is None:
        return None
    return [skill.strip() for skill in match.group(1).split(",") if skill.strip()]


def build_job_profile_chain(llm: BaseChatModel) -> Runnable:
    return JOB_PROFILE_PROMPT | llm.with_structured_output(_JobProfileOutput)


class _SkillCriteriaOutput(BaseModel):
    skill: str
    instructions: str
    strong: str
    adequate: str
    weak: str
    none: str


class _TotalYearsCriteriaOutput(BaseModel):
    instructions: str


class _AdaptiveRequirementOutput(BaseModel):
    source: Literal["education", "seniority", "other"]
    requirement: str
    criteria_meets: str
    criteria_partially_meets: str
    criteria_does_not_meet: str


class _RubricOutput(BaseModel):
    reasoning: str = Field(min_length=1)
    skill_criteria: list[_SkillCriteriaOutput] = Field(default_factory=list)
    total_years_criteria: _TotalYearsCriteriaOutput | None = None
    adaptive_requirements: list[_AdaptiveRequirementOutput] = Field(default_factory=list)


RUBRIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior technical recruiter building a detailed evaluation rubric for a "
            "specific job profile, to brief an AI evaluator that will assess candidate CVs "
            "against it.\n\n"
            "In `reasoning` (max 3 sentences), note anything non-obvious before answering.\n\n"
            "For EACH skill listed below (must-have and nice-to-have together), produce one "
            "`skill_criteria` entry with the EXACT same `skill` name given to you, verbatim. "
            "`instructions` should give role-specific guidance for judging evidence of that "
            "skill in a CV; `strong`/`adequate`/`weak`/`none` should each describe what that "
            "evidence-strength level looks like FOR THIS SKILL IN THIS ROLE, calibrated to the "
            "role's seniority and domain.\n\n"
            "Produce exactly one `total_years_criteria` entry whose `instructions` calibrate "
            "what counts as 'relevant' experience for THIS role, and what years-of-experience "
            "ranges correspond to junior/mid/senior for this specific role's seniority level.\n\n"
            "For EACH requirement listed below (education, seniority, and each 'other' "
            "requirement), produce one `adaptive_requirements` entry with the EXACT same "
            "`source` and `requirement` text given to you, verbatim, plus `criteria_meets`, "
            "`criteria_partially_meets`, and `criteria_does_not_meet` describing what each "
            "match level looks like in a candidate's CV for that specific requirement.",
        ),
        (
            "human",
            "Job Title: {title}\n\n"
            "Skills to calibrate:\n{skills_block}\n\n"
            "Requirements to calibrate:\n{requirements_block}",
        ),
    ]
)


def build_rubric_chain(llm: BaseChatModel) -> Runnable:
    return RUBRIC_PROMPT | llm.with_structured_output(_RubricOutput)


def _format_skills_block(must_have_skills: list[str], nice_to_have_skills: list[str]) -> str:
    lines = [f"- {skill} (must-have)" for skill in must_have_skills]
    lines += [f"- {skill} (nice-to-have)" for skill in nice_to_have_skills]
    return "\n".join(lines) if lines else "(none)"


def _format_requirements_block(
    education_requirement: str | None,
    seniority_requirement: str | None,
    other_requirements: list[str],
) -> str:
    lines = []
    if education_requirement:
        lines.append(f"- education: {education_requirement}")
    if seniority_requirement:
        lines.append(f"- seniority: {seniority_requirement}")
    lines += [f"- other: {req}" for req in other_requirements]
    return "\n".join(lines) if lines else "(none)"


def _finalize_skill_criteria(
    skills: list[str],
    rubric_skills: list[_SkillCriteriaOutput],
    jp_id: str,
) -> list[SkillCriteria]:
    # Keyed on a normalized form — the model is told to echo the skill name
    # back verbatim, but whitespace/case drift shouldn't drop a real match
    # into the fallback path.
    by_name = {s.skill.strip().casefold(): s for s in rubric_skills}
    result = []
    for skill in skills:
        match = by_name.get(skill.strip().casefold())
        if match is None:
            logger.warning(
                "Rubric generation for %s: no skill-criteria rubric returned for skill %r — "
                "using generic fallback", jp_id, skill,
            )
            result.append(fallback_skill_criteria(jp_id, skill))
            continue
        result.append(SkillCriteria(
            skill=skill, instructions=match.instructions,
            strong=match.strong, adequate=match.adequate, weak=match.weak, none=match.none,
        ))
    return result


def _finalize_total_years_criteria(
    rubric_total_years: _TotalYearsCriteriaOutput | None,
) -> TotalYearsCriteria:
    if rubric_total_years is None:
        logger.warning("Rubric generation: no total_years_criteria returned — using generic fallback")
        return fallback_total_years_criteria()
    return TotalYearsCriteria(instructions=rubric_total_years.instructions)


def _finalize_adaptive_requirements(
    education_requirement: str | None,
    seniority_requirement: str | None,
    other_requirements: list[str],
    rubric_requirements: list[_AdaptiveRequirementOutput],
    jp_id: str,
) -> list[AdaptiveRequirement]:
    expected: list[tuple[str, str]] = []
    if education_requirement:
        expected.append(("education", education_requirement))
    if seniority_requirement:
        expected.append(("seniority", seniority_requirement))
    expected += [("other", req) for req in other_requirements]

    # Same normalized-key rationale as _finalize_skill_criteria above.
    by_key = {(r.source, r.requirement.strip().casefold()): r for r in rubric_requirements}
    result = []
    for source, requirement in expected:
        match = by_key.get((source, requirement.strip().casefold()))
        if match is None:
            logger.warning(
                "Rubric generation for %s: no adaptive-requirement rubric returned for %s "
                "requirement %r — dropping this question", jp_id, source, requirement,
            )
            continue
        result.append(AdaptiveRequirement(
            source=source, requirement=requirement,
            criteria_meets=match.criteria_meets,
            criteria_partially_meets=match.criteria_partially_meets,
            criteria_does_not_meet=match.criteria_does_not_meet,
        ))
    return result


def generate_job_profile(
    jp: JobProfile,
    chain: Runnable,
    rubric_chain: Runnable,
    model_name: str,
) -> JobProfile:
    try:
        result = invoke_and_validate(
            chain,
            {"title": jp.title, "description": jp.raw_text},
            _JobProfileOutput,
            lambda _: None,
            log_context=f"Job profile extraction for {jp.id}",
        )
    except GenerationError as exc:
        raise JobProfileGenerationError(str(exc)) from exc

    explicit_must_have = _parse_explicit_must_have(jp.raw_text)
    must_have_skills = explicit_must_have if explicit_must_have is not None else result.must_have_skills
    must_have_normalized = {skill.strip().lower() for skill in must_have_skills}
    nice_to_have_skills = [
        skill for skill in result.nice_to_have_skills
        if skill.strip().lower() not in must_have_normalized
    ]

    try:
        rubric = invoke_and_validate(
            rubric_chain,
            {
                "title": jp.title,
                "skills_block": _format_skills_block(must_have_skills, nice_to_have_skills),
                "requirements_block": _format_requirements_block(
                    result.education_requirement, result.seniority_requirement, result.other_requirements,
                ),
            },
            _RubricOutput,
            lambda _: None,
            log_context=f"Adaptive rubric generation for {jp.id}",
        )
    except GenerationError as exc:
        # Call 1 already succeeded — a rubric hiccup shouldn't discard the
        # whole JP (the same reasoning that motivates the per-skill/per-field
        # fallbacks below). Degrade to an empty rubric; _finalize_* fills
        # every skill/field in with its generic fallback.
        logger.warning(
            "Adaptive rubric generation for %s failed, falling back to generic rubric for every "
            "field: %s", jp.id, exc,
        )
        rubric = _RubricOutput(reasoning="rubric generation failed")

    skill_criteria = _finalize_skill_criteria(must_have_skills + nice_to_have_skills, rubric.skill_criteria, jp.id)
    total_years_criteria = _finalize_total_years_criteria(rubric.total_years_criteria)
    adaptive_requirements = _finalize_adaptive_requirements(
        result.education_requirement, result.seniority_requirement, result.other_requirements,
        rubric.adaptive_requirements, jp.id,
    )

    return JobProfile(
        id=jp.id,
        title=jp.title,
        raw_text=jp.raw_text,
        source_path=jp.source_path,
        generated_by_model=model_name,
        must_have_skills=must_have_skills,
        nice_to_have_skills=nice_to_have_skills,
        education_requirement=result.education_requirement,
        seniority_requirement=result.seniority_requirement,
        other_requirements=result.other_requirements,
        skill_criteria=skill_criteria,
        total_years_criteria=total_years_criteria,
        adaptive_requirements=adaptive_requirements,
    )


def _cache_key(jp: JobProfile, model_name: str) -> str:
    # Deliberately unchanged in shape by the adaptive-rubric feature (Call 2):
    # it's keyed only on Call 1's formatted prompt + model name, so a cache
    # hit skips both calls. One consequence: a job_profile.json cached by a
    # pre-adaptive-rubric build is still a valid cache hit here, and will
    # silently come back with empty skill_criteria/adaptive_requirements
    # (generic fallback questions, no adaptive ones) until its JP text or
    # model name changes. Delete cache/job_profile.json after pulling this
    # change if you want the adaptive rubric regenerated immediately.
    prompt_repr = repr(
        JOB_PROFILE_PROMPT.format_messages(title=jp.title, description=jp.raw_text)
    )
    return hashlib.sha256(f"{prompt_repr}||{model_name}".encode("utf-8")).hexdigest()


def load_or_generate_job_profile(
    jp: JobProfile,
    chain: Runnable,
    rubric_chain: Runnable,
    model_name: str,
    cache_path: Path,
) -> JobProfile:
    key = _cache_key(jp, model_name)

    with _cache_lock:
        cache = load_json_cache(cache_path, "job profile cache")
        cached_entry = cache.get(jp.id)
        if cached_entry is not None and cached_entry.get("cache_key") == key:
            try:
                return JobProfile.model_validate(cached_entry["job_profile"])
            except (KeyError, ValidationError) as exc:
                logger.warning("Discarding invalid cached job profile for %s: %s", jp.id, exc)

    job_profile = generate_job_profile(jp, chain, rubric_chain, model_name)

    with _cache_lock:
        cache = load_json_cache(cache_path, "job profile cache")
        cache[jp.id] = {"cache_key": key, "job_profile": job_profile.model_dump()}
        save_json_cache(cache_path, cache)

    return job_profile
