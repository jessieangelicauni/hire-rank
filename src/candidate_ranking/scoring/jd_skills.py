from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, ValidationError

from candidate_ranking.generation import GenerationError, invoke_and_validate
from candidate_ranking.json_cache import load_json_cache, save_json_cache
from candidate_ranking.models import JDSkills, JobDescription

logger = logging.getLogger(__name__)

_cache_lock = threading.Lock()

JD_SKILL_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert HR consultant extracting structured requirements a job description asks for.\n\n"
            "Task:\n"
            "- Read the job title and description and identify every technology and technical subject explicitly mentioned.\n"
            "- Reason through the description section by section in the `reasoning` field before giving your final answer.\n"
            "- Note which text backs each skill, certification, seniority requirement, or education requirement you identify.\n"
            "- List the identified skills in the `technical_skills` field.\n"
            "- Classify the must-have subset of those skills into `must_have_skills` (see Must-Have Classification below).\n"
            "- List any named professional certifications (e.g. \"AWS Certified Solutions Architect\", \"PMP\") in the `certifications` field.\n"
            "- If the description states a seniority or years-of-experience requirement, summarize it in one sentence in the `seniority_requirement` field; otherwise leave it null.\n"
            "- If that requirement states an explicit minimum number of years, put that number in `seniority_min_years` (e.g., \"5+ years\" or \"3-5 years\" both give 5 and 3 respectively -- use the minimum stated); otherwise leave it null.\n"
            "- If the description states an education requirement, summarize it in one sentence in the `education_requirement` field; otherwise leave it null.\n\n"
            "Must-Have Classification:\n"
            "- `must_have_skills` is the small core the role is built around: the main language(s) plus 1-3 defining frameworks or platforms.\n"
            "- Infrastructure and cross-cutting tooling (Docker, Kubernetes, cloud platforms, CI/CD, version control, BI tools) is nice-to-have, not must-have, unless that infrastructure is the role itself (e.g., DevOps, Cloud Engineer).\n"
            "- Most skills in a job description are nice-to-have; if everything looks essential, reconsider which ones the role truly cannot function without.\n\n"
            "Constraints:\n"
            "- Name each skill as the single atomic technology or subject it refers to.\n"
            "- Write each skill the way it would appear as a standalone item on a resume.\n"
            "- Do not phrase a skill as a description of proficiency, usage, or context.\n"
            "- Omit generic process or methodology phrases that do not name a specific technology or subject.\n"
            "- Every entry in `must_have_skills` must also appear in `technical_skills`.\n"
            "- Do not name a certification as a technical skill, or a technical skill as a certification.\n"
            "- A degree requirement belongs only in `education_requirement`, never in `technical_skills`.",
        ),
        ("human", "Job Title: {title}\n\nJob Description:\n{description}"),
    ]
)


class _GeneratedSkills(BaseModel):
    reasoning: str = Field(min_length=1)
    technical_skills: list[str] = Field(default_factory=list)
    must_have_skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    seniority_requirement: str | None = None
    seniority_min_years: float | None = None
    education_requirement: str | None = None


class JDSkillsGenerationError(Exception):
    pass


def _filter_must_have_skills(must_have_skills: list[str], technical_skills: list[str], jd_id: str) -> list[str]:
    technical_normalized = {s.strip().lower() for s in technical_skills}
    filtered: list[str] = []
    for skill in must_have_skills:
        if skill.strip().lower() in technical_normalized:
            filtered.append(skill)
        else:
            logger.warning(
                "Discarding must-have skill %r for JD %s: not present in extracted technical_skills",
                skill, jd_id,
            )
    return filtered


def build_jd_skills_chain(llm: BaseChatModel) -> Runnable:
    return JD_SKILL_EXTRACTION_PROMPT | llm.with_structured_output(_GeneratedSkills)


def generate_jd_skills(
    jd: JobDescription,
    chain: Runnable,
    model_name: str,
) -> JDSkills:
    def validate(result: _GeneratedSkills) -> str | None:
        if not result.technical_skills:
            return "generated an empty technical_skills list"
        return None

    last_error: GenerationError | None = None
    for attempt in range(2):
        try:
            result = invoke_and_validate(
                chain,
                {"title": jd.title, "description": jd.raw_text},
                _GeneratedSkills,
                validate,
                log_context=f"JD skill extraction for {jd.id} (attempt {attempt + 1}/2)",
            )
            break
        except GenerationError as exc:
            last_error = exc
    else:
        raise JDSkillsGenerationError(str(last_error)) from last_error

    return JDSkills(
        job_description_id=jd.id,
        generated_by_model=model_name,
        technical_skills=result.technical_skills,
        must_have_skills=_filter_must_have_skills(result.must_have_skills, result.technical_skills, jd.id),
        certifications=result.certifications,
        seniority_requirement=result.seniority_requirement,
        seniority_min_years=result.seniority_min_years,
        education_requirement=result.education_requirement,
    )


def _cache_key(jd: JobDescription, model_name: str) -> str:
    prompt_repr = repr(
        JD_SKILL_EXTRACTION_PROMPT.format_messages(title=jd.title, description=jd.raw_text)
    )
    digest_input = f"{prompt_repr}||{model_name}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()


def load_or_generate_jd_skills(
    jd: JobDescription,
    chain: Runnable,
    model_name: str,
    cache_path: Path,
) -> JDSkills:
    key = _cache_key(jd, model_name)

    with _cache_lock:
        cache = load_json_cache(cache_path, "JD skills cache")
        cached_entry = cache.get(jd.id)
        if cached_entry is not None and cached_entry.get("cache_key") == key:
            try:
                return JDSkills.model_validate(cached_entry["jd_skills"])
            except (KeyError, ValidationError) as exc:
                logger.warning("Discarding invalid cached JD skills for %s: %s", jd.id, exc)

    jd_skills = generate_jd_skills(jd, chain, model_name)

    with _cache_lock:
        cache = load_json_cache(cache_path, "JD skills cache")
        cache[jd.id] = {"cache_key": key, "jd_skills": jd_skills.model_dump()}
        save_json_cache(cache_path, cache)

    return jd_skills
