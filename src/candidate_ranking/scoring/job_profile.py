from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, ValidationError

from candidate_ranking.utils import GenerationError, invoke_and_validate, load_json_cache, save_json_cache
from candidate_ranking.models import JobDescription, JobProfile

logger = logging.getLogger(__name__)

_cache_lock = threading.Lock()

JOB_PROFILE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior technical recruiter extracting a structured job profile "
            "from a job description, for candidate evaluation and skill matching.\n\n"
            "In `reasoning` (max 3 sentences), note anything non-obvious before answering.\n\n"
            "Skills → list every named technology, split into:\n"
            "- `must_have_skills`: the core the role is built around — main language(s) plus "
            "1-3 defining frameworks or platforms. Infra/tooling (Docker, cloud, CI/CD) is not "
            "core unless it IS the role.\n"
            "- `nice_to_have_skills`: every other named technology. Most skills belong here.\n\n"
            "Non-skill requirements:\n"
            "- `education_requirement`: one sentence summary if the JD states one, else null.\n"
            "- `seniority_requirement`: one sentence summary of seniority/years-of-experience "
            "if stated, else null.\n"
            "- `other_requirements`: anything else the JD requires or prefers (certifications, "
            "languages, domain knowledge, etc.), each as a short phrase.",
        ),
        ("human", "Job Title: {title}\n\nJob Description:\n{description}"),
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


def build_job_profile_chain(llm: BaseChatModel) -> Runnable:
    return JOB_PROFILE_PROMPT | llm.with_structured_output(_JobProfileOutput)


def generate_job_profile(
    jd: JobDescription,
    chain: Runnable,
    model_name: str,
) -> JobProfile:
    try:
        result = invoke_and_validate(
            chain,
            {"title": jd.title, "description": jd.raw_text},
            _JobProfileOutput,
            lambda _: None,
            log_context=f"Job profile extraction for {jd.id}",
        )
    except GenerationError as exc:
        raise JobProfileGenerationError(str(exc)) from exc

    return JobProfile(
        job_description_id=jd.id,
        generated_by_model=model_name,
        must_have_skills=result.must_have_skills,
        nice_to_have_skills=result.nice_to_have_skills,
        education_requirement=result.education_requirement,
        seniority_requirement=result.seniority_requirement,
        other_requirements=result.other_requirements,
    )


def _cache_key(jd: JobDescription, model_name: str) -> str:
    prompt_repr = repr(
        JOB_PROFILE_PROMPT.format_messages(title=jd.title, description=jd.raw_text)
    )
    return hashlib.sha256(f"{prompt_repr}||{model_name}".encode("utf-8")).hexdigest()


def load_or_generate_job_profile(
    jd: JobDescription,
    chain: Runnable,
    model_name: str,
    cache_path: Path,
) -> JobProfile:
    key = _cache_key(jd, model_name)

    with _cache_lock:
        cache = load_json_cache(cache_path, "job profile cache")
        cached_entry = cache.get(jd.id)
        if cached_entry is not None and cached_entry.get("cache_key") == key:
            try:
                return JobProfile.model_validate(cached_entry["job_profile"])
            except (KeyError, ValidationError) as exc:
                logger.warning("Discarding invalid cached job profile for %s: %s", jd.id, exc)

    job_profile = generate_job_profile(jd, chain, model_name)

    with _cache_lock:
        cache = load_json_cache(cache_path, "job profile cache")
        cache[jd.id] = {"cache_key": key, "job_profile": job_profile.model_dump()}
        save_json_cache(cache_path, cache)

    return job_profile
