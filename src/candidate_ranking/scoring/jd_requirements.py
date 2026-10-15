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
from candidate_ranking.models import JDRequirement, JDRequirements, JobDescription

logger = logging.getLogger(__name__)

_cache_lock = threading.Lock()

JD_REQUIREMENTS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior technical recruiter parsing a job description.\n\n"
            "Extract all requirements and classify each as:\n"
            "- MUST_HAVE: non-negotiable; absence alone disqualifies a candidate\n"
            "- NICE_TO_HAVE: preferred but absence does not disqualify\n\n"
            "Rules:\n"
            "- Be conservative with MUST_HAVE — only use it if the JD treats the requirement as essential\n"
            "- 'Preferred', 'a plus', 'bonus' in JD text → NICE_TO_HAVE at most\n"
            "- Minimum years of experience and seniority level → MUST_HAVE if stated\n"
            "- Degree/education → MUST_HAVE only if explicitly required, not just preferred\n"
            "- Include context: the brief reason why you classified it this way",
        ),
        ("human", "Job Title: {title}\n\nJob Description:\n{description}"),
    ]
)


class _RequirementItem(BaseModel):
    requirement: str
    context: str


class _JDRequirementsOutput(BaseModel):
    must_have: list[_RequirementItem] = Field(default_factory=list)
    nice_to_have: list[_RequirementItem] = Field(default_factory=list)


class JDRequirementsGenerationError(Exception):
    pass


def build_jd_requirements_chain(llm: BaseChatModel) -> Runnable:
    return JD_REQUIREMENTS_PROMPT | llm.with_structured_output(_JDRequirementsOutput)


def generate_jd_requirements(
    jd: JobDescription,
    chain: Runnable,
    model_name: str,
) -> JDRequirements:
    try:
        result = invoke_and_validate(
            chain,
            {"title": jd.title, "description": jd.raw_text},
            _JDRequirementsOutput,
            lambda _: None,
            log_context=f"JD requirements extraction for {jd.id}",
        )
    except GenerationError as exc:
        raise JDRequirementsGenerationError(str(exc)) from exc

    return JDRequirements(
        job_description_id=jd.id,
        generated_by_model=model_name,
        must_have=[JDRequirement(requirement=r.requirement, context=r.context) for r in result.must_have],
        nice_to_have=[JDRequirement(requirement=r.requirement, context=r.context) for r in result.nice_to_have],
    )


def _cache_key(jd: JobDescription, model_name: str) -> str:
    prompt_repr = repr(
        JD_REQUIREMENTS_PROMPT.format_messages(title=jd.title, description=jd.raw_text)
    )
    return hashlib.sha256(f"{prompt_repr}||{model_name}".encode("utf-8")).hexdigest()


def load_or_generate_jd_requirements(
    jd: JobDescription,
    chain: Runnable,
    model_name: str,
    cache_path: Path,
) -> JDRequirements:
    key = _cache_key(jd, model_name)

    with _cache_lock:
        cache = load_json_cache(cache_path, "JD requirements cache")
        cached_entry = cache.get(jd.id)
        if cached_entry is not None and cached_entry.get("cache_key") == key:
            try:
                return JDRequirements.model_validate(cached_entry["jd_requirements"])
            except (KeyError, ValidationError) as exc:
                logger.warning("Discarding invalid cached JD requirements for %s: %s", jd.id, exc)

    jd_requirements = generate_jd_requirements(jd, chain, model_name)

    with _cache_lock:
        cache = load_json_cache(cache_path, "JD requirements cache")
        cache[jd.id] = {"cache_key": key, "jd_requirements": jd_requirements.model_dump()}
        save_json_cache(cache_path, cache)

    return jd_requirements
