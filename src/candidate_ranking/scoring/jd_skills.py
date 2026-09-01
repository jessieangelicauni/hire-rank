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
            "You are an expert HR consultant extracting the technical skill vocabulary a job description asks for.\n\n"
            "Task:\n"
            "- Read the job title and description and identify every technology and technical subject explicitly mentioned.\n"
            "- Reason through the description section by section in the `reasoning` field before giving your final answer.\n"
            "- Note which text backs each skill you identify.\n"
            "- List the identified skills in the `technical_skills` field.\n\n"
            "Constraints:\n"
            "- Name each skill as the single atomic technology or subject it refers to.\n"
            "- Write each skill the way it would appear as a standalone item on a resume.\n"
            "- Do not phrase a skill as a description of proficiency, usage, or context.\n"
            "- Omit generic process or methodology phrases that do not name a specific technology or subject.\n"
            "- List required and nice-to-have skills together, without distinguishing between them.",
        ),
        ("human", "Job Title: {title}\n\nJob Description:\n{description}"),
    ]
)


class _GeneratedSkills(BaseModel):
    reasoning: str = Field(min_length=1)
    technical_skills: list[str] = Field(default_factory=list)


class JDSkillsGenerationError(Exception):
    pass


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
