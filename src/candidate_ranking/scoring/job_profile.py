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
from candidate_ranking.models import EvaluationCriterion, JobProfile
from candidate_ranking.scoring.assessment import fallback_evaluation_criterion

logger = logging.getLogger(__name__)

_cache_lock = threading.Lock()


class _EvaluationCriterionOutput(BaseModel):
    label: str
    instructions: str
    strong: str
    partial: str
    none: str


class _RubricOutput(BaseModel):
    reasoning: str = Field(min_length=1)
    evaluation_criteria: list[_EvaluationCriterionOutput] = Field(default_factory=list)


RUBRIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior technical recruiter. Extract a structured evaluation rubric from "
            "the job posting below, to brief an AI evaluator that will score candidate CVs "
            "against it.\n\n"
            "Task: identify every concrete technical skill, tool, technology, language, "
            "framework, or platform worth evaluating a candidate on. Produce one "
            "`evaluation_criteria` entry per item.\n\n"
            "Atomicity is critical: each `label` must name exactly ONE skill, as it would "
            "appear verbatim on a resume (e.g. 'Python', 'AWS', 'Kubernetes'). Two violations "
            "to avoid, both common in real postings:\n"
            "  - a list disguised as one requirement -- 'Python or Java Programming', 'AWS/GCP', "
            "'AWS Services (EC2, EBS, S3, Lambda)' -- split each into its own entry, one per "
            "named skill, even if the posting phrases it as a single bullet.\n"
            "  - a category standing in for a skill -- 'Cloud-native technologies and frameworks', "
            "'Service resilience and scalability' -- replace it with the specific skills the "
            "posting actually names or clearly implies (e.g. Kubernetes, Docker, Helm), never "
            "the category label itself.\n\n"
            "For each entry:\n"
            "- `label`: the single skill's canonical name, 1-3 words.\n"
            "- `instructions`: role-specific guidance for judging evidence of this skill in a CV.\n"
            "- `strong`: what strong evidence for THIS skill in THIS role looks like.\n"
            "- `partial`: what partial evidence looks like (mentioned, but not clearly demonstrated).\n"
            "- `none`: what counts as no evidence.\n\n"
            "Calibrate `strong`/`partial`/`none` to this role's seniority and domain — do not "
            "reuse generic text across different criteria.\n\n"
            "In `reasoning` (max 3 sentences), note anything non-obvious about the posting "
            "before answering.",
        ),
        ("human", "Job Posting:\n{raw_text}"),
    ]
)


def build_rubric_chain(llm: BaseChatModel) -> Runnable:
    return RUBRIC_PROMPT | llm.with_structured_output(_RubricOutput)


class JobProfileGenerationError(Exception):
    pass


def generate_job_profile(
    jp: JobProfile,
    rubric_chain: Runnable,
    model_name: str,
) -> JobProfile:
    try:
        rubric = invoke_and_validate(
            rubric_chain,
            {"raw_text": jp.raw_text},
            _RubricOutput,
            lambda _: None,
            log_context=f"Evaluation-criteria generation for {jp.id}",
        )
        evaluation_criteria = [
            EvaluationCriterion(
                label=c.label, instructions=c.instructions,
                strong=c.strong, partial=c.partial, none=c.none,
            )
            for c in rubric.evaluation_criteria
        ]
    except GenerationError as exc:
        logger.warning("Evaluation-criteria generation for %s failed: %s", jp.id, exc)
        evaluation_criteria = []

    if not evaluation_criteria:
        logger.warning(
            "JobProfile %s: no evaluation criteria generated — falling back to one generic "
            "technical-skill criterion", jp.id,
        )
        evaluation_criteria = [fallback_evaluation_criterion(jp.id, "Relevant Technical Skills")]

    technical_skills: list[str] = []
    seen: set[str] = set()
    for criterion in evaluation_criteria:
        normalized = criterion.label.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            technical_skills.append(criterion.label)

    return JobProfile(
        id=jp.id,
        raw_text=jp.raw_text,
        technical_skills=technical_skills,
        evaluation_criteria=evaluation_criteria,
    )


_RUBRIC_LOGIC_VERSION = 5


def _cache_key(jp: JobProfile, model_name: str) -> str:
    rubric_prompt_repr = repr(RUBRIC_PROMPT.format_messages(raw_text=jp.raw_text))
    return hashlib.sha256(
        f"{rubric_prompt_repr}||{model_name}||v{_RUBRIC_LOGIC_VERSION}".encode("utf-8")
    ).hexdigest()


def load_or_generate_job_profile(
    jp: JobProfile,
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

    job_profile = generate_job_profile(jp, rubric_chain, model_name)

    with _cache_lock:
        cache = load_json_cache(cache_path, "job profile cache")
        cache[jp.id] = {"cache_key": key, "job_profile": job_profile.model_dump()}
        save_json_cache(cache_path, cache)

    return job_profile
