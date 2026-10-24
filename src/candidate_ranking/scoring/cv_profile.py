from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, ValidationError

from candidate_ranking.models import Candidate, CandidateProfile, SkillEntry
from candidate_ranking.utils import GenerationError, invoke_and_validate, load_json_cache, save_json_cache

logger = logging.getLogger(__name__)

_cache_lock = threading.Lock()

CV_PROFILE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert technical recruiter extracting structured data from a CV.\n\n"
            "List every concrete technical skill, tool, technology, language, or framework "
            "mentioned anywhere in the CV, once each. For each, write `context`: the exact "
            "verbatim sentence or phrase where it's mentioned -- never invented, paraphrased, "
            "or summarized. If a skill appears in multiple places, pick the mention that most "
            "substantively demonstrates it (a sentence showing it used in a project, over a "
            "bare skills-list entry).",
        ),
        ("human", "CV:\n{cv_text}"),
    ]
)


class _SkillOutput(BaseModel):
    name: str
    context: str


class _CvProfileOutput(BaseModel):
    skills: list[_SkillOutput] = Field(default_factory=list)


class CvProfileGenerationError(Exception):
    pass


def build_cv_profile_chain(llm: BaseChatModel) -> Runnable:
    return CV_PROFILE_PROMPT | llm.with_structured_output(_CvProfileOutput)


def generate_cv_profile(candidate: Candidate, chain: Runnable) -> CandidateProfile:
    try:
        result = invoke_and_validate(
            chain,
            {"cv_text": candidate.raw_text},
            _CvProfileOutput,
            lambda _result: None,
            log_context=f"CV profile extraction for {candidate.id}",
        )
    except GenerationError as exc:
        raise CvProfileGenerationError(str(exc)) from exc

    return CandidateProfile(
        skills=[SkillEntry(**s.model_dump()) for s in result.skills],
    )


def _cache_key(candidate: Candidate, model_name: str) -> str:
    prompt_repr = repr(CV_PROFILE_PROMPT.format_messages(cv_text=candidate.raw_text))
    digest_input = f"{prompt_repr}||{model_name}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()


def load_or_generate_cv_profile(
    candidate: Candidate,
    chain: Runnable,
    model_name: str,
    cache_path: Path,
) -> CandidateProfile:
    key = _cache_key(candidate, model_name)

    with _cache_lock:
        cache = load_json_cache(cache_path, "CV profile cache")
        cached_entry = cache.get(candidate.id)
        if cached_entry is not None and cached_entry.get("cache_key") == key:
            try:
                return CandidateProfile.model_validate(cached_entry["profile"])
            except ValidationError as exc:
                logger.warning("Discarding invalid cached CV profile for %s: %s", candidate.id, exc)

    profile = generate_cv_profile(candidate, chain)

    with _cache_lock:
        cache = load_json_cache(cache_path, "CV profile cache")
        cache[candidate.id] = {"cache_key": key, "profile": profile.model_dump()}
        save_json_cache(cache_path, cache)

    return profile
