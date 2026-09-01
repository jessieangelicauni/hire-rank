from __future__ import annotations

import hashlib
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, ValidationError
from pypdf import PdfReader

from candidate_ranking.generation import GenerationError, invoke_and_validate
from candidate_ranking.json_cache import load_json_cache, save_json_cache
from candidate_ranking.models import Candidate, slugify

logger = logging.getLogger(__name__)

_cache_lock = threading.Lock()


def _candidate_from_cache(cached_entry: dict) -> Candidate | None:
    try:
        return Candidate.model_validate(cached_entry["candidate"])
    except (KeyError, ValidationError) as exc:
        logger.warning("Discarding invalid cache entry: %s", exc)
        return None


def _parse_pdf(path: Path) -> tuple[str, int]:
    reader = PdfReader(str(path))
    pages_text = [page.extract_text(extraction_mode="layout") or "" for page in reader.pages]
    raw_text = re.sub(r"[ \t]{2,}", " ", "\n".join(pages_text)).strip()
    return raw_text, len(reader.pages)


def _cv_prompt_cache_key(prompt: ChatPromptTemplate, candidate: Candidate, model_name: str) -> str:
    prompt_repr = repr(prompt.format_messages(cv_text=candidate.raw_text))
    digest_input = f"{prompt_repr}||{model_name}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()


def _extract_field(
    candidate: Candidate,
    chain: Runnable,
    result_type: type[BaseModel],
    field_name: str,
    error_type: type[Exception],
    log_context_label: str,
):
    try:
        result = invoke_and_validate(
            chain,
            {"cv_text": candidate.raw_text},
            result_type,
            lambda _result: None,
            log_context=f"{log_context_label} for {candidate.id}",
        )
    except GenerationError as exc:
        raise error_type(str(exc)) from exc

    return getattr(result, field_name)


def _load_or_extract_field(
    candidate: Candidate,
    cache_path: Path,
    cache_key: str,
    cache_label: str,
    field_name: str,
    extract: Callable[[], object],
):
    with _cache_lock:
        cache = load_json_cache(cache_path, cache_label)
        cached_entry = cache.get(candidate.id)
        if cached_entry is not None and cached_entry.get("cache_key") == cache_key:
            return cached_entry[field_name]

    value = extract()

    with _cache_lock:
        cache = load_json_cache(cache_path, cache_label)
        cache[candidate.id] = {"cache_key": cache_key, field_name: value}
        save_json_cache(cache_path, cache)

    return value


def load_candidates(cv_dir: Path, cache_path: Path) -> list[Candidate]:
    cache = load_json_cache(cache_path, "CV cache")
    candidates: list[Candidate] = []
    cache_dirty = False

    for path in sorted(cv_dir.glob("*.pdf")):
        stat = path.stat()
        cache_key = path.name
        cached_entry = cache.get(cache_key)

        if (
            cached_entry is not None
            and cached_entry.get("mtime") == stat.st_mtime
            and cached_entry.get("size") == stat.st_size
        ):
            candidate_from_cache = _candidate_from_cache(cached_entry)
            if candidate_from_cache is not None:
                candidates.append(candidate_from_cache)
                continue

        try:
            raw_text, num_pages = _parse_pdf(path)
            candidate = Candidate(
                id=slugify(path.stem),
                source_path=str(path),
                raw_text=raw_text,
                num_pages=num_pages,
                char_count=len(raw_text),
                parse_status="ok",
                error_message=None,
            )
        except Exception as exc:
            logger.warning("Failed to parse CV %s: %s", path, exc)
            candidate = Candidate(
                id=slugify(path.stem),
                source_path=str(path),
                raw_text="",
                num_pages=0,
                char_count=0,
                parse_status="failed",
                error_message=str(exc),
            )

        candidates.append(candidate)
        if candidate.parse_status == "ok":
            cache[cache_key] = {
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "candidate": candidate.model_dump(),
            }
            cache_dirty = True

    if cache_dirty:
        save_json_cache(cache_path, cache)

    return candidates


SKILL_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert technical recruiter extracting a candidate's technical skills from their CV.\n\n"
            "Task:\n"
            "- Read the CV text and identify every technology and technical subject the candidate has direct, hands-on experience with.\n"
            "- Reason through the CV section by section in the `reasoning` field before giving your final answer.\n"
            "- Note which CV text backs each skill you identify.\n"
            "- List the identified skills in the `skills` field.\n\n"
            "Constraints:\n"
            "- Name each skill as the single atomic technology or subject it refers to.\n"
            "- Write each skill the way it would appear as a standalone item on a resume.\n"
            "- Do not phrase a skill as a description of proficiency, usage, or context.\n"
            "- Include a skill only if a specific technology or subject naming it appears in the CV text.\n"
            "- Never infer a skill from general job duties, or unrelated context.\n"
        ),
        ("human", "Candidate CV:\n{cv_text}"),
    ]
)


class _ExtractedSkills(BaseModel):
    reasoning: str = Field(min_length=1)
    skills: list[str]


class SkillExtractionError(Exception):
    pass


def build_skill_extraction_chain(llm: BaseChatModel) -> Runnable:
    return SKILL_EXTRACTION_PROMPT | llm.with_structured_output(_ExtractedSkills)


def extract_cv_skills(candidate: Candidate, chain: Runnable) -> list[str]:
    return _extract_field(candidate, chain, _ExtractedSkills, "skills", SkillExtractionError, "Skill extraction")


def load_or_extract_cv_skills(candidate: Candidate, chain: Runnable, model_name: str, cache_path: Path) -> list[str]:
    key = _cv_prompt_cache_key(SKILL_EXTRACTION_PROMPT, candidate, model_name)
    return _load_or_extract_field(
        candidate,
        cache_path,
        key,
        "CV skill cache",
        "skills",
        lambda: extract_cv_skills(candidate, chain),
    )


def _skills_for_candidate(candidate: Candidate, chain: Runnable, model_name: str, cache_path: Path) -> Candidate:
    if candidate.parse_status != "ok":
        return candidate
    try:
        skills = load_or_extract_cv_skills(candidate, chain, model_name, cache_path)
    except SkillExtractionError as exc:
        logger.warning("Skill extraction failed for %s, leaving skills empty: %s", candidate.id, exc)
        skills = []
    return candidate.model_copy(update={"skills": skills})


def enrich_candidates_with_skills(
    candidates: list[Candidate], chain: Runnable, model_name: str, cache_path: Path, max_workers: int = 1
) -> list[Candidate]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            executor.map(
                lambda candidate: _skills_for_candidate(candidate, chain, model_name, cache_path), candidates
            )
        )


NAME_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert technical recruiter identifying a candidate's full name from their CV.\n\n"
            "Task:\n"
            "- Locate the section of the CV that states the candidate's identity.\n"
            "- Reason in the `reasoning` field whether that section names a person, not a company or document title.\n"
            "- Write the candidate's name in the `name` field.\n\n"
            "Constraints:\n"
            "- Extract the name exactly as written in the CV text.\n"
            "- Do not change the name's spelling or capitalization.\n"
            "- Never derive the name from an email address, filename, or other indirect signal.\n"
            "- Set `name` to null if no full personal name is stated in the CV text.",
        ),
        ("human", "Candidate CV:\n{cv_text}"),
    ]
)


class _ExtractedName(BaseModel):
    reasoning: str = Field(min_length=1)
    name: str | None = None


class NameExtractionError(Exception):
    pass


def build_name_extraction_chain(llm: BaseChatModel) -> Runnable:
    return NAME_EXTRACTION_PROMPT | llm.with_structured_output(_ExtractedName)


def extract_candidate_name(candidate: Candidate, chain: Runnable) -> str | None:
    return _extract_field(candidate, chain, _ExtractedName, "name", NameExtractionError, "Name extraction")


def _normalize_name_casing(name: str) -> str:
    return name.title() if name.isupper() else name


def load_or_extract_candidate_name(
    candidate: Candidate, chain: Runnable, model_name: str, cache_path: Path
) -> str | None:
    key = _cv_prompt_cache_key(NAME_EXTRACTION_PROMPT, candidate, model_name)
    name = _load_or_extract_field(
        candidate,
        cache_path,
        key,
        "CV name cache",
        "name",
        lambda: extract_candidate_name(candidate, chain),
    )
    return _normalize_name_casing(name) if name else name
