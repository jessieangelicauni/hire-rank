from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, ValidationError

from candidate_ranking.generation import GenerationError, invoke_and_validate
from candidate_ranking.models import Assessment, Candidate, JDSkills, JobDescription
from candidate_ranking.scoring.skills import bridge_negated_jd_skills_to_candidate, find_negated_skill_mentions

logger = logging.getLogger(__name__)

ASSESSMENT_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert HR assessor evaluating a candidate's CV against a job description.\n\n"
            "Task:\n"
            "- Compare the candidate's CV against the job description.\n"
            "- Reason through the comparison in the `reasoning` field before giving your final answer.\n"
            "- Work through the role's requirements one by one in the `reasoning` field.\n"
            "- Note what the CV explicitly supports or fails to support for each requirement.\n"
            "- Identify any further CV content directly relevant to the role, such as projects, prior roles, skills, or metrics.\n"
            "- Place each strength as its own item in the `strengths` field.\n"
            "- Place each weakness as its own item in the `weaknesses` field.\n"
            "- If a skill or tool appears in the CV only as a bare list item (e.g. in a skills/tools list, with no sentence describing how it was used), place it in the `additional_skills` field instead of `strengths` -- as the bare skill name only, with no invented verb, usage context, or outcome.\n\n"
            "Constraints:\n"
            "- Every strength, weakness, and additional skill must be a direct, factual statement grounded in the CV text.\n"
            "- Do not speculate, infer, or state anything not present in the CV.\n"
            "- `strengths` is for facts the CV states with sentence-level context (a role, a project, a described outcome) -- a bare skill-list mention with no such context belongs in `additional_skills`, never in `strengths`.\n"
            "- Do not combine two separate CV statements (e.g. two different bullet points, achievements, or metrics) into a single strength -- state each fact as specifically, and only as jointly, as the CV itself states it.\n"
            "- Every specific tool, skill, or concept you name inside a strength or weakness (including ones listed after \"including\"/\"such as\") must appear verbatim in the CV -- never add a related but unlisted one to round out the list.\n"
            "- Do not attach a purpose, use case, or extra detail to a fact unless the CV states that detail together with it -- e.g. if the CV lists \"Python\" and, separately and unconnected, describes data work elsewhere, do not write \"Python for data processing\" as if the CV connected the two.\n"
            "- Never hedge or admit uncertainty in a strength or weakness (e.g. \"implied by\", \"not explicitly stated but\", \"likely\", \"presumably\") -- if a fact can't be stated as directly and plainly supported by the CV, leave it out entirely.\n"
            "- State a relevant gap as a weakness plainly and directly.\n"
            "- Do not soften or omit a gap that is relevant to the role.\n"
            "- Do not state the same fact as both a strength and a weakness.\n"
            "- Include every strength, weakness, and additional skill the CV supports for this role.\n"
            "- Do not pad the list with restatements or trivial detail.\n"
            "- Before stating a weakness that the candidate lacks a skill, check it against the `Candidate's identified skills` list below -- never claim a skill is absent or not mentioned if it appears in that list. This applies even when the skill only appears as a bare list item with no sentence-level context: a skill placed in `additional_skills` for lacking context is still a skill the candidate has, not a gap -- do not also write it up as a weakness under softened phrasing such as \"lacks specific/detailed experience with X\", \"no explicit mention of X\", \"only listed without detailed context\", or similar. A skill in the candidate's identified skills list is either a strength (with context) or an additional skill (without context); it is never also a weakness.\n"
            "- `strengths` must contain at least one item.\n"
            "- `weaknesses` may be empty only if the CV shows no relevant gap for this role.",
        ),
        (
            "human",
            "Job Title: {job_title}\n\nJob Description:\n{job_description}\n\n"
            "Candidate's identified skills: {candidate_skills}\n\n"
            "Candidate CV:\n{cv_text}{retry_feedback}",
        ),
    ]
)


class _GeneratedAssessment(BaseModel):
    reasoning: str = Field(min_length=1)
    strengths: list[str] = Field(min_length=1)
    weaknesses: list[str] = Field(default_factory=list)
    additional_skills: list[str] = Field(default_factory=list)


class AssessmentGenerationError(Exception):
    pass


ASSESSMENT_SCOPE_VERSION = "strengths-weaknesses-additional_skills"


def build_assessment_chain(llm: BaseChatModel) -> Runnable:
    return ASSESSMENT_GENERATION_PROMPT | llm.with_structured_output(_GeneratedAssessment)


def _format_candidate_skills(candidate: Candidate) -> str:
    return ", ".join(candidate.skills) if candidate.skills else "(none extracted)"


def _find_contradictions(
    weaknesses: list[str],
    candidate_skills: list[str],
    jd_technical_skills: list[str] | None = None,
    skill_embedder: Callable[[list[str]], np.ndarray] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Returns (contradicted candidate-skill names, JD-skill -> candidate-skill bridge map)."""
    if not candidate_skills:
        return [], {}
    contradicted: set[str] = set()
    for weakness in weaknesses:
        contradicted.update(find_negated_skill_mentions(weakness, candidate_skills))
    bridged: dict[str, str] = {}
    if jd_technical_skills and skill_embedder is not None:
        bridged = bridge_negated_jd_skills_to_candidate(
            weaknesses, candidate_skills, jd_technical_skills, skill_embedder
        )
        contradicted.update(bridged.values())
    return sorted(contradicted), bridged


def _drop_contradicting_weaknesses(
    weaknesses: list[str],
    candidate_skills: list[str],
    bridged_jd_skill_names: list[str],
) -> list[str]:
    return [
        w
        for w in weaknesses
        if not find_negated_skill_mentions(w, candidate_skills)
        and not find_negated_skill_mentions(w, bridged_jd_skill_names)
    ]


def _build_retry_feedback(contradicted_skills: list[str]) -> str:
    return (
        "\n\nYour previous answer incorrectly claimed the candidate lacks the following "
        f"skill(s), which are explicitly listed in `Candidate's identified skills` above: "
        f"{', '.join(contradicted_skills)}. Remove these weakness items entirely -- do not "
        "keep them under softened wording (e.g. \"lacks specific/detailed experience with "
        "X\", \"only listed without context\"), since that is still the same incorrect claim. "
        "If any of these skills only appear as a bare list item with no sentence-level "
        "context in the CV, that belongs in `additional_skills`, not in `weaknesses`."
    )


def generate_assessment(
    jd: JobDescription,
    candidate: Candidate,
    chain: Runnable,
    model_name: str,
    jd_skills: JDSkills | None = None,
    skill_embedder: Callable[[list[str]], np.ndarray] | None = None,
) -> Assessment:
    jd_technical_skills = jd_skills.technical_skills if jd_skills is not None else None
    retry_feedback = ""
    contradicted: list[str] = []
    bridged: dict[str, str] = {}
    for attempt in range(2):
        try:
            result = invoke_and_validate(
                chain,
                {
                    "job_title": jd.title,
                    "job_description": jd.raw_text,
                    "candidate_skills": _format_candidate_skills(candidate),
                    "cv_text": candidate.raw_text,
                    "retry_feedback": retry_feedback,
                },
                _GeneratedAssessment,
                lambda _result: None,
                log_context=f"Assessment generation for {jd.id}/{candidate.id} (attempt {attempt + 1}/2)",
            )
        except GenerationError as exc:
            raise AssessmentGenerationError(str(exc)) from exc

        contradicted, bridged = _find_contradictions(
            result.weaknesses, candidate.skills, jd_technical_skills, skill_embedder
        )
        if not contradicted:
            break
        retry_feedback = _build_retry_feedback(contradicted)

    weaknesses = result.weaknesses
    if contradicted:
        logger.warning(
            "Assessment for %s/%s still contradicts the candidate's own skills after retry: %s "
            "-- dropping the contradicting weakness item(s) rather than keeping them",
            jd.id, candidate.id, contradicted,
        )
        weaknesses = _drop_contradicting_weaknesses(weaknesses, candidate.skills, list(bridged))

    return Assessment(
        job_description_id=jd.id,
        candidate_id=candidate.id,
        generated_by_model=model_name,
        strengths=result.strengths,
        weaknesses=weaknesses,
        additional_skills=result.additional_skills,
        reasoning=result.reasoning,
    )


def filter_assessable_candidates(candidates: list[Candidate]) -> list[Candidate]:
    assessable = [c for c in candidates if c.parse_status == "ok"]
    excluded_count = len(candidates) - len(assessable)
    if excluded_count > 0:
        logger.warning(
            "Excluded %d candidate(s) with failed CV parsing from the assessable pool",
            excluded_count,
        )
    return assessable


def _assessment_cache_path(cache_dir: Path, jd_id: str, candidate_id: str) -> Path:
    return cache_dir / "assessments" / jd_id / f"{candidate_id}.json"


def _assessment_cache_key(
    jd: JobDescription,
    candidate: Candidate,
    model_name: str,
) -> str:
    prompt_repr = repr(
        ASSESSMENT_GENERATION_PROMPT.format_messages(
            job_title=jd.title, job_description=jd.raw_text,
            candidate_skills=_format_candidate_skills(candidate), cv_text=candidate.raw_text,
            retry_feedback="",
        )
    )
    digest_input = f"{prompt_repr}||{model_name}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()


def _read_cached_assessment(path: Path, expected_key: str) -> Assessment | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Discarding unreadable assessment cache %s: %s", path, exc)
        return None
    if not isinstance(data, dict) or data.get("cache_key") != expected_key:
        return None
    try:
        return Assessment.model_validate(data["assessment"])
    except (KeyError, ValidationError) as exc:
        logger.warning("Discarding invalid cached assessment %s: %s", path, exc)
        return None


def _write_cached_assessment(path: Path, cache_key: str, assessment: Assessment) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"cache_key": cache_key, "assessment": assessment.model_dump()}, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_or_generate_assessment(
    jd: JobDescription,
    candidate: Candidate,
    chain: Runnable,
    model_name: str,
    cache_dir: Path,
    jd_skills: JDSkills | None = None,
    skill_embedder: Callable[[list[str]], np.ndarray] | None = None,
) -> Assessment:
    path = _assessment_cache_path(cache_dir, jd.id, candidate.id)
    key = _assessment_cache_key(jd, candidate, model_name)

    cached = _read_cached_assessment(path, key)
    if cached is not None:
        return cached

    assessment = generate_assessment(jd, candidate, chain, model_name, jd_skills, skill_embedder)
    _write_cached_assessment(path, key, assessment)
    return assessment
