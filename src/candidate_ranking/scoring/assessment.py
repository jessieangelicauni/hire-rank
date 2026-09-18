from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from pydantic import ValidationError

from candidate_ranking.models import Assessment, Candidate, JDSkills, JobDescription
from candidate_ranking.scoring.jev_client import JEV_MODEL_ID, JevAnswer, JevClient, JevClientError, JevQuestion

logger = logging.getLogger(__name__)

JEV_MODEL_NAME = JEV_MODEL_ID

ASSESSMENT_SCOPE_VERSION = "jev-score-recommendation-v1"

_REQUIREMENT_KEY_PREFIX = "requirement::"
_OVERALL_FIT_KEY = "overall_fit_score"
_RECOMMENDATION_KEY = "overall_recommendation"
_MIN_QUALIFICATIONS_KEY = "meets_min_qualifications"
_RETRY_ON_LOW_CONFIDENCE_KEYS = (_OVERALL_FIT_KEY, _RECOMMENDATION_KEY, _MIN_QUALIFICATIONS_KEY)
_CONFIDENCE_RETRY_THRESHOLD = 0.5

_OVERALL_FIT_CRITERIA = [
    "Shows almost no relevant skills or experience for this role",
    "Has some relevant skills but significant gaps in the role's core requirements",
    "Meets roughly half of the role's core requirements with moderate relevant experience",
    "Meets most of the role's core requirements with solid relevant experience",
    "Meets or exceeds nearly all of the role's core requirements with strong, demonstrated experience",
]
_REQUIREMENT_FIT_CRITERIA = [
    "Not mentioned anywhere in the CV",
    "Mentioned only as a bare skill-list item, with no sentence describing real usage",
    "Used with some described context, but limited depth, duration, or unclear proficiency",
    "Used substantively in a real role or project with clear responsibility",
    "Extensively and expertly demonstrated, with strong measurable outcomes or deep ownership",
]
assert len(_OVERALL_FIT_CRITERIA) == len(_REQUIREMENT_FIT_CRITERIA)
_SCORE_MAX_INDEX = len(_OVERALL_FIT_CRITERIA) - 1


class AssessmentGenerationError(Exception):
    pass


def _requirement_question_key(requirement: str) -> str:
    return f"{_REQUIREMENT_KEY_PREFIX}{requirement}"


def _score_to_percent(raw_score: float) -> float:
    return max(0.0, min(100.0, raw_score * (100.0 / _SCORE_MAX_INDEX)))


def _format_candidate_skills(candidate: Candidate) -> str:
    return ", ".join(candidate.skills) if candidate.skills else "(none extracted)"


def _build_state(jd: JobDescription, candidate: Candidate, low_confidence_note: str = "") -> str:
    return (
        f"Job Title: {jd.title}\n\n"
        f"Job Description:\n{jd.raw_text}\n\n"
        f"Candidate's identified skills: {_format_candidate_skills(candidate)}\n\n"
        f"Candidate CV:\n{candidate.raw_text}{low_confidence_note}"
    )


def _build_questions(jd_technical_skills: list[str] | None) -> list[JevQuestion]:
    questions = [
        JevQuestion(
            key=_OVERALL_FIT_KEY,
            kind="score",
            instructions="How well does this candidate's CV fit the job description overall?",
            criteria=_OVERALL_FIT_CRITERIA,
        ),
        JevQuestion(
            key=_RECOMMENDATION_KEY,
            kind="choice",
            instructions="What is the hiring recommendation for this candidate against this job description?",
            criteria={
                "hire": "Candidate clearly meets or exceeds the role's requirements",
                "maybe": "Candidate partially meets the role's requirements",
                "no": "Candidate does not meet the role's requirements",
            },
        ),
        JevQuestion(
            key=_MIN_QUALIFICATIONS_KEY,
            kind="noul",
            instructions="Does the candidate meet the job description's minimum qualifications?",
            criteria={
                "true": "Meets every minimum qualification stated in the job description",
                "false": "Fails at least one minimum qualification stated in the job description",
            },
        ),
    ]
    for requirement in jd_technical_skills or []:
        questions.append(
            JevQuestion(
                key=_requirement_question_key(requirement),
                kind="score",
                instructions=f"How well does the candidate's CV support the requirement '{requirement}'?",
                criteria=_REQUIREMENT_FIT_CRITERIA,
            )
        )
    return questions


def _answers_to_assessment(
    jd: JobDescription, candidate: Candidate, model_name: str, answers: list[JevAnswer]
) -> Assessment:
    by_key = {a.key: a for a in answers}
    confidence = {a.key: a.confidence for a in answers}
    requirement_scores = {
        key[len(_REQUIREMENT_KEY_PREFIX):]: _score_to_percent(a.value)
        for key, a in by_key.items()
        if key.startswith(_REQUIREMENT_KEY_PREFIX)
    }
    return Assessment(
        job_description_id=jd.id,
        candidate_id=candidate.id,
        generated_by_model=model_name,
        overall_fit_score=_score_to_percent(by_key[_OVERALL_FIT_KEY].value),
        overall_recommendation=by_key[_RECOMMENDATION_KEY].value,
        meets_min_qualifications=by_key[_MIN_QUALIFICATIONS_KEY].value,
        requirement_scores=requirement_scores,
        confidence=confidence,
    )


def generate_assessment(
    jd: JobDescription,
    candidate: Candidate,
    jev_client: JevClient,
    model_name: str = JEV_MODEL_NAME,
    jd_skills: JDSkills | None = None,
) -> Assessment:
    jd_technical_skills = jd_skills.technical_skills if jd_skills is not None else None
    questions = _build_questions(jd_technical_skills)

    low_confidence_note = ""
    assessment: Assessment | None = None
    for _attempt in range(2):
        state = _build_state(jd, candidate, low_confidence_note)
        try:
            answers = jev_client.evaluate(state, questions)
        except JevClientError as exc:
            raise AssessmentGenerationError(str(exc)) from exc

        try:
            assessment = _answers_to_assessment(jd, candidate, model_name, answers)
        except (KeyError, ValidationError) as exc:
            raise AssessmentGenerationError(
                f"Jev response for {jd.id}/{candidate.id} was unusable: {exc}"
            ) from exc
        low_confidence_keys = [
            key for key in _RETRY_ON_LOW_CONFIDENCE_KEYS
            if assessment.confidence.get(key, 1.0) < _CONFIDENCE_RETRY_THRESHOLD
        ]
        if not low_confidence_keys:
            break
        logger.warning(
            "Low-confidence Jev answer(s) for %s/%s: %s -- retrying once",
            jd.id, candidate.id, low_confidence_keys,
        )
        low_confidence_note = (
            "\n\nNote: a previous evaluation of this same candidate/job pair returned "
            f"low-confidence answers for: {', '.join(low_confidence_keys)}. Re-evaluate carefully."
        )

    if assessment is None:
        raise AssessmentGenerationError(
            f"Jev evaluation for {jd.id}/{candidate.id} produced no assessment"
        )
    return assessment


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
    jd_skills: JDSkills | None = None,
) -> str:
    jd_technical_skills = jd_skills.technical_skills if jd_skills is not None else None
    state = _build_state(jd, candidate)
    questions_repr = repr([q.model_dump() for q in _build_questions(jd_technical_skills)])
    digest_input = f"{state}||{questions_repr}||{model_name}".encode("utf-8")
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
    jev_client: JevClient,
    model_name: str,
    cache_dir: Path,
    jd_skills: JDSkills | None = None,
) -> Assessment:
    path = _assessment_cache_path(cache_dir, jd.id, candidate.id)
    key = _assessment_cache_key(jd, candidate, model_name, jd_skills)

    cached = _read_cached_assessment(path, key)
    if cached is not None:
        return cached

    assessment = generate_assessment(jd, candidate, jev_client, model_name, jd_skills)
    _write_cached_assessment(path, key, assessment)
    return assessment
