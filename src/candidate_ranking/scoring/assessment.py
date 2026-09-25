from __future__ import annotations

import hashlib
import json
import logging
import statistics
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from candidate_ranking.models import Assessment, Candidate, JDSkills, JobDescription
from candidate_ranking.scoring.jev_client import JEV_MODEL_ID, JevAnswer, JevClient, JevClientError, JevQuestion

logger = logging.getLogger(__name__)

JEV_MODEL_NAME = JEV_MODEL_ID

ASSESSMENT_SCOPE_VERSION = "jev-score-recommendation-v1"

_REQUIREMENT_KEY_PREFIX = "requirement::"
_CERTIFICATION_KEY_PREFIX = "certification::"
_RECOMMENDATION_KEY = "overall_recommendation"
_MIN_QUALIFICATIONS_KEY = "meets_min_qualifications"
_SENIORITY_YEARS_KEY = "seniority_years"
_EDUCATION_KEY = "education"
_RETRY_ON_LOW_CONFIDENCE_KEYS = (
    _RECOMMENDATION_KEY, _MIN_QUALIFICATIONS_KEY,
    _SENIORITY_YEARS_KEY, _EDUCATION_KEY,
)
_CONFIDENCE_RETRY_THRESHOLD = 0.5

DEFAULT_N_CALLS = 3

_REQUIREMENT_FIT_CRITERIA = [
    "Not mentioned anywhere in the CV",
    "Mentioned only as a bare skill-list item, with no sentence describing real usage",
    "Used with some described context, but limited depth, duration, or unclear proficiency",
    "Used substantively in a real role or project with clear responsibility",
    "Extensively and expertly demonstrated, with strong measurable outcomes or deep ownership",
]
_SENIORITY_YEARS_FIT_CRITERIA = [
    "No years of total experience stated or inferable from the CV",
    "Fewer than half the required years of total experience",
    "At least half but still below the required years of total experience",
    "Meets the required years of total experience",
    "Exceeds the required years of total experience",
]
_EDUCATION_FIT_CRITERIA = [
    "No relevant education mentioned",
    "Education mentioned but neither the field nor the degree level matches the requirement",
    "Matches the requirement on field or degree level, but not both",
    "Matches the requirement's field and degree level",
    "Exceeds the requirement (higher degree level) in the same or a closely related field",
]
assert (
    len(_REQUIREMENT_FIT_CRITERIA) == len(_SENIORITY_YEARS_FIT_CRITERIA)
    == len(_EDUCATION_FIT_CRITERIA)
)
_SCORE_MAX_INDEX = len(_REQUIREMENT_FIT_CRITERIA) - 1


class AssessmentGenerationError(Exception):
    pass


def _requirement_question_key(requirement: str) -> str:
    return f"{_REQUIREMENT_KEY_PREFIX}{requirement}"


def _certification_question_key(certification: str) -> str:
    return f"{_CERTIFICATION_KEY_PREFIX}{certification}"


def _seniority_years_instructions(seniority_requirement: str, min_years: float) -> str:
    return (
        f"The job requires at least {min_years:g} years of total experience "
        f"(stated as: '{seniority_requirement}'). How many total years of experience does the "
        "candidate's CV show?"
    )


def _education_instructions(education_requirement: str) -> str:
    return (
        "How well does the candidate meet the job description's stated education "
        f"requirement: '{education_requirement}'?"
    )


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


def _build_questions(jd_skills: JDSkills | None) -> list[JevQuestion]:
    questions = [
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
    for requirement in jd_skills.technical_skills if jd_skills else []:
        questions.append(
            JevQuestion(
                key=_requirement_question_key(requirement),
                kind="score",
                instructions=f"How well does the candidate's CV support the requirement '{requirement}'?",
                criteria=_REQUIREMENT_FIT_CRITERIA,
            )
        )
    for certification in jd_skills.certifications if jd_skills else []:
        questions.append(
            JevQuestion(
                key=_certification_question_key(certification),
                kind="noul",
                instructions=f"Does the candidate's CV show possession of the certification '{certification}'?",
                criteria={
                    "true": f"The CV states the candidate holds the '{certification}' certification",
                    "false": f"The CV does not state the candidate holds the '{certification}' certification",
                },
            )
        )
    if jd_skills and jd_skills.seniority_requirement:
        if jd_skills.seniority_min_years is not None:
            questions.append(
                JevQuestion(
                    key=_SENIORITY_YEARS_KEY,
                    kind="score",
                    instructions=_seniority_years_instructions(
                        jd_skills.seniority_requirement, jd_skills.seniority_min_years
                    ),
                    criteria=_SENIORITY_YEARS_FIT_CRITERIA,
                )
            )
    if jd_skills and jd_skills.education_requirement:
        questions.append(
            JevQuestion(
                key=_EDUCATION_KEY,
                kind="score",
                instructions=_education_instructions(jd_skills.education_requirement),
                criteria=_EDUCATION_FIT_CRITERIA,
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
    certification_results = {
        key[len(_CERTIFICATION_KEY_PREFIX):]: a.value
        for key, a in by_key.items()
        if key.startswith(_CERTIFICATION_KEY_PREFIX)
    }
    requirement_probabilities = {
        key[len(_REQUIREMENT_KEY_PREFIX):]: a.probabilities
        for key, a in by_key.items()
        if key.startswith(_REQUIREMENT_KEY_PREFIX) and a.probabilities
    }
    return Assessment(
        job_description_id=jd.id,
        candidate_id=candidate.id,
        generated_by_model=model_name,
        overall_recommendation=by_key[_RECOMMENDATION_KEY].value,
        meets_min_qualifications=by_key[_MIN_QUALIFICATIONS_KEY].value,
        requirement_scores=requirement_scores,
        confidence=confidence,
        certification_results=certification_results,
        seniority_years_fit_score=(
            _score_to_percent(by_key[_SENIORITY_YEARS_KEY].value) if _SENIORITY_YEARS_KEY in by_key else None
        ),
        education_fit_score=_score_to_percent(by_key[_EDUCATION_KEY].value) if _EDUCATION_KEY in by_key else None,
        recommendation_probabilities=by_key[_RECOMMENDATION_KEY].probabilities or {},
        requirement_probabilities=requirement_probabilities,
        seniority_probabilities=(
            by_key[_SENIORITY_YEARS_KEY].probabilities if _SENIORITY_YEARS_KEY in by_key else None
        ),
        education_probabilities=by_key[_EDUCATION_KEY].probabilities if _EDUCATION_KEY in by_key else None,
    )


def _generate_single_assessment(
    jd: JobDescription,
    candidate: Candidate,
    jev_client: JevClient,
    model_name: str,
    questions: list[JevQuestion],
) -> tuple[Assessment, list[JevAnswer]]:
    low_confidence_note = ""
    assessment: Assessment | None = None
    answers: list[JevAnswer] = []
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
    return assessment, answers


def _aggregate_recommendation(calls: list[Assessment]) -> str:
    counts = Counter(a.overall_recommendation for a in calls)
    top_count = max(counts.values())
    tied = [label for label, count in counts.items() if count == top_count]
    if len(tied) == 1:
        return tied[0]

    def mean_confidence_for(label: str) -> float:
        confidences = [
            a.confidence.get(_RECOMMENDATION_KEY, 0.0) for a in calls if a.overall_recommendation == label
        ]
        return statistics.mean(confidences) if confidences else 0.0

    return max(tied, key=mean_confidence_for)


def _aggregate_meets_min_qualifications(calls: list[Assessment]) -> bool:
    return sum(a.meets_min_qualifications for a in calls) > len(calls) / 2


def _aggregate_mean_dict(calls: list[Assessment], field: str) -> dict[str, float]:
    keys = {key for a in calls for key in getattr(a, field)}
    return {key: statistics.mean(getattr(a, field)[key] for a in calls if key in getattr(a, field)) for key in keys}


def _aggregate_bool_dict(calls: list[Assessment], field: str) -> dict[str, bool]:
    keys = {key for a in calls for key in getattr(a, field)}
    result: dict[str, bool] = {}
    for key in keys:
        votes = [getattr(a, field)[key] for a in calls if key in getattr(a, field)]
        result[key] = sum(votes) > len(votes) / 2
    return result


def _aggregate_mean_optional(calls: list[Assessment], field: str) -> float | None:
    values = [v for a in calls if (v := getattr(a, field)) is not None]
    if not values:
        return None
    return statistics.mean(values)


def _aggregate_mean_nested_dict(calls: list[Assessment], field: str) -> dict[str, dict[str, float]]:
    outer_keys = {key for a in calls for key in getattr(a, field)}
    result: dict[str, dict[str, float]] = {}
    for outer_key in outer_keys:
        inner_dicts = [getattr(a, field)[outer_key] for a in calls if outer_key in getattr(a, field)]
        inner_keys = {k for d in inner_dicts for k in d}
        result[outer_key] = {
            inner_key: statistics.mean(d[inner_key] for d in inner_dicts if inner_key in d)
            for inner_key in inner_keys
        }
    return result


def _aggregate_mean_optional_dict(calls: list[Assessment], field: str) -> dict[str, float] | None:
    dicts = [d for a in calls if (d := getattr(a, field)) is not None]
    if not dicts:
        return None
    keys = {k for d in dicts for k in d}
    return {key: statistics.mean(d[key] for d in dicts if key in d) for key in keys}


def generate_assessment(
    jd: JobDescription,
    candidate: Candidate,
    jev_client: JevClient,
    model_name: str = JEV_MODEL_NAME,
    jd_skills: JDSkills | None = None,
    n_calls: int = DEFAULT_N_CALLS,
) -> Assessment:
    if n_calls < 1:
        raise ValueError(f"n_calls must be >= 1, got {n_calls}")

    questions = _build_questions(jd_skills)

    calls = [
        _generate_single_assessment(jd, candidate, jev_client, model_name, questions)[0]
        for _ in range(n_calls)
    ]
    if n_calls == 1:
        return calls[0]

    return Assessment(
        job_description_id=jd.id,
        candidate_id=candidate.id,
        generated_by_model=model_name,
        overall_recommendation=_aggregate_recommendation(calls),
        meets_min_qualifications=_aggregate_meets_min_qualifications(calls),
        requirement_scores=_aggregate_mean_dict(calls, "requirement_scores"),
        confidence=_aggregate_mean_dict(calls, "confidence"),
        certification_results=_aggregate_bool_dict(calls, "certification_results"),
        seniority_years_fit_score=_aggregate_mean_optional(calls, "seniority_years_fit_score"),
        education_fit_score=_aggregate_mean_optional(calls, "education_fit_score"),
        recommendation_probabilities=_aggregate_mean_dict(calls, "recommendation_probabilities"),
        requirement_probabilities=_aggregate_mean_nested_dict(calls, "requirement_probabilities"),
        seniority_probabilities=_aggregate_mean_optional_dict(calls, "seniority_probabilities"),
        education_probabilities=_aggregate_mean_optional_dict(calls, "education_probabilities"),
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
    jd_skills: JDSkills | None = None,
    n_calls: int = DEFAULT_N_CALLS,
) -> str:
    state = _build_state(jd, candidate)
    questions_repr = repr([q.model_dump() for q in _build_questions(jd_skills)])
    digest_input = f"{state}||{questions_repr}||{model_name}||n_calls={n_calls}".encode("utf-8")
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
    n_calls: int = DEFAULT_N_CALLS,
) -> Assessment:
    path = _assessment_cache_path(cache_dir, jd.id, candidate.id)
    key = _assessment_cache_key(jd, candidate, model_name, jd_skills, n_calls)

    cached = _read_cached_assessment(path, key)
    if cached is not None:
        return cached

    assessment = generate_assessment(jd, candidate, jev_client, model_name, jd_skills, n_calls)
    _write_cached_assessment(path, key, assessment)
    return assessment
