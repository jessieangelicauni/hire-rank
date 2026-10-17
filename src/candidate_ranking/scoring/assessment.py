from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import uuid
from pathlib import Path
from pydantic import ValidationError

from candidate_ranking.models import (
    Assessment, Candidate, JobProfile, RequirementEvaluation, SkillCriteria, TotalYearsCriteria,
    AdaptiveRequirementEvaluation,
)
from candidate_ranking.scoring.jev_client import JEV_MODEL_ID, JevClientError, JevQuestion
logger = logging.getLogger(__name__)

_cache_lock = threading.Lock()

_YEARS_TICKS = [0, 1, 2, 3, 4, 5]

_SKILL_EVIDENCE_CRITERIA: dict[str, str] = {
    "strong": (
        "Skill demonstrated in work experience with a specific system, project, or measurable outcome. "
        "E.g. 'Built REST APIs using Django at PT X for 3 years' -> STRONG for Django."
    ),
    "adequate": (
        "Skill mentioned in an experience context but without a named project or specific system. "
        "E.g. 'Used Python for data processing tasks' -> ADEQUATE."
    ),
    "weak": (
        "Skill appears ONLY in Skills / Summary / Certifications sections — never shown in any "
        "work experience or project. Listed but never demonstrated in action."
    ),
    "none": (
        "No mention of the skill anywhere in the CV. "
        "Check ALL sections (experience, projects, education, skills, summary, certifications, volunteer) "
        "before answering NONE. "
        "Remember implied skills: 'Django' implies 'Python'; 'Spring Boot' implies 'Java'; "
        "'React' implies 'JavaScript'; 'Kubernetes' implies 'Docker'."
    ),
}

_GENERIC_SKILL_INSTRUCTION_TEMPLATE = (
    "Role context: {jp_role}.\n"
    "Assess the evidence strength for the skill '{skill}' in this CV.\n\n"
    "EVIDENCE RULES (apply strictly and consistently across all roles and industries):\n"
    "  STRONG   — demonstrated in work experience with a specific system, project, or measurable outcome\n"
    "  ADEQUATE — mentioned in an experience context but without a named project or specific system\n"
    "  WEAK     — listed in Skills/Summary/Certifications only, never shown in any experience\n"
    "  NONE     — no mention anywhere; check ALL sections before answering NONE; "
    "consider implied skills (Django->Python, Spring Boot->Java, React->JavaScript)"
)

_TOTAL_YEARS_FALLBACK_INSTRUCTIONS = (
    "How many total years of RELEVANT experience does this candidate have for this role?\n\n"
    "COUNT: Only years where the candidate actively used skills directly applicable to this role. "
    "Exclude career breaks and education years. Parallel full-time roles count separately only if "
    "genuinely concurrent.\n\n"
    "CALIBRATION:\n"
    " 0 years  — student, bootcamp grad, or career changer with no relevant work history\n"
    " 1 year   — junior, first year in the field\n"
    " 2 years  — early-mid, building independence on domain tasks\n"
    " 3 years  — solid, works independently on most domain tasks\n"
    " 4 years  — experienced, has seen multiple project cycles\n"
    " 5+ years — very experienced, likely has led teams or major projects"
)


def fallback_skill_criteria(jp_id: str, skill: str) -> SkillCriteria:
    """Generic, role-agnostic skill-evidence rubric — used when the adaptive rubric
    (scoring/job_profile.py Call 2) doesn't have an entry for this skill, and as the
    defensive fallback here when a JobProfile has no skill_criteria at all (e.g. one
    cached before this feature existed)."""
    return SkillCriteria(
        skill=skill,
        instructions=_GENERIC_SKILL_INSTRUCTION_TEMPLATE.format(jp_role=jp_id, skill=skill),
        strong=_SKILL_EVIDENCE_CRITERIA["strong"],
        adequate=_SKILL_EVIDENCE_CRITERIA["adequate"],
        weak=_SKILL_EVIDENCE_CRITERIA["weak"],
        none=_SKILL_EVIDENCE_CRITERIA["none"],
    )


def fallback_total_years_criteria() -> TotalYearsCriteria:
    """Generic, role-agnostic years-of-experience calibration — same fallback role as
    fallback_skill_criteria, for total_years_criteria."""
    return TotalYearsCriteria(instructions=_TOTAL_YEARS_FALLBACK_INSTRUCTIONS)


def _score_to_int(value: float, ticks: list[int]) -> int:
    n = len(ticks) - 1
    clamped = max(0.0, min(float(n), float(value)))
    idx_low = int(clamped)
    idx_high = min(idx_low + 1, n)
    frac = clamped - idx_low
    return round(ticks[idx_low] * (1 - frac) + ticks[idx_high] * frac)


def _build_jev_questions(jp: JobProfile) -> list[JevQuestion]:
    questions: list[JevQuestion] = []

    criteria_by_skill = {c.skill: c for c in jp.skill_criteria}

    def _skill_question(key: str, skill: str) -> JevQuestion:
        criteria = criteria_by_skill.get(skill)
        if criteria is None:
            logger.warning(
                "JobProfile %s: no skill_criteria entry for %r — using generic fallback", jp.id, skill
            )
            criteria = fallback_skill_criteria(jp.id, skill)
        return JevQuestion(
            key=key,
            kind="choice",
            instructions=criteria.instructions,
            criteria={
                "strong": criteria.strong, "adequate": criteria.adequate,
                "weak": criteria.weak, "none": criteria.none,
            },
        )

    for i, skill in enumerate(jp.must_have_skills):
        questions.append(_skill_question(f"must_have_{i}_evidence_strength", skill))
    for i, skill in enumerate(jp.nice_to_have_skills):
        questions.append(_skill_question(f"nice_to_have_{i}_evidence_strength", skill))

    years_criteria = jp.total_years_criteria
    if years_criteria is None:
        logger.warning("JobProfile %s: no total_years_criteria — using generic fallback", jp.id)
        years_criteria = fallback_total_years_criteria()
    questions.append(JevQuestion(
        key="total_relevant_years",
        kind="score",
        instructions=years_criteria.instructions,
        criteria=["0 years", "1 year", "2 years", "3 years", "4 years", "5+ years"],
    ))

    for i, req in enumerate(jp.adaptive_requirements):
        questions.append(JevQuestion(
            key=f"adaptive_{i}_match_level",
            kind="choice",
            instructions=(
                f"Requirement ({req.source}): {req.requirement}\n\n"
                "Assess whether this candidate meets this requirement based on their CV."
            ),
            criteria={
                "meets": req.criteria_meets,
                "partially_meets": req.criteria_partially_meets,
                "does_not_meet": req.criteria_does_not_meet,
            },
        ))

    return questions


def _extract_bm25_evidence(cv_text: str, query: str, top_k: int = 3) -> str:
    from rank_bm25 import BM25Okapi
    sentences = [s.strip() for s in re.split(r"[.!\n|]+", cv_text) if len(s.strip()) > 15]
    if not sentences:
        return ""
    tokenized = [s.lower().split() for s in sentences]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    # A score of 0 means none of the query's tokens appear in that sentence at
    # all. When every sentence ties at 0 (the query term never appears
    # anywhere in the CV), the sort above falls back to original CV order —
    # i.e. the header/contact block — which is not evidence of anything.
    top_indices = sorted(i for i in ranked if scores[i] > 0)
    if not top_indices:
        return ""
    return " ... ".join(sentences[i] for i in top_indices)


class AssessmentGenerationError(Exception):
    pass


def _answers_to_assessment(
    answers: list,
    jp: JobProfile,
    candidate: Candidate,
) -> Assessment:
    by_key = {a.key: a for a in answers}
    cv = candidate.raw_text

    must_have_evals = []
    for i, skill in enumerate(jp.must_have_skills):
        answer = by_key[f"must_have_{i}_evidence_strength"]
        must_have_evals.append(RequirementEvaluation(
            requirement=skill,
            evidence=_extract_bm25_evidence(cv, skill),
            evidence_strength=answer.value,
            confidence=answer.confidence,
        ))

    nih_evals = []
    for i, skill in enumerate(jp.nice_to_have_skills):
        answer = by_key[f"nice_to_have_{i}_evidence_strength"]
        nih_evals.append(RequirementEvaluation(
            requirement=skill,
            evidence=_extract_bm25_evidence(cv, skill),
            evidence_strength=answer.value,
            confidence=answer.confidence,
        ))

    qualified = all(ev.evidence_strength != "none" for ev in must_have_evals)
    failed = [ev.requirement for ev in must_have_evals if ev.evidence_strength == "none"]
    disqualification_reason = (
        f"Failed must-have requirements: {', '.join(failed)}" if failed else None
    )

    total_years = _score_to_int(by_key["total_relevant_years"].value, _YEARS_TICKS)

    adaptive_evals = []
    for i, req in enumerate(jp.adaptive_requirements):
        answer = by_key[f"adaptive_{i}_match_level"]
        adaptive_evals.append(AdaptiveRequirementEvaluation(
            source=req.source,
            requirement=req.requirement,
            evidence=_extract_bm25_evidence(cv, req.requirement),
            match_level=answer.value,
            confidence=answer.confidence,
        ))

    return Assessment(
        job_profile_id=jp.id,
        candidate_id=candidate.id,
        generated_by_model=f"jev/{JEV_MODEL_ID}",
        must_have_evaluations=must_have_evals,
        nice_to_have_evaluations=nih_evals,
        total_relevant_years=total_years,
        adaptive_requirement_evaluations=adaptive_evals,
        qualified=qualified,
        disqualification_reason=disqualification_reason,
    )


def _assessment_cache_key(
    jp: JobProfile,
    candidate: Candidate,
    questions: list,
) -> str:
    questions_repr = json.dumps(
        [q.model_dump() for q in questions], sort_keys=True
    )
    digest_input = (
        f"{jp.id}||{jp.raw_text}||{candidate.id}||{candidate.raw_text}"
        f"||{questions_repr}||{JEV_MODEL_ID}"
    )
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


def generate_assessment(
    jp: JobProfile,
    candidate: Candidate,
    jev_client: "JevClient",
) -> Assessment:
    questions = _build_jev_questions(jp)

    try:
        answers = jev_client.evaluate(candidate.raw_text, questions)
    except JevClientError as exc:
        raise AssessmentGenerationError(
            f"Jev call failed for {jp.id}/{candidate.id}: {exc}"
        ) from exc

    return _answers_to_assessment(answers, jp, candidate)


def _assessment_cache_path(cache_dir: Path, jp_id: str, candidate_id: str) -> Path:
    return cache_dir / "assessments" / jp_id / f"{candidate_id}.json"


def load_or_generate_assessment(
    jp: JobProfile,
    candidate: Candidate,
    jev_client: "JevClient",
    cache_dir: Path,
) -> Assessment:
    questions = _build_jev_questions(jp)
    path = _assessment_cache_path(cache_dir, jp.id, candidate.id)
    key = _assessment_cache_key(jp, candidate, questions)

    with _cache_lock:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("cache_key") == key:
                    return Assessment.model_validate(data["assessment"])
            except (json.JSONDecodeError, KeyError, ValidationError, OSError) as exc:
                logger.warning("Discarding unreadable assessment cache %s: %s", path, exc)

    assessment = generate_assessment(jp, candidate, jev_client)

    with _cache_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(
            json.dumps(
                {"cache_key": key, "assessment": assessment.model_dump()}, indent=2
            ),
            encoding="utf-8",
        )
        tmp.replace(path)
    return assessment


def filter_assessable_candidates(candidates: list[Candidate]) -> list[Candidate]:
    assessable = [c for c in candidates if c.parse_status == "ok"]
    excluded = len(candidates) - len(assessable)
    if excluded > 0:
        logger.warning("Excluded %d candidate(s) with failed CV parsing", excluded)
    return assessable
