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
    Assessment, Candidate, CareerTrajectory, JobDescription, JobProfile, RequirementEvaluation,
)
from candidate_ranking.scoring.jev_client import JEV_MODEL_ID, JevClientError, JevQuestion
logger = logging.getLogger(__name__)

_cache_lock = threading.Lock()

# ── Jev question builder ──────────────────────────────────────────────────────

_EVIDENCE_TO_VERDICT: dict[str, str] = {
    "strong": "MET",
    "adequate": "MET",
    "weak": "PARTIALLY_MET",
    "none": "NOT_MET",
}

# Scale tick values for each score question (index i → real-world value for score i).
_YEARS_TICKS = [0, 3, 6, 10, 15, 20]
_MONTHS_TICKS = [6, 12, 24, 36, 48, 72]
_RECENT_MONTHS_TICKS = [1, 6, 12, 24, 36, 60]

_SKILL_EVIDENCE_CRITERIA: dict[str, str] = {
    "strong": (
        "Skill demonstrated in work experience with a specific system, project, or measurable outcome. "
        "E.g. 'Built REST APIs using Django at PT X for 3 years' → STRONG for Django."
    ),
    "adequate": (
        "Skill mentioned in an experience context but without a named project or specific system. "
        "E.g. 'Used Python for data processing tasks' → ADEQUATE."
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



def _score_to_int(value: float, ticks: list[int]) -> int:
    """Convert a Jev score (0..n-1 float) to an integer via linear interpolation."""
    n = len(ticks) - 1
    clamped = max(0.0, min(float(n), float(value)))
    idx_low = int(clamped)
    idx_high = min(idx_low + 1, n)
    frac = clamped - idx_low
    return round(ticks[idx_low] * (1 - frac) + ticks[idx_high] * frac)


def _build_jev_questions(profile: JobProfile) -> list[JevQuestion]:
    """Build a dynamic question set for the given JobProfile.

    The set is adaptive: count and content vary per JD. Every call with
    the same profile produces the same questions in the same order so the
    cache key (which hashes the rendered questions) is deterministic.
    """
    questions: list[JevQuestion] = []

    # ── Skill evidence strength (1 per must-have, 1 per nice-to-have) ────────
    skill_instruction = (
        "Role context: {jd_role}.\n"
        "Assess the evidence strength for the skill '{skill}' in this CV.\n\n"
        "EVIDENCE RULES (apply strictly and consistently across all roles and industries):\n"
        "  STRONG   — demonstrated in work experience with a specific system, project, or measurable outcome\n"
        "  ADEQUATE — mentioned in an experience context but without a named project or specific system\n"
        "  WEAK     — listed in Skills/Summary/Certifications only, never shown in any experience\n"
        "  NONE     — no mention anywhere; check ALL sections before answering NONE; "
        "consider implied skills (Django→Python, Spring Boot→Java, React→JavaScript)"
    )
    for i, skill in enumerate(profile.must_have_skills):
        questions.append(JevQuestion(
            key=f"must_have_{i}_evidence_strength",
            kind="choice",
            instructions=skill_instruction.format(jd_role=profile.job_description_id, skill=skill),
            criteria=_SKILL_EVIDENCE_CRITERIA,
        ))
    for i, skill in enumerate(profile.nice_to_have_skills):
        questions.append(JevQuestion(
            key=f"nice_to_have_{i}_evidence_strength",
            kind="choice",
            instructions=skill_instruction.format(jd_role=profile.job_description_id, skill=skill),
            criteria=_SKILL_EVIDENCE_CRITERIA,
        ))

    # ── Education (conditional) ───────────────────────────────────────────────
    if profile.education_requirement:
        questions.append(JevQuestion(
            key="education_requirement",
            kind="choice",
            instructions=(
                f"Required education: {profile.education_requirement}\n\n"
                "Does this candidate meet the education requirement?\n"
                "Check: degree type, field of study, institution, professional certifications.\n"
                "A candidate without the exact degree but with 5+ years of directly relevant experience "
                "per missing degree level qualifies as EQUIVALENT_EXPERIENCE."
            ),
            criteria={
                "meets": "Holds the specified degree or higher",
                "equivalent_experience": (
                    "Does not hold the exact degree but demonstrates substantial compensating experience "
                    "or professional certifications (e.g. AWS Professional, CFA, PMP)"
                ),
                "does_not_meet": (
                    "Neither holds the degree nor has clearly compensating experience"
                ),
            },
        ))

    # ── Seniority (conditional) ───────────────────────────────────────────────
    if profile.seniority_requirement:
        questions.append(JevQuestion(
            key="seniority_requirement",
            kind="choice",
            instructions=(
                f"Required seniority: {profile.seniority_requirement}\n\n"
                "Does this candidate's seniority level meet the requirement?\n"
                "Assess: job title progression, scope of responsibilities, team size, "
                "architectural/strategic decision involvement, and total years in relevant roles.\n"
                "Title alone is insufficient — assess scope of work."
            ),
            criteria={
                "meets_or_exceeds": (
                    "Current title, scope of responsibilities, and experience level are at or above "
                    "the stated requirement"
                ),
                "below": (
                    "Current title is lower, scope is narrower, or experience is insufficient "
                    "for the stated level"
                ),
            },
        ))

    # ── Other requirements (1 noul per item) ─────────────────────────────────
    for i, req in enumerate(profile.other_requirements):
        questions.append(JevQuestion(
            key=f"other_req_{i}",
            kind="noul",
            instructions=(
                f"Requirement: {req}\n\n"
                "Does this candidate demonstrate or meet this specific requirement?\n"
                "For requirements like 'willingness to travel' or 'portfolio required', "
                "check for indirect signals (past travel in experience, GitHub/portfolio links, etc.) "
                "before answering FALSE."
            ),
            criteria={
                "true": "Clear evidence in the CV that this requirement is met",
                "false": "No evidence found, or the CV directly contradicts the requirement",
            },
        ))

    # ── Total relevant years ──────────────────────────────────────────────────
    questions.append(JevQuestion(
        key="total_relevant_years",
        kind="score",
        instructions=(
            "How many total years of RELEVANT experience does this candidate have for this role?\n\n"
            "COUNT: Only years where the candidate actively used skills directly applicable to this role. "
            "Exclude career breaks and education years. Parallel full-time roles count separately only if "
            "genuinely concurrent.\n\n"
            "CALIBRATION:\n"
            "  0 years  — student, bootcamp grad, or career changer with no relevant work history\n"
            "  3 years  — junior-to-mid, first few years in the field\n"
            "  6 years  — solid mid-level, works independently on most domain tasks\n"
            " 10 years  — senior tenure, has seen multiple tech cycles\n"
            " 15 years  — very experienced senior, likely has led teams or major projects\n"
            " 20+ years — career specialist, principal/distinguished level"
        ),
        criteria=["0 years", "3 years", "6 years", "10 years", "15 years", "20+ years"],
    ))

    # ── Career trajectory (5 fixed questions) ────────────────────────────────
    questions.append(JevQuestion(
        key="career_direction",
        kind="choice",
        instructions=(
            "Based on the progression of roles — titles, responsibilities, scope, and team size "
            "— what is this candidate's career trajectory? Focus on the last 5 years if the career is longer."
        ),
        criteria={
            "ascending": (
                "Clear upward progression: increasing scope, seniority, team size, or "
                "strategic responsibility over time"
            ),
            "lateral": (
                "Movement across domains or roles without a clear upward trajectory — "
                "specialization change, industry change, or horizontal moves at the same level"
            ),
            "descending": (
                "Move to lower seniority, smaller scope, or smaller organization without a "
                "clear strategic reason (e.g. demotion, stepping down from leadership)"
            ),
            "unclear": (
                "Insufficient information, or trajectory is mixed and inconsistent"
            ),
        },
    ))
    questions.append(JevQuestion(
        key="tenure_pattern",
        kind="choice",
        instructions=(
            "Based on the duration of each role listed, what is the candidate's tenure pattern?\n"
            "Note: mark VARIABLE (not CONCERNING) if short stints are clearly contract, "
            "freelance, or project-based."
        ),
        criteria={
            "stable": "Most roles lasted 2+ years; average tenure suggests reliable commitment",
            "variable": (
                "Mix of short and long tenures — some stints under 18 months alongside longer ones; "
                "may be contextually acceptable (contracts, startups, project-based work)"
            ),
            "concerning": (
                "Predominantly short tenures (<18 months) across multiple consecutive roles "
                "without context suggesting contract or project-based work"
            ),
        },
    ))
    questions.append(JevQuestion(
        key="average_tenure_months",
        kind="score",
        instructions=(
            "What is the average length of each role this candidate has held (in months)?\n"
            "Count each distinct role/employer as one unit. "
            "Exclude the current role if it started less than 6 months ago.\n\n"
            "CALIBRATION:\n"
            "  6 months  — very short average, mostly <6 month stints\n"
            " 12 months  — short, mostly ~1 year roles\n"
            " 24 months  — typical, most roles ~2 years\n"
            " 36 months  — good average, most roles ~3 years\n"
            " 48 months  — solid long-tenured professional\n"
            " 72+ months — rare, typically senior specialists"
        ),
        criteria=["6 months", "12 months", "24 months", "36 months", "48 months", "72+ months"],
    ))
    questions.append(JevQuestion(
        key="most_recent_role_duration_months",
        kind="score",
        instructions=(
            "How long has the candidate been in their MOST RECENT role (in months)?\n"
            "If still employed: count from start date to today (approximately).\n"
            "A very recent start (1–3 months) is normal for active job seekers.\n\n"
            "CALIBRATION:\n"
            " 1 month  — just started or left after a brief stint\n"
            " 6 months — early stage\n"
            "12 months — one year in\n"
            "24 months — meaningfully invested\n"
            "36 months — owns significant scope\n"
            "60+ months — deeply established"
        ),
        criteria=["1 month", "6 months", "12 months", "24 months", "36 months", "60+ months"],
    ))
    questions.append(JevQuestion(
        key="has_multinational_experience",
        kind="noul",
        instructions=(
            "Has this candidate worked internationally or for multinational organizations?\n"
            "Check: company names and nationalities, job locations, project scopes, "
            "mentions of international clients or offices, remote work for foreign companies."
        ),
        criteria={
            "true": (
                "Has worked in more than one country, remotely for a foreign company, or held "
                "roles explicitly spanning multiple international markets"
            ),
            "false": "All experience appears to be within a single country, for domestic companies",
        },
    ))

    return questions


# ── BM25 evidence extractor ───────────────────────────────────────────────────

def _extract_bm25_evidence(cv_text: str, query: str, top_k: int = 3) -> str:
    """Return the top-k most relevant CV sentences for the query, verbatim."""
    from rank_bm25 import BM25Okapi
    sentences = [s.strip() for s in re.split(r"[.!\n]+", cv_text) if len(s.strip()) > 15]
    if not sentences:
        return ""
    tokenized = [s.lower().split() for s in sentences]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(
        sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    )
    return " … ".join(sentences[i] for i in top_indices)


class AssessmentGenerationError(Exception):
    pass


def _answers_to_assessment(
    answers: list,
    profile: JobProfile,
    jd: JobDescription,
    candidate: Candidate,
) -> Assessment:
    by_key = {a.key: a for a in answers}
    cv = candidate.raw_text

    must_have_evals = []
    for i, skill in enumerate(profile.must_have_skills):
        es = by_key[f"must_have_{i}_evidence_strength"].value
        verdict = _EVIDENCE_TO_VERDICT[es]
        must_have_evals.append(RequirementEvaluation(
            requirement=skill, verdict=verdict,
            evidence=_extract_bm25_evidence(cv, skill),
            evidence_strength=es, recency="unknown", years_demonstrated=0,
        ))

    nih_evals = []
    for i, skill in enumerate(profile.nice_to_have_skills):
        es = by_key[f"nice_to_have_{i}_evidence_strength"].value
        verdict = _EVIDENCE_TO_VERDICT[es]
        nih_evals.append(RequirementEvaluation(
            requirement=skill, verdict=verdict,
            evidence=_extract_bm25_evidence(cv, skill),
            evidence_strength=es, recency="unknown", years_demonstrated=0,
        ))

    # §1.2: compute qualified from verdicts — never trust a separately-reported boolean
    qualified = all(ev.verdict != "NOT_MET" for ev in must_have_evals)
    failed = [ev.requirement for ev in must_have_evals if ev.verdict == "NOT_MET"]
    disqualification_reason = (
        f"Failed must-have requirements: {', '.join(failed)}" if failed else None
    )

    total_years = _score_to_int(by_key["total_relevant_years"].value, _YEARS_TICKS)
    avg_tenure = _score_to_int(by_key["average_tenure_months"].value, _MONTHS_TICKS)
    recent_tenure = _score_to_int(
        by_key["most_recent_role_duration_months"].value, _RECENT_MONTHS_TICKS
    )
    trajectory = CareerTrajectory(
        direction=by_key["career_direction"].value,
        tenure_pattern=by_key["tenure_pattern"].value,
        average_tenure_months=avg_tenure,
        most_recent_role_duration_months=recent_tenure,
        has_multinational_experience=bool(by_key["has_multinational_experience"].value),
    )

    education_eval = None
    if profile.education_requirement and "education_requirement" in by_key:
        education_eval = by_key["education_requirement"].value

    seniority_eval = None
    if profile.seniority_requirement and "seniority_requirement" in by_key:
        seniority_eval = by_key["seniority_requirement"].value

    other_evals: dict[str, bool] = {}
    for i, req in enumerate(profile.other_requirements):
        key = f"other_req_{i}"
        if key in by_key:
            other_evals[req] = bool(by_key[key].value)

    return Assessment(
        job_description_id=jd.id,
        candidate_id=candidate.id,
        generated_by_model=f"jev/{JEV_MODEL_ID}",
        must_have_evaluations=must_have_evals,
        nice_to_have_evaluations=nih_evals,
        total_relevant_years=total_years,
        career_trajectory=trajectory,
        qualified=qualified,
        disqualification_reason=disqualification_reason,
        education_evaluation=education_eval,
        seniority_evaluation=seniority_eval,
        other_requirement_evaluations=other_evals,
    )


def _assessment_cache_key(
    jd: JobDescription,
    candidate: Candidate,
    questions: list,
) -> str:
    questions_repr = json.dumps(
        [q.model_dump() for q in questions], sort_keys=True
    )
    digest_input = (
        f"{jd.id}||{jd.raw_text}||{candidate.id}||{candidate.raw_text}"
        f"||{questions_repr}||{JEV_MODEL_ID}"
    )
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


def generate_assessment(
    jd: JobDescription,
    candidate: Candidate,
    profile: JobProfile | None,
    jev_client: "JevClient",
) -> Assessment:
    if profile is None:
        raise AssessmentGenerationError(
            f"Cannot assess {jd.id}/{candidate.id}: JobProfile is required for Jev gate"
        )

    questions = _build_jev_questions(profile)

    try:
        answers = jev_client.evaluate(candidate.raw_text, questions)
    except JevClientError as exc:
        raise AssessmentGenerationError(
            f"Jev call failed for {jd.id}/{candidate.id}: {exc}"
        ) from exc

    return _answers_to_assessment(answers, profile, jd, candidate)


def _assessment_cache_path(cache_dir: Path, jd_id: str, candidate_id: str) -> Path:
    return cache_dir / "assessments" / jd_id / f"{candidate_id}.json"


def load_or_generate_assessment(
    jd: JobDescription,
    candidate: Candidate,
    profile: JobProfile | None,
    jev_client: "JevClient",
    cache_dir: Path,
) -> Assessment:
    questions = _build_jev_questions(profile) if profile else []
    path = _assessment_cache_path(cache_dir, jd.id, candidate.id)
    key = _assessment_cache_key(jd, candidate, questions)

    with _cache_lock:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("cache_key") == key:
                    return Assessment.model_validate(data["assessment"])
            except (json.JSONDecodeError, KeyError, ValidationError, OSError) as exc:
                logger.warning("Discarding unreadable assessment cache %s: %s", path, exc)

    assessment = generate_assessment(jd, candidate, profile, jev_client)

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
