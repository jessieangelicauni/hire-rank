from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, ValidationError

from candidate_ranking.utils import invoke_and_validate
from candidate_ranking.models import (
    Assessment, Candidate, CareerTrajectory, JobDescription, JobProfile, RedFlag, RequirementEvaluation,
)
from candidate_ranking.scoring.jev_client import JEV_MODEL_ID, JevClientError, JevQuestion
logger = logging.getLogger(__name__)

_cache_lock = threading.Lock()

# ── Jev question builder ──────────────────────────────────────────────────────

_RED_FLAG_TYPES: list[str] = [
    "employment_gap", "skill_mismatch", "regression", "job_hopping", "other",
]

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
_FLAG_MONTHS_TICKS = [0, 3, 6, 12, 24, 36]

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


_FLAG_INSTRUCTIONS: dict[str, dict[str, str]] = {
    "employment_gap": {
        "has": (
            "Is there an unexplained employment gap of 3+ months in this CV?\n"
            "Count periods between end-of-role and start-of-next-role that are not explained "
            "by education, parental leave, caregiving, health, freelancing, or founding a company.\n"
            "Explicitly stated reasons are NOT gaps."
        ),
        "severity": (
            "How significant is the employment gap concern?\n"
            "  MINOR    — single gap of 3–6 months, or clearly transitional\n"
            "  MODERATE — gap of 6–18 months, or multiple shorter unexplained gaps totaling 6+ months\n"
            "  MAJOR    — gap of 18+ months, or multiple large gaps suggesting long-term disengagement"
        ),
        "duration": (
            "What is the total duration (in months) of unexplained employment gaps? "
            "Sum all gaps ≥3 months that lack explanation."
        ),
    },
    "skill_mismatch": {
        "has": (
            "Does this candidate's recent experience (last 3–5 years) show a significant "
            "domain mismatch with the role's core requirements?\n"
            "A mismatch means the background is predominantly in a DIFFERENT domain, language stack, "
            "or job function — not merely missing a few skills."
        ),
        "severity": (
            "How severe is the domain mismatch?\n"
            "  MINOR    — adjacent domain, 1 tool generation away; typical upskilling expected\n"
            "  MODERATE — different domain but transferable core skills; requires targeted upskilling\n"
            "  MAJOR    — fundamentally different field; very little overlap with requirements"
        ),
        "duration": (
            "For how many months has the candidate been in a mismatched domain? "
            "Count from when their career diverged from this role's domain."
        ),
    },
    "regression": {
        "has": (
            "Has the candidate taken a step back in seniority, scope, or responsibility "
            "without a clear strategic reason (e.g. co-founding a startup, relocating, caregiving)?"
        ),
        "severity": (
            "How significant is the career regression?\n"
            "  MINOR    — slight step back in one dimension (title or scope, not both)\n"
            "  MODERATE — clear step back in both title and scope\n"
            "  MAJOR    — significant drop in seniority or from leadership to individual contributor"
        ),
        "duration": "How many months has the candidate been in the regressed position?",
    },
    "job_hopping": {
        "has": (
            "Does this candidate show a pattern of job hopping — multiple short tenures (<18 months) "
            "across consecutive non-contract roles, suggesting a retention risk?"
        ),
        "severity": (
            "How concerning is the job hopping pattern?\n"
            "  MINOR    — 2–3 short stints with longer tenures elsewhere; may be early career\n"
            "  MODERATE — 3–4 consecutive short stints; raises retention questions\n"
            "  MAJOR    — 5+ consecutive short stints with no apparent pattern justification"
        ),
        "duration": (
            "Over how many months does the job hopping pattern span? "
            "Count from the first short-tenure role to the most recent."
        ),
    },
    "other": {
        "has": (
            "Is there any other red flag in this CV not covered by the specific categories above? "
            "(e.g. unexplained credential inconsistency, suspicious date overlaps, "
            "vague role descriptions suggesting possible embellishment, repeated industry exits)"
        ),
        "severity": (
            "How significant is the other red flag?\n"
            "  MINOR    — worth noting, unlikely to be decisive\n"
            "  MODERATE — requires clarification in an interview\n"
            "  MAJOR    — serious concern that could be disqualifying"
        ),
        "duration": (
            "If duration-based, over how many months does this flag span? "
            "Use 0 if not duration-based."
        ),
    },
}

_SEVERITY_CRITERIA: dict[str, str] = {
    "minor": "Minor concern — worth noting but unlikely to be decisive",
    "moderate": "Notable concern — requires clarification or monitoring",
    "major": "Significant red flag — could be disqualifying",
}

_FLAG_DURATION_CRITERIA: list[str] = [
    "0 months", "3 months", "6 months", "12 months", "24 months", "36+ months",
]

# Jev noul criteria keys are strings "true"/"false", not Python bools — Jev API requirement.
_PRESENT_CRITERIA: dict[str, str] = {
    "true": "Clear evidence of this flag is present in the CV",
    "false": "No evidence of this flag, or all instances are clearly explained",
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

    # ── Red flags (5 types × 3 questions each) ────────────────────────────────
    for flag_type in _RED_FLAG_TYPES:
        instr = _FLAG_INSTRUCTIONS[flag_type]
        questions.append(JevQuestion(
            key=f"has_{flag_type}",
            kind="noul",
            instructions=instr["has"],
            criteria=_PRESENT_CRITERIA,
        ))
        questions.append(JevQuestion(
            key=f"{flag_type}_severity",
            kind="choice",
            instructions=instr["severity"],
            criteria=_SEVERITY_CRITERIA,
        ))
        questions.append(JevQuestion(
            key=f"{flag_type}_duration_months",
            kind="score",
            instructions=instr["duration"],
            criteria=_FLAG_DURATION_CRITERIA,
        ))

    return questions


# ── Free-text Ollama chain ────────────────────────────────────────────────────

FREE_TEXT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior technical recruiter extracting evidence from a CV.\n\n"
        "Your task is purely EXTRACTION and SUMMARIZATION — do not make hiring judgments.\n"
        "Quote relevant CV text directly where possible. Be concise and specific.\n\n"
        "For skills marked as active red flags, provide a brief factual description "
        "of what the CV shows. For career trajectory evidence, quote the most relevant "
        "role transitions. If a field has no applicable content, leave it empty or null.",
    ),
    (
        "human",
        "ROLE: {jd_title}\n\n"
        "MUST-HAVE SKILLS: {must_have_skills}\n"
        "NICE-TO-HAVE SKILLS: {nice_to_have_skills}\n\n"
        "ACTIVE RED FLAGS TO DESCRIBE: {active_flags}\n\n"
        "CV:\n{cv_text}",
    ),
])


class _SkillEvidenceItem(BaseModel):
    requirement: str
    evidence: str = Field(min_length=1)


class _RedFlagDetailItem(BaseModel):
    type: str
    detail: str = Field(min_length=1)


class _FreeTextOutput(BaseModel):
    must_have_evidence: list[_SkillEvidenceItem] = Field(default_factory=list)
    nice_to_have_evidence: list[_SkillEvidenceItem] = Field(default_factory=list)
    strongest_achievements: list[str] = Field(default_factory=list)
    notable_signals: str | None = None
    red_flag_details: list[_RedFlagDetailItem] = Field(default_factory=list)
    career_trajectory_evidence: str = ""
    multinational_evidence: str | None = None


def build_free_text_chain(llm: BaseChatModel) -> Runnable:
    return FREE_TEXT_PROMPT | llm.with_structured_output(_FreeTextOutput)


# ── Prompt 2: Qualification Gate ─────────────────────────────────────────────

QUALIFICATION_GATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior technical recruiter screening a CV.\n\n"
            "EVIDENCE RULES — apply consistently:\n"
            "  STRONG   = skill demonstrated in work experience with specific system or project\n"
            "  ADEQUATE = skill mentioned in experience context but without project specifics\n"
            "  WEAK     = skill listed in skills/summary section ONLY, never shown in any experience or project\n\n"
            "Verdict mapping:\n"
            "  MET           = strong or adequate evidence\n"
            "  PARTIALLY_MET = weak evidence only\n"
            "  NOT_MET       = no evidence anywhere in CV\n\n"
            "NON-COMPENSATORY RULE:\n"
            "  If any MUST_HAVE verdict is NOT_MET → candidate is disqualified immediately.\n"
            "  Strong performance elsewhere does not compensate.\n\n"
            "EVIDENCE CITATION RULE:\n"
            "  For every verdict, quote or reference the exact section of the CV where the evidence appears.\n"
            "  If you cannot find specific text → verdict is PARTIALLY_MET (weak) or NOT_MET, never MET.\n\n"
            "MISSED EVIDENCE RULE:\n"
            "  Before concluding NOT_MET for any requirement, re-read ALL sections:\n"
            "  skills, experience, projects, education, certifications, and summary.\n"
            "  Skills are often implied — 'Django' implies Python, 'Spring Boot' implies Java.\n\n"
            "DURATION RULE:\n"
            "  Compute total_relevant_years by summing durations of roles where the candidate used\n"
            "  skills relevant to this JD. Use CV dates directly.\n"
            "  years_demonstrated per requirement = years that skill appears in active use.\n\n"
            "RED_FLAG RULE:\n"
            "  type must be exactly one of: EMPLOYMENT_GAP, SKILL_MISMATCH, REGRESSION, JOB_HOPPING, OTHER.\n"
            "  severity must be exactly one of: MINOR, MODERATE, MAJOR.\n\n"
            "Evaluate every MUST_HAVE requirement first. If candidate qualifies, also evaluate\n"
            "NICE_TO_HAVE requirements and fill all remaining fields.",
        ),
        (
            "human",
            "ROLE: {jd_title}\n\n"
            "MUST_HAVE REQUIREMENTS:\n{must_have_json}\n\n"
            "NICE_TO_HAVE REQUIREMENTS:\n{nice_to_have_json}\n\n"
            "CV:\n{cv_text}",
        ),
    ]
)


class _RequirementEvaluationOutput(BaseModel):
    requirement: str
    verdict: Literal["MET", "PARTIALLY_MET", "NOT_MET"]
    evidence: str = Field(min_length=1)
    evidence_strength: Literal["strong", "adequate", "weak"]
    recency: Literal["current", "recent", "dated", "unknown"] = "unknown"
    years_demonstrated: int = Field(ge=0, default=0)


class _CareerTrajectoryOutput(BaseModel):
    direction: Literal["ascending", "lateral", "descending", "unclear"]
    evidence: str
    tenure_pattern: Literal["stable", "variable", "concerning"]
    average_tenure_months: int = Field(ge=0, default=0)
    most_recent_role_duration_months: int = Field(ge=0, default=0)
    has_multinational_experience: bool
    multinational_evidence: str | None = None


class _RedFlagOutput(BaseModel):
    type: Literal["employment_gap", "skill_mismatch", "regression", "job_hopping", "other"]
    detail: str
    duration_months: int | None = None
    severity: Literal["minor", "moderate", "major"]


class _QualificationOutput(BaseModel):
    must_have_evaluations: list[_RequirementEvaluationOutput]
    nice_to_have_evaluations: list[_RequirementEvaluationOutput] = Field(default_factory=list)
    total_relevant_years: int = Field(ge=0, default=0)
    career_trajectory: _CareerTrajectoryOutput | None = None
    strongest_achievements: list[str] = Field(default_factory=list)
    notable_signals: str | None = None
    red_flags: list[_RedFlagOutput] = Field(default_factory=list)
    qualified: bool
    disqualification_reason: str | None = None


class AssessmentGenerationError(Exception):
    pass


def build_qualification_chain(llm: BaseChatModel) -> Runnable:
    return QUALIFICATION_GATE_PROMPT | llm.with_structured_output(_QualificationOutput)


def _output_to_assessment(
    jd: JobDescription,
    candidate: Candidate,
    model_name: str,
    output: _QualificationOutput,
) -> Assessment:
    must_have_evals = [
        RequirementEvaluation(
            requirement=ev.requirement,
            verdict=ev.verdict,
            evidence=ev.evidence,
            evidence_strength=ev.evidence_strength,
            recency=ev.recency,
            years_demonstrated=ev.years_demonstrated,
        )
        for ev in output.must_have_evaluations
    ]
    nice_to_have_evals = [
        RequirementEvaluation(
            requirement=ev.requirement,
            verdict=ev.verdict,
            evidence=ev.evidence,
            evidence_strength=ev.evidence_strength,
            recency=ev.recency,
            years_demonstrated=ev.years_demonstrated,
        )
        for ev in output.nice_to_have_evaluations
    ]
    traj = None
    if output.career_trajectory:
        t = output.career_trajectory
        traj = CareerTrajectory(
            direction=t.direction,
            evidence=t.evidence,
            tenure_pattern=t.tenure_pattern,
            average_tenure_months=t.average_tenure_months,
            most_recent_role_duration_months=t.most_recent_role_duration_months,
            has_multinational_experience=t.has_multinational_experience,
            multinational_evidence=t.multinational_evidence,
        )
    red_flags = [
        RedFlag(
            type=rf.type,
            detail=rf.detail,
            duration_months=rf.duration_months,
            severity=rf.severity,
        )
        for rf in output.red_flags
    ]
    return Assessment(
        job_description_id=jd.id,
        candidate_id=candidate.id,
        generated_by_model=model_name,
        must_have_evaluations=must_have_evals,
        nice_to_have_evaluations=nice_to_have_evals,
        total_relevant_years=output.total_relevant_years,
        career_trajectory=traj,
        strongest_achievements=output.strongest_achievements,
        notable_signals=output.notable_signals,
        red_flags=red_flags,
        qualified=output.qualified,
        disqualification_reason=output.disqualification_reason,
    )


def generate_assessment(
    jd: JobDescription,
    candidate: Candidate,
    profile: JobProfile | None,
    chain: Runnable,
    model_name: str,
) -> Assessment:
    must_have_json = (
        json.dumps(
            [{"requirement": s, "context": "Core skill required by the job description"} for s in profile.must_have_skills],
            indent=2,
        )
        if profile else "[]"
    )
    nice_to_have_json = (
        json.dumps(
            [{"requirement": s, "context": "Skill mentioned in the job description"} for s in profile.nice_to_have_skills],
            indent=2,
        )
        if profile else "[]"
    )

    def validate(result: _QualificationOutput) -> str | None:
        # Only enforce non-empty evaluations when a profile was actually provided;
        # if profile is None the LLM received empty lists and may return no evals.
        if profile and not result.must_have_evaluations:
            return "must_have_evaluations is empty"
        return None

    log_context = f"Qualification gate for {jd.id}/{candidate.id}"
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            result = invoke_and_validate(
                chain,
                {
                    "jd_title": jd.title,
                    "must_have_json": must_have_json,
                    "nice_to_have_json": nice_to_have_json,
                    "cv_text": candidate.raw_text,
                },
                _QualificationOutput,
                validate,
                log_context=f"{log_context} (attempt {attempt + 1}/2)",
            )
            break
        except Exception as exc:  # noqa: BLE001 — translate ALL failures to domain error
            last_error = exc
    else:
        raise AssessmentGenerationError(str(last_error)) from last_error

    return _output_to_assessment(jd, candidate, model_name, result)


def _assessment_cache_path(cache_dir: Path, jd_id: str, candidate_id: str) -> Path:
    return cache_dir / "assessments" / jd_id / f"{candidate_id}.json"


def _assessment_cache_key(
    jd: JobDescription,
    candidate: Candidate,
    profile: JobProfile | None,
    model_name: str,
) -> str:
    profile_repr = profile.model_dump_json() if profile else "null"
    digest_input = f"{jd.id}||{jd.raw_text}||{candidate.id}||{candidate.raw_text}||{profile_repr}||{model_name}"
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


def load_or_generate_assessment(
    jd: JobDescription,
    candidate: Candidate,
    profile: JobProfile | None,
    chain: Runnable,
    model_name: str,
    cache_dir: Path,
) -> Assessment:
    path = _assessment_cache_path(cache_dir, jd.id, candidate.id)
    key = _assessment_cache_key(jd, candidate, profile, model_name)

    with _cache_lock:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("cache_key") == key:
                    return Assessment.model_validate(data["assessment"])
            except (json.JSONDecodeError, KeyError, ValidationError, OSError) as exc:
                logger.warning("Discarding unreadable assessment cache %s: %s", path, exc)

    # LLM call happens outside the lock to avoid blocking other threads.
    assessment = generate_assessment(jd, candidate, profile, chain, model_name)

    with _cache_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        # uuid suffix avoids collision when multiple processes write concurrently.
        tmp = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(
            json.dumps({"cache_key": key, "assessment": assessment.model_dump()}, indent=2),
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
