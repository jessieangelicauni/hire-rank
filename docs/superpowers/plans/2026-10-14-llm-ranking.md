# LLM Ranking System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Jev-based scoring pipeline with a 3-prompt self-hosted LLM pipeline: JD requirement extraction (Prompt 1), CV qualification gate (Prompt 2), and listwise ranking (Prompt 3) with Plackett-Luce aggregation.

**Architecture:** Prompt 1 extracts must_have/nice_to_have requirements per JD (1 call/JD). Prompt 2 evaluates each shortlisted CV against those requirements with evidence citations and a non-compensatory qualified verdict (1 call/pair). Prompt 3 ranks qualified candidates listwise — one-shot ×3 + P-L for pool ≤ 20, MC-KG tournament + P-L for pool > 20.

**Tech Stack:** LangChain + LangGraph, Pydantic, Ollama (Qwen2.5-14B-Instruct q4_K_M), `ranking/plackett_luce.py` (existing), `ranking/tournament.py` (updated), `ranking/mc_kg.py` (unchanged)

---

## File Map

```
CREATE:
  src/candidate_ranking/scoring/jd_requirements.py
  src/candidate_ranking/scoring/llm_assessment.py
  tests/scoring/test_jd_requirements.py
  tests/scoring/test_llm_assessment.py

REWRITE:
  src/candidate_ranking/models.py
  src/candidate_ranking/graphs/pipeline.py
  src/candidate_ranking/cli.py
  src/candidate_ranking/output/formatter.py

UPDATE:
  src/candidate_ranking/config.py
  src/candidate_ranking/ranking/tournament.py
  src/candidate_ranking/output/console_export.py
  src/candidate_ranking/output/run_output.py
  tests/test_models.py
  tests/output/test_formatter.py
  tests/graphs/test_pipeline.py
  .env.example

DELETE:
  src/candidate_ranking/scoring/jev_client.py
  src/candidate_ranking/scoring/assessment.py
  tests/scoring/test_jev_client.py
  tests/scoring/test_assessment.py
```

---

## Task 1: Rewrite models.py — new Assessment + JDRequirements dataclasses

**Files:**
- Rewrite: `src/candidate_ranking/models.py`
- Rewrite: `tests/test_models.py`

- [ ] **Step 1: Write the failing tests for new models**

```python
# tests/test_models.py
from __future__ import annotations
import pytest
from pydantic import ValidationError
from candidate_ranking.models import (
    Assessment, CareerTrajectory, JDRequirement, JDRequirements,
    JDSkills, RedFlag, RequirementEvaluation,
)

def _req_eval(verdict="MET") -> RequirementEvaluation:
    return RequirementEvaluation(
        requirement="Python", verdict=verdict, evidence="FastAPI at PT XYZ",
        evidence_strength="strong", recency="current", years_demonstrated=3,
    )

def _trajectory() -> CareerTrajectory:
    return CareerTrajectory(
        direction="ascending", evidence="Junior→Senior 2019-2023",
        tenure_pattern="stable", average_tenure_months=24,
        most_recent_role_duration_months=18,
        has_multinational_experience=False, multinational_evidence=None,
    )

def _assessment(qualified=True) -> Assessment:
    return Assessment(
        job_description_id="jd-1", candidate_id="cand-1",
        generated_by_model="qwen2.5:14b",
        must_have_evaluations=[_req_eval()],
        qualified=qualified,
    )

def test_assessment_qualified_flag():
    assert _assessment(qualified=True).qualified is True
    assert _assessment(qualified=False).qualified is False

def test_assessment_disqualification_reason_none_when_qualified():
    a = _assessment(qualified=True)
    assert a.disqualification_reason is None

def test_assessment_has_no_jev_fields():
    assert "requirement_scores" not in Assessment.model_fields
    assert "composite_fit_score" not in Assessment.model_fields
    assert "confidence" not in Assessment.model_fields
    assert "seniority_years_fit_score" not in Assessment.model_fields

def test_requirement_evaluation_verdict_literals():
    for v in ("MET", "PARTIALLY_MET", "NOT_MET"):
        ev = _req_eval(verdict=v)
        assert ev.verdict == v
    with pytest.raises(ValidationError):
        _req_eval(verdict="UNKNOWN")

def test_career_trajectory_roundtrips():
    t = _trajectory()
    assert t.direction == "ascending"
    assert t.has_multinational_experience is False

def test_red_flag_duration_optional():
    rf = RedFlag(type="employment_gap", detail="6-month gap", severity="minor")
    assert rf.duration_months is None

def test_jd_requirements_lists():
    jdr = JDRequirements(
        job_description_id="jd-1", generated_by_model="qwen2.5:14b",
        must_have=[JDRequirement(requirement="Python", context="Primary language")],
        nice_to_have=[JDRequirement(requirement="Docker", context="Preferred")],
    )
    assert len(jdr.must_have) == 1
    assert jdr.must_have[0].requirement == "Python"

def test_jd_requirements_defaults_empty():
    jdr = JDRequirements(job_description_id="jd-1", generated_by_model="qwen2.5:14b")
    assert jdr.must_have == []
    assert jdr.nice_to_have == []

def test_jd_skills_unchanged():
    jds = JDSkills(
        job_description_id="jd-1", generated_by_model="qwen2.5:14b",
        technical_skills=["Python"], must_have_skills=["Python"],
    )
    assert jds.must_have_skills == ["Python"]
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_models.py -v
```
Expected: FAIL — `RequirementEvaluation`, `CareerTrajectory`, `RedFlag`, `JDRequirement`, `JDRequirements` not found; `Assessment` still has Jev fields.

- [ ] **Step 3: Rewrite models.py**

```python
# src/candidate_ranking/models.py
from __future__ import annotations
import re
from typing import Literal
from pydantic import BaseModel, Field


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()


class JobDescription(BaseModel):
    id: str
    title: str
    raw_text: str
    source_path: str


class Candidate(BaseModel):
    id: str
    source_path: str
    raw_text: str
    num_pages: int
    char_count: int
    parse_status: Literal["ok", "failed"]
    error_message: str | None = None
    skills: list[str] = Field(default_factory=list)


class JDSkills(BaseModel):
    job_description_id: str
    generated_by_model: str
    technical_skills: list[str] = Field(default_factory=list)
    must_have_skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    seniority_requirement: str | None = None
    seniority_min_years: float | None = Field(default=None, ge=0)
    education_requirement: str | None = None


class JDRequirement(BaseModel):
    requirement: str
    context: str


class JDRequirements(BaseModel):
    job_description_id: str
    generated_by_model: str
    must_have: list[JDRequirement] = Field(default_factory=list)
    nice_to_have: list[JDRequirement] = Field(default_factory=list)


class RequirementEvaluation(BaseModel):
    requirement: str
    verdict: Literal["MET", "PARTIALLY_MET", "NOT_MET"]
    evidence: str
    evidence_strength: Literal["strong", "adequate", "weak"]
    recency: Literal["current", "recent", "dated", "unknown"]
    years_demonstrated: int = Field(ge=0, default=0)


class CareerTrajectory(BaseModel):
    direction: Literal["ascending", "lateral", "descending", "unclear"]
    evidence: str
    tenure_pattern: Literal["stable", "variable", "concerning"]
    average_tenure_months: int = Field(ge=0, default=0)
    most_recent_role_duration_months: int = Field(ge=0, default=0)
    has_multinational_experience: bool
    multinational_evidence: str | None = None


class RedFlag(BaseModel):
    type: Literal["employment_gap", "skill_mismatch", "regression", "job_hopping"]
    detail: str
    duration_months: int | None = None
    severity: Literal["minor", "moderate", "major"]


class Assessment(BaseModel):
    job_description_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    generated_by_model: str = Field(min_length=1)
    must_have_evaluations: list[RequirementEvaluation] = Field(default_factory=list)
    nice_to_have_evaluations: list[RequirementEvaluation] = Field(default_factory=list)
    total_relevant_years: int = Field(ge=0, default=0)
    career_trajectory: CareerTrajectory | None = None
    strongest_achievements: list[str] = Field(default_factory=list)
    notable_signals: str | None = None
    red_flags: list[RedFlag] = Field(default_factory=list)
    qualified: bool = False
    disqualification_reason: str | None = None


class TournamentIterationRecord(BaseModel):
    iteration: int = Field(ge=0)
    subset_candidate_ids: list[str] = Field(min_length=1)
    ranking: list[str] = Field(min_length=1)
    delta_u: float


class TournamentResult(BaseModel):
    job_description_id: str = Field(min_length=1)
    repeat_index: int = Field(ge=0)
    candidate_ids: list[str]
    final_utilities: dict[str, float]
    final_utility_variance: dict[str, float] = Field(default_factory=dict)
    iteration_history: list[TournamentIterationRecord]
    status: Literal["ok", "failed"]
    stop_reason: Literal["budget_exhausted", "no_unseen_subset"] | None = None


class ConvergencePoint(BaseModel):
    iteration: int = Field(ge=0)
    kendall_tau: float | None = Field(default=None, ge=-1.0, le=1.0)
    delta_u: float


class JDEvaluation(BaseModel):
    job_description_id: str = Field(min_length=1)
    convergence_trace: list[ConvergencePoint] = Field(default_factory=list)
    mean_kendall_tau: float | None = Field(default=None, ge=-1.0, le=1.0)
    mean_abs_delta_u: float | None = None
    kendall_tau_std: float | None = Field(default=None, ge=0.0)
    abs_delta_u_std: float | None = Field(default=None, ge=0.0)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_models.py -v
```
Expected: PASS (all 10 tests)

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/models.py tests/test_models.py
git commit -m "feat: replace Jev Assessment with qualification-gate Assessment dataclasses"
```

---

## Task 2: Create scoring/jd_requirements.py (Prompt 1)

**Files:**
- Create: `src/candidate_ranking/scoring/jd_requirements.py`
- Create: `tests/scoring/test_jd_requirements.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/scoring/test_jd_requirements.py
from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from candidate_ranking.models import JDRequirement, JDRequirements, JobDescription
from candidate_ranking.scoring.jd_requirements import (
    JDRequirementsGenerationError,
    generate_jd_requirements,
    load_or_generate_jd_requirements,
)


def _jd() -> JobDescription:
    return JobDescription(id="jd-1", title="Backend Engineer",
                          raw_text="Requires Python 3+ years. Docker preferred.", source_path="jd.pdf")


def _mock_chain(must_have=None, nice_to_have=None):
    from candidate_ranking.scoring.jd_requirements import _JDRequirementsOutput, _RequirementItem
    output = _JDRequirementsOutput(
        must_have=must_have or [_RequirementItem(requirement="Python", context="Primary language")],
        nice_to_have=nice_to_have or [_RequirementItem(requirement="Docker", context="Preferred")],
    )
    chain = MagicMock()
    chain.invoke.return_value = output
    return chain


def test_generate_jd_requirements_returns_jd_requirements():
    result = generate_jd_requirements(_jd(), _mock_chain(), "qwen2.5:14b")
    assert isinstance(result, JDRequirements)
    assert result.job_description_id == "jd-1"
    assert result.generated_by_model == "qwen2.5:14b"
    assert result.must_have[0].requirement == "Python"
    assert result.nice_to_have[0].requirement == "Docker"


def test_generate_jd_requirements_raises_on_empty_must_have():
    from candidate_ranking.scoring.jd_requirements import _JDRequirementsOutput
    chain = MagicMock()
    chain.invoke.return_value = _JDRequirementsOutput(must_have=[], nice_to_have=[])
    with pytest.raises(JDRequirementsGenerationError):
        generate_jd_requirements(_jd(), chain, "qwen2.5:14b")


def test_load_or_generate_uses_cache_on_second_call(tmp_path: Path):
    chain = _mock_chain()
    cache_path = tmp_path / "jd_requirements.json"
    first = load_or_generate_jd_requirements(_jd(), chain, "qwen2.5:14b", cache_path)
    second = load_or_generate_jd_requirements(_jd(), chain, "qwen2.5:14b", cache_path)
    assert first == second
    assert chain.invoke.call_count == 1


def test_load_or_generate_regenerates_on_model_change(tmp_path: Path):
    chain = _mock_chain()
    cache_path = tmp_path / "jd_requirements.json"
    load_or_generate_jd_requirements(_jd(), chain, "model-a", cache_path)
    load_or_generate_jd_requirements(_jd(), chain, "model-b", cache_path)
    assert chain.invoke.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/scoring/test_jd_requirements.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Create scoring/jd_requirements.py**

```python
# src/candidate_ranking/scoring/jd_requirements.py
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
from candidate_ranking.models import JDRequirement, JDRequirements, JobDescription

logger = logging.getLogger(__name__)

_cache_lock = threading.Lock()

JD_REQUIREMENTS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior technical recruiter parsing a job description.\n\n"
            "Extract all requirements and classify each as:\n"
            "- MUST_HAVE: non-negotiable; absence alone disqualifies a candidate\n"
            "- NICE_TO_HAVE: preferred but absence does not disqualify\n\n"
            "Rules:\n"
            "- Be conservative with MUST_HAVE — only use it if the JD treats the requirement as essential\n"
            "- 'Preferred', 'a plus', 'bonus' in JD text → NICE_TO_HAVE at most\n"
            "- Minimum years of experience and seniority level → MUST_HAVE if stated\n"
            "- Degree/education → MUST_HAVE only if explicitly required, not just preferred\n"
            "- Include context: the brief reason why you classified it this way",
        ),
        ("human", "Job Title: {title}\n\nJob Description:\n{description}"),
    ]
)


class _RequirementItem(BaseModel):
    requirement: str
    context: str


class _JDRequirementsOutput(BaseModel):
    must_have: list[_RequirementItem] = Field(default_factory=list)
    nice_to_have: list[_RequirementItem] = Field(default_factory=list)


class JDRequirementsGenerationError(Exception):
    pass


def build_jd_requirements_chain(llm: BaseChatModel) -> Runnable:
    return JD_REQUIREMENTS_PROMPT | llm.with_structured_output(_JDRequirementsOutput)


def generate_jd_requirements(
    jd: JobDescription,
    chain: Runnable,
    model_name: str,
) -> JDRequirements:
    def validate(result: _JDRequirementsOutput) -> str | None:
        if not result.must_have:
            return "generated an empty must_have list — at least one must-have requirement expected"
        return None

    last_error: GenerationError | None = None
    for attempt in range(2):
        try:
            result = invoke_and_validate(
                chain,
                {"title": jd.title, "description": jd.raw_text},
                _JDRequirementsOutput,
                validate,
                log_context=f"JD requirements extraction for {jd.id} (attempt {attempt + 1}/2)",
            )
            break
        except GenerationError as exc:
            last_error = exc
    else:
        raise JDRequirementsGenerationError(str(last_error)) from last_error

    return JDRequirements(
        job_description_id=jd.id,
        generated_by_model=model_name,
        must_have=[JDRequirement(requirement=r.requirement, context=r.context) for r in result.must_have],
        nice_to_have=[JDRequirement(requirement=r.requirement, context=r.context) for r in result.nice_to_have],
    )


def _cache_key(jd: JobDescription, model_name: str) -> str:
    prompt_repr = repr(
        JD_REQUIREMENTS_PROMPT.format_messages(title=jd.title, description=jd.raw_text)
    )
    return hashlib.sha256(f"{prompt_repr}||{model_name}".encode("utf-8")).hexdigest()


def load_or_generate_jd_requirements(
    jd: JobDescription,
    chain: Runnable,
    model_name: str,
    cache_path: Path,
) -> JDRequirements:
    key = _cache_key(jd, model_name)

    with _cache_lock:
        cache = load_json_cache(cache_path, "JD requirements cache")
        cached_entry = cache.get(jd.id)
        if cached_entry is not None and cached_entry.get("cache_key") == key:
            try:
                return JDRequirements.model_validate(cached_entry["jd_requirements"])
            except (KeyError, ValidationError) as exc:
                logger.warning("Discarding invalid cached JD requirements for %s: %s", jd.id, exc)

    jd_requirements = generate_jd_requirements(jd, chain, model_name)

    with _cache_lock:
        cache = load_json_cache(cache_path, "JD requirements cache")
        cache[jd.id] = {"cache_key": key, "jd_requirements": jd_requirements.model_dump()}
        save_json_cache(cache_path, cache)

    return jd_requirements
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/scoring/test_jd_requirements.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/scoring/jd_requirements.py tests/scoring/test_jd_requirements.py
git commit -m "feat: add JD requirement extractor (Prompt 1)"
```

---

## Task 3: Create scoring/llm_assessment.py — Prompt 2 (qualification gate)

**Files:**
- Create: `src/candidate_ranking/scoring/llm_assessment.py`
- Create: `tests/scoring/test_llm_assessment.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/scoring/test_llm_assessment.py
from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from candidate_ranking.models import (
    Assessment, Candidate, JDRequirement, JDRequirements, JobDescription,
)
from candidate_ranking.scoring.llm_assessment import (
    AssessmentGenerationError,
    filter_assessable_candidates,
    generate_assessment,
    load_or_generate_assessment,
)


def _jd() -> JobDescription:
    return JobDescription(id="jd-1", title="Backend Engineer",
                          raw_text="Python 3+ yrs required.", source_path="jd.pdf")


def _candidate(parse_status="ok") -> Candidate:
    return Candidate(id="cand-1", source_path="cv.pdf", raw_text="I built FastAPI services.",
                     num_pages=1, char_count=30, parse_status=parse_status)


def _jd_requirements() -> JDRequirements:
    return JDRequirements(
        job_description_id="jd-1", generated_by_model="qwen2.5:14b",
        must_have=[JDRequirement(requirement="Python", context="Primary language")],
        nice_to_have=[JDRequirement(requirement="Docker", context="Preferred")],
    )


def _make_qualification_output(qualified=True):
    from candidate_ranking.scoring.llm_assessment import (
        _QualificationOutput, _RequirementEvaluationOutput,
    )
    verdict = "MET" if qualified else "NOT_MET"
    return _QualificationOutput(
        must_have_evaluations=[
            _RequirementEvaluationOutput(
                requirement="Python", verdict=verdict,
                evidence="FastAPI at PT XYZ" if qualified else "Only listed in skills section",
                evidence_strength="strong" if qualified else "weak",
                recency="current", years_demonstrated=3 if qualified else 0,
            )
        ],
        qualified=qualified,
        disqualification_reason=None if qualified else "Python not demonstrated in experience",
    )


def _mock_chain(qualified=True):
    chain = MagicMock()
    chain.invoke.return_value = _make_qualification_output(qualified)
    return chain


def test_generate_assessment_returns_qualified_assessment():
    result = generate_assessment(_jd(), _candidate(), _jd_requirements(), _mock_chain(), "qwen2.5:14b")
    assert isinstance(result, Assessment)
    assert result.qualified is True
    assert result.must_have_evaluations[0].requirement == "Python"
    assert result.must_have_evaluations[0].verdict == "MET"


def test_generate_assessment_returns_disqualified_assessment():
    result = generate_assessment(_jd(), _candidate(), _jd_requirements(), _mock_chain(False), "qwen2.5:14b")
    assert result.qualified is False
    assert result.disqualification_reason is not None


def test_generate_assessment_raises_on_chain_error():
    chain = MagicMock()
    chain.invoke.side_effect = ValueError("schema violation")
    with pytest.raises(AssessmentGenerationError):
        generate_assessment(_jd(), _candidate(), _jd_requirements(), chain, "qwen2.5:14b")


def test_load_or_generate_assessment_uses_cache(tmp_path: Path):
    chain = _mock_chain()
    first = load_or_generate_assessment(_jd(), _candidate(), _jd_requirements(), chain, "qwen2.5:14b", tmp_path)
    second = load_or_generate_assessment(_jd(), _candidate(), _jd_requirements(), chain, "qwen2.5:14b", tmp_path)
    assert first == second
    assert chain.invoke.call_count == 1
    assert (tmp_path / "assessments" / "jd-1" / "cand-1.json").exists()


def test_load_or_generate_assessment_regenerates_when_requirements_change(tmp_path: Path):
    chain = _mock_chain()
    reqs_v1 = _jd_requirements()
    reqs_v2 = JDRequirements(
        job_description_id="jd-1", generated_by_model="qwen2.5:14b",
        must_have=[JDRequirement(requirement="Go", context="Changed")],
    )
    load_or_generate_assessment(_jd(), _candidate(), reqs_v1, chain, "qwen2.5:14b", tmp_path)
    load_or_generate_assessment(_jd(), _candidate(), reqs_v2, chain, "qwen2.5:14b", tmp_path)
    assert chain.invoke.call_count == 2


def test_filter_assessable_candidates_excludes_failed_parse():
    candidates = [
        _candidate("ok"),
        Candidate(id="bad", source_path="x.pdf", raw_text="", num_pages=0,
                  char_count=0, parse_status="failed"),
    ]
    result = filter_assessable_candidates(candidates)
    assert len(result) == 1
    assert result[0].id == "cand-1"
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/scoring/test_llm_assessment.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Create scoring/llm_assessment.py (Prompt 2 part)**

```python
# src/candidate_ranking/scoring/llm_assessment.py
from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Callable

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, ValidationError, create_model

from candidate_ranking.generation import GenerationError, invoke_and_validate
from candidate_ranking.models import (
    Assessment, Candidate, CareerTrajectory, JDRequirements,
    JobDescription, RedFlag, RequirementEvaluation,
)
from candidate_ranking.ranking.plackett_luce import fit_utilities

logger = logging.getLogger(__name__)

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
    verdict: str  # MET | PARTIALLY_MET | NOT_MET
    evidence: str = Field(min_length=1)
    evidence_strength: str  # strong | adequate | weak
    recency: str = "unknown"  # current | recent | dated | unknown
    years_demonstrated: int = Field(ge=0, default=0)


class _CareerTrajectoryOutput(BaseModel):
    direction: str  # ascending | lateral | descending | unclear
    evidence: str
    tenure_pattern: str  # stable | variable | concerning
    average_tenure_months: int = Field(ge=0, default=0)
    most_recent_role_duration_months: int = Field(ge=0, default=0)
    has_multinational_experience: bool
    multinational_evidence: str | None = None


class _RedFlagOutput(BaseModel):
    type: str  # employment_gap | skill_mismatch | regression | job_hopping
    detail: str
    duration_months: int | None = None
    severity: str  # minor | moderate | major


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
            verdict=ev.verdict,  # type: ignore[arg-type]
            evidence=ev.evidence,
            evidence_strength=ev.evidence_strength,  # type: ignore[arg-type]
            recency=ev.recency,  # type: ignore[arg-type]
            years_demonstrated=ev.years_demonstrated,
        )
        for ev in output.must_have_evaluations
    ]
    nice_to_have_evals = [
        RequirementEvaluation(
            requirement=ev.requirement,
            verdict=ev.verdict,  # type: ignore[arg-type]
            evidence=ev.evidence,
            evidence_strength=ev.evidence_strength,  # type: ignore[arg-type]
            recency=ev.recency,  # type: ignore[arg-type]
            years_demonstrated=ev.years_demonstrated,
        )
        for ev in output.nice_to_have_evaluations
    ]
    traj = None
    if output.career_trajectory:
        t = output.career_trajectory
        traj = CareerTrajectory(
            direction=t.direction,  # type: ignore[arg-type]
            evidence=t.evidence,
            tenure_pattern=t.tenure_pattern,  # type: ignore[arg-type]
            average_tenure_months=t.average_tenure_months,
            most_recent_role_duration_months=t.most_recent_role_duration_months,
            has_multinational_experience=t.has_multinational_experience,
            multinational_evidence=t.multinational_evidence,
        )
    red_flags = [
        RedFlag(
            type=rf.type,  # type: ignore[arg-type]
            detail=rf.detail,
            duration_months=rf.duration_months,
            severity=rf.severity,  # type: ignore[arg-type]
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
    jd_requirements: JDRequirements | None,
    chain: Runnable,
    model_name: str,
) -> Assessment:
    must_have_json = (
        json.dumps([{"requirement": r.requirement, "context": r.context} for r in jd_requirements.must_have], indent=2)
        if jd_requirements else "[]"
    )
    nice_to_have_json = (
        json.dumps([{"requirement": r.requirement, "context": r.context} for r in jd_requirements.nice_to_have], indent=2)
        if jd_requirements else "[]"
    )

    def validate(result: _QualificationOutput) -> str | None:
        if not result.must_have_evaluations:
            return "must_have_evaluations is empty"
        return None

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
            log_context=f"Qualification gate for {jd.id}/{candidate.id}",
        )
    except GenerationError as exc:
        raise AssessmentGenerationError(str(exc)) from exc

    return _output_to_assessment(jd, candidate, model_name, result)


def _assessment_cache_path(cache_dir: Path, jd_id: str, candidate_id: str) -> Path:
    return cache_dir / "assessments" / jd_id / f"{candidate_id}.json"


def _assessment_cache_key(
    jd: JobDescription,
    candidate: Candidate,
    jd_requirements: JDRequirements | None,
    model_name: str,
) -> str:
    reqs_repr = jd_requirements.model_dump_json() if jd_requirements else "null"
    digest_input = f"{jd.id}||{jd.raw_text}||{candidate.id}||{candidate.raw_text}||{reqs_repr}||{model_name}"
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


def load_or_generate_assessment(
    jd: JobDescription,
    candidate: Candidate,
    jd_requirements: JDRequirements | None,
    chain: Runnable,
    model_name: str,
    cache_dir: Path,
) -> Assessment:
    path = _assessment_cache_path(cache_dir, jd.id, candidate.id)
    key = _assessment_cache_key(jd, candidate, jd_requirements, model_name)

    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("cache_key") == key:
                return Assessment.model_validate(data["assessment"])
        except (json.JSONDecodeError, KeyError, ValidationError, OSError) as exc:
            logger.warning("Discarding unreadable assessment cache %s: %s", path, exc)

    assessment = generate_assessment(jd, candidate, jd_requirements, chain, model_name)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
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
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/scoring/test_llm_assessment.py -v
```
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/scoring/llm_assessment.py tests/scoring/test_llm_assessment.py
git commit -m "feat: add CV qualification gate runner (Prompt 2)"
```

---

## Task 4: Add Prompt 3 (one-shot ranking) to llm_assessment.py

**Files:**
- Modify: `src/candidate_ranking/scoring/llm_assessment.py`
- Modify: `tests/scoring/test_llm_assessment.py`

- [ ] **Step 1: Add failing tests for Prompt 3**

Append to `tests/scoring/test_llm_assessment.py`:

```python
from candidate_ranking.scoring.llm_assessment import (
    build_ranking_chain_factory,
    rank_one_shot,
)


def _qualified_assessment(candidate_id: str, years: int = 3) -> Assessment:
    from candidate_ranking.models import RequirementEvaluation
    return Assessment(
        job_description_id="jd-1", candidate_id=candidate_id,
        generated_by_model="qwen2.5:14b", qualified=True,
        total_relevant_years=years,
        must_have_evaluations=[
            RequirementEvaluation(
                requirement="Python", verdict="MET", evidence=f"FastAPI at {candidate_id}",
                evidence_strength="strong", recency="current", years_demonstrated=years,
            )
        ],
    )


def _make_ranking_chain(ordering: list[int]):
    """ordering = list of 1-based tokens in ranked order (first = best)."""
    chain = MagicMock()
    def _invoke(payload):
        from pydantic import BaseModel as PydanticModel
        class _Result(PydanticModel):
            ranking: list[int]
            overall_reasoning: str
        return _Result(ranking=ordering, overall_reasoning="test")
    chain.invoke.side_effect = _invoke
    return chain


def test_rank_one_shot_returns_all_candidate_ids():
    jd = _jd()
    assessments = {
        "cand-a": _qualified_assessment("cand-a"),
        "cand-b": _qualified_assessment("cand-b"),
        "cand-c": _qualified_assessment("cand-c"),
    }

    # chain always returns same order: 1,2,3 regardless of shuffle
    # (P-L will still produce a deterministic result since all runs agree)
    call_count = [0]
    def _build_chain():
        chain = MagicMock()
        def _invoke(payload):
            call_count[0] += 1
            # Return candidates in the shuffled order (positional: 1=best, 2=second, 3=worst)
            size = payload["candidates_text"].count("\n\n") + 1
            from pydantic import BaseModel as M
            class R(M):
                ranking: list[int]
                overall_reasoning: str
            return R(ranking=list(range(1, size + 1)), overall_reasoning="ok")
        chain.invoke.side_effect = _invoke
        return chain

    result = rank_one_shot(jd, assessments, _build_chain, n_runs=3)
    assert len(result) == 3
    assert set(result) == {"cand-a", "cand-b", "cand-c"}
    assert call_count[0] == 3  # 3 independent runs


def test_rank_one_shot_single_candidate_returns_immediately():
    jd = _jd()
    assessments = {"cand-a": _qualified_assessment("cand-a")}
    build_chain = MagicMock()
    result = rank_one_shot(jd, assessments, build_chain, n_runs=3)
    assert result == ["cand-a"]
    build_chain.assert_not_called()
```

- [ ] **Step 2: Run to verify they fail**

```
pytest tests/scoring/test_llm_assessment.py::test_rank_one_shot_returns_all_candidate_ids tests/scoring/test_llm_assessment.py::test_rank_one_shot_single_candidate_returns_immediately -v
```
Expected: FAIL — `build_ranking_chain_factory`, `rank_one_shot` not found.

- [ ] **Step 3: Append Prompt 3 code to llm_assessment.py**

Append the following to `src/candidate_ranking/scoring/llm_assessment.py`:

```python
# ── Prompt 3: Listwise Ranking ────────────────────────────────────────────────

LISTWISE_RANKING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a panel of senior HR professionals making a final ranking decision.\n\n"
            "All candidates below have already passed qualification screening — they all meet the minimum requirements.\n"
            "Rank them from STRONGEST to WEAKEST fit for this specific role.\n\n"
            "RANKING CRITERIA (in order of importance):\n"
            "1. Relevance — whose experience most directly matches what this role does daily\n"
            "2. Evidence quality — concrete achievements and measurable impact over vague responsibilities\n"
            "3. Career trajectory — growing scope, promotions, increasing seniority over time\n"
            "4. Recency — recent hands-on experience with core stack outweighs older exposure\n"
            "5. Nice-to-have coverage — breadth of preferred skills as a tiebreaker\n\n"
            "RULES:\n"
            "- Rank ALL candidates. No ties. Forced ordering.\n"
            "- Candidates are numbered starting at 1. `ranking` must contain every listed number exactly once.\n"
            "- Judge only on information provided. Do not assume unlisted skills.\n"
            "- When two candidates feel equal: 'Who would I call for interview first?' — that person ranks higher.",
        ),
        (
            "human",
            "ROLE: {jd_title}\n\nJOB SUMMARY: {jd_summary}\n\nCANDIDATES:\n{candidates_text}",
        ),
    ]
)


def _extract_jd_summary(jd: JobDescription) -> str:
    words = jd.raw_text.split()
    return f"{jd.title}\n\n" + " ".join(words[:200])


def _render_evidence_for_ranking(candidate_ids: list[str], assessments: dict[str, Assessment]) -> str:
    blocks = []
    for index, candidate_id in enumerate(candidate_ids, start=1):
        a = assessments[candidate_id]
        must_text = "\n".join(
            f"    [{ev.verdict}] {ev.requirement} ({ev.evidence_strength}, {ev.years_demonstrated}yr): {ev.evidence}"
            for ev in a.must_have_evaluations
        ) or "    (none)"
        met_nice = [ev for ev in a.nice_to_have_evaluations if ev.verdict in ("MET", "PARTIALLY_MET")]
        nice_text = ", ".join(f"{ev.requirement} [{ev.verdict}]" for ev in met_nice) or "(none)"
        ach_text = "\n".join(f"    - {ach}" for ach in a.strongest_achievements) or "    (none)"
        traj = a.career_trajectory
        traj_text = (
            f"{traj.direction}, avg {traj.average_tenure_months}mo/role"
            + (", multinational exp." if traj.has_multinational_experience else "")
        ) if traj else "unknown"
        blocks.append(
            f"{index}:\n"
            f"  Must-have evaluations:\n{must_text}\n"
            f"  Nice-to-have met: {nice_text}\n"
            f"  Achievements:\n{ach_text}\n"
            f"  Trajectory: {traj_text} | Relevant years: {a.total_relevant_years}"
        )
    return "\n\n".join(blocks)


def _ranking_model(size: int) -> type[BaseModel]:
    return create_model(
        "_RankingResult",
        ranking=(list[int], Field(min_length=size, max_length=size)),
        overall_reasoning=(str, Field(min_length=1)),
    )


def build_ranking_chain_factory(llm: BaseChatModel) -> Callable[[], Runnable]:
    """Returns a factory that builds a new chain per ranking run (needed for temperature diversity)."""
    def _build() -> Runnable:
        size_placeholder_model = create_model("_SizePlaceholder", ranking=(list[int], ...), overall_reasoning=(str, ...))
        return LISTWISE_RANKING_PROMPT | llm.with_structured_output(size_placeholder_model)

    # We can't bind the exact list size at factory time, so we return a partial chain that
    # accepts size at call time.
    def _factory() -> Runnable:
        return LISTWISE_RANKING_PROMPT | llm

    return _factory


def _single_ranking_run(
    jd: JobDescription,
    candidate_ids: list[str],
    assessments: dict[str, Assessment],
    build_chain: Callable[[], Runnable],
) -> list[str]:
    """Returns candidate_ids ordered from best (rank 1) to worst."""
    size = len(candidate_ids)
    token_to_id = {i + 1: cid for i, cid in enumerate(candidate_ids)}
    expected_tokens = set(token_to_id)
    ranking_model = _ranking_model(size)

    chain = LISTWISE_RANKING_PROMPT | build_chain().steps[-1].with_structured_output(ranking_model) \
        if hasattr(build_chain(), "steps") else \
        LISTWISE_RANKING_PROMPT | build_chain().with_structured_output(ranking_model)

    def validate(result: BaseModel) -> str | None:
        ranking = result.ranking  # type: ignore[attr-defined]
        if set(ranking) != expected_tokens or len(ranking) != size:
            return f"ranking is not a permutation of {sorted(expected_tokens)}, got {ranking}"
        return None

    try:
        result = invoke_and_validate(
            chain,
            {
                "jd_title": jd.title,
                "jd_summary": _extract_jd_summary(jd),
                "candidates_text": _render_evidence_for_ranking(candidate_ids, assessments),
            },
            ranking_model,
            validate,
            log_context=f"Listwise ranking for {jd.id} ({size} candidates)",
        )
    except GenerationError as exc:
        logger.warning("Listwise ranking run failed for %s: %s — using input order as fallback", jd.id, exc)
        return list(candidate_ids)

    return [token_to_id[token] for token in result.ranking]  # type: ignore[attr-defined]


def rank_one_shot(
    jd: JobDescription,
    assessments: dict[str, Assessment],
    build_chain: Callable[[], Runnable],
    n_runs: int = 3,
    pl_prior_variance: float = 1.0,
    rng_seed: int | None = None,
) -> list[str]:
    """One-shot listwise ranking × n_runs with P-L aggregation. Returns sorted candidate_ids."""
    candidate_ids = list(assessments.keys())
    if len(candidate_ids) <= 1:
        return candidate_ids

    rng = random.Random(rng_seed)
    rankings: list[list[str]] = []
    for _ in range(n_runs):
        shuffled = candidate_ids.copy()
        rng.shuffle(shuffled)
        ranking = _single_ranking_run(jd, shuffled, assessments, build_chain)
        rankings.append(ranking)

    utilities = fit_utilities(rankings, candidate_ids, pl_prior_variance)
    return [cid for cid, _ in sorted(zip(candidate_ids, utilities), key=lambda x: -x[1])]
```

> **Note on `build_ranking_chain_factory`**: The current implementation returns a factory that builds a chain wrapping the LLM. The `_single_ranking_run` function uses `with_structured_output(ranking_model)` at call time, binding the size-specific schema. Rewrite `build_ranking_chain_factory` and `_single_ranking_run` to use this cleaner pattern:

```python
def build_ranking_chain_factory(llm: BaseChatModel) -> Callable[[type[BaseModel]], Runnable]:
    """Returns a factory that builds a size-specific chain for each run."""
    def _build(ranking_model: type[BaseModel]) -> Runnable:
        return LISTWISE_RANKING_PROMPT | llm.with_structured_output(ranking_model)
    return _build


def _single_ranking_run(
    jd: JobDescription,
    candidate_ids: list[str],
    assessments: dict[str, Assessment],
    build_chain: Callable[[type[BaseModel]], Runnable],
) -> list[str]:
    size = len(candidate_ids)
    token_to_id = {i + 1: cid for i, cid in enumerate(candidate_ids)}
    expected_tokens = set(token_to_id)
    ranking_model = _ranking_model(size)
    chain = build_chain(ranking_model)

    def validate(result: BaseModel) -> str | None:
        ranking = result.ranking  # type: ignore[attr-defined]
        if set(ranking) != expected_tokens or len(ranking) != size:
            return f"ranking is not a permutation of {sorted(expected_tokens)}, got {ranking}"
        return None

    try:
        result = invoke_and_validate(
            chain,
            {
                "jd_title": jd.title,
                "jd_summary": _extract_jd_summary(jd),
                "candidates_text": _render_evidence_for_ranking(candidate_ids, assessments),
            },
            ranking_model,
            validate,
            log_context=f"Listwise ranking for {jd.id} ({size} candidates)",
        )
    except GenerationError as exc:
        logger.warning("Listwise ranking run failed for %s: %s — using input order as fallback", jd.id, exc)
        return list(candidate_ids)

    return [token_to_id[token] for token in result.ranking]  # type: ignore[attr-defined]


def rank_one_shot(
    jd: JobDescription,
    assessments: dict[str, Assessment],
    build_chain: Callable[[type[BaseModel]], Runnable],
    n_runs: int = 3,
    pl_prior_variance: float = 1.0,
    rng_seed: int | None = None,
) -> list[str]:
    candidate_ids = list(assessments.keys())
    if len(candidate_ids) <= 1:
        return candidate_ids

    rng = random.Random(rng_seed)
    rankings: list[list[str]] = []
    for _ in range(n_runs):
        shuffled = candidate_ids.copy()
        rng.shuffle(shuffled)
        ranking = _single_ranking_run(jd, shuffled, assessments, build_chain)
        rankings.append(ranking)

    utilities = fit_utilities(rankings, candidate_ids, pl_prior_variance)
    return [cid for cid, _ in sorted(zip(candidate_ids, utilities), key=lambda x: -x[1])]
```

Replace the earlier `build_ranking_chain_factory` and `_single_ranking_run` stubs with this cleaner version.

Also add `from pydantic import create_model` to the imports at the top (it's already imported above via `create_model` — make sure the import exists).

- [ ] **Step 4: Run all llm_assessment tests**

```
pytest tests/scoring/test_llm_assessment.py -v
```
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/scoring/llm_assessment.py tests/scoring/test_llm_assessment.py
git commit -m "feat: add one-shot listwise ranking with Plackett-Luce aggregation (Prompt 3)"
```

---

## Task 5: Update ranking/tournament.py — render from new Assessment fields

**Files:**
- Modify: `src/candidate_ranking/ranking/tournament.py`

- [ ] **Step 1: Replace `_render_candidates_text`**

In `tournament.py`, find the existing `_render_candidates_text` function (lines 72-90) and replace it entirely:

```python
def _render_candidates_text(subset: tuple[str, ...], assessments: dict[str, Assessment]) -> str:
    blocks = []
    for index, candidate_id in enumerate(subset, start=1):
        a = assessments[candidate_id]
        must_text = "\n".join(
            f"    [{ev.verdict}] {ev.requirement} ({ev.evidence_strength}, {ev.years_demonstrated}yr): {ev.evidence}"
            for ev in a.must_have_evaluations
        ) or "    (none)"
        met_nice = [ev for ev in a.nice_to_have_evaluations if ev.verdict in ("MET", "PARTIALLY_MET")]
        nice_text = ", ".join(f"{ev.requirement} [{ev.verdict}]" for ev in met_nice) or "(none)"
        ach_text = "\n".join(f"    - {ach}" for ach in a.strongest_achievements) or "    (none)"
        traj = a.career_trajectory
        traj_text = (
            f"{traj.direction}, avg {traj.average_tenure_months}mo/role"
            + (", multinational exp." if traj.has_multinational_experience else "")
        ) if traj else "unknown"
        blocks.append(
            f"{index}:\n"
            f"  Must-have evaluations:\n{must_text}\n"
            f"  Nice-to-have met: {nice_text}\n"
            f"  Achievements:\n{ach_text}\n"
            f"  Trajectory: {traj_text} | Relevant years: {a.total_relevant_years}"
        )
    return "\n\n".join(blocks)
```

- [ ] **Step 2: Update `LISTWISE_RANKING_PROMPT` in tournament.py**

Replace the system message content (the string in `LISTWISE_RANKING_PROMPT`) at lines 30-48:

```python
LISTWISE_RANKING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert hiring panel member ranking a small group of qualified candidates for a role.\n\n"
            "Task:\n"
            "- Read the job description and each candidate's qualification evidence.\n"
            "- Reason through each candidate in the `reasoning` field.\n"
            "- Compare candidates against each other on: relevance, evidence quality, trajectory, recency, nice-to-have coverage.\n"
            "- Order the candidates from strongest to weakest overall fit.\n"
            "- Put only the candidates' numbers, in that order, in the `ranking` field.\n\n"
            "Constraints:\n"
            "- Candidates are numbered starting at 1 in the list below.\n"
            "- `ranking` must contain every listed candidate number exactly once.\n"
            "- `ranking` must not contain any number that is not listed below.\n"
            "- Never substitute a candidate's name or any other text for their number in `ranking`.",
        ),
        ("human", "Job Description:\n{jd_text}\n\nCandidates:\n{candidates_text}{retry_feedback}"),
    ]
)
```

- [ ] **Step 3: Verify existing tournament tests still pass**

```
pytest tests/ -k "tournament or mc_kg or plackett" -v
```
Expected: PASS — if any tournament tests exist, they should pass. If they fail because they create old Assessment objects, fix the test fixtures to use new Assessment fields.

- [ ] **Step 4: Commit**

```bash
git add src/candidate_ranking/ranking/tournament.py
git commit -m "fix: update tournament candidate rendering for new Assessment evidence fields"
```

---

## Task 6: Update output/formatter.py — P-L utility-based ranking

**Files:**
- Rewrite: `src/candidate_ranking/output/formatter.py`
- Update: `tests/output/test_formatter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/output/test_formatter.py  (full rewrite)
from __future__ import annotations
import json
from pathlib import Path
from candidate_ranking.models import Assessment, JobDescription, RequirementEvaluation
from candidate_ranking.output.formatter import format_jd_ranking, write_jd_ranking


def _jd() -> JobDescription:
    return JobDescription(id="jd-1", title="Backend Engineer", raw_text="...", source_path="jd.pdf")


def _assessment(candidate_id: str) -> Assessment:
    return Assessment(
        job_description_id="jd-1", candidate_id=candidate_id,
        generated_by_model="qwen2.5:14b", qualified=True,
        must_have_evaluations=[
            RequirementEvaluation(requirement="Python", verdict="MET", evidence="FastAPI",
                                  evidence_strength="strong", recency="current", years_demonstrated=3)
        ],
    )


def test_format_jd_ranking_orders_by_utility():
    assessments = {
        "cand-a": _assessment("cand-a"),
        "cand-b": _assessment("cand-b"),
        "cand-c": _assessment("cand-c"),
    }
    utilities = {"cand-a": 0.5, "cand-b": 2.1, "cand-c": 1.0}

    _, json_payload = format_jd_ranking(_jd(), assessments, utilities)

    rows = json_payload["rankings"]
    assert [r["candidate_id"] for r in rows] == ["cand-b", "cand-c", "cand-a"]
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert rows[0]["utility"] == 2.1
    assert "composite_fit_score" not in rows[0]


def test_format_jd_ranking_includes_qualified_and_years():
    assessments = {"cand-a": _assessment("cand-a")}
    utilities = {"cand-a": 1.0}

    _, json_payload = format_jd_ranking(_jd(), assessments, utilities)

    row = json_payload["rankings"][0]
    assert row["qualified"] is True
    assert row["total_relevant_years"] == 0


def test_write_jd_ranking_writes_markdown_and_json(tmp_path: Path):
    assessments = {"cand-a": _assessment("cand-a")}
    utilities = {"cand-a": 1.5}

    write_jd_ranking(tmp_path, _jd(), assessments, utilities)

    assert (tmp_path / "jd-1" / "ranking.md").exists()
    json_payload = json.loads((tmp_path / "jd-1" / "ranking.json").read_text(encoding="utf-8"))
    assert json_payload["rankings"][0]["candidate_id"] == "cand-a"
    assert json_payload["rankings"][0]["utility"] == 1.5
```

- [ ] **Step 2: Run to verify they fail**

```
pytest tests/output/test_formatter.py -v
```
Expected: FAIL — `format_jd_ranking` doesn't accept `utilities` parameter.

- [ ] **Step 3: Rewrite formatter.py**

```python
# src/candidate_ranking/output/formatter.py
from __future__ import annotations
import json
from pathlib import Path
from candidate_ranking.models import Assessment, JobDescription


def format_jd_ranking(
    jd: JobDescription,
    assessments: dict[str, Assessment],
    utilities: dict[str, float],
) -> tuple[str, dict]:
    ranked = sorted(utilities, key=utilities.__getitem__, reverse=True)

    lines = [
        f"# Ranking: {jd.title} ({jd.id})",
        "",
        "> Ranked by Plackett-Luce utility over LLM listwise comparisons.",
        "",
    ]
    rows = []
    for rank, candidate_id in enumerate(ranked, start=1):
        a = assessments.get(candidate_id)
        utility = utilities[candidate_id]
        lines.append(f"{rank}. **{candidate_id}** (utility={utility:.4f})")
        rows.append({
            "rank": rank,
            "candidate_id": candidate_id,
            "utility": utility,
            "qualified": a.qualified if a else None,
            "total_relevant_years": a.total_relevant_years if a else None,
        })

    return "\n".join(lines) + "\n", {"job_description_id": jd.id, "rankings": rows}


def write_jd_ranking(
    run_dir: Path,
    jd: JobDescription,
    assessments: dict[str, Assessment],
    utilities: dict[str, float],
) -> None:
    markdown_text, json_payload = format_jd_ranking(jd, assessments, utilities)
    jd_dir = run_dir / jd.id
    jd_dir.mkdir(parents=True, exist_ok=True)
    (jd_dir / "ranking.md").write_text(markdown_text, encoding="utf-8")
    (jd_dir / "ranking.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run tests**

```
pytest tests/output/test_formatter.py -v
```
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/output/formatter.py tests/output/test_formatter.py
git commit -m "feat: update ranking formatter to use P-L utilities instead of composite score"
```

---

## Task 7: Update config.py and .env.example

**Files:**
- Modify: `src/candidate_ranking/config.py`
- Modify: `.env.example`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Update config.py**

In `RunConfig`, remove `jev_api_key: str` and add two new fields:

```python
@dataclass(frozen=True)
class RunConfig:
    preset: str
    jd_dir: Path
    cv_dir: Path
    cache_dir: Path
    runs_dir: Path
    max_jds: int | None
    max_candidates: int | None
    ollama_model: str
    ollama_base_url: str
    ollama_num_parallel: int
    faithfulness_model: str | None = None
    skill_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_num_ctx: int = 8192
    ranking_temperature: float = 0.4
    ranking_pool_threshold: int = 20
```

In `RunConfig.full()`, remove `jev_api_key=""`.

In `_ENV_OVERRIDES`, remove `"CANDIDATE_RANKING_JEV_API_KEY"` entry and add:

```python
"CANDIDATE_RANKING_RANKING_TEMPERATURE": ("ranking_temperature", float),
"CANDIDATE_RANKING_RANKING_POOL_THRESHOLD": ("ranking_pool_threshold", int),
```

- [ ] **Step 2: Update .env.example**

Remove the first three lines (Jev keys). Update to:

```
CANDIDATE_RANKING_OLLAMA_BASE_URL=http://localhost:11434
CANDIDATE_RANKING_OLLAMA_NUM_PARALLEL=4
CANDIDATE_RANKING_OLLAMA_NUM_CTX=16384
CANDIDATE_RANKING_SKILL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CANDIDATE_RANKING_RANKING_TEMPERATURE=0.4
CANDIDATE_RANKING_RANKING_POOL_THRESHOLD=20

CANDIDATE_RANKING_JD_DIR=/home/jessie/AXPilot-Internal/job-description
CANDIDATE_RANKING_CV_DIR=/home/jessie/AXPilot-Internal/cv
CANDIDATE_RANKING_CACHE_DIR=/home/jessie/AXPilot-Internal/runs/_cache
CANDIDATE_RANKING_RUNS_DIR=/home/jessie/AXPilot-Internal/runs
CANDIDATE_RANKING_MAX_JDS=
CANDIDATE_RANKING_MAX_CANDIDATES=
```

- [ ] **Step 3: Update tests/test_config.py**

Read the current `tests/test_config.py` first, then update any test that references `jev_api_key` to use the new fields. Add tests for `ranking_temperature` and `ranking_pool_threshold`.

- [ ] **Step 4: Run config tests**

```
pytest tests/test_config.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/config.py .env.example tests/test_config.py
git commit -m "fix: remove jev_api_key from config, add ranking_temperature and ranking_pool_threshold"
```

---

## Task 8: Rewrite graphs/pipeline.py — wire new stages

**Files:**
- Rewrite: `src/candidate_ranking/graphs/pipeline.py`
- Update: `tests/graphs/test_pipeline.py`

- [ ] **Step 1: Write failing tests for new pipeline structure**

```python
# tests/graphs/test_pipeline.py  (partial — add these new tests, delete old Jev-specific ones)
from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
from candidate_ranking.config import RunConfig
from candidate_ranking.graphs.pipeline import (
    PipelineState,
    assessments_by_jd,
    build_pipeline_graph,
)
from candidate_ranking.models import Assessment, JDRequirements, JobDescription, RequirementEvaluation


def _qualified_assessment(jd_id: str, candidate_id: str) -> Assessment:
    return Assessment(
        job_description_id=jd_id, candidate_id=candidate_id,
        generated_by_model="qwen2.5:14b", qualified=True,
        must_have_evaluations=[
            RequirementEvaluation(requirement="Python", verdict="MET", evidence="FastAPI",
                                  evidence_strength="strong", recency="current", years_demonstrated=3)
        ],
    )


def _ok_result(jd_id: str, candidate_id: str) -> dict:
    return {
        "jd_id": jd_id, "candidate_id": candidate_id, "status": "ok",
        "assessment": _qualified_assessment(jd_id, candidate_id), "error": None,
    }


def _failed_result(jd_id: str, candidate_id: str) -> dict:
    return {"jd_id": jd_id, "candidate_id": candidate_id, "status": "failed",
            "assessment": None, "error": "boom"}


def test_pipeline_state_has_jd_requirements_key():
    assert "jd_requirements" in PipelineState.__annotations__


def test_pipeline_state_has_no_jev_fields():
    # No Jev-specific fields remain
    for banned in ("jev_client", "jev_results"):
        assert banned not in PipelineState.__annotations__


def test_assessments_by_jd_groups_ok_and_skips_failed():
    results = [_ok_result("jd-1", "a"), _ok_result("jd-1", "b"), _failed_result("jd-1", "c")]
    grouped = assessments_by_jd(results)
    assert set(grouped["jd-1"]) == {"a", "b"}
    assert grouped["jd-1"]["a"].qualified is True


def test_build_pipeline_graph_has_requirements_and_no_jev_nodes(tmp_path: Path):
    cfg = RunConfig.full(tmp_path)
    graph = build_pipeline_graph(
        cfg=cfg,
        jd_skills_chain=MagicMock(),
        jd_requirements_chain=MagicMock(),
        qualification_chain=MagicMock(),
        build_ranking_chain=MagicMock(),
        skill_index=np.zeros((1, 10)),
        skill_row_map=[("jd-1", "cand-1")],
        skill_embedder=MagicMock(return_value=np.zeros((1, 10))),
        run_id="test-run",
    )
    node_names = set(graph.nodes)
    assert "extract_requirements_for_jd" in node_names
    assert "generate_assessment_for_pair" in node_names
    assert "rank_and_format_jd_node" in node_names
    # No Jev node
    assert not any("jev" in n.lower() for n in node_names)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/graphs/test_pipeline.py -v
```
Expected: FAIL — `jd_requirements` key not in `PipelineState`, `build_pipeline_graph` has wrong signature.

- [ ] **Step 3: Rewrite pipeline.py**

```python
# src/candidate_ranking/graphs/pipeline.py
from __future__ import annotations

import logging
import operator
from pathlib import Path
from typing import Annotated, Callable, Literal, TypedDict

import numpy as np
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from pydantic import BaseModel

from candidate_ranking.config import RunConfig
from candidate_ranking.models import Assessment, Candidate, JDRequirements, JDSkills, JobDescription
from candidate_ranking.output.formatter import write_jd_ranking
from candidate_ranking.ranking.tournament import run_tournament_for_jd_repeat
from candidate_ranking.scoring.jd_requirements import (
    JDRequirementsGenerationError,
    load_or_generate_jd_requirements,
)
from candidate_ranking.scoring.jd_skills import JDSkillsGenerationError, load_or_generate_jd_skills
from candidate_ranking.scoring.llm_assessment import (
    AssessmentGenerationError,
    load_or_generate_assessment,
    rank_one_shot,
)
from candidate_ranking.scoring.skills import shortlist_candidates

logger = logging.getLogger(__name__)


class AssessmentResult(TypedDict):
    jd_id: str
    candidate_id: str
    status: Literal["ok", "failed"]
    assessment: Assessment | None
    error: str | None


class PipelineState(TypedDict):
    jds: list[JobDescription]
    candidates: list[Candidate]
    jd_skills: Annotated[dict[str, JDSkills], operator.or_]
    jd_requirements: Annotated[dict[str, JDRequirements], operator.or_]
    shortlists: Annotated[dict[str, list[str]], operator.or_]
    assessment_results: Annotated[list[AssessmentResult], operator.add]


def assessments_by_jd(assessment_results: list[AssessmentResult]) -> dict[str, dict[str, Assessment]]:
    grouped: dict[str, dict[str, Assessment]] = {}
    for result in assessment_results:
        if result["status"] != "ok":
            continue
        grouped.setdefault(result["jd_id"], {})[result["candidate_id"]] = result["assessment"]
    return grouped


def build_pipeline_graph(
    cfg: RunConfig,
    jd_skills_chain: Runnable,
    jd_requirements_chain: Runnable,
    qualification_chain: Runnable,
    build_ranking_chain: Callable[[type[BaseModel]], Runnable],
    skill_index: np.ndarray,
    skill_row_map: list[tuple[str, str]],
    skill_embedder: Callable[[list[str]], np.ndarray],
    run_id: str,
    skill_match_threshold: float = 0.8,
    min_skill_matches: int = 5,
    min_must_have_matches: int = 2,
) -> StateGraph:
    jd_skills_cache_path = cfg.cache_dir / "jd_skills.json"
    jd_requirements_cache_path = cfg.cache_dir / "jd_requirements.json"

    def _noop(state: PipelineState) -> dict:
        return {}

    def fanout_skill_extraction(state: PipelineState) -> list[Send]:
        return [Send("extract_skills_for_jd", {"jd": jd}) for jd in state["jds"]]

    def extract_skills_for_jd(payload: dict) -> dict:
        jd = payload["jd"]
        try:
            jd_skills = load_or_generate_jd_skills(jd, jd_skills_chain, cfg.ollama_model, jd_skills_cache_path)
        except JDSkillsGenerationError as exc:
            logger.warning("Excluding %s from this run: %s", jd.id, exc)
            return {}
        return {"jd_skills": {jd.id: jd_skills}}

    def fanout_shortlist(state: PipelineState) -> list[Send]:
        return [
            Send("build_shortlist_for_jd", {"jd": jd, "jd_skills": state["jd_skills"][jd.id]})
            for jd in state["jds"] if jd.id in state["jd_skills"]
        ]

    def build_shortlist_for_jd(payload: dict) -> dict:
        jd, jd_skills = payload["jd"], payload["jd_skills"]
        candidate_ids = shortlist_candidates(
            jd, jd_skills, skill_index, skill_row_map, skill_embedder,
            threshold=skill_match_threshold, min_matches=min_skill_matches,
            min_must_have_matches=min_must_have_matches,
        )
        return {"shortlists": {jd.id: candidate_ids}}

    def fanout_requirements_extraction(state: PipelineState) -> list[Send]:
        return [
            Send("extract_requirements_for_jd", {"jd": jd})
            for jd in state["jds"] if jd.id in state["jd_skills"]
        ]

    def extract_requirements_for_jd(payload: dict) -> dict:
        jd = payload["jd"]
        try:
            jd_requirements = load_or_generate_jd_requirements(
                jd, jd_requirements_chain, cfg.ollama_model, jd_requirements_cache_path
            )
        except JDRequirementsGenerationError as exc:
            logger.warning("Skipping requirements for %s: %s", jd.id, exc)
            return {}
        return {"jd_requirements": {jd.id: jd_requirements}}

    def fanout_assessments(state: PipelineState) -> list[Send]:
        candidates_by_id = {c.id: c for c in state["candidates"]}
        return [
            Send(
                "generate_assessment_for_pair",
                {
                    "jd": jd,
                    "candidate": candidates_by_id[cid],
                    "jd_requirements": state["jd_requirements"].get(jd.id),
                },
            )
            for jd in state["jds"]
            for cid in state["shortlists"].get(jd.id, [])
        ]

    def generate_assessment_for_pair(payload: dict) -> dict:
        jd, candidate = payload["jd"], payload["candidate"]
        jd_requirements = payload.get("jd_requirements")
        try:
            assessment = load_or_generate_assessment(
                jd, candidate, jd_requirements, qualification_chain, cfg.ollama_model, cfg.cache_dir
            )
            result: AssessmentResult = {
                "jd_id": jd.id, "candidate_id": candidate.id, "status": "ok",
                "assessment": assessment, "error": None,
            }
        except AssessmentGenerationError as exc:
            result = {
                "jd_id": jd.id, "candidate_id": candidate.id, "status": "failed",
                "assessment": None, "error": str(exc),
            }
        return {"assessment_results": [result]}

    def fanout_rank_and_format(state: PipelineState) -> list[Send]:
        grouped = assessments_by_jd(state["assessment_results"])
        return [
            Send("rank_and_format_jd_node", {"jd": jd, "assessments": grouped.get(jd.id, {})})
            for jd in state["jds"] if grouped.get(jd.id)
        ]

    def rank_and_format_jd_node(payload: dict) -> dict:
        jd: JobDescription = payload["jd"]
        all_assessments: dict[str, Assessment] = payload["assessments"]

        qualified = {cid: a for cid, a in all_assessments.items() if a.qualified}
        disqualified_count = len(all_assessments) - len(qualified)
        if disqualified_count > 0:
            logger.info("JD %s: %d/%d candidates disqualified", jd.id, disqualified_count, len(all_assessments))

        if not qualified:
            logger.warning("No qualified candidates for JD %s — skipping ranking", jd.id)
            write_jd_ranking(cfg.runs_dir / run_id, jd, {}, {})
            return {}

        if len(qualified) <= cfg.ranking_pool_threshold:
            utilities_list = rank_one_shot(jd, qualified, build_ranking_chain)
            utilities = {cid: float(len(qualified) - i) for i, cid in enumerate(utilities_list)}
        else:
            checkpoint_path = cfg.cache_dir / "tournament" / jd.id / "0.json"
            result = run_tournament_for_jd_repeat(
                jd=jd,
                assessments=qualified,
                repeat_index=0,
                tournament_iterations=35,
                tournament_subset_size=10,
                num_subset_samples=50,
                num_mc_draws=100,
                pl_prior_variance=1.0,
                build_ranking_chain=build_ranking_chain,
                checkpoint_path=checkpoint_path,
                rng_seed=42,
            )
            utilities = result.final_utilities

        write_jd_ranking(cfg.runs_dir / run_id, jd, qualified, utilities)
        return {}

    graph = StateGraph(PipelineState)
    graph.add_node("start", _noop)
    graph.add_node("extract_skills_for_jd", extract_skills_for_jd)
    graph.add_node("skills_barrier", _noop)
    graph.add_node("build_shortlist_for_jd", build_shortlist_for_jd)
    graph.add_node("extract_requirements_for_jd", extract_requirements_for_jd)
    graph.add_node("shortlist_requirements_barrier", _noop)
    graph.add_node("generate_assessment_for_pair", generate_assessment_for_pair)
    graph.add_node("assessments_barrier", _noop)
    graph.add_node("rank_and_format_jd_node", rank_and_format_jd_node)

    graph.set_entry_point("start")
    graph.add_conditional_edges("start", fanout_skill_extraction, ["extract_skills_for_jd"])
    graph.add_edge("extract_skills_for_jd", "skills_barrier")
    graph.add_conditional_edges(
        "skills_barrier",
        lambda s: (
            [Send("build_shortlist_for_jd", {"jd": jd, "jd_skills": s["jd_skills"][jd.id]})
             for jd in s["jds"] if jd.id in s["jd_skills"]]
            + [Send("extract_requirements_for_jd", {"jd": jd})
               for jd in s["jds"] if jd.id in s["jd_skills"]]
        ),
        ["build_shortlist_for_jd", "extract_requirements_for_jd"],
    )
    graph.add_edge("build_shortlist_for_jd", "shortlist_requirements_barrier")
    graph.add_edge("extract_requirements_for_jd", "shortlist_requirements_barrier")
    graph.add_conditional_edges("shortlist_requirements_barrier", fanout_assessments, ["generate_assessment_for_pair"])
    graph.add_edge("generate_assessment_for_pair", "assessments_barrier")
    graph.add_conditional_edges("assessments_barrier", fanout_rank_and_format, ["rank_and_format_jd_node"])
    graph.add_edge("rank_and_format_jd_node", END)

    return graph
```

- [ ] **Step 4: Run pipeline tests**

```
pytest tests/graphs/test_pipeline.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/graphs/pipeline.py tests/graphs/test_pipeline.py
git commit -m "feat: rewire pipeline for 3-prompt LLM ranking — add requirement extraction and qualification gate stages"
```

---

## Task 9: Rewrite cli.py — remove Jev, wire new LLM chains

**Files:**
- Rewrite: `src/candidate_ranking/cli.py`

- [ ] **Step 1: Rewrite cli.py**

```python
# src/candidate_ranking/cli.py
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.graphs.pipeline import build_pipeline_graph
from candidate_ranking.ingestion.cv import build_skill_extraction_chain, enrich_candidates_with_skills, load_candidates
from candidate_ranking.ingestion.jd import load_job_descriptions
from candidate_ranking.logging_setup import configure_logging
from candidate_ranking.models import Assessment, Candidate, JDRequirements, JobDescription
from candidate_ranking.output.console_export import export_console_web_data, seed_console_web_roles
from candidate_ranking.output.run_output import write_run_output
from candidate_ranking.scoring.jd_requirements import build_jd_requirements_chain
from candidate_ranking.scoring.jd_skills import build_jd_skills_chain
from candidate_ranking.scoring.llm_assessment import (
    build_qualification_chain,
    build_ranking_chain_factory,
    filter_assessable_candidates,
)
from candidate_ranking.scoring.skills import build_candidate_skill_index, build_skill_embedder

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ASSESSMENT_SCOPE_VERSION = "llm-qualification-gate-v1"


def load_corpus(cfg: RunConfig) -> tuple[list[JobDescription], list[Candidate]]:
    jds = load_job_descriptions(cfg.jd_dir)
    candidates = load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")
    if cfg.max_jds is not None:
        jds = jds[: cfg.max_jds]
    if cfg.max_candidates is not None:
        candidates = candidates[: cfg.max_candidates]
    return jds, candidates


_CHECKPOINT_ALLOWED_MSGPACK_MODULES = [
    ("candidate_ranking.models", name)
    for name in ("JobDescription", "Candidate", "JDSkills", "Assessment", "JDRequirements")
]


@contextmanager
def sqlite_checkpointer(db_path: Path) -> Iterator[SqliteSaver]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    serde = JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_ALLOWED_MSGPACK_MODULES)
    with closing(sqlite3.connect(str(db_path), check_same_thread=False)) as conn:
        yield SqliteSaver(conn, serde=serde)


def run(
    run_id: str | None = None,
    skill_match_threshold: float = 0.8,
    min_skill_matches: int = 5,
    min_must_have_matches: int = 2,
) -> Path:
    cfg = RunConfig.full(PROJECT_ROOT)
    cfg = apply_env_overrides(cfg)

    run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    print(f"Run ID: {run_id}")
    configure_logging(cfg.runs_dir / run_id / "warnings.json")

    jds, candidates = load_corpus(cfg)
    try:
        seed_console_web_roles(cfg)
    except Exception as exc:
        print(f"WARNING: console-web role seeding failed: {exc}", file=sys.stderr)
    candidates = filter_assessable_candidates(candidates)

    llm = ChatOllama(
        model=cfg.ollama_model, base_url=cfg.ollama_base_url, temperature=0, num_ctx=cfg.ollama_num_ctx
    )
    llm_ranking = ChatOllama(
        model=cfg.ollama_model, base_url=cfg.ollama_base_url,
        temperature=cfg.ranking_temperature, num_ctx=cfg.ollama_num_ctx,
    )

    jd_skills_chain = build_jd_skills_chain(llm)
    jd_requirements_chain = build_jd_requirements_chain(llm)
    qualification_chain = build_qualification_chain(llm)
    build_ranking_chain = build_ranking_chain_factory(llm_ranking)

    skill_extraction_chain = build_skill_extraction_chain(llm)
    candidates = enrich_candidates_with_skills(
        candidates, skill_extraction_chain, cfg.ollama_model,
        cfg.cache_dir / "cv_skills.json", max_workers=cfg.ollama_num_parallel,
    )
    skill_embedder = build_skill_embedder(cfg.skill_embedding_model)
    skill_index, skill_row_map = build_candidate_skill_index(candidates, skill_embedder)

    graph = build_pipeline_graph(
        cfg=cfg,
        jd_skills_chain=jd_skills_chain,
        jd_requirements_chain=jd_requirements_chain,
        qualification_chain=qualification_chain,
        build_ranking_chain=build_ranking_chain,
        skill_index=skill_index,
        skill_row_map=skill_row_map,
        skill_embedder=skill_embedder,
        run_id=run_id,
        skill_match_threshold=skill_match_threshold,
        min_skill_matches=min_skill_matches,
        min_must_have_matches=min_must_have_matches,
    )

    db_path = cfg.cache_dir / "checkpoints.db"
    initial_state = {
        "jds": jds,
        "candidates": candidates,
        "jd_skills": {},
        "jd_requirements": {},
        "shortlists": {},
        "assessment_results": [],
    }

    with sqlite_checkpointer(db_path) as saver:
        compiled = graph.compile(checkpointer=saver)
        thread_config = {"configurable": {"thread_id": run_id}, "max_concurrency": cfg.ollama_num_parallel}
        resuming = saver.get_tuple(thread_config) is not None
        final_state = compiled.invoke(None if resuming else initial_state, config=thread_config)

    for jd in jds:
        if len(final_state["shortlists"].get(jd.id, [])) == 0:
            print(
                f"WARNING: JD {jd.id}'s shortlist is empty — no candidates assessed.",
                file=sys.stderr,
            )

    manifest_path = cfg.runs_dir / run_id / "manifest.json"
    if resuming and manifest_path.exists():
        try:
            prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            skill_match_threshold = prior_manifest.get("skill_match_threshold", skill_match_threshold)
            min_skill_matches = prior_manifest.get("min_skill_matches", min_skill_matches)
            min_must_have_matches = prior_manifest.get("min_must_have_matches", min_must_have_matches)
        except (json.JSONDecodeError, OSError):
            pass

    manifest = {
        "run_id": run_id,
        "preset": cfg.preset,
        "ollama_model": cfg.ollama_model,
        "assessment_scope": ASSESSMENT_SCOPE_VERSION,
        "skill_embedding_model": cfg.skill_embedding_model,
        "skill_match_threshold": skill_match_threshold,
        "min_skill_matches": min_skill_matches,
        "min_must_have_matches": min_must_have_matches,
        "ranking_temperature": cfg.ranking_temperature,
        "ranking_pool_threshold": cfg.ranking_pool_threshold,
        "shortlist_sizes": {jd_id: len(cids) for jd_id, cids in final_state["shortlists"].items()},
        "jd_ids": [jd.id for jd in jds],
        "candidate_ids": [c.id for c in candidates],
    }
    run_dir = write_run_output(cfg.runs_dir, run_id, manifest, final_state["assessment_results"])

    total_shortlisted = sum(len(v) for v in final_state["shortlists"].values())
    if total_shortlisted == 0:
        raise RuntimeError(
            f"Run produced zero shortlisted candidates across {len(jds)} JDs — no assessments generated."
        )

    try:
        exported_path = export_console_web_data(cfg, run_id)
        if exported_path:
            print(f"Exported console-web data to {exported_path}")
    except Exception as exc:
        print(f"WARNING: console-web export failed: {exc}", file=sys.stderr)

    return run_dir


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="candidate_ranking.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--run-id", default=None)
    run_parser.add_argument("--skill-match-threshold", type=float, default=0.8)
    run_parser.add_argument("--min-skill-matches", type=int, default=5)
    run_parser.add_argument("--min-must-have-matches", type=int, default=2)
    args = parser.parse_args()

    if args.command == "run":
        run_dir = run(
            run_id=args.run_id,
            skill_match_threshold=args.skill_match_threshold,
            min_skill_matches=args.min_skill_matches,
            min_must_have_matches=args.min_must_have_matches,
        )
        print(f"Run complete: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify imports work**

```
python -c "from candidate_ranking.cli import run; print('ok')"
```
Expected: `ok` (no import errors)

- [ ] **Step 3: Commit**

```bash
git add src/candidate_ranking/cli.py
git commit -m "feat: remove Jev from CLI, wire 3-prompt LLM pipeline"
```

---

## Task 10: Update console_export.py and run_output.py

**Files:**
- Modify: `src/candidate_ranking/output/console_export.py`
- Modify: `src/candidate_ranking/output/run_output.py`

- [ ] **Step 1: Update console_export.py**

In `console_export.py`, make the following targeted changes:

**Replace** `_composite_fit_score` function (lines 27-34) with:

```python
def _utility_from_ranking(ranking_rows: list[dict], cv_id: str) -> float | None:
    for row in ranking_rows:
        if row.get("candidate_id") == cv_id:
            return row.get("utility")
    return None
```

**Replace** the `score_by_cv_id` line in `export_console_web_data` (currently `score_by_cv_id = {row["candidate_id"]: row["composite_fit_score"] ...}`) with:

```python
ranking_rows = ranking.get("rankings", [])
score_by_cv_id = {row["candidate_id"]: row.get("utility", 0.0) for row in ranking_rows}
```

**Replace** the `assessments[row_id] = {...}` block with:

```python
assessments[row_id] = {
    "qualified": assessment_entry.get("qualified"),
    "total_relevant_years": assessment_entry.get("total_relevant_years", 0),
    "must_have_met": sum(
        1 for ev in assessment_entry.get("must_have_evaluations", [])
        if ev.get("verdict") == "MET"
    ),
    "nice_to_have_met": sum(
        1 for ev in assessment_entry.get("nice_to_have_evaluations", [])
        if ev.get("verdict") == "MET"
    ),
    "has_multinational_experience": (
        assessment_entry.get("career_trajectory") or {}
    ).get("has_multinational_experience"),
}
```

**Replace** `_comparison_from_assessments` (lines 40-48):

```python
def _comparison_from_assessments(jd_id: str, jd_assessments: dict[str, dict], ranking_stability: float | None) -> dict:
    if not jd_assessments:
        return _null_comparison(jd_id)
    qualified_count = sum(1 for a in jd_assessments.values() if a.get("qualified"))
    return {
        "jdId": jd_id,
        "qualifiedCount": qualified_count,
        "totalAssessed": len(jd_assessments),
        "rankingStability": ranking_stability,
    }
```

Also remove the import of `statistics` (line 5) if it's no longer used after removing `_composite_fit_score`.

- [ ] **Step 2: Update run_output.py**

`run_output.py` has its own `AssessmentResultRecord` TypedDict that duplicates `graphs/pipeline.py`'s `AssessmentResult`. Both reference `Assessment`. Since `Assessment` is now the qualification model, no structural change is needed — `model_dump()` will serialize the new fields automatically.

No code changes needed in `run_output.py` beyond confirming `Assessment` import still resolves. Run:

```
python -c "from candidate_ranking.output.run_output import write_run_output; print('ok')"
```

- [ ] **Step 3: Run console export tests**

```
pytest tests/output/test_console_export.py -v
```
Read the current `tests/output/test_console_export.py` first. Update any test fixtures that create old Assessment fields (`requirement_scores`, `composite_fit_score`) to use new fields (`qualified`, `must_have_evaluations`). Also update any assertions that check for old keys in the exported JSON.

- [ ] **Step 4: Commit**

```bash
git add src/candidate_ranking/output/console_export.py src/candidate_ranking/output/run_output.py tests/output/test_console_export.py
git commit -m "fix: update console export to use new qualification Assessment fields"
```

---

## Task 11: Delete old files and tests

**Files:**
- Delete: `src/candidate_ranking/scoring/jev_client.py`
- Delete: `src/candidate_ranking/scoring/assessment.py`
- Delete: `tests/scoring/test_jev_client.py`
- Delete: `tests/scoring/test_assessment.py`

- [ ] **Step 1: Verify nothing imports the deleted files**

```bash
grep -r "jev_client\|from candidate_ranking.scoring.assessment import" src/ tests/ --include="*.py"
```
Expected: no matches (all imports should have been updated in previous tasks).

- [ ] **Step 2: Delete the files**

```bash
git rm src/candidate_ranking/scoring/jev_client.py
git rm src/candidate_ranking/scoring/assessment.py
git rm tests/scoring/test_jev_client.py
git rm tests/scoring/test_assessment.py
```

- [ ] **Step 3: Update tests/test_scripts_import.py if needed**

Read `tests/test_scripts_import.py`. If it imports from `scoring.assessment` or `scoring.jev_client`, update those imports to `scoring.llm_assessment`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove Jev client and old assessment runner — replaced by llm_assessment.py"
```

---

## Task 12: Full test suite green

- [ ] **Step 1: Run full suite**

```
pytest --tb=short -q
```

- [ ] **Step 2: Fix any remaining failures**

Common failure patterns to look for and fix:
- Any test creating `Assessment(requirement_scores=..., composite_fit_score=...)` → update to new fields
- Any test asserting `ranking["rankings"][0]["composite_fit_score"]` → use `"utility"` key
- Any test passing `jev_client=` to `build_pipeline_graph` → remove that kwarg
- Tournament tests using `assessment.strengths` → update to new Assessment fields

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "test: fix remaining test fixtures after Jev → LLM qualification gate migration"
```

---

## Self-Review Checklist

**Spec coverage:**
- Prompt 1 (JD extractor) → Task 2 ✓
- Prompt 2 (qualification gate) → Task 3 ✓
- Prompt 3 (one-shot ranking) → Task 4 ✓
- Tournament path (pool > 20) → Task 8, `rank_and_format_jd_node` ✓
- P-L aggregation → `rank_one_shot` uses `fit_utilities` from `ranking/plackett_luce.py` ✓
- Temperature 0.4 for Prompt 3 → `llm_ranking` in Task 9 ✓
- Non-compensatory gate → `qualified` field + filter in `rank_and_format_jd_node` ✓
- Evidence citation → `evidence` field required in `_RequirementEvaluationOutput` ✓
- No Jev → deleted in Task 11 ✓
- Shortlisting unchanged → `shortlist_candidates` call preserved in pipeline ✓

**Type consistency note:** `build_ranking_chain_factory` returns `Callable[[type[BaseModel]], Runnable]`. The tournament.py `run_tournament_for_jd_repeat` accepts `build_ranking_chain: Callable[[type[BaseModel]], Runnable]` — these match. The `rank_one_shot` in `llm_assessment.py` also accepts `Callable[[type[BaseModel]], Runnable]` — these match.

**One gap to confirm:** In `rank_and_format_jd_node`, the one-shot path returns a utility dict computed as `{cid: float(len(qualified) - i) for i, cid in enumerate(utilities_list)}` — this converts the ranked list to ordinal utilities (e.g., 5, 4, 3, 2, 1 for 5 candidates). This is correct for display but loses the P-L probability meaning. Consider storing the actual P-L utilities from `rank_one_shot` instead. Fix: update `rank_one_shot` to return both the sorted list AND the utility dict, or have it return only the sorted list and compute utilities from `fit_utilities` directly in the caller.

Simplest fix — update `rank_one_shot` signature to return `dict[str, float]` (utility per candidate_id):

```python
def rank_one_shot(...) -> dict[str, float]:
    ...
    utilities = fit_utilities(rankings, candidate_ids, pl_prior_variance)
    return dict(zip(candidate_ids, utilities.tolist()))
```

And in pipeline.py, change `rank_and_format_jd_node` to:

```python
utilities = rank_one_shot(jd, qualified, build_ranking_chain)
# utilities is already dict[str, float]
```

Apply this fix in Task 4 (or as a patch in Task 8).
