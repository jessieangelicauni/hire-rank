# Job Profile Extraction Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two per-JD LLM calls (`jd_requirements.py`, `jd_skills.py`) with one merged call producing a single `JobProfile`, cutting LLM calls for JD processing from 2 to 1 and dropping four fields that have zero downstream consumers.

**Architecture:** Add a new `JobProfile` model and `scoring/job_profile.py` module alongside the existing code first, migrate each downstream consumer (`skills.py`, `llm_assessment.py`, `pipeline.py`, `cli.py`) one at a time, then delete the now-unused `jd_requirements.py`/`jd_skills.py` files, their models, and their tests in a final cleanup task. This keeps every task's own test suite green throughout, even though the full pipeline is only fully consistent after the last task.

**Tech Stack:** Python, pydantic, LangChain (`ChatPromptTemplate`, `with_structured_output`), LangGraph (`StateGraph`, `Send`), pytest.

## Global Constraints

- No retry loop in `generate_job_profile` — one LLM call attempt, matching the already-applied removal of the retry loop in the current `jd_requirements.py` (see `docs/superpowers/specs/2026-10-15-job-profile-extraction-merge-design.md`, Motivation).
- `must_have_skills` and `nice_to_have_skills` are a disjoint partition — every extracted skill lands in exactly one list, never both and never neither (when present at all).
- `education_requirement`, `seniority_requirement`, `other_requirements` are informational only — never sent to the qualification gate (`llm_assessment.py`). Only `must_have_skills`/`nice_to_have_skills` feed the gate's `must_have_json`/`nice_to_have_json`.
- Every new/changed function signature below is exact — later tasks depend on these names and types verbatim.

---

### Task 1: Add `JobProfile` model

**Files:**
- Modify: `src/candidate_ranking/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `JobProfile(BaseModel)` with fields `job_description_id: str`, `generated_by_model: str`, `must_have_skills: list[str]`, `nice_to_have_skills: list[str]`, `education_requirement: str | None`, `seniority_requirement: str | None`, `other_requirements: list[str]` — all list fields default to `[]`, both optional strings default to `None`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_models.py`, change the import block at the top of the file:

```python
from candidate_ranking.models import (
    Assessment, CareerTrajectory, JDRequirement, JDRequirements,
    JDSkills, JobProfile, RedFlag, RequirementEvaluation,
)
```

Add these two test functions at the end of the file:

```python
def test_job_profile_defaults_empty():
    jp = JobProfile(job_description_id="jd-1", generated_by_model="qwen2.5:14b")
    assert jp.must_have_skills == []
    assert jp.nice_to_have_skills == []
    assert jp.education_requirement is None
    assert jp.seniority_requirement is None
    assert jp.other_requirements == []

def test_job_profile_fields_roundtrip():
    jp = JobProfile(
        job_description_id="jd-1", generated_by_model="qwen2.5:14b",
        must_have_skills=["Python"], nice_to_have_skills=["Docker"],
        education_requirement="Bachelor's degree", seniority_requirement="5+ years",
        other_requirements=["AWS Certified Solutions Architect"],
    )
    assert jp.must_have_skills == ["Python"]
    assert jp.nice_to_have_skills == ["Docker"]
    assert jp.education_requirement == "Bachelor's degree"
    assert jp.seniority_requirement == "5+ years"
    assert jp.other_requirements == ["AWS Certified Solutions Architect"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models.py -v`
Expected: FAIL — `ImportError: cannot import name 'JobProfile' from 'candidate_ranking.models'`

- [ ] **Step 3: Implement `JobProfile`**

In `src/candidate_ranking/models.py`, insert immediately after the `JDRequirements` class (after its `nice_to_have` field, before `class RequirementEvaluation`):

```python
class JobProfile(BaseModel):
    job_description_id: str
    generated_by_model: str
    must_have_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    education_requirement: str | None = None
    seniority_requirement: str | None = None
    other_requirements: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models.py -v`
Expected: PASS — all tests in the file, including the pre-existing `JDRequirements`/`JDSkills` ones.

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/models.py tests/test_models.py
git commit -m "feat: add JobProfile model alongside existing JD models"
```

---

### Task 2: Create `scoring/job_profile.py`

**Files:**
- Create: `src/candidate_ranking/scoring/job_profile.py`
- Test: Create `tests/scoring/test_job_profile.py`

**Interfaces:**
- Consumes: `JobProfile` (Task 1), `GenerationError`/`invoke_and_validate`/`load_json_cache`/`save_json_cache` from `candidate_ranking.utils` (existing).
- Produces: `JOB_PROFILE_PROMPT`, `_JobProfileOutput`, `JobProfileGenerationError`, `build_job_profile_chain(llm: BaseChatModel) -> Runnable`, `generate_job_profile(jd: JobDescription, chain: Runnable, model_name: str) -> JobProfile`, `load_or_generate_job_profile(jd: JobDescription, chain: Runnable, model_name: str, cache_path: Path) -> JobProfile`.

- [ ] **Step 1: Write the failing tests**

Create `tests/scoring/test_job_profile.py`:

```python
from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from candidate_ranking.models import JobDescription, JobProfile
from candidate_ranking.scoring.job_profile import (
    JobProfileGenerationError,
    _JobProfileOutput,
    generate_job_profile,
    load_or_generate_job_profile,
)


def _jd() -> JobDescription:
    return JobDescription(id="jd-1", title="Backend Engineer",
                          raw_text="Requires Python 3+ years. Docker preferred.", source_path="jd.pdf")


def _mock_chain(**overrides) -> MagicMock:
    output = _JobProfileOutput(
        reasoning="analysis",
        must_have_skills=overrides.get("must_have_skills", ["Python"]),
        nice_to_have_skills=overrides.get("nice_to_have_skills", ["Docker"]),
        education_requirement=overrides.get("education_requirement"),
        seniority_requirement=overrides.get("seniority_requirement"),
        other_requirements=overrides.get("other_requirements", []),
    )
    chain = MagicMock()
    chain.invoke.return_value = output
    return chain


def test_generate_job_profile_returns_job_profile():
    result = generate_job_profile(_jd(), _mock_chain(), "qwen2.5:14b")
    assert isinstance(result, JobProfile)
    assert result.job_description_id == "jd-1"
    assert result.generated_by_model == "qwen2.5:14b"
    assert result.must_have_skills == ["Python"]
    assert result.nice_to_have_skills == ["Docker"]


def test_generate_job_profile_carries_education_and_seniority():
    chain = _mock_chain(education_requirement="Bachelor's degree", seniority_requirement="3+ years")
    result = generate_job_profile(_jd(), chain, "qwen2.5:14b")
    assert result.education_requirement == "Bachelor's degree"
    assert result.seniority_requirement == "3+ years"


def test_generate_job_profile_raises_on_chain_error_without_retry():
    chain = MagicMock()
    chain.invoke.side_effect = ValueError("schema violation")
    with pytest.raises(JobProfileGenerationError):
        generate_job_profile(_jd(), chain, "qwen2.5:14b")
    assert chain.invoke.call_count == 1  # no retry loop


def test_load_or_generate_uses_cache_on_second_call(tmp_path: Path):
    chain = _mock_chain()
    cache_path = tmp_path / "job_profile.json"
    first = load_or_generate_job_profile(_jd(), chain, "qwen2.5:14b", cache_path)
    second = load_or_generate_job_profile(_jd(), chain, "qwen2.5:14b", cache_path)
    assert first == second
    assert chain.invoke.call_count == 1


def test_load_or_generate_regenerates_on_model_change(tmp_path: Path):
    chain = _mock_chain()
    cache_path = tmp_path / "job_profile.json"
    load_or_generate_job_profile(_jd(), chain, "model-a", cache_path)
    load_or_generate_job_profile(_jd(), chain, "model-b", cache_path)
    assert chain.invoke.call_count == 2


def test_load_or_generate_discards_corrupt_cache(tmp_path: Path):
    chain = _mock_chain()
    cache_path = tmp_path / "job_profile.json"
    cache_path.write_text(
        json.dumps({"jd-1": {"cache_key": "wrong-key", "job_profile": {"bad": "data"}}}), encoding="utf-8"
    )
    result = load_or_generate_job_profile(_jd(), chain, "qwen2.5:14b", cache_path)
    assert isinstance(result, JobProfile)
    assert chain.invoke.call_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scoring/test_job_profile.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'candidate_ranking.scoring.job_profile'`

- [ ] **Step 3: Implement `job_profile.py`**

Create `src/candidate_ranking/scoring/job_profile.py`:

```python
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
from candidate_ranking.models import JobDescription, JobProfile

logger = logging.getLogger(__name__)

_cache_lock = threading.Lock()

JOB_PROFILE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior technical recruiter extracting a structured job profile "
            "from a job description, for candidate evaluation and skill matching.\n\n"
            "In `reasoning` (max 3 sentences), note anything non-obvious before answering.\n\n"
            "Skills → list every named technology, split into:\n"
            "- `must_have_skills`: the core the role is built around — main language(s) plus "
            "1-3 defining frameworks or platforms. Infra/tooling (Docker, cloud, CI/CD) is not "
            "core unless it IS the role.\n"
            "- `nice_to_have_skills`: every other named technology. Most skills belong here.\n\n"
            "Non-skill requirements:\n"
            "- `education_requirement`: one sentence summary if the JD states one, else null.\n"
            "- `seniority_requirement`: one sentence summary of seniority/years-of-experience "
            "if stated, else null.\n"
            "- `other_requirements`: anything else the JD requires or prefers (certifications, "
            "languages, domain knowledge, etc.), each as a short phrase.",
        ),
        ("human", "Job Title: {title}\n\nJob Description:\n{description}"),
    ]
)


class _JobProfileOutput(BaseModel):
    reasoning: str = Field(min_length=1)
    must_have_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    education_requirement: str | None = None
    seniority_requirement: str | None = None
    other_requirements: list[str] = Field(default_factory=list)


class JobProfileGenerationError(Exception):
    pass


def build_job_profile_chain(llm: BaseChatModel) -> Runnable:
    return JOB_PROFILE_PROMPT | llm.with_structured_output(_JobProfileOutput)


def generate_job_profile(
    jd: JobDescription,
    chain: Runnable,
    model_name: str,
) -> JobProfile:
    try:
        result = invoke_and_validate(
            chain,
            {"title": jd.title, "description": jd.raw_text},
            _JobProfileOutput,
            lambda _: None,
            log_context=f"Job profile extraction for {jd.id}",
        )
    except GenerationError as exc:
        raise JobProfileGenerationError(str(exc)) from exc

    return JobProfile(
        job_description_id=jd.id,
        generated_by_model=model_name,
        must_have_skills=result.must_have_skills,
        nice_to_have_skills=result.nice_to_have_skills,
        education_requirement=result.education_requirement,
        seniority_requirement=result.seniority_requirement,
        other_requirements=result.other_requirements,
    )


def _cache_key(jd: JobDescription, model_name: str) -> str:
    prompt_repr = repr(
        JOB_PROFILE_PROMPT.format_messages(title=jd.title, description=jd.raw_text)
    )
    return hashlib.sha256(f"{prompt_repr}||{model_name}".encode("utf-8")).hexdigest()


def load_or_generate_job_profile(
    jd: JobDescription,
    chain: Runnable,
    model_name: str,
    cache_path: Path,
) -> JobProfile:
    key = _cache_key(jd, model_name)

    with _cache_lock:
        cache = load_json_cache(cache_path, "job profile cache")
        cached_entry = cache.get(jd.id)
        if cached_entry is not None and cached_entry.get("cache_key") == key:
            try:
                return JobProfile.model_validate(cached_entry["job_profile"])
            except (KeyError, ValidationError) as exc:
                logger.warning("Discarding invalid cached job profile for %s: %s", jd.id, exc)

    job_profile = generate_job_profile(jd, chain, model_name)

    with _cache_lock:
        cache = load_json_cache(cache_path, "job profile cache")
        cache[jd.id] = {"cache_key": key, "job_profile": job_profile.model_dump()}
        save_json_cache(cache_path, cache)

    return job_profile
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scoring/test_job_profile.py -v`
Expected: PASS — all 6 tests.

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/scoring/job_profile.py tests/scoring/test_job_profile.py
git commit -m "feat: add job_profile.py, a merged JD requirements+skills extraction"
```

---

### Task 3: Migrate `skills.py` to `JobProfile`

**Files:**
- Modify: `src/candidate_ranking/scoring/skills.py`
- Modify: `tests/scoring/test_skills.py`

**Interfaces:**
- Consumes: `JobProfile` (Task 1).
- Produces: `match_candidate_skills(profile: JobProfile, index, row_map, embedder, threshold=0.8) -> dict[str, set[str]]`, `shortlist_candidates(jd, profile: JobProfile, index, row_map, embedder, threshold=0.8, min_matches=2, min_must_have_matches=2) -> list[str]` — both renamed from a `jd_skills: JDSkills` parameter, reading `profile.must_have_skills + profile.nice_to_have_skills` wherever `jd_skills.technical_skills` was read before, and `profile.must_have_skills` wherever `jd_skills.must_have_skills` was read before.

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `tests/scoring/test_skills.py`:

```python
from __future__ import annotations

from typing import Callable

import numpy as np

from candidate_ranking.models import Candidate, JobDescription, JobProfile
from candidate_ranking.scoring.skills import build_candidate_skill_index, shortlist_candidates

_JD_SKILLS = ["Python", "SQL", "AWS", "Docker", "Kubernetes", "Terraform", "Jenkins"]
_MUST_HAVE = ["Python", "SQL"]


def _one_hot_embedder(vocab: list[str]) -> Callable[[list[str]], np.ndarray]:
    """Deterministic stand-in for a real sentence-transformer embedder: an exact
    (case-insensitive) string match gets cosine similarity 1.0, anything else 0.0."""
    index = {v.strip().lower(): i for i, v in enumerate(vocab)}
    dim = len(index)

    def embed(texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), dim), dtype="float32")
        for row, text in enumerate(texts):
            key = text.strip().lower()
            if key in index:
                vectors[row, index[key]] = 1.0
        return vectors

    return embed


def _jd() -> JobDescription:
    return JobDescription(id="jd-1", title="Backend Engineer", raw_text="...", source_path="jd.pdf")


def _profile(must_have: list[str]) -> JobProfile:
    nice_to_have = [s for s in _JD_SKILLS if s not in must_have]
    return JobProfile(
        job_description_id="jd-1",
        generated_by_model="qwen2.5:14b",
        must_have_skills=must_have,
        nice_to_have_skills=nice_to_have,
    )


def _candidate(candidate_id: str, skills: list[str]) -> Candidate:
    return Candidate(
        id=candidate_id, source_path="cv.pdf", raw_text="...", num_pages=1, char_count=10,
        parse_status="ok", skills=skills,
    )


def test_shortlist_excludes_candidate_who_meets_total_but_not_must_have():
    peripheral_only = _candidate("peripheral-only", ["AWS", "Docker", "Kubernetes", "Terraform", "Jenkins"])
    embedder = _one_hot_embedder(_JD_SKILLS)
    index, row_map = build_candidate_skill_index([peripheral_only], embedder)

    shortlisted = shortlist_candidates(
        _jd(), _profile(_MUST_HAVE), index, row_map, embedder,
        min_matches=5, min_must_have_matches=2,
    )

    assert shortlisted == []


def test_shortlist_excludes_candidate_who_meets_must_have_but_not_total():
    core_only = _candidate("core-only", ["Python", "SQL", "AWS"])
    embedder = _one_hot_embedder(_JD_SKILLS)
    index, row_map = build_candidate_skill_index([core_only], embedder)

    shortlisted = shortlist_candidates(
        _jd(), _profile(_MUST_HAVE), index, row_map, embedder,
        min_matches=5, min_must_have_matches=2,
    )

    assert shortlisted == []


def test_shortlist_includes_candidate_who_meets_both():
    both = _candidate("both", ["Python", "SQL", "AWS", "Docker", "Kubernetes"])
    embedder = _one_hot_embedder(_JD_SKILLS)
    index, row_map = build_candidate_skill_index([both], embedder)

    shortlisted = shortlist_candidates(
        _jd(), _profile(_MUST_HAVE), index, row_map, embedder,
        min_matches=5, min_must_have_matches=2,
    )

    assert shortlisted == ["both"]


def test_shortlist_empty_must_have_skills_preserves_prior_behavior():
    peripheral_only = _candidate("peripheral-only", ["AWS", "Docker", "Kubernetes", "Terraform", "Jenkins"])
    embedder = _one_hot_embedder(_JD_SKILLS)
    index, row_map = build_candidate_skill_index([peripheral_only], embedder)

    shortlisted = shortlist_candidates(
        _jd(), _profile([]), index, row_map, embedder,
        min_matches=5, min_must_have_matches=2,
    )

    assert shortlisted == ["peripheral-only"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scoring/test_skills.py -v`
Expected: FAIL — `TypeError` (shortlist_candidates still expects `JDSkills`-shaped `technical_skills`/`must_have_skills`, or `ImportError` for `JobProfile` usage mismatch), since `skills.py` has not been updated yet.

- [ ] **Step 3: Update `skills.py`**

In `src/candidate_ranking/scoring/skills.py`, change the import line:

```python
from candidate_ranking.models import Candidate, JobDescription, JobProfile
```

Replace `match_candidate_skills`:

```python
def match_candidate_skills(
    profile: JobProfile,
    index: np.ndarray,
    row_map: list[tuple[str, str]],
    embedder: Callable[[list[str]], np.ndarray],
    threshold: float = 0.8,
) -> dict[str, set[str]]:
    all_skills = profile.must_have_skills + profile.nice_to_have_skills
    if not all_skills:
        return {}

    deduplicated_skills: list[str] = []
    seen: set[str] = set()
    for skill in all_skills:
        normalized = skill.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            deduplicated_skills.append(skill)

    query_vectors = embedder(deduplicated_skills)
    similarities = query_vectors @ index.T

    matched: dict[str, set[str]] = {}
    for row in range(len(deduplicated_skills)):
        for row_index in np.flatnonzero(similarities[row] >= threshold):
            candidate_id, _skill = row_map[row_index]
            matched.setdefault(candidate_id, set()).add(deduplicated_skills[row])
    return matched
```

Replace `shortlist_candidates`:

```python
def shortlist_candidates(
    jd: JobDescription,
    profile: JobProfile,
    index: np.ndarray,
    row_map: list[tuple[str, str]],
    embedder: Callable[[list[str]], np.ndarray],
    threshold: float = 0.8,
    min_matches: int = 2,
    min_must_have_matches: int = 2,
) -> list[str]:
    all_skills = profile.must_have_skills + profile.nice_to_have_skills
    if not all_skills:
        logger.warning("JD %s has no extracted technical skills; shortlist is empty", jd.id)
        return []

    matched = match_candidate_skills(profile, index, row_map, embedder, threshold)

    deduplicated_count = len({s.strip().lower() for s in all_skills})
    required = min(min_matches, deduplicated_count)

    must_have_normalized = {s.strip().lower() for s in profile.must_have_skills}
    required_must_have = min(min_must_have_matches, len(must_have_normalized))

    def must_have_match_count(skills: set[str]) -> int:
        return sum(1 for skill in skills if skill.strip().lower() in must_have_normalized)

    shortlisted = sorted(
        candidate_id
        for candidate_id, skills in matched.items()
        if len(skills) >= required and must_have_match_count(skills) >= required_must_have
    )
    if not shortlisted:
        logger.warning(
            "JD %s's skill-match shortlist is empty at threshold=%s, required matches=%s, "
            "required must-have matches=%s", jd.id, threshold, required, required_must_have,
        )
    return shortlisted
```

`bridge_negated_jd_skills_to_candidate` and `find_negated_skill_mentions` are unchanged — they already take plain `list[str]`, not `JDSkills`/`JobProfile`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scoring/test_skills.py -v`
Expected: PASS — all 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/scoring/skills.py tests/scoring/test_skills.py
git commit -m "refactor: migrate skills.py shortlisting from JDSkills to JobProfile"
```

---

### Task 4: Migrate `llm_assessment.py` to `JobProfile`

**Files:**
- Modify: `src/candidate_ranking/scoring/llm_assessment.py`
- Modify: `tests/scoring/test_llm_assessment.py`

**Interfaces:**
- Consumes: `JobProfile` (Task 1).
- Produces: `generate_assessment(jd, candidate, profile: JobProfile | None, chain, model_name) -> Assessment`, `load_or_generate_assessment(jd, candidate, profile: JobProfile | None, chain, model_name, cache_dir) -> Assessment` — both renamed from a `jd_requirements: JDRequirements | None` parameter. The gate's `must_have_json`/`nice_to_have_json` are now built from `profile.must_have_skills`/`profile.nice_to_have_skills` with a fixed placeholder `context` string per item (no requirement-specific context is available from a bare skill name).

- [ ] **Step 1: Write the failing tests**

In `tests/scoring/test_llm_assessment.py`, change the import block at the top:

```python
from candidate_ranking.models import (
    Assessment, Candidate, JobDescription, JobProfile, RequirementEvaluation,
)
```

Replace the `_jd_requirements` helper with `_job_profile`:

```python
def _job_profile() -> JobProfile:
    return JobProfile(
        job_description_id="jd-1", generated_by_model="qwen2.5:14b",
        must_have_skills=["Python"], nice_to_have_skills=["Docker"],
    )
```

Replace every call site of `_jd_requirements()` with `_job_profile()` (in `test_generate_assessment_returns_qualified_assessment`, `test_generate_assessment_returns_disqualified_assessment`, `test_generate_assessment_raises_on_chain_error`, `test_generate_assessment_raises_on_unexpected_exception`, `test_load_or_generate_assessment_uses_cache`).

Replace `test_load_or_generate_assessment_regenerates_when_requirements_change`:

```python
def test_load_or_generate_assessment_regenerates_when_profile_changes(tmp_path: Path):
    """Changing the job profile must invalidate the cache and trigger a new LLM call."""
    profile_v1 = _job_profile()
    profile_v2 = JobProfile(
        job_description_id="jd-1", generated_by_model="qwen2.5:14b",
        must_have_skills=["Go"],
    )
    with patch(
        "candidate_ranking.scoring.llm_assessment.generate_assessment",
        return_value=_assessment(),
    ) as mock_gen:
        load_or_generate_assessment(_jd(), _candidate(), profile_v1, MagicMock(), "qwen2.5:14b", tmp_path)
        load_or_generate_assessment(_jd(), _candidate(), profile_v2, MagicMock(), "qwen2.5:14b", tmp_path)
    assert mock_gen.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scoring/test_llm_assessment.py -v`
Expected: FAIL — `ImportError: cannot import name 'JobProfile'` (not yet exported the way the test expects at this layer — actually `JobProfile` already exists from Task 1, so the failure here is a `TypeError`/`AttributeError` from `generate_assessment`/`load_or_generate_assessment` still expecting `JDRequirements`-shaped objects, e.g. `AttributeError: 'JobProfile' object has no attribute 'must_have'`).

- [ ] **Step 3: Update `llm_assessment.py`**

Change the import line:

```python
from candidate_ranking.models import (
    Assessment, Candidate, CareerTrajectory, JobDescription, JobProfile, RedFlag, RequirementEvaluation,
)
```

In `generate_assessment`, rename the parameter and rebuild the JSON blocks:

```python
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
```

(The retry loop here is unchanged — it belongs to the qualification gate's own validation, not to the job-profile extraction removed earlier, and is out of scope per the design doc.)

Update `_assessment_cache_key`:

```python
def _assessment_cache_key(
    jd: JobDescription,
    candidate: Candidate,
    profile: JobProfile | None,
    model_name: str,
) -> str:
    profile_repr = profile.model_dump_json() if profile else "null"
    digest_input = f"{jd.id}||{jd.raw_text}||{candidate.id}||{candidate.raw_text}||{profile_repr}||{model_name}"
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
```

Update `load_or_generate_assessment`'s signature and body (only the parameter name and pass-through change; the caching logic is otherwise identical):

```python
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
        tmp = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(
            json.dumps({"cache_key": key, "assessment": assessment.model_dump()}, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)
    return assessment
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scoring/test_llm_assessment.py -v`
Expected: PASS — all tests.

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/scoring/llm_assessment.py tests/scoring/test_llm_assessment.py
git commit -m "refactor: migrate qualification gate from JDRequirements to JobProfile"
```

---

### Task 5: Merge pipeline nodes in `graphs/pipeline.py`

**Files:**
- Modify: `src/candidate_ranking/graphs/pipeline.py`
- Modify: `tests/graphs/test_pipeline.py`

**Interfaces:**
- Consumes: `load_or_generate_job_profile`/`JobProfileGenerationError` (Task 2), `shortlist_candidates` (Task 3), `load_or_generate_assessment`/`AssessmentGenerationError` (Task 4).
- Produces: `PipelineState` with a single `job_profiles: Annotated[dict[str, JobProfile], operator.or_]` field (replacing `jd_skills`/`jd_requirements`); `build_pipeline_graph(cfg, job_profile_chain: Runnable, qualification_chain, ranking_llm, skill_index, skill_row_map, skill_embedder, run_id, skill_match_threshold=0.8, min_skill_matches=5, min_must_have_matches=2) -> StateGraph` (replacing the `jd_skills_chain`/`jd_requirements_chain` parameters with one `job_profile_chain`); one node `extract_job_profile_for_jd` (replacing `extract_skills_for_jd` + `extract_requirements_for_jd`).

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `tests/graphs/test_pipeline.py`:

```python
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
from candidate_ranking.models import Assessment, JobDescription, RequirementEvaluation


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


def test_pipeline_state_has_job_profiles_key():
    assert "job_profiles" in PipelineState.__annotations__


def test_pipeline_state_has_no_jev_fields():
    for banned in ("jev_client", "jev_results"):
        assert banned not in PipelineState.__annotations__


def test_assessments_by_jd_groups_ok_and_skips_failed():
    results = [_ok_result("jd-1", "a"), _ok_result("jd-1", "b"), _failed_result("jd-1", "c")]
    grouped = assessments_by_jd(results)
    assert set(grouped["jd-1"]) == {"a", "b"}
    assert grouped["jd-1"]["a"].qualified is True


def test_build_pipeline_graph_has_profile_node_and_no_jev_nodes(tmp_path: Path):
    cfg = RunConfig.full(tmp_path)
    graph = build_pipeline_graph(
        cfg=cfg,
        job_profile_chain=MagicMock(),
        qualification_chain=MagicMock(),
        ranking_llm=MagicMock(),
        skill_index=np.zeros((1, 10)),
        skill_row_map=[("jd-1", "cand-1")],
        skill_embedder=MagicMock(return_value=np.zeros((1, 10))),
        run_id="test-run",
    )
    node_names = set(graph.nodes)
    assert "extract_job_profile_for_jd" in node_names
    assert "generate_assessment_for_pair" in node_names
    assert "rank_and_format_jd_node" in node_names
    assert "extract_skills_for_jd" not in node_names
    assert "extract_requirements_for_jd" not in node_names
    assert not any("jev" in n.lower() for n in node_names)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/graphs/test_pipeline.py -v`
Expected: FAIL — `TypeError: build_pipeline_graph() got an unexpected keyword argument 'job_profile_chain'` (current signature still takes `jd_skills_chain`/`jd_requirements_chain`), and `test_pipeline_state_has_job_profiles_key` fails since `PipelineState` doesn't have that key yet.

- [ ] **Step 3: Replace `pipeline.py`**

Replace the full contents of `src/candidate_ranking/graphs/pipeline.py`:

```python
# src/candidate_ranking/graphs/pipeline.py
from __future__ import annotations

import logging
import operator
from typing import TYPE_CHECKING, Annotated, Callable, Literal, TypedDict

import numpy as np
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph
from langgraph.types import Send

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from candidate_ranking.config import RunConfig
from candidate_ranking.models import Assessment, Candidate, JobDescription, JobProfile
from candidate_ranking.output.writer import write_jd_ranking
from candidate_ranking.scoring.job_profile import (
    JobProfileGenerationError,
    load_or_generate_job_profile,
)
from candidate_ranking.scoring.llm_assessment import (
    AssessmentGenerationError,
    load_or_generate_assessment,
)
from candidate_ranking.scoring.active_listwise_ranking import rank_active_listwise
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
    job_profiles: Annotated[dict[str, JobProfile], operator.or_]
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
    job_profile_chain: Runnable,
    qualification_chain: Runnable,
    ranking_llm: "BaseChatModel",
    skill_index: np.ndarray,
    skill_row_map: list[tuple[str, str]],
    skill_embedder: Callable[[list[str]], np.ndarray],
    run_id: str,
    skill_match_threshold: float = 0.8,
    min_skill_matches: int = 5,
    min_must_have_matches: int = 2,
) -> StateGraph:
    job_profile_cache_path = cfg.cache_dir / "job_profile.json"

    def _noop(state: PipelineState) -> dict:
        return {}

    def fanout_job_profile_extraction(state: PipelineState) -> list[Send]:
        return [Send("extract_job_profile_for_jd", {"jd": jd}) for jd in state["jds"]]

    def extract_job_profile_for_jd(payload: dict) -> dict:
        jd = payload["jd"]
        try:
            profile = load_or_generate_job_profile(jd, job_profile_chain, cfg.ollama_model, job_profile_cache_path)
        except JobProfileGenerationError as exc:
            logger.warning("Excluding %s from this run: %s", jd.id, exc)
            return {}
        return {"job_profiles": {jd.id: profile}}

    def fanout_shortlist(state: PipelineState) -> list[Send]:
        return [
            Send("build_shortlist_for_jd", {"jd": jd, "profile": state["job_profiles"][jd.id]})
            for jd in state["jds"]
            if jd.id in state["job_profiles"]
        ]

    def build_shortlist_for_jd(payload: dict) -> dict:
        jd, profile = payload["jd"], payload["profile"]
        candidate_ids = shortlist_candidates(
            jd, profile, skill_index, skill_row_map, skill_embedder,
            threshold=skill_match_threshold, min_matches=min_skill_matches,
            min_must_have_matches=min_must_have_matches,
        )
        return {"shortlists": {jd.id: candidate_ids}}

    def fanout_assessments(state: PipelineState) -> list[Send]:
        candidates_by_id = {c.id: c for c in state["candidates"]}
        return [
            Send(
                "generate_assessment_for_pair",
                {
                    "jd": jd,
                    "candidate": candidates_by_id[cid],
                    "job_profile": state["job_profiles"].get(jd.id),
                },
            )
            for jd in state["jds"]
            for cid in state["shortlists"].get(jd.id, [])
        ]

    def generate_assessment_for_pair(payload: dict) -> dict:
        jd, candidate = payload["jd"], payload["candidate"]
        profile = payload.get("job_profile")
        try:
            assessment = load_or_generate_assessment(
                jd, candidate, profile, qualification_chain, cfg.ollama_model, cfg.cache_dir
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

        if not all_assessments:
            logger.warning("No assessed candidates for JD %s — skipping ranking", jd.id)
            write_jd_ranking(cfg.runs_dir / run_id, jd, {}, {})
            return {}

        utilities = rank_active_listwise(
            jd,
            all_assessments,
            ranking_llm,
            subset_size=cfg.ranking_subset_size,
            max_iterations=cfg.ranking_max_iterations,
            min_delta_u=cfg.ranking_min_delta_u,
            pl_prior_variance=cfg.ranking_pl_prior_variance,
        )

        write_jd_ranking(cfg.runs_dir / run_id, jd, all_assessments, utilities)
        return {}

    graph = StateGraph(PipelineState)
    graph.add_node("start", _noop)
    graph.add_node("extract_job_profile_for_jd", extract_job_profile_for_jd)
    graph.add_node("profile_barrier", _noop)
    graph.add_node("build_shortlist_for_jd", build_shortlist_for_jd)
    graph.add_node("shortlist_barrier", _noop)
    graph.add_node("generate_assessment_for_pair", generate_assessment_for_pair)
    graph.add_node("assessments_barrier", _noop)
    graph.add_node("rank_and_format_jd_node", rank_and_format_jd_node)

    graph.set_entry_point("start")
    graph.add_conditional_edges("start", fanout_job_profile_extraction, ["extract_job_profile_for_jd"])
    graph.add_edge("extract_job_profile_for_jd", "profile_barrier")
    graph.add_conditional_edges("profile_barrier", fanout_shortlist, ["build_shortlist_for_jd"])
    graph.add_edge("build_shortlist_for_jd", "shortlist_barrier")
    graph.add_conditional_edges("shortlist_barrier", fanout_assessments, ["generate_assessment_for_pair"])
    graph.add_edge("generate_assessment_for_pair", "assessments_barrier")
    graph.add_conditional_edges("assessments_barrier", fanout_rank_and_format, ["rank_and_format_jd_node"])
    graph.add_edge("rank_and_format_jd_node", END)

    return graph
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/graphs/test_pipeline.py -v`
Expected: PASS — all 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/graphs/pipeline.py tests/graphs/test_pipeline.py
git commit -m "refactor: merge skill+requirement extraction into one pipeline node"
```

---

### Task 6: Wire `cli.py` to the merged chain

**Files:**
- Modify: `src/candidate_ranking/cli.py`

**Interfaces:**
- Consumes: `build_job_profile_chain` (Task 2), `build_pipeline_graph(..., job_profile_chain=...)` (Task 5).
- Produces: `run()` constructs one `job_profile_chain` instead of `jd_skills_chain` + `jd_requirements_chain`; `_CHECKPOINT_ALLOWED_MSGPACK_MODULES` lists `"JobProfile"` instead of `"JDSkills"`/`"JDRequirements"`; `initial_state` has a `"job_profiles": {}` key instead of `"jd_skills": {}` + `"jd_requirements": {}`.

There is no dedicated `test_cli.py` in this repo (confirmed: `find tests -iname "*cli*"` returns nothing) — verification for this task is an import/construction smoke check, not pytest.

- [ ] **Step 1: Update imports**

In `src/candidate_ranking/cli.py`, change:

```python
from candidate_ranking.models import Assessment, Candidate, JDRequirements, JobDescription
```

to:

```python
from candidate_ranking.models import Assessment, Candidate, JobDescription
```

Change:

```python
from candidate_ranking.scoring.jd_requirements import build_jd_requirements_chain
from candidate_ranking.scoring.jd_skills import build_jd_skills_chain
```

to:

```python
from candidate_ranking.scoring.job_profile import build_job_profile_chain
```

- [ ] **Step 2: Update the checkpoint allow-list**

Change:

```python
_CHECKPOINT_ALLOWED_MSGPACK_MODULES = [
    ("candidate_ranking.models", name)
    for name in ("JobDescription", "Candidate", "JDSkills", "Assessment", "JDRequirements")
]
```

to:

```python
_CHECKPOINT_ALLOWED_MSGPACK_MODULES = [
    ("candidate_ranking.models", name)
    for name in ("JobDescription", "Candidate", "JobProfile", "Assessment")
]
```

- [ ] **Step 3: Update chain construction and graph wiring in `run()`**

Change:

```python
    jd_skills_chain = build_jd_skills_chain(llm)
    jd_requirements_chain = build_jd_requirements_chain(llm)
    qualification_chain = build_qualification_chain(llm)
```

to:

```python
    job_profile_chain = build_job_profile_chain(llm)
    qualification_chain = build_qualification_chain(llm)
```

Change the `build_pipeline_graph(...)` call:

```python
    graph = build_pipeline_graph(
        cfg=cfg,
        jd_skills_chain=jd_skills_chain,
        jd_requirements_chain=jd_requirements_chain,
        qualification_chain=qualification_chain,
```

to:

```python
    graph = build_pipeline_graph(
        cfg=cfg,
        job_profile_chain=job_profile_chain,
        qualification_chain=qualification_chain,
```

- [ ] **Step 4: Update `initial_state`**

Change:

```python
    initial_state = {
        "jds": jds,
        "candidates": candidates,
        "jd_skills": {},
        "jd_requirements": {},
        "shortlists": {},
        "assessment_results": [],
    }
```

to:

```python
    initial_state = {
        "jds": jds,
        "candidates": candidates,
        "job_profiles": {},
        "shortlists": {},
        "assessment_results": [],
    }
```

- [ ] **Step 5: Smoke-check the module imports cleanly**

Run: `python -c "import candidate_ranking.cli"`
Expected: no output, exit code 0 (no `ImportError`/`NameError`).

- [ ] **Step 6: Commit**

```bash
git add src/candidate_ranking/cli.py
git commit -m "refactor: wire cli.py to the merged job_profile_chain"
```

---

### Task 7: Delete the superseded requirements/skills modules

**Files:**
- Delete: `src/candidate_ranking/scoring/jd_requirements.py`
- Delete: `src/candidate_ranking/scoring/jd_skills.py`
- Delete: `tests/scoring/test_jd_requirements.py`
- Delete: `tests/scoring/test_jd_skills.py`
- Modify: `src/candidate_ranking/models.py` (remove `JDRequirement`, `JDRequirements`, `JDSkills`)
- Modify: `tests/test_models.py` (remove the now-dead import names and their tests)

**Interfaces:** None produced — this task only removes now-unreferenced code. (Confirmed via `grep -rln "JDRequirement\|JDSkills\|jd_requirements\|jd_skills" --include=*.py src scripts` before Task 1: every match was in a file this plan already modifies or deletes — `skills.py`, `jd_requirements.py`, `pipeline.py`, `models.py`, `cli.py`, `jd_skills.py`, `llm_assessment.py`. Nothing outside this set references them.)

- [ ] **Step 1: Delete the superseded source files**

```bash
git rm src/candidate_ranking/scoring/jd_requirements.py src/candidate_ranking/scoring/jd_skills.py
git rm tests/scoring/test_jd_requirements.py tests/scoring/test_jd_skills.py
```

- [ ] **Step 2: Remove the superseded models**

In `src/candidate_ranking/models.py`, delete the `JDSkills`, `JDRequirement`, and `JDRequirements` class definitions (everything `JobProfile` was inserted after in Task 1 replaces them — `JobProfile` itself stays).

- [ ] **Step 3: Remove the superseded model tests**

In `tests/test_models.py`, change the import block:

```python
from candidate_ranking.models import (
    Assessment, CareerTrajectory, JobProfile, RedFlag, RequirementEvaluation,
)
```

Delete these three test functions: `test_jd_requirements_lists`, `test_jd_requirements_defaults_empty`, `test_jd_skills_unchanged`.

- [ ] **Step 4: Run the full test suite**

Run: `pytest -v`
Expected: PASS — every test in the suite, with zero references to `JDRequirement`, `JDRequirements`, or `JDSkills` remaining anywhere.

Run: `grep -rn "JDRequirement\|JDSkills\|jd_requirements\|jd_skills" --include=*.py src tests scripts`
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/models.py tests/test_models.py
git commit -m "refactor: delete jd_requirements.py and jd_skills.py, superseded by job_profile.py"
```
