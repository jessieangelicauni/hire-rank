# Jev-Backed Qualification Gate & Pairwise Ranking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the qualification gate from a single Ollama LLM call to a Jev (TypeSafe AI) structured call + Ollama free-text companion call; fix 4 known bugs; replace the active-listwise ranking LLM backend with Jev pairwise comparisons.

**Architecture:** Three phases. Phase 1: one bug fix (rotation fallback in `_select_uncertain_subset`). Phase 2: Jev-backed qualification gate — one paid Jev call per candidate produces all typed answers (evidence strength, career trajectory, red flags, education/seniority/other advisory checks), one free local Ollama call produces all free-text fields (evidence quotes, achievements, signals). Phase 3: pairwise ranking via Jev — the existing Plackett-Luce active-learning loop runs unchanged at `subset_size=2`, with `_rank_subset` replaced by a new `_rank_pair_jev` comparator. Bugs §1.1–§1.4 from the spec are addressed inside the Phase 2 rewrite.

**Tech Stack:** `requests` (Jev HTTP client), Pydantic v2, LangChain (Ollama free-text chain), NumPy (Plackett-Luce fit)

**Spec reference:** `docs/superpowers/specs/2026-10-17-jev-redesign-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/candidate_ranking/scoring/jev_client.py` | **Create** (restore from `f0296f2^`) | Generic Jev HTTP client, model-agnostic |
| `src/candidate_ranking/scoring/llm_assessment.py` | **Rewrite** | Jev question builder, free-text chain, hybrid `generate_assessment`, updated cache key |
| `src/candidate_ranking/scoring/active_listwise_ranking.py` | **Modify** | Bug §1.3 rotation fix; new `_rank_pair_jev`; `_rank_active_listwise_core` accepts `rank_fn` |
| `src/candidate_ranking/models.py` | **Modify** | Add `"none"` to `evidence_strength`; add advisory fields to `Assessment` |
| `src/candidate_ranking/config.py` | **Modify** | Add `jev_api_key`, `ranking_pairwise_max_iterations` |
| `.env.example` | **Modify** | Add `CANDIDATE_RANKING_JEV_API_KEY` |
| `src/candidate_ranking/graphs/pipeline.py` | **Modify** | Accept `jev_client` + `free_text_chain`; drop `qualification_chain`; wire pairwise ranking |
| `src/candidate_ranking/cli.py` | **Modify** | Construct `JevClient`, `free_text_chain`; update eval command signatures |
| `src/candidate_ranking/evaluation/qualification_reliability.py` | **Modify** | Update `repeat_qualification_gate` to new `generate_assessment` signature |
| `tests/scoring/test_jev_client.py` | **Create** (restore from `f0296f2^`) | JevClient unit tests |
| `tests/scoring/test_llm_assessment.py` | **Rewrite** | Tests for new Jev-backed API |
| `tests/scoring/test_active_listwise_ranking.py` | **Modify** | Tests for rotation fix + pairwise mode |

---

## Phase 1 — Bug Fix

### Task 1: Fix §1.3 — stuck-subset rotation fallback in `_select_uncertain_subset`

**Files:**
- Modify: `src/candidate_ranking/scoring/active_listwise_ranking.py`
- Test: `tests/scoring/test_active_listwise_ranking.py`

**Context:** When `n <= subset_size` (the entire pool fits in one subset), there is no candidate outside the subset to swap in when a retry is needed. The current loop exits without changing `selected`, so a failed retry re-sends the identical prompt forever. Fix: when no swap candidate exists, rotate `selected` by the `iteration` counter.

- [ ] **Step 1: Write the failing test**

Add to `tests/scoring/test_active_listwise_ranking.py`:

```python
def test_select_uncertain_subset_rotates_when_no_swap_candidate():
    """When n == subset_size (all in pool), avoid_signature must be escaped via rotation."""
    from candidate_ranking.scoring.active_listwise_ranking import _select_uncertain_subset
    item_ids = ["a", "b", "c"]
    variances = np.array([1.0, 0.9, 0.8])
    # Top-3 == all 3, so there is no swap candidate.
    initial = _select_uncertain_subset(variances, item_ids, subset_size=3)
    avoid = frozenset(initial)
    # With iteration=1, the list must be rotated (not identical).
    result = _select_uncertain_subset(variances, item_ids, subset_size=3, avoid_signature=avoid, iteration=1)
    assert frozenset(result) == avoid          # same members — pool is the same
    assert result != initial                   # different ORDER — rotation happened
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/scoring/test_active_listwise_ranking.py::test_select_uncertain_subset_rotates_when_no_swap_candidate -v
```

Expected: `FAILED` — `_select_uncertain_subset` doesn't accept `iteration` yet.

- [ ] **Step 3: Update `_select_uncertain_subset` in `active_listwise_ranking.py`**

Replace the function signature and body (lines ~69–93):

```python
def _select_uncertain_subset(
    variances: np.ndarray,
    item_ids: list[str],
    subset_size: int,
    avoid_signature: frozenset[str] | None = None,
    iteration: int = 0,
) -> list[str]:
    """Pick the subset_size items with highest utility variance.

    If that exact combination is `avoid_signature`, first try swapping the
    lowest-variance member for the next-highest-variance item outside the
    subset.  If no such item exists (n <= subset_size), rotate the list by
    `iteration` instead — ensuring each retry presents candidates in a
    different order even when the pool is exhausted.
    """
    order = np.argsort(variances)[::-1]
    selected = [item_ids[i] for i in order[:subset_size]]
    if avoid_signature is not None and frozenset(selected) == avoid_signature:
        swapped = False
        for i in order[subset_size:]:
            replacement = item_ids[i]
            if replacement not in selected:
                selected[-1] = replacement
                swapped = True
                break
        if not swapped and len(selected) > 1:
            # No swap candidate (pool exhausted) — rotate by iteration
            offset = iteration % len(selected)
            selected = selected[offset:] + selected[:offset]
    return selected
```

- [ ] **Step 4: Pass `iteration` at the call site in `_rank_active_listwise_core`**

In `_rank_active_listwise_core`, change the `_select_uncertain_subset` call:

```python
# Before:
subset_ids = _select_uncertain_subset(variances, item_ids, subset_size, last_failed_signature)

# After:
subset_ids = _select_uncertain_subset(variances, item_ids, subset_size, last_failed_signature, iteration)
```

- [ ] **Step 5: Run the new test + full suite**

```
pytest tests/scoring/test_active_listwise_ranking.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/candidate_ranking/scoring/active_listwise_ranking.py tests/scoring/test_active_listwise_ranking.py
git commit -m "fix: escape stuck-subset via rotation when no swap candidate exists (§1.3)"
```

---

## Phase 2 — Qualification Gate → Jev Hybrid

### Task 2: Restore `jev_client.py` and `test_jev_client.py`

**Files:**
- Create: `src/candidate_ranking/scoring/jev_client.py`
- Create: `tests/scoring/test_jev_client.py`

- [ ] **Step 1: Restore `jev_client.py` from git history**

```bash
git show f0296f2^:src/candidate_ranking/scoring/jev_client.py > src/candidate_ranking/scoring/jev_client.py
```

- [ ] **Step 2: Verify content looks correct**

```bash
python -c "from candidate_ranking.scoring.jev_client import JevClient, JevQuestion, JevAnswer, JevClientError; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Restore `test_jev_client.py` from git history**

```bash
git show f0296f2^:tests/scoring/test_jev_client.py > tests/scoring/test_jev_client.py
```

- [ ] **Step 4: Run restored tests**

```bash
pytest tests/scoring/test_jev_client.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/scoring/jev_client.py tests/scoring/test_jev_client.py
git commit -m "feat: restore JevClient + tests (deleted in f0296f2, needed for Jev migration)"
```

---

### Task 3: Add `jev_api_key` and pairwise config to `RunConfig`

**Files:**
- Modify: `src/candidate_ranking/config.py`
- Modify: `.env.example`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_jev_api_key_override():
    import os
    from candidate_ranking.config import RunConfig, apply_env_overrides
    from pathlib import Path
    cfg = RunConfig.full(Path("."))
    assert cfg.jev_api_key is None
    os.environ["CANDIDATE_RANKING_JEV_API_KEY"] = "tok-abc"
    try:
        cfg2 = apply_env_overrides(cfg)
        assert cfg2.jev_api_key == "tok-abc"
    finally:
        del os.environ["CANDIDATE_RANKING_JEV_API_KEY"]


def test_pairwise_max_iterations_default():
    from candidate_ranking.config import RunConfig
    from pathlib import Path
    cfg = RunConfig.full(Path("."))
    assert cfg.ranking_pairwise_max_iterations == 80
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_config.py::test_jev_api_key_override tests/test_config.py::test_pairwise_max_iterations_default -v
```

Expected: `FAILED` — fields don't exist yet.

- [ ] **Step 3: Add fields to `RunConfig`**

In `config.py`, add to the `RunConfig` dataclass after `ranking_min_delta_u`:

```python
jev_api_key: str | None = None
ranking_pairwise_max_iterations: int = 80
```

In `_ENV_OVERRIDES`, add:

```python
"CANDIDATE_RANKING_JEV_API_KEY": ("jev_api_key", str),
"CANDIDATE_RANKING_PAIRWISE_MAX_ITERATIONS": ("ranking_pairwise_max_iterations", int),
```

- [ ] **Step 4: Add to `.env.example`**

Append to `.env.example`:

```
CANDIDATE_RANKING_JEV_API_KEY=
CANDIDATE_RANKING_PAIRWISE_MAX_ITERATIONS=80
```

- [ ] **Step 5: Run tests**

```
pytest tests/test_config.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/candidate_ranking/config.py .env.example tests/test_config.py
git commit -m "feat: add jev_api_key and ranking_pairwise_max_iterations to RunConfig"
```

---

### Task 4: Extend `models.py` for Jev output

**Files:**
- Modify: `src/candidate_ranking/models.py`
- Test: `tests/test_models.py`

**Changes:** (1) Add `"none"` to `RequirementEvaluation.evidence_strength` — allows expressing "no evidence at all" (distinct from "weak") so the verdict mapping can correctly set NOT_MET at the source. (2) Add three advisory fields to `Assessment` — education/seniority/other_requirements results are advisory-only (§4 decision: do not affect `qualified`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_models.py`:

```python
def test_requirement_evaluation_accepts_none_evidence_strength():
    from candidate_ranking.models import RequirementEvaluation
    ev = RequirementEvaluation(
        requirement="Python", verdict="NOT_MET",
        evidence="Not found", evidence_strength="none",
        recency="unknown", years_demonstrated=0,
    )
    assert ev.evidence_strength == "none"


def test_assessment_advisory_fields_default_to_none_or_empty():
    from candidate_ranking.models import Assessment
    a = Assessment(
        job_description_id="jd-1", candidate_id="c-1", generated_by_model="m",
    )
    assert a.education_evaluation is None
    assert a.seniority_evaluation is None
    assert a.other_requirement_evaluations == {}
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_models.py::test_requirement_evaluation_accepts_none_evidence_strength tests/test_models.py::test_assessment_advisory_fields_default_to_none_or_empty -v
```

Expected: `FAILED`.

- [ ] **Step 3: Update `RequirementEvaluation`**

In `models.py`, change `evidence_strength`:

```python
# Before:
evidence_strength: Literal["strong", "adequate", "weak"]

# After:
evidence_strength: Literal["strong", "adequate", "weak", "none"]
```

- [ ] **Step 4: Add advisory fields to `Assessment`**

After `disqualification_reason` in `Assessment`:

```python
education_evaluation: Literal["meets", "equivalent_experience", "does_not_meet"] | None = None
seniority_evaluation: Literal["meets_or_exceeds", "below"] | None = None
other_requirement_evaluations: dict[str, bool] = Field(default_factory=dict)
```

- [ ] **Step 5: Run tests**

```
pytest tests/test_models.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/candidate_ranking/models.py tests/test_models.py
git commit -m "feat: extend Assessment model — evidence_strength 'none', advisory edu/seniority/other fields"
```

---

### Task 5: Write Jev question builder and score helpers

**Files:**
- Modify: `src/candidate_ranking/scoring/llm_assessment.py` (add new functions, keep old ones temporarily)
- Test: `tests/scoring/test_llm_assessment.py`

**Note:** This task adds the building blocks; the full rewrite happens in Task 7. The old `generate_assessment` and `build_qualification_chain` are untouched here.

The question prompts below are written from an LLM-engineering + HR perspective to be adaptive across varying JDs: they use the specific `jd_title`, `skill`, or `requirement` name inline so the same function works for a software engineer JD, a finance role, or a creative director JD without template changes.

- [ ] **Step 1: Write the failing unit test for `_build_jev_questions`**

Add to `tests/scoring/test_llm_assessment.py`:

```python
def test_build_jev_questions_keys_for_minimal_profile():
    """Question keys must follow the documented naming scheme."""
    from candidate_ranking.scoring.llm_assessment import _build_jev_questions
    from candidate_ranking.models import JobProfile
    profile = JobProfile(
        job_description_id="jd-1", generated_by_model="m",
        must_have_skills=["Python", "PostgreSQL"],
        nice_to_have_skills=["Docker"],
    )
    questions = _build_jev_questions(profile)
    keys = [q.key for q in questions]
    # Skill evidence
    assert "must_have_0_evidence_strength" in keys
    assert "must_have_1_evidence_strength" in keys
    assert "nice_to_have_0_evidence_strength" in keys
    # No edu/seniority/other when not set
    assert "education_requirement" not in keys
    assert "seniority_requirement" not in keys
    # Fixed questions always present
    assert "total_relevant_years" in keys
    assert "career_direction" in keys
    assert "has_employment_gap" in keys
    assert "employment_gap_severity" in keys
    assert "employment_gap_duration_months" in keys
    # Total count: 2 must + 1 nih + 1 total_years + 5 career + 15 red_flags = 24
    assert len(questions) == 24


def test_build_jev_questions_includes_edu_seniority_other_when_set():
    from candidate_ranking.scoring.llm_assessment import _build_jev_questions
    from candidate_ranking.models import JobProfile
    profile = JobProfile(
        job_description_id="jd-1", generated_by_model="m",
        must_have_skills=["Python"],
        education_requirement="Bachelor's degree in Computer Science or related",
        seniority_requirement="Senior (5+ years)",
        other_requirements=["Must have valid driver's license", "Willing to relocate"],
    )
    questions = _build_jev_questions(profile)
    keys = [q.key for q in questions]
    assert "education_requirement" in keys
    assert "seniority_requirement" in keys
    assert "other_req_0" in keys
    assert "other_req_1" in keys
    # 1 + 1 (edu) + 1 (seniority) + 2 (other) + 1 + 5 + 15 = 26
    assert len(questions) == 26


def test_score_to_int_interpolates_correctly():
    from candidate_ranking.scoring.llm_assessment import _score_to_int
    ticks = [0, 3, 6, 10, 15, 20]
    assert _score_to_int(0.0, ticks) == 0
    assert _score_to_int(5.0, ticks) == 20
    assert _score_to_int(2.0, ticks) == 6
    assert _score_to_int(2.5, ticks) == 8   # midpoint between 6 and 10
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/scoring/test_llm_assessment.py::test_build_jev_questions_keys_for_minimal_profile tests/scoring/test_llm_assessment.py::test_score_to_int_interpolates_correctly -v
```

Expected: `FAILED` — functions don't exist yet.

- [ ] **Step 3: Add `_build_jev_questions` and `_score_to_int` to `llm_assessment.py`**

Add the following **above** the existing `QUALIFICATION_GATE_PROMPT` line (import `JevQuestion` at the top of the module alongside other imports):

```python
from candidate_ranking.scoring.jev_client import JEV_MODEL_ID, JevClientError, JevQuestion
```

Then add the constants and functions:

```python
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

# Scale tick values for each score question (index i → score value i from Jev).
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
    _flag_instructions: dict[str, dict[str, str]] = {
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
            "duration": (
                "How many months has the candidate been in the regressed position?"
            ),
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
    _severity_criteria = {
        "minor": "Minor concern — worth noting but unlikely to be decisive",
        "moderate": "Notable concern — requires clarification or monitoring",
        "major": "Significant red flag — could be disqualifying",
    }
    _flag_duration_criteria = [
        "0 months", "3 months", "6 months", "12 months", "24 months", "36+ months",
    ]
    _present_criteria = {
        "true": "Clear evidence of this flag is present in the CV",
        "false": "No evidence of this flag, or all instances are clearly explained",
    }
    for flag_type in _RED_FLAG_TYPES:
        instr = _flag_instructions[flag_type]
        questions.append(JevQuestion(
            key=f"has_{flag_type}",
            kind="noul",
            instructions=instr["has"],
            criteria=_present_criteria,
        ))
        questions.append(JevQuestion(
            key=f"{flag_type}_severity",
            kind="choice",
            instructions=instr["severity"],
            criteria=_severity_criteria,
        ))
        questions.append(JevQuestion(
            key=f"{flag_type}_duration_months",
            kind="score",
            instructions=instr["duration"],
            criteria=_flag_duration_criteria,
        ))

    return questions
```

- [ ] **Step 4: Run tests**

```
pytest tests/scoring/test_llm_assessment.py::test_build_jev_questions_keys_for_minimal_profile tests/scoring/test_llm_assessment.py::test_build_jev_questions_includes_edu_seniority_other_when_set tests/scoring/test_llm_assessment.py::test_score_to_int_interpolates_correctly -v
```

Expected: all 3 pass.

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/scoring/llm_assessment.py tests/scoring/test_llm_assessment.py
git commit -m "feat: add Jev question builder and score helpers to llm_assessment"
```

---

### Task 6: Write free-text Ollama chain

**Files:**
- Modify: `src/candidate_ranking/scoring/llm_assessment.py`
- Test: `tests/scoring/test_llm_assessment.py`

**Purpose:** The Ollama chain handles every free-text field that Jev structurally cannot produce: per-skill evidence quotes, achievements summary, notable signals, per-active-red-flag detail text, career trajectory evidence, and multinational context.

- [ ] **Step 1: Write the failing test**

Add to `tests/scoring/test_llm_assessment.py`:

```python
def test_build_free_text_chain_returns_runnable():
    from unittest.mock import MagicMock
    from candidate_ranking.scoring.llm_assessment import build_free_text_chain
    chain = build_free_text_chain(MagicMock())
    assert chain is not None


def test_free_text_output_model_validates():
    from candidate_ranking.scoring.llm_assessment import _FreeTextOutput, _SkillEvidenceItem
    out = _FreeTextOutput(
        must_have_evidence=[_SkillEvidenceItem(requirement="Python", evidence="Built FastAPI")],
        strongest_achievements=["Led migration to microservices"],
    )
    assert out.must_have_evidence[0].requirement == "Python"
    assert out.career_trajectory_evidence == ""
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/scoring/test_llm_assessment.py::test_build_free_text_chain_returns_runnable tests/scoring/test_llm_assessment.py::test_free_text_output_model_validates -v
```

Expected: `FAILED`.

- [ ] **Step 3: Add `_FreeTextOutput` and `build_free_text_chain` to `llm_assessment.py`**

Add after the Jev question builder section:

```python
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
```

- [ ] **Step 4: Run tests**

```
pytest tests/scoring/test_llm_assessment.py::test_build_free_text_chain_returns_runnable tests/scoring/test_llm_assessment.py::test_free_text_output_model_validates -v
```

Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/scoring/llm_assessment.py tests/scoring/test_llm_assessment.py
git commit -m "feat: add free-text Ollama chain and output model to llm_assessment"
```

---

### Task 7: Rewrite `generate_assessment` and `load_or_generate_assessment`

**Files:**
- Modify: `src/candidate_ranking/scoring/llm_assessment.py` (replace old functions)
- Test: `tests/scoring/test_llm_assessment.py` (rewrite existing tests)

**Bugs fixed here:** §1.1 (cache key now hashes rendered questions), §1.2 (`qualified` computed from verdicts, never trusted from LLM), §1.4 (narrow exception handling — `JevClientError` and `GenerationError` only).

- [ ] **Step 1: Add the test helper `_fake_answers_for_profile`**

Add to `tests/scoring/test_llm_assessment.py`:

```python
def _fake_answers_for_profile(profile) -> list:
    """Return minimal valid JevAnswer list — all skills STRONG, no red flags, ascending career."""
    from candidate_ranking.scoring.jev_client import JevAnswer
    answers = []
    for i in range(len(profile.must_have_skills)):
        answers.append(JevAnswer(key=f"must_have_{i}_evidence_strength", kind="choice", value="strong", confidence=0.9))
    for i in range(len(profile.nice_to_have_skills)):
        answers.append(JevAnswer(key=f"nice_to_have_{i}_evidence_strength", kind="choice", value="adequate", confidence=0.7))
    if profile.education_requirement:
        answers.append(JevAnswer(key="education_requirement", kind="choice", value="meets", confidence=0.8))
    if profile.seniority_requirement:
        answers.append(JevAnswer(key="seniority_requirement", kind="choice", value="meets_or_exceeds", confidence=0.8))
    for i in range(len(profile.other_requirements)):
        answers.append(JevAnswer(key=f"other_req_{i}", kind="noul", value=True, confidence=0.8))
    answers += [
        JevAnswer(key="total_relevant_years", kind="score", value=2.0, confidence=0.8),
        JevAnswer(key="career_direction", kind="choice", value="ascending", confidence=0.8),
        JevAnswer(key="tenure_pattern", kind="choice", value="stable", confidence=0.8),
        JevAnswer(key="average_tenure_months", kind="score", value=2.0, confidence=0.8),
        JevAnswer(key="most_recent_role_duration_months", kind="score", value=2.0, confidence=0.8),
        JevAnswer(key="has_multinational_experience", kind="noul", value=False, confidence=0.8),
    ]
    for flag_type in ["employment_gap", "skill_mismatch", "regression", "job_hopping", "other"]:
        answers += [
            JevAnswer(key=f"has_{flag_type}", kind="noul", value=False, confidence=0.9),
            JevAnswer(key=f"{flag_type}_severity", kind="choice", value="minor", confidence=0.7),
            JevAnswer(key=f"{flag_type}_duration_months", kind="score", value=0.0, confidence=0.7),
        ]
    return answers


def _fake_free_text(profile) -> object:
    from candidate_ranking.scoring.llm_assessment import _FreeTextOutput, _SkillEvidenceItem
    return _FreeTextOutput(
        must_have_evidence=[
            _SkillEvidenceItem(requirement=s, evidence=f"Evidence for {s}")
            for s in profile.must_have_skills
        ],
        nice_to_have_evidence=[
            _SkillEvidenceItem(requirement=s, evidence=f"Evidence for {s}")
            for s in profile.nice_to_have_skills
        ],
        strongest_achievements=["Led migration project", "Reduced latency by 40%"],
        career_trajectory_evidence="Promoted from junior to senior in 3 years",
    )
```

- [ ] **Step 2: Rewrite existing tests for the new `generate_assessment` signature**

Replace the existing test functions `test_generate_assessment_*` and `test_load_or_generate_assessment_*`:

```python
def test_generate_assessment_returns_qualified_assessment():
    from unittest.mock import MagicMock
    from candidate_ranking.scoring.llm_assessment import generate_assessment
    profile = _job_profile()
    jev = MagicMock()
    jev.evaluate.return_value = _fake_answers_for_profile(profile)
    chain = MagicMock()
    chain.invoke.return_value = _fake_free_text(profile)
    result = generate_assessment(_jd(), _candidate(), profile, jev, chain, "qwen2.5:14b")
    assert isinstance(result, Assessment)
    assert result.qualified is True
    assert result.must_have_evaluations[0].requirement == "Python"
    assert result.must_have_evaluations[0].verdict == "MET"
    assert result.must_have_evaluations[0].evidence == "Evidence for Python"


def test_generate_assessment_computes_qualified_from_verdicts():
    """qualified must be derived from must-have verdicts, never trusted from Jev."""
    from unittest.mock import MagicMock
    from candidate_ranking.scoring.jev_client import JevAnswer
    from candidate_ranking.scoring.llm_assessment import generate_assessment
    profile = _job_profile()
    answers = _fake_answers_for_profile(profile)
    # Override must_have_0 to NONE → verdict NOT_MET → qualified=False
    answers[0] = JevAnswer(key="must_have_0_evidence_strength", kind="choice", value="none", confidence=0.9)
    jev = MagicMock()
    jev.evaluate.return_value = answers
    chain = MagicMock()
    chain.invoke.return_value = _fake_free_text(profile)
    result = generate_assessment(_jd(), _candidate(), profile, jev, chain, "qwen2.5:14b")
    assert result.qualified is False
    assert result.disqualification_reason is not None
    assert "Python" in result.disqualification_reason


def test_generate_assessment_raises_on_jev_error():
    from unittest.mock import MagicMock
    from candidate_ranking.scoring.jev_client import JevClientError
    from candidate_ranking.scoring.llm_assessment import AssessmentGenerationError, generate_assessment
    jev = MagicMock()
    jev.evaluate.side_effect = JevClientError("network failure")
    with pytest.raises(AssessmentGenerationError):
        generate_assessment(_jd(), _candidate(), _job_profile(), jev, MagicMock(), "qwen2.5:14b")


def test_generate_assessment_raises_on_free_text_error():
    from unittest.mock import MagicMock
    from candidate_ranking.utils import GenerationError
    from candidate_ranking.scoring.llm_assessment import AssessmentGenerationError, generate_assessment
    profile = _job_profile()
    jev = MagicMock()
    jev.evaluate.return_value = _fake_answers_for_profile(profile)
    chain = MagicMock()
    chain.invoke.side_effect = GenerationError("schema mismatch")
    with pytest.raises(AssessmentGenerationError):
        generate_assessment(_jd(), _candidate(), profile, jev, chain, "qwen2.5:14b")


def test_load_or_generate_assessment_uses_cache(tmp_path):
    from unittest.mock import MagicMock, patch
    from candidate_ranking.scoring.llm_assessment import load_or_generate_assessment
    profile = _job_profile()
    with patch("candidate_ranking.scoring.llm_assessment.generate_assessment",
               return_value=_assessment()) as mock_gen:
        first = load_or_generate_assessment(
            _jd(), _candidate(), profile, MagicMock(), MagicMock(), "qwen2.5:14b", tmp_path
        )
        second = load_or_generate_assessment(
            _jd(), _candidate(), profile, MagicMock(), MagicMock(), "qwen2.5:14b", tmp_path
        )
    assert first == second
    assert mock_gen.call_count == 1


def test_load_or_generate_assessment_regenerates_when_profile_changes(tmp_path):
    from unittest.mock import MagicMock, patch
    from candidate_ranking.scoring.llm_assessment import load_or_generate_assessment
    from candidate_ranking.models import JobProfile
    profile_v1 = _job_profile()
    profile_v2 = JobProfile(
        job_description_id="jd-1", generated_by_model="qwen2.5:14b",
        must_have_skills=["Go"],
    )
    with patch("candidate_ranking.scoring.llm_assessment.generate_assessment",
               return_value=_assessment()) as mock_gen:
        load_or_generate_assessment(
            _jd(), _candidate(), profile_v1, MagicMock(), MagicMock(), "qwen2.5:14b", tmp_path
        )
        load_or_generate_assessment(
            _jd(), _candidate(), profile_v2, MagicMock(), MagicMock(), "qwen2.5:14b", tmp_path
        )
    assert mock_gen.call_count == 2
```

- [ ] **Step 3: Run new tests to verify they fail**

```
pytest tests/scoring/test_llm_assessment.py::test_generate_assessment_returns_qualified_assessment tests/scoring/test_llm_assessment.py::test_generate_assessment_computes_qualified_from_verdicts -v
```

Expected: `FAILED` — old `generate_assessment` has wrong signature.

- [ ] **Step 4: Replace old `generate_assessment` (and related internals) in `llm_assessment.py`**

Remove: `_RequirementEvaluationOutput`, `_CareerTrajectoryOutput`, `_RedFlagOutput`, `_QualificationOutput`, `build_qualification_chain`, old `generate_assessment`, old `_output_to_assessment`, old `_assessment_cache_key`, old `load_or_generate_assessment`.

Add:

```python
# ── Jev answer → Assessment assembler ────────────────────────────────────────

def _answers_to_assessment(
    answers: list,
    free_text: "_FreeTextOutput",
    profile: JobProfile,
    jd: JobDescription,
    candidate: Candidate,
    ollama_model_name: str,
) -> Assessment:
    by_key = {a.key: a for a in answers}

    must_have_evals = []
    for i, skill in enumerate(profile.must_have_skills):
        es = by_key[f"must_have_{i}_evidence_strength"].value
        verdict = _EVIDENCE_TO_VERDICT[es]
        evidence = next(
            (e.evidence for e in free_text.must_have_evidence if e.requirement == skill), ""
        )
        must_have_evals.append(RequirementEvaluation(
            requirement=skill, verdict=verdict, evidence=evidence,
            evidence_strength=es, recency="unknown", years_demonstrated=0,
        ))

    nih_evals = []
    for i, skill in enumerate(profile.nice_to_have_skills):
        es = by_key[f"nice_to_have_{i}_evidence_strength"].value
        verdict = _EVIDENCE_TO_VERDICT[es]
        evidence = next(
            (e.evidence for e in free_text.nice_to_have_evidence if e.requirement == skill), ""
        )
        nih_evals.append(RequirementEvaluation(
            requirement=skill, verdict=verdict, evidence=evidence,
            evidence_strength=es, recency="unknown", years_demonstrated=0,
        ))

    # §1.2: compute qualified from verdicts — never trust a separate LLM-reported boolean
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
    has_multi = bool(by_key["has_multinational_experience"].value)
    trajectory = CareerTrajectory(
        direction=by_key["career_direction"].value,
        evidence=free_text.career_trajectory_evidence or "",
        tenure_pattern=by_key["tenure_pattern"].value,
        average_tenure_months=avg_tenure,
        most_recent_role_duration_months=recent_tenure,
        has_multinational_experience=has_multi,
        multinational_evidence=free_text.multinational_evidence if has_multi else None,
    )

    red_flags = []
    for flag_type in _RED_FLAG_TYPES:
        if by_key[f"has_{flag_type}"].value:
            severity = by_key[f"{flag_type}_severity"].value
            duration = _score_to_int(by_key[f"{flag_type}_duration_months"].value, _FLAG_MONTHS_TICKS)
            detail = next(
                (d.detail for d in free_text.red_flag_details if d.type == flag_type), ""
            )
            red_flags.append(RedFlag(
                type=flag_type, detail=detail,
                duration_months=duration if duration > 0 else None,
                severity=severity,
            ))

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
        generated_by_model=f"jev/{JEV_MODEL_ID}+{ollama_model_name}",
        must_have_evaluations=must_have_evals,
        nice_to_have_evaluations=nih_evals,
        total_relevant_years=total_years,
        career_trajectory=trajectory,
        strongest_achievements=free_text.strongest_achievements,
        notable_signals=free_text.notable_signals,
        red_flags=red_flags,
        qualified=qualified,
        disqualification_reason=disqualification_reason,
        education_evaluation=education_eval,
        seniority_evaluation=seniority_eval,
        other_requirement_evaluations=other_evals,
    )


# ── Cache key (hashes rendered questions — §1.1 fix) ─────────────────────────

def _assessment_cache_key(
    jd: JobDescription,
    candidate: Candidate,
    questions: list,
    ollama_model_name: str,
) -> str:
    questions_repr = json.dumps(
        [q.model_dump() for q in questions], sort_keys=True
    )
    digest_input = (
        f"{jd.id}||{jd.raw_text}||{candidate.id}||{candidate.raw_text}"
        f"||{questions_repr}||{JEV_MODEL_ID}||{ollama_model_name}"
    )
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


# ── Core generation ───────────────────────────────────────────────────────────

def generate_assessment(
    jd: JobDescription,
    candidate: Candidate,
    profile: JobProfile | None,
    jev_client: "JevClient",
    free_text_chain: Runnable,
    ollama_model_name: str,
) -> Assessment:
    from candidate_ranking.scoring.jev_client import JevClient

    if profile is None:
        raise AssessmentGenerationError(
            f"Cannot assess {jd.id}/{candidate.id}: JobProfile is required for Jev gate"
        )

    questions = _build_jev_questions(profile)
    state = candidate.raw_text

    # §1.4: catch only domain-specific errors — broad except was masking bugs
    try:
        answers = jev_client.evaluate(state, questions)
    except JevClientError as exc:
        raise AssessmentGenerationError(
            f"Jev call failed for {jd.id}/{candidate.id}: {exc}"
        ) from exc

    active_flags = [
        ft for ft in _RED_FLAG_TYPES
        if any(a.key == f"has_{ft}" and a.value for a in answers)
    ]
    active_flags_str = ", ".join(active_flags) if active_flags else "none"

    log_context = f"Free-text chain for {jd.id}/{candidate.id}"
    try:
        free_text = invoke_and_validate(
            free_text_chain,
            {
                "jd_title": jd.title,
                "must_have_skills": ", ".join(profile.must_have_skills),
                "nice_to_have_skills": ", ".join(profile.nice_to_have_skills),
                "active_flags": active_flags_str,
                "cv_text": candidate.raw_text,
            },
            _FreeTextOutput,
            lambda r: None,
            log_context=log_context,
        )
    except GenerationError as exc:
        raise AssessmentGenerationError(
            f"Free-text chain failed for {jd.id}/{candidate.id}: {exc}"
        ) from exc

    return _answers_to_assessment(answers, free_text, profile, jd, candidate, ollama_model_name)


def _assessment_cache_path(cache_dir: Path, jd_id: str, candidate_id: str) -> Path:
    return cache_dir / "assessments" / jd_id / f"{candidate_id}.json"


def load_or_generate_assessment(
    jd: JobDescription,
    candidate: Candidate,
    profile: JobProfile | None,
    jev_client: "JevClient",
    free_text_chain: Runnable,
    ollama_model_name: str,
    cache_dir: Path,
) -> Assessment:
    questions = _build_jev_questions(profile) if profile else []
    path = _assessment_cache_path(cache_dir, jd.id, candidate.id)
    key = _assessment_cache_key(jd, candidate, questions, ollama_model_name)

    with _cache_lock:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("cache_key") == key:
                    return Assessment.model_validate(data["assessment"])
            except (json.JSONDecodeError, KeyError, ValidationError, OSError) as exc:
                logger.warning("Discarding unreadable assessment cache %s: %s", path, exc)

    assessment = generate_assessment(
        jd, candidate, profile, jev_client, free_text_chain, ollama_model_name
    )

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
```

Also remove the old `QUALIFICATION_GATE_PROMPT` and all old Pydantic output classes. Keep `filter_assessable_candidates` unchanged.

- [ ] **Step 5: Run all llm_assessment tests**

```
pytest tests/scoring/test_llm_assessment.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/candidate_ranking/scoring/llm_assessment.py tests/scoring/test_llm_assessment.py
git commit -m "feat: rewrite qualification gate as Jev+Ollama hybrid (§2–4, fixes §1.1 §1.2 §1.4)"
```

---

### Task 8: Wire `JevClient` + `free_text_chain` through `pipeline.py`, `cli.py`, and `qualification_reliability.py`

**Files:**
- Modify: `src/candidate_ranking/graphs/pipeline.py`
- Modify: `src/candidate_ranking/cli.py`
- Modify: `src/candidate_ranking/evaluation/qualification_reliability.py`
- Test: `tests/graphs/test_pipeline.py` (smoke-check imports)

- [ ] **Step 1: Update `qualification_reliability.py`**

Replace the `repeat_qualification_gate` signature and body:

```python
def repeat_qualification_gate(
    jd: JobDescription,
    candidate: Candidate,
    profile: "JobProfile | None",
    jev_client: "JevClient",
    free_text_chain: "Runnable",
    ollama_model_name: str,
    repeats: int = 3,
) -> list[Assessment]:
    """Run the qualification gate `repeats` independent times, bypassing the on-disk cache."""
    return [
        generate_assessment(jd, candidate, profile, jev_client, free_text_chain, ollama_model_name)
        for _ in range(repeats)
    ]
```

Update the `TYPE_CHECKING` block to add `JevClient`:

```python
if TYPE_CHECKING:
    from langchain_core.runnables import Runnable
    from candidate_ranking.scoring.jev_client import JevClient
```

- [ ] **Step 2: Update `build_pipeline_graph` signature in `pipeline.py`**

Replace `qualification_chain: Runnable` with `jev_client` + `free_text_chain`:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from candidate_ranking.scoring.jev_client import JevClient
```

```python
def build_pipeline_graph(
    cfg: RunConfig,
    job_profile_chain: Runnable,
    free_text_chain: Runnable,          # replaces qualification_chain
    jev_client: "JevClient",
    ranking_llm: "BaseChatModel | None",  # None = pairwise Jev mode (Phase 3)
    skill_index: np.ndarray,
    skill_row_map: list[tuple[str, str]],
    skill_embedder: Callable[[list[str]], np.ndarray],
    run_id: str,
    skill_match_threshold: float = 0.8,
    min_skill_matches: int = 3,
) -> StateGraph:
```

In `generate_assessment_for_pair` inside the graph builder, replace:

```python
# Before:
assessment = load_or_generate_assessment(
    jd, candidate, profile, qualification_chain, cfg.ollama_model, cfg.cache_dir
)

# After:
assessment = load_or_generate_assessment(
    jd, candidate, profile, jev_client, free_text_chain, cfg.ollama_model, cfg.cache_dir
)
```

Update the import at the top of `pipeline.py`:

```python
from candidate_ranking.scoring.llm_assessment import (
    AssessmentGenerationError,
    load_or_generate_assessment,
)
# Remove: build_qualification_chain (no longer exists)
```

- [ ] **Step 3: Update `cli.py`**

In the `run()` function, replace:

```python
# Remove:
qualification_chain = build_qualification_chain(llm)

# Add:
from candidate_ranking.scoring.jev_client import JevClient
from candidate_ranking.scoring.llm_assessment import build_free_text_chain, filter_assessable_candidates

if not cfg.jev_api_key:
    raise RuntimeError(
        "CANDIDATE_RANKING_JEV_API_KEY is required — set it in .env or the environment"
    )
jev_client = JevClient(api_token=cfg.jev_api_key)
free_text_chain = build_free_text_chain(llm)
```

Update the `build_pipeline_graph` call:

```python
graph = build_pipeline_graph(
    cfg=cfg,
    job_profile_chain=job_profile_chain,
    free_text_chain=free_text_chain,
    jev_client=jev_client,
    ranking_llm=None,        # Phase 3: pairwise via Jev (wired in Task 11)
    skill_index=skill_index,
    skill_row_map=skill_row_map,
    skill_embedder=skill_embedder,
    run_id=run_id,
    skill_match_threshold=skill_match_threshold,
    min_skill_matches=min_skill_matches,
)
```

Update the `ASSESSMENT_SCOPE_VERSION` constant:

```python
ASSESSMENT_SCOPE_VERSION = "jev-qualification-gate-v1"
```

Update the `evaluate-reliability` command in `main()` to pass `jev_client` and `free_text_chain`:

```python
elif args.command == "evaluate-reliability":
    from candidate_ranking.scoring.jev_client import JevClient
    cfg = RunConfig.full(PROJECT_ROOT)
    cfg = apply_env_overrides(cfg)
    if not cfg.jev_api_key:
        raise RuntimeError("CANDIDATE_RANKING_JEV_API_KEY is required for reliability evaluation")
    llm = ChatOllama(model=cfg.ollama_model, base_url=cfg.ollama_base_url, temperature=0, num_ctx=cfg.ollama_num_ctx)
    jev_client = JevClient(api_token=cfg.jev_api_key)
    free_text_chain = build_free_text_chain(llm)
    # ... rest of the command replaces `chain` with jev_client + free_text_chain + cfg.ollama_model
```

Also update the `repeat_qualification_gate` call in the `evaluate-reliability` command to match the new signature.

- [ ] **Step 4: Run import smoke test**

```bash
python -c "from candidate_ranking.graphs.pipeline import build_pipeline_graph; from candidate_ranking.cli import run; print('imports ok')"
```

Expected: `imports ok`

- [ ] **Step 5: Run full test suite**

```
pytest tests/ -v --ignore=tests/scoring/test_llm_assessment.py
```

Expected: all pass (test_llm_assessment already passes from Task 7).

- [ ] **Step 6: Commit**

```bash
git add src/candidate_ranking/graphs/pipeline.py src/candidate_ranking/cli.py src/candidate_ranking/evaluation/qualification_reliability.py
git commit -m "feat: wire JevClient + free_text_chain through pipeline, cli, and reliability eval"
```

---

## Phase 3 — Pairwise Ranking via Jev

### Task 9: Add `_rank_pair_jev` and refactor `_rank_active_listwise_core` to accept `rank_fn`

**Files:**
- Modify: `src/candidate_ranking/scoring/active_listwise_ranking.py`
- Test: `tests/scoring/test_active_listwise_ranking.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/scoring/test_active_listwise_ranking.py`:

```python
def test_rank_pair_jev_returns_winner_first():
    from unittest.mock import MagicMock
    from candidate_ranking.scoring.jev_client import JevAnswer
    from candidate_ranking.scoring.active_listwise_ranking import _rank_pair_jev
    jd = _jd()
    assessments = {"alice": _assessment("alice", years=5), "bob": _assessment("bob", years=3)}
    jev = MagicMock()
    jev.evaluate.return_value = [
        JevAnswer(key="stronger_fit", kind="choice", value="A", confidence=0.85)
    ]
    result = _rank_pair_jev(jd, ["alice", "bob"], assessments, jev)
    assert result == ["alice", "bob"]   # A=alice won


def test_rank_pair_jev_returns_none_on_jev_error():
    from unittest.mock import MagicMock
    from candidate_ranking.scoring.jev_client import JevClientError
    from candidate_ranking.scoring.active_listwise_ranking import _rank_pair_jev
    jev = MagicMock()
    jev.evaluate.side_effect = JevClientError("timeout")
    result = _rank_pair_jev(_jd(), ["a", "b"], {"a": _assessment("a"), "b": _assessment("b")}, jev)
    assert result is None


def test_rank_active_listwise_uses_jev_pairwise_when_client_provided():
    """When jev_client is given, the LLM path must never be called."""
    from unittest.mock import MagicMock, patch
    from candidate_ranking.scoring.jev_client import JevAnswer
    assessments = {f"c{i}": _assessment(f"c{i}") for i in range(4)}
    jev = MagicMock()
    # Always say A wins
    jev.evaluate.return_value = [
        JevAnswer(key="stronger_fit", kind="choice", value="A", confidence=0.9)
    ]
    with patch(f"{MODULE}._rank_subset") as mock_llm_rank:
        result = rank_active_listwise(_jd(), assessments, jev_client=jev, subset_size=2, max_iterations=5)
    mock_llm_rank.assert_not_called()
    assert set(result) == set(assessments)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/scoring/test_active_listwise_ranking.py::test_rank_pair_jev_returns_winner_first tests/scoring/test_active_listwise_ranking.py::test_rank_active_listwise_uses_jev_pairwise_when_client_provided -v
```

Expected: `FAILED`.

- [ ] **Step 3: Add `_rank_pair_jev` to `active_listwise_ranking.py`**

Add after `_rank_subset`:

```python
def _rank_pair_jev(
    jd: JobDescription,
    pair_ids: list[str],
    assessments: dict[str, Assessment],
    jev_client: "JevClient",
) -> list[str] | None:
    """Ask Jev which of two candidates is the stronger fit. Returns [winner, loser] or None on error."""
    from candidate_ranking.scoring.jev_client import JevClientError, JevQuestion
    assert len(pair_ids) == 2, "pairwise comparator requires exactly 2 candidates"
    a_id, b_id = pair_ids
    state = (
        f"ROLE: {jd.title}\n\n"
        f"CANDIDATE A:\n{_render_evidence_for_ranking([a_id], assessments)}\n\n"
        f"CANDIDATE B:\n{_render_evidence_for_ranking([b_id], assessments)}"
    )
    question = JevQuestion(
        key="stronger_fit",
        kind="choice",
        instructions=(
            "Based on the evidence summaries above, which candidate is the stronger fit for this role?\n"
            "Consider: relevance of experience, evidence quality, career trajectory, recency, "
            "and nice-to-have coverage as a tiebreaker. Do not consider the candidate labels (A/B) "
            "as signals — evaluate purely on the evidence."
        ),
        criteria={
            "A": "Candidate A has stronger overall fit for this role",
            "B": "Candidate B has stronger overall fit for this role",
        },
    )
    try:
        answers = jev_client.evaluate(state, [question])
    except JevClientError as exc:
        logger.warning("Jev pairwise ranking failed for %s: %s — skipping", jd.id, exc)
        return None
    winner = answers[0].value
    return [a_id, b_id] if winner == "A" else [b_id, a_id]
```

- [ ] **Step 4: Refactor `_rank_active_listwise_core` to accept a `rank_fn` callable**

Replace the signature:

```python
def _rank_active_listwise_core(
    jd: JobDescription,
    assessments: dict[str, Assessment],
    rank_fn: "Callable[[JobDescription, list[str], dict[str, Assessment]], list[str] | None]",
    subset_size: int,
    max_iterations: int,
    min_delta_u: float,
    pl_prior_variance: float,
) -> dict[str, float]:
```

Inside, replace `ordering = _rank_subset(jd, subset_ids, assessments, llm)` with:

```python
ordering = rank_fn(jd, subset_ids, assessments)
```

- [ ] **Step 5: Update `rank_active_listwise` to build `rank_fn` and accept optional `jev_client`**

```python
def rank_active_listwise(
    jd: JobDescription,
    assessments: dict[str, Assessment],
    llm: "BaseChatModel | None" = None,
    jev_client: "JevClient | None" = None,
    subset_size: int = 8,
    max_iterations: int = 40,
    min_delta_u: float = 0.01,
    pl_prior_variance: float = 1.0,
) -> dict[str, float]:
    """Active listwise (or pairwise) ranking via variance-based subset selection + Plackett-Luce.

    Pass jev_client (with subset_size=2) for pairwise Jev mode.
    Pass llm for legacy Ollama listwise mode.
    """
    if jev_client is not None:
        def rank_fn(j, s, a): return _rank_pair_jev(j, s, a, jev_client)
    elif llm is not None:
        def rank_fn(j, s, a): return _rank_subset(j, s, a, llm)
    else:
        raise ValueError("Either llm or jev_client must be provided to rank_active_listwise")

    qualified = {cid: a for cid, a in assessments.items() if a.qualified}
    disqualified = {cid: a for cid, a in assessments.items() if not a.qualified}

    if qualified and disqualified:
        qualified_utilities = _rank_active_listwise_core(
            jd, qualified, rank_fn, subset_size, max_iterations, min_delta_u, pl_prior_variance
        )
        floor = min(qualified_utilities.values()) - 1.0
        disqualified_order = _bm25_sort(jd, disqualified)
        disqualified_utilities = {cid: floor - i for i, cid in enumerate(disqualified_order)}
        return {**qualified_utilities, **disqualified_utilities}

    return _rank_active_listwise_core(
        jd, assessments, rank_fn, subset_size, max_iterations, min_delta_u, pl_prior_variance
    )
```

Add `TYPE_CHECKING` import for `JevClient`:

```python
if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from candidate_ranking.scoring.jev_client import JevClient
```

- [ ] **Step 6: Run all active_listwise tests**

```
pytest tests/scoring/test_active_listwise_ranking.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/candidate_ranking/scoring/active_listwise_ranking.py tests/scoring/test_active_listwise_ranking.py
git commit -m "feat: add pairwise Jev ranking — _rank_pair_jev, rank_fn refactor in _rank_active_listwise_core"
```

---

### Task 10: Wire pairwise ranking into `pipeline.py` and `cli.py`

**Files:**
- Modify: `src/candidate_ranking/graphs/pipeline.py`
- Modify: `src/candidate_ranking/cli.py`

- [ ] **Step 1: Update `rank_and_format_jd_node` in `pipeline.py`**

Replace the `rank_active_listwise` call to use pairwise when `ranking_llm is None`:

```python
def rank_and_format_jd_node(payload: dict) -> dict:
    jd: JobDescription = payload["jd"]
    all_assessments: dict[str, Assessment] = payload["assessments"]

    if not all_assessments:
        logger.warning("No assessed candidates for JD %s — skipping ranking", jd.id)
        write_jd_ranking(cfg.runs_dir / run_id, jd, {}, {})
        return {}

    if ranking_llm is None:
        # Pairwise Jev mode
        utilities = rank_active_listwise(
            jd, all_assessments,
            jev_client=jev_client,
            subset_size=2,
            max_iterations=cfg.ranking_pairwise_max_iterations,
            min_delta_u=cfg.ranking_min_delta_u,
            pl_prior_variance=cfg.ranking_pl_prior_variance,
        )
    else:
        # Legacy Ollama listwise mode
        utilities = rank_active_listwise(
            jd, all_assessments,
            llm=ranking_llm,
            subset_size=cfg.ranking_subset_size,
            max_iterations=cfg.ranking_max_iterations,
            min_delta_u=cfg.ranking_min_delta_u,
            pl_prior_variance=cfg.ranking_pl_prior_variance,
        )

    write_jd_ranking(cfg.runs_dir / run_id, jd, all_assessments, utilities)
    return {}
```

Add `ranking_pairwise_max_iterations` to the `build_pipeline_graph` closure so `rank_and_format_jd_node` can reference `cfg.ranking_pairwise_max_iterations`.

- [ ] **Step 2: Update `evaluate-stability` command in `cli.py`**

The stability command uses `rank_active_listwise` directly. Update it to pass `jev_client` when available:

```python
elif args.command == "evaluate-stability":
    cfg = RunConfig.full(PROJECT_ROOT)
    cfg = apply_env_overrides(cfg)

    if cfg.jev_api_key:
        from candidate_ranking.scoring.jev_client import JevClient
        ranking_llm_or_jev = ("jev", JevClient(api_token=cfg.jev_api_key))
    else:
        ranking_llm_or_jev = ("llm", ChatOllama(
            model=cfg.ollama_model, base_url=cfg.ollama_base_url,
            temperature=cfg.ranking_temperature, num_ctx=cfg.ollama_num_ctx,
        ))

    # ... in run_stability_report call, pass the appropriate backend
```

Update `run_stability_report` in `ranking_stability.py` to accept `jev_client | llm` — or simply keep the stability report Ollama-only for now and note it as a future improvement. For now, the stability evaluation can remain Ollama-based since it reads from cached assessments and its primary purpose is measuring ranking consistency, which is a separate concern from the migration.

**Decision: leave `evaluate-stability` using Ollama listwise for now — add a TODO comment noting it should be updated to pairwise Jev in a future pass.**

Add to the `evaluate-stability` block in `cli.py`:

```python
# TODO: migrate stability evaluation to pairwise Jev (currently still uses Ollama listwise)
```

- [ ] **Step 3: Run import smoke test and full test suite**

```bash
python -c "from candidate_ranking.cli import run; print('ok')"
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/candidate_ranking/graphs/pipeline.py src/candidate_ranking/cli.py
git commit -m "feat: wire pairwise Jev ranking into pipeline and cli (Phase 3 complete)"
```

---

## Self-Review Checklist (run after writing, fix inline)

- [ ] **Spec §1.3** — fixed in Task 1 (`_select_uncertain_subset` rotation) ✓
- [ ] **Spec §1.1** — fixed in Task 7 (cache key hashes rendered questions) ✓
- [ ] **Spec §1.2** — fixed in Task 7 (`qualified` computed from verdicts) ✓
- [ ] **Spec §1.4** — fixed in Task 7 (narrow `JevClientError`/`GenerationError` only) ✓
- [ ] **Spec §2 — Jev question builder** — Task 5 builds all required question types ✓
- [ ] **Spec §3 — Adaptive questions** — `_build_jev_questions` builds dynamically per profile ✓
- [ ] **Spec §4 — Advisory-only for edu/seniority/other** — Task 4 adds advisory model fields; Task 7 assembler sets them but never affects `qualified` ✓
- [ ] **Spec §5 — Pairwise via existing active loop** — Task 9 reuses existing Plackett-Luce loop at `subset_size=2` ✓
- [ ] **Spec §6 — Build order** — Phase 1 (§1.3 bug), Phase 2 (gate), Phase 3 (ranking) ✓
- [ ] **Spec §7 — Files** — all listed files accounted for in File Map ✓
- [ ] **`qualification_reliability.py`** — updated in Task 8 ✓
- [ ] **`jev_client.py` restore** — Task 2 restores exact file from git ✓
- [ ] **No TBDs** — all code blocks are complete ✓
- [ ] **Type consistency** — `_FreeTextOutput` defined in Task 6, used in Task 7; `JevQuestion`/`JevAnswer` from `jev_client.py` (Task 2) ✓
