# Remove Multi-Call Ensemble Averaging (C2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire the `n_calls=3` multi-call ensemble-averaging mechanism from the production Jev assessment pipeline and its corresponding claim ("C2") in `docs/paper2_jev.tex`, so the pipeline makes one Jev call per (job, applicant) pair and the paper reports two contributions instead of three.

**Architecture:** `assessment.py`'s `generate_assessment()` currently fans out to `n_calls` (default 3) independent Jev calls and statistically aggregates them; it becomes a thin wrapper around the existing single-call path (`_generate_single_assessment`), and every now-unreachable aggregation helper is deleted. Two research scripts whose entire purpose was validating the 3-call-vs-1-call ranking-convergence gain (`scripts/validate_multi_call_averaging.py`, `scripts/generate_convergence_figure.py`) are deleted outright, along with their generated figure. The paper is edited section by section to drop the "Multi-Call Ensemble Averaging" contribution, renumber the remaining contribution from C3 to C2, renumber pipeline stages 4→gone/5→4/6→5, and replace every ensemble-derived number (convergence 0.982→0.964, total calls 819→273) with the corresponding single-call number that is already published in the paper's own abstract/results text — no new experiments or Jev calls are needed.

**Tech Stack:** Python 3 (pydantic, pytest, unittest.mock), LaTeX (IEEEtran), no build step for the paper (never compile to PDF).

## Global Constraints

- Never run `pdflatex` or otherwise compile `docs/paper2_jev.tex`, and never commit a generated `docs/paper2_jev.pdf` — edit the `.tex` source only; the user compiles it themselves.
- `docs/` is gitignored at the repo root (`.gitignore` line 4: `docs/`); every file under it that must be committed (specs, plans, the paper, its figures) needs `git add -f` / `git rm -f`, matching how the existing tracked files under `docs/` were added.
- Retry-on-low-confidence in `_generate_single_assessment` (`src/candidate_ranking/scoring/assessment.py`) is a separate error-recovery mechanism, not part of the ensembling being removed — do not touch it.
- `collect_ablation` in `scripts/run_jev_evaluation_study.py` (its own unrelated `n_calls=3` default, for a vague-vs-concrete question-phrasing ablation) is out of scope — do not modify it.

---

### Task 1: Remove multi-call ensembling from `assessment.py`

**Files:**
- Modify: `src/candidate_ranking/scoring/assessment.py`
- Modify: `tests/scoring/test_assessment.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `generate_assessment(jd, candidate, jev_client, model_name=JEV_MODEL_NAME, jd_skills=None) -> Assessment` (no `n_calls` parameter — later tasks/callers must not pass one). `load_or_generate_assessment(jd, candidate, jev_client, model_name, cache_dir, jd_skills=None) -> Assessment` (same: no `n_calls`). Both were already called without `n_calls` from `src/candidate_ranking/graphs/pipeline.py:120`, so that call site needs no change.

- [ ] **Step 1: Rewrite `tests/scoring/test_assessment.py` to drop the ensembling tests and `n_calls` kwargs**

Replace the entire file with:

```python
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from candidate_ranking.models import Candidate, JDSkills, JobDescription
from candidate_ranking.scoring.assessment import (
    AssessmentGenerationError,
    JEV_MODEL_NAME,
    _build_questions,
    generate_assessment,
    load_or_generate_assessment,
)
from candidate_ranking.scoring.jev_client import JevAnswer, JevClientError


def _jd() -> JobDescription:
    return JobDescription(id="jd-1", title="Backend Engineer", raw_text="Needs Python and SQL.", source_path="jd.pdf")


def _candidate() -> Candidate:
    return Candidate(
        id="cand-1", source_path="cv.pdf", raw_text="I know Python.", num_pages=1, char_count=20,
        parse_status="ok", skills=["Python"],
    )


def _jd_skills() -> JDSkills:
    return JDSkills(job_description_id="jd-1", generated_by_model="qwen2.5:14b", technical_skills=["Python", "SQL"])


def _jd_skills_full() -> JDSkills:
    return JDSkills(
        job_description_id="jd-1",
        generated_by_model="qwen2.5:14b",
        technical_skills=["Python"],
        certifications=["AWS Certified Solutions Architect"],
        seniority_requirement="5+ years of backend development experience",
        seniority_min_years=5.0,
        education_requirement="Bachelor's degree in Computer Science or related field",
    )


def _high_confidence_answers() -> list[JevAnswer]:
    return [
        JevAnswer(key="overall_recommendation", kind="choice", value="hire", confidence=0.85),
        JevAnswer(key="meets_min_qualifications", kind="noul", value=True, confidence=0.95),
        JevAnswer(key="requirement::Python", kind="score", value=4.0, confidence=0.9),
        JevAnswer(key="requirement::SQL", kind="score", value=1.0, confidence=0.8),
    ]


def test_generate_assessment_maps_jev_answers_onto_assessment():
    jev_client = Mock()
    jev_client.evaluate.return_value = _high_confidence_answers()

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills())

    assert assessment.job_description_id == "jd-1"
    assert assessment.candidate_id == "cand-1"
    assert assessment.generated_by_model == JEV_MODEL_NAME
    assert assessment.overall_recommendation == "hire"
    assert assessment.meets_min_qualifications is True
    assert assessment.requirement_scores == {"Python": 100.0, "SQL": 25.0}
    assert assessment.composite_fit_score == 62.5  # mean(100, 25)
    assert assessment.confidence["overall_recommendation"] == 0.85
    jev_client.evaluate.assert_called_once()
    state, questions = jev_client.evaluate.call_args.args
    assert "Backend Engineer" in state
    assert "I know Python." in state
    question_keys = {q.key for q in questions}
    assert question_keys == {
        "overall_recommendation", "meets_min_qualifications",
        "requirement::Python", "requirement::SQL",
    }


def test_generate_assessment_skips_requirement_scores_without_jd_skills():
    jev_client = Mock()
    jev_client.evaluate.return_value = [
        JevAnswer(key="overall_recommendation", kind="choice", value="maybe", confidence=0.7),
        JevAnswer(key="meets_min_qualifications", kind="noul", value=False, confidence=0.8),
    ]

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, jd_skills=None)

    assert assessment.requirement_scores == {}
    assert assessment.composite_fit_score == 0.0
    _, questions = jev_client.evaluate.call_args.args
    assert all(not q.key.startswith("requirement::") for q in questions)


def test_generate_assessment_retries_once_on_low_confidence_then_accepts():
    jev_client = Mock()
    low_confidence_answers = [
        JevAnswer(key="overall_recommendation", kind="choice", value="hire", confidence=0.85),
        JevAnswer(key="meets_min_qualifications", kind="noul", value=True, confidence=0.3),
    ]
    jev_client.evaluate.side_effect = [low_confidence_answers, _high_confidence_answers()]

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills())

    assert jev_client.evaluate.call_count == 2
    assert assessment.confidence["meets_min_qualifications"] == 0.95
    second_state, _ = jev_client.evaluate.call_args_list[1].args
    assert "low-confidence" in second_state


def test_generate_assessment_wraps_jev_client_error():
    jev_client = Mock()
    jev_client.evaluate.side_effect = JevClientError("network down")

    with pytest.raises(AssessmentGenerationError, match="network down"):
        generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills())


def test_generate_assessment_wraps_unusable_jev_response():
    missing_key_client = Mock()
    missing_key_client.evaluate.return_value = [
        JevAnswer(key="overall_recommendation", kind="choice", value="hire", confidence=0.85),
    ]
    with pytest.raises(AssessmentGenerationError, match="jd-1/cand-1"):
        generate_assessment(_jd(), _candidate(), missing_key_client, JEV_MODEL_NAME, _jd_skills())

    invalid_value_client = Mock()
    invalid_value_client.evaluate.return_value = [
        JevAnswer(key="overall_recommendation", kind="choice", value="strongly_hire", confidence=0.85),
        JevAnswer(key="meets_min_qualifications", kind="noul", value=True, confidence=0.95),
    ]
    with pytest.raises(AssessmentGenerationError, match="jd-1/cand-1"):
        generate_assessment(_jd(), _candidate(), invalid_value_client, JEV_MODEL_NAME, _jd_skills())


def test_generate_assessment_preserves_probability_distributions():
    jev_client = Mock()
    jev_client.evaluate.return_value = [
        JevAnswer(key="overall_recommendation", kind="choice", value="hire", confidence=0.8,
                   probabilities={"hire": 0.8, "maybe": 0.15, "no": 0.05}),
        JevAnswer(key="meets_min_qualifications", kind="noul", value=True, confidence=0.9),
        JevAnswer(key="requirement::Python", kind="score", value=4.0, confidence=0.9,
                   probabilities={"0": 0, "1": 0, "2": 0, "3": 0.1, "4": 0.9}),
    ]

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills())

    assert assessment.recommendation_probabilities == {"hire": 0.8, "maybe": 0.15, "no": 0.05}
    assert assessment.requirement_probabilities["Python"] == {"0": 0, "1": 0, "2": 0, "3": 0.1, "4": 0.9}
    assert assessment.seniority_probabilities is None
    assert assessment.education_probabilities is None


def test_load_or_generate_assessment_uses_cache_on_second_call(tmp_path: Path):
    jev_client = Mock()
    jev_client.evaluate.return_value = _high_confidence_answers()

    first = load_or_generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, tmp_path, _jd_skills())
    second = load_or_generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, tmp_path, _jd_skills())

    assert first == second
    jev_client.evaluate.assert_called_once()
    cache_file = tmp_path / "assessments" / "jd-1" / "cand-1.json"
    assert cache_file.exists()
    cached = json.loads(cache_file.read_text(encoding="utf-8"))
    assert cached["assessment"]["composite_fit_score"] == 62.5


def test_generate_assessment_builds_certification_seniority_education_questions():
    jev_client = Mock()
    jev_client.evaluate.return_value = [
        JevAnswer(key="overall_recommendation", kind="choice", value="hire", confidence=0.85),
        JevAnswer(key="meets_min_qualifications", kind="noul", value=True, confidence=0.95),
        JevAnswer(key="requirement::Python", kind="score", value=4.0, confidence=0.9),
        JevAnswer(key="certification::AWS Certified Solutions Architect", kind="noul", value=True, confidence=0.9),
        JevAnswer(key="seniority_years", kind="score", value=3.0, confidence=0.85),
        JevAnswer(key="education", kind="score", value=1.0, confidence=0.8),
    ]

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills_full())

    _, questions = jev_client.evaluate.call_args.args
    question_keys = {q.key for q in questions}
    assert "certification::AWS Certified Solutions Architect" in question_keys
    assert "seniority_years" in question_keys
    assert "education" in question_keys
    assert assessment.certification_results == {"AWS Certified Solutions Architect": True}
    assert assessment.seniority_years_fit_score == 75.0
    assert assessment.education_fit_score == 25.0


def test_generate_assessment_omits_seniority_education_questions_when_not_stated():
    jev_client = Mock()
    jev_client.evaluate.return_value = [
        JevAnswer(key="overall_recommendation", kind="choice", value="hire", confidence=0.85),
        JevAnswer(key="meets_min_qualifications", kind="noul", value=True, confidence=0.95),
    ]

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills())

    _, questions = jev_client.evaluate.call_args.args
    question_keys = {q.key for q in questions}
    assert "seniority_years" not in question_keys
    assert "education" not in question_keys
    assert not any(k.startswith("certification::") for k in question_keys)
    assert assessment.certification_results == {}
    assert assessment.seniority_years_fit_score is None
    assert assessment.education_fit_score is None


def test_seniority_and_education_criteria_are_five_level_evidence_based_scores():
    jd_skills = JDSkills(
        job_description_id="jd-1", generated_by_model="qwen2.5:14b", technical_skills=[],
        seniority_requirement="5+ years of backend experience",
        seniority_min_years=5.0,
        education_requirement="Bachelor's degree in Computer Science",
    )
    questions = _build_questions(jd_skills)
    by_key = {q.key: q for q in questions}

    assert by_key["seniority_years"].kind == "score"
    assert by_key["education"].kind == "score"
    assert len(by_key["seniority_years"].criteria) == 5
    assert len(by_key["education"].criteria) == 5

    seniority_years_criteria = " ".join(by_key["seniority_years"].criteria).lower()
    education_criteria = " ".join(by_key["education"].criteria).lower()

    # The old Noul criteria just restated the question ("supports"/"does not support" this
    # requirement) -- circular, giving Jev no concrete evidence to look for. The Score criteria
    # must instead name the kind of evidence that distinguishes each level.
    for banned_phrase in ("supports that the candidate", "does not support that the candidate"):
        assert banned_phrase not in seniority_years_criteria
        assert banned_phrase not in education_criteria

    assert any(term in seniority_years_criteria for term in ("total experience", "years"))
    assert any(term in education_criteria for term in ("degree", "field", "credential"))
```

This drops five tests that existed only to exercise the removed ensembling behavior (`test_generate_assessment_defaults_to_three_calls_and_averages`, `test_generate_assessment_averages_probability_distributions_across_calls`, `test_generate_assessment_n_calls_one_skips_averaging`, `test_generate_assessment_aggregates_certification_and_seniority_across_calls`, `test_load_or_generate_assessment_cache_key_depends_on_n_calls`), removes every `n_calls=...` keyword argument from the remaining calls, and renames `test_generate_assessment_single_call_preserves_probability_distributions` to `test_generate_assessment_preserves_probability_distributions` (the "single_call" qualifier is meaningless once every call is single-call).

- [ ] **Step 2: Run the tests and confirm they fail against the current implementation**

Run: `uv run pytest tests/scoring/test_assessment.py -v`

Expected: several `FAIL`s — in particular `test_generate_assessment_maps_jev_answers_onto_assessment` fails on `jev_client.evaluate.assert_called_once()` (current code still defaults to `n_calls=3`, so the mock is called 3 times), and `test_load_or_generate_assessment_uses_cache_on_second_call` fails the same way.

- [ ] **Step 3: Remove the ensembling mechanism from `src/candidate_ranking/scoring/assessment.py`**

Apply these edits in order (all within the same file):

Edit 1 — drop now-unused imports (`statistics` and `Counter` are only used by the aggregation helpers being deleted below):

```python
# OLD
from __future__ import annotations

import hashlib
import json
import logging
import statistics
from collections import Counter
from pathlib import Path

# NEW
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
```

Edit 2 — drop the `DEFAULT_N_CALLS` constant:

```python
# OLD
_CONFIDENCE_RETRY_THRESHOLD = 0.5

DEFAULT_N_CALLS = 3

_REQUIREMENT_FIT_CRITERIA = [

# NEW
_CONFIDENCE_RETRY_THRESHOLD = 0.5

_REQUIREMENT_FIT_CRITERIA = [
```

Edit 3 — delete every cross-call aggregation helper (`_aggregate_recommendation` through `_aggregate_mean_optional_dict`), replacing the whole block with nothing (leave `def generate_assessment(` as the very next thing after `_generate_single_assessment`'s closing `return assessment, answers`):

```python
# OLD (delete this entire block, from "def _aggregate_recommendation" up to
# but not including "def generate_assessment")
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

# NEW
def generate_assessment(
```

Edit 4 — simplify `generate_assessment` to a single call, no aggregation:

```python
# OLD
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

# NEW
def generate_assessment(
    jd: JobDescription,
    candidate: Candidate,
    jev_client: JevClient,
    model_name: str = JEV_MODEL_NAME,
    jd_skills: JDSkills | None = None,
) -> Assessment:
    questions = _build_questions(jd_skills)
    assessment, _answers = _generate_single_assessment(jd, candidate, jev_client, model_name, questions)
    return assessment
```

Edit 5 — drop `n_calls` from the cache key:

```python
# OLD
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

# NEW
def _assessment_cache_key(
    jd: JobDescription,
    candidate: Candidate,
    model_name: str,
    jd_skills: JDSkills | None = None,
) -> str:
    state = _build_state(jd, candidate)
    questions_repr = repr([q.model_dump() for q in _build_questions(jd_skills)])
    digest_input = f"{state}||{questions_repr}||{model_name}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()
```

Edit 6 — drop `n_calls` from `load_or_generate_assessment`:

```python
# OLD
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

# NEW
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
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/scoring/test_assessment.py -v`

Expected: all tests `PASS`.

- [ ] **Step 5: Run the full test suite to catch any other dependent**

Run: `uv run pytest`

Expected: all tests `PASS` (in particular nothing in `tests/` outside `test_assessment.py` should reference `DEFAULT_N_CALLS`, `n_calls`, or any `_aggregate_*` helper — this was confirmed absent before writing this plan, but re-verify since the suite is the ground truth).

- [ ] **Step 6: Commit**

```bash
git add src/candidate_ranking/scoring/assessment.py tests/scoring/test_assessment.py
git commit -m "$(cat <<'EOF'
refactor: remove multi-call ensemble averaging from Jev assessment

generate_assessment() now makes exactly one Jev call per (job,
applicant) pair instead of n_calls=3 plus cross-call aggregation. This
retires the paper's C2 contribution (see docs/superpowers/specs/
2026-09-25-remove-multicall-ensemble-c2-design.md) and cuts Jev token
spend to a third of what it was.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Delete the C2 validation scripts, generated figure, and fix documentation references

**Files:**
- Delete: `scripts/validate_multi_call_averaging.py`
- Delete: `scripts/generate_convergence_figure.py`
- Delete: `docs/fig_convergence.png`
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing from Task 1 (these files don't call `generate_assessment`/`load_or_generate_assessment` — they call `JevClient.evaluate()` directly via `collect_test_retest`, which is unchanged and stays).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Confirm nothing else references the files being deleted**

Run: `grep -rn "validate_multi_call_averaging\|generate_convergence_figure\|fig_convergence" --include="*.py" --include="*.md" --include="*.tex" .`

Expected output: only `README.md` (handled in Step 3 below) and `docs/paper2_jev.tex` (handled in Task 3). If anything else shows up, stop and investigate before deleting.

- [ ] **Step 2: Delete the two scripts and the generated figure**

```bash
git rm scripts/validate_multi_call_averaging.py scripts/generate_convergence_figure.py
git rm -f docs/fig_convergence.png
```

(`docs/` is gitignored so `docs/fig_convergence.png` needs `-f`; `scripts/` is not gitignored so the first `git rm` needs no `-f`.)

- [ ] **Step 3: Update `README.md`**

Edit 1 — remove the `validate_multi_call_averaging.py` invocation from the evaluation-harness command block:

```markdown
# OLD
```bash
uv run python scripts/run_jev_evaluation_study.py --run-id 20260918-104531
uv run python scripts/analyze_jev_evaluation_study.py --run-id 20260918-104531
uv run python scripts/validate_multi_call_averaging.py --run-id 20260918-104531
```

# NEW
```bash
uv run python scripts/run_jev_evaluation_study.py --run-id 20260918-104531
uv run python scripts/analyze_jev_evaluation_study.py --run-id 20260918-104531
```
```

Edit 2 — remove the `validate_multi_call_averaging.py` bullet from the description list right after that block:

```markdown
# OLD
- `analyze_jev_evaluation_study.py` computes the actual statistics (Kendall-
  tau ranking convergence, Spearman internal coherence, Mann-Whitney U
  group separation, Wilcoxon signed-rank ablation effects) from that raw
  data and writes `runs/<run_id>/evaluation/report.{json,md}`.
- `validate_multi_call_averaging.py` collects two more independent 3-call
  trials on top of the test-retest data, then compares single-call ranking
  convergence against 3-call-averaged ranking convergence, writing
  `runs/<run_id>/evaluation/multi_call_averaging_validation.json`.

Each script requires `CANDIDATE_RANKING_JEV_API_KEY` except

# NEW
- `analyze_jev_evaluation_study.py` computes the actual statistics (Kendall-
  tau ranking convergence, Spearman internal coherence, Mann-Whitney U
  group separation, Wilcoxon signed-rank ablation effects) from that raw
  data and writes `runs/<run_id>/evaluation/report.{json,md}`.

Each script requires `CANDIDATE_RANKING_JEV_API_KEY` except
```

Edit 3 — fix the `scoring/` directory description (it mentions "multi-call aggregation", which no longer happens there) and remove the deleted script from the project-layout listing:

```markdown
# OLD
  scoring/                      # JD skill extraction, skill matching, and Jev-based
                                  # structured assessment + multi-call aggregation

# NEW
  scoring/                      # JD skill extraction, skill matching, and Jev-based
                                  # structured assessment
```

```markdown
# OLD
scripts/
  run_jev_evaluation_study.py       # collects test-retest + ablation raw data, see above
  analyze_jev_evaluation_study.py    # computes statistics from that raw data
  validate_multi_call_averaging.py    # single-call vs. 3-call-averaged ranking convergence

# NEW
scripts/
  run_jev_evaluation_study.py       # collects test-retest + ablation raw data, see above
  analyze_jev_evaluation_study.py    # computes statistics from that raw data
```

- [ ] **Step 4: Verify no dangling references remain**

Run: `grep -rn "validate_multi_call_averaging\|generate_convergence_figure\|fig_convergence" --include="*.py" --include="*.md" .`

Expected: no output (the only remaining reference, in `docs/paper2_jev.tex`, is handled by Task 3 — this grep intentionally excludes `.tex`).

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
chore: remove C2 multi-call-averaging validation scripts

validate_multi_call_averaging.py and generate_convergence_figure.py
existed only to validate the multi-call ensemble-averaging mechanism
removed in the previous commit; nothing is left for them to validate.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

(The `git rm` from Step 2 stages the deletions; this commit picks those up along with the `README.md` change staged here.)

---

### Task 3: Rewrite `docs/paper2_jev.tex` to remove the C2 claim

**Files:**
- Modify: `docs/paper2_jev.tex`

**Interfaces:**
- Consumes: nothing (pure text edit).
- Produces: nothing consumed by later tasks.

Apply every edit below in order. Each "OLD" block is the exact current text (verified against the file as of this plan's writing); each "NEW" block is what it becomes. Do not compile the file at any point (see Global Constraints).

- [ ] **Step 1: Abstract — drop the third mechanism and the ensemble-averaging numeric claim**

```latex
% OLD
This research's objective is to propose three architecture-level mechanisms for reliably deploying Jev in a real applicant-ranking pipeline: criteria-grounded question design, multi-call ensemble averaging, and cost/latency efficiency from a single parallel pass outperforming an iterative tournament. The research method transforms an existing LLM-and-tournament applicant-ranking pipeline, stage by stage, into a fully structured typed-question architecture built on Jev and these three mechanisms, evaluated on 273 shortlisted (job, applicant) pairs across 10 job profiles. The results show that criteria-grounded design raises per-requirement answer confidence from 0.672 to 0.894 and pooled seniority/education confidence from 0.636 to 0.766; that three-call ensemble averaging, run independently and in parallel, recovers ranking convergence from 0.964 to 0.982 -- exceeding the prior tournament-based system's 0.957 -- using well under half as many model calls and roughly 6--10$\times$ lower per-candidate latency, with recommendation agreement holding steady across independent repeats.

% NEW
This research's objective is to propose two architecture-level mechanisms for reliably deploying Jev in a real applicant-ranking pipeline: criteria-grounded question design, and cost/latency efficiency from a single parallel pass outperforming an iterative tournament. The research method transforms an existing LLM-and-tournament applicant-ranking pipeline, stage by stage, into a fully structured typed-question architecture built on Jev and these two mechanisms, evaluated on 273 shortlisted (job, applicant) pairs across 10 job profiles. The results show that criteria-grounded design raises per-requirement answer confidence from 0.672 to 0.894 and pooled seniority/education confidence from 0.636 to 0.766; that single-call ranking convergence reaches 0.964, exceeding the prior tournament-based system's 0.957, using well under half as many model calls and roughly 6--10$\times$ lower per-candidate latency, with recommendation agreement holding steady across independent repeats.
```

- [ ] **Step 2: Contribution list — drop the "Multi-Call Ensemble Averaging" bullet**

```latex
% OLD
This paper makes the following three architecture-level contributions:

\begin{enumerate}
\item \textbf{Criteria-Grounded Question Design}: the first large-sample statistical test of TypeSafe's own criteria-grounded Score-question guidance, extended to seniority and education Score questions, each validated with concrete-situation criteria in place of bare ordinal labels.
\item \textbf{Multi-Call Ensemble Averaging}: combining several independent Jev calls per applicant -- arithmetic mean for score-valued answers, confidence-weighted majority vote for categorical ones -- to match or exceed an iterative tournament's ranking-order stability, validated against single-call rankings so the gain reflects recovered stability rather than hidden disagreement.
\item \textbf{Cost and Latency Efficiency}: a direct sort on the ensemble-averaged score outperforming the prior system's iterative listwise tournament -- no iterative ranking process, all calls parallelize, total model calls cut to well under half, and per-candidate wall-clock latency lower by roughly 6--10$\times$.
\end{enumerate}

% NEW
This paper makes the following two architecture-level contributions:

\begin{enumerate}
\item \textbf{Criteria-Grounded Question Design}: the first large-sample statistical test of TypeSafe's own criteria-grounded Score-question guidance, extended to seniority and education Score questions, each validated with concrete-situation criteria in place of bare ordinal labels.
\item \textbf{Cost and Latency Efficiency}: a direct sort on Jev's per-call composite score outperforming the prior system's iterative listwise tournament -- no iterative ranking process, all calls parallelize, total model calls cut to well under half, and per-candidate wall-clock latency lower by roughly 6--10$\times$.
\end{enumerate}
```

- [ ] **Step 3: Section II — drop the ensembling justification clause**

```latex
% OLD
Every question in a call is answered independently against the same state, so adding a question to a call costs negligibly more than the call alone \cite{ref11} -- the reason multi-call ensembling (Section III-E) triples every call's answer count at close to one call's marginal cost.

% NEW
Every question in a call is answered independently against the same state, so adding a question to a call costs negligibly more than the call alone \cite{ref11}.
```

- [ ] **Step 4: Pipeline stage diagram — remove Stage 4, renumber, drop `$n{=}3$`**

```latex
% OLD
\fbox{\begin{minipage}{0.85\columnwidth}\centering Stage 3: Structured Assessment via Jev, $n{=}3$ calls \textbf{[C1: Criteria-Grounded Design]}\end{minipage}} \\[1pt]
$\downarrow$ \\[1pt]
\fbox{\begin{minipage}{0.85\columnwidth}\centering Stage 4: Multi-Call Ensemble Aggregation \textbf{[C2: Ensemble Averaging]}\end{minipage}} \\[1pt]
$\downarrow$ \\[1pt]
\fbox{\begin{minipage}{0.85\columnwidth}\centering Stage 5: Ranking, direct sort, no tournament \textbf{[C3: Cost/Latency Efficiency]}\end{minipage}} \\[1pt]
$\downarrow$ \\[1pt]
\fbox{\begin{minipage}{0.85\columnwidth}\centering Stage 6: Score Breakdown Diagnostic\end{minipage}}
\end{tabular}
\caption{End-to-end pipeline architecture, presented stage by stage, with the three contributions (C1--C3) annotated at their point of intervention.}

% NEW
\fbox{\begin{minipage}{0.85\columnwidth}\centering Stage 3: Structured Assessment via Jev \textbf{[C1: Criteria-Grounded Design]}\end{minipage}} \\[1pt]
$\downarrow$ \\[1pt]
\fbox{\begin{minipage}{0.85\columnwidth}\centering Stage 4: Ranking, direct sort, no tournament \textbf{[C2: Cost/Latency Efficiency]}\end{minipage}} \\[1pt]
$\downarrow$ \\[1pt]
\fbox{\begin{minipage}{0.85\columnwidth}\centering Stage 5: Score Breakdown Diagnostic\end{minipage}}
\end{tabular}
\caption{End-to-end pipeline architecture, presented stage by stage, with the two contributions (C1--C2) annotated at their point of intervention.}
```

- [ ] **Step 5: Paragraph after the diagram — five stages, III-B--III-F, C1--C2**

```latex
% OLD
Fig.~\ref{fig_pipeline} shows the end-to-end pipeline as six sequential stages, described in turn in Sections III-B--III-G; C1--C3 mark where each contribution intervenes.

% NEW
Fig.~\ref{fig_pipeline} shows the end-to-end pipeline as five sequential stages, described in turn in Sections III-B--III-F; C1--C2 mark where each contribution intervenes.
```

- [ ] **Step 6: Section III-D (Stage 3) — one call, not `$n$` calls**

```latex
% OLD
Where the prior system's free text lived, the pipeline sends $n$ independent calls to Jev per shortlisted pair (C1).

% NEW
Where the prior system's free text lived, the pipeline sends one call to Jev per shortlisted pair (C1).
```

- [ ] **Step 7: Delete the entire "Stage 4: Multi-Call Ensemble Aggregation" subsection (old Section III-E)**

```latex
% OLD (delete this whole block, including the two paragraphs, leaving
% "\subsection{Stage 5: Ranking}" — renamed in the next step — immediately
% after the preceding subsection)
\subsection{Stage 4: Multi-Call Ensemble Aggregation}
A single Jev call is a single sample, carrying noise that can shift a ranking. Since calls are architecturally independent and cheap to add (Section II), the pipeline issues $n$ calls per pair -- three by default, matching \cite{ref2}'s stability-repeat count -- combined per field: numeric fields by arithmetic mean, the recommendation by majority vote (ties toward higher mean confidence), and binary answers by simple majority, defaulting to the conservative (negative) outcome on a tie (C2).

Averaging could also hide real disagreement rather than remove noise: ranking convergence was measured both from single, unaggregated calls (Kendall-$\tau$ across three per-pair rankings) and from three three-call averages. If averaging only hid disagreement, the second would not consistently beat the first.

\subsection{Stage 5: Ranking}

% NEW
\subsection{Stage 4: Ranking}
```

- [ ] **Step 8: Rewrite the (now Stage 4) Ranking subsection body — no more combined calls, renumber C3→C2 and the section refs**

```latex
% OLD
The $n$ combined calls from Stage 4 (Section III-E) yield one aggregated composite score per (job, applicant) pair; applicants are ranked by a direct sort on this score. This direct sort outperforms the prior system's iterative Monte Carlo knowledge-gradient tournament with Bayesian Plackett-Luce aggregation: no ranking process, no waiting on a prior iteration's result. Because Jev's calls need no iterative dependency, they also parallelize, unlike the prior system's tournament (C3); Section V-D quantifies the resulting reduction in model calls and wall-clock latency. The ranked output then feeds a diagnostic check (Section III-G) that reads per-requirement scores afterward for upstream errors.

% NEW
Stage 3 (Section III-D) yields one composite score per (job, applicant) pair; applicants are ranked by a direct sort on this score. This direct sort outperforms the prior system's iterative Monte Carlo knowledge-gradient tournament with Bayesian Plackett-Luce aggregation: no ranking process, no waiting on a prior iteration's result. Because Jev's calls need no iterative dependency, they also parallelize, unlike the prior system's tournament (C2); Section V-C quantifies the resulting reduction in model calls and wall-clock latency. The ranked output then feeds a diagnostic check (Section III-F) that reads per-requirement scores afterward for upstream errors.
```

- [ ] **Step 9: Score Breakdown Diagnostic subsection — Stage 6→5, Section V-C→V-B, "three"→"two"**

```latex
% OLD
\subsection{Stage 6: Score Breakdown Diagnostic}
Because Jev reports a separate score per extracted requirement rather than a single free-text verdict, a job profile's shortlisted pool can be inspected at the level of individual skills, not only an aggregate score. A requirement is informative only if the pool's median score on it is reasonably high; a shortlisted applicant scoring near-zero on most informative requirements is flagged -- not because Jev judged them harshly, but because peers could clear a bar this applicant could not. Applied under a total-only shortlisting baseline (Stage 2's cosine threshold without the must-have gate), this diagnostic quantifies the must-have gate's necessity; Section V-C reports the corpus-wide result. This diagnostic validates Stage 2's must-have gate; it is not one of this paper's three architecture-level contributions.

% NEW
\subsection{Stage 5: Score Breakdown Diagnostic}
Because Jev reports a separate score per extracted requirement rather than a single free-text verdict, a job profile's shortlisted pool can be inspected at the level of individual skills, not only an aggregate score. A requirement is informative only if the pool's median score on it is reasonably high; a shortlisted applicant scoring near-zero on most informative requirements is flagged -- not because Jev judged them harshly, but because peers could clear a bar this applicant could not. Applied under a total-only shortlisting baseline (Stage 2's cosine threshold without the must-have gate), this diagnostic quantifies the must-have gate's necessity; Section V-B reports the corpus-wide result. This diagnostic validates Stage 2's must-have gate; it is not one of this paper's two architecture-level contributions.
```

- [ ] **Step 10: Pipeline-configuration table — remove the "Independent calls averaged" row**

```latex
% OLD
Must-have skill-match minimum & $\geq$ 2 (auto-reduced to the job profile's must-have count if smaller)\\
Independent calls averaged per assessment ($n$) & 3\\
Score question levels & 5 (concrete situational criteria)\\

% NEW
Must-have skill-match minimum & $\geq$ 2 (auto-reduced to the job profile's must-have count if smaller)\\
Score question levels & 5 (concrete situational criteria)\\
```

- [ ] **Step 11: Methodology paragraph after the config table — convergence is single-call only, III-G→III-F**

```latex
% OLD
Table~\ref{tab_config} shows the full configuration. Three evaluation procedures are applied: repeatability (three independent single calls per pair); ranking convergence (Kendall-$\tau$, Section III-E, from single calls and three-call averages); and the criteria-design ablations. Spearman is used wherever the size of a score gap carries meaning, such as the per-requirement score diagnostic of Section III-G; Kendall-$\tau$ is used only for ranking convergence, where relative candidate order is what matters.

% NEW
Table~\ref{tab_config} shows the full configuration. Three evaluation procedures are applied: repeatability (three independent single calls per pair); ranking convergence (Kendall-$\tau$ across those same three single-call rankings); and the criteria-design ablations. Spearman is used wherever the size of a score gap carries meaning, such as the per-requirement score diagnostic of Section III-F; Kendall-$\tau$ is used only for ranking convergence, where relative candidate order is what matters.
```

- [ ] **Step 12: Delete the entire "Multi-Call Ensemble Averaging (C2)" results subsection**

```latex
% OLD (delete this whole block -- subsection heading, figure, both
% paragraphs, the reliability table and its paragraph -- leaving
% "\subsection{Must-Have Gate: Diagnostic Validation}" as the next thing
% after the preceding subsection, "Criteria-Grounded Question Design (C1)")
\subsection{Multi-Call Ensemble Averaging (C2)}
\begin{figure}[H]
\centering
\includegraphics[width=\columnwidth]{fig_convergence}
\caption{Ranking convergence (Kendall-$\tau$) by job profile, single-call ($\tau_1$) vs.\ three-call-averaged ($\tau_3$).}
\label{fig_convergence_diag}
\end{figure}
Fig.~\ref{fig_convergence_diag} shows $\tau_1$ (single unaggregated calls) and $\tau_3$ (three-call averages, Section III-E) per job profile: three-call averaging raises mean convergence, with nearly every profile improving. The one exception, ui-ux-designer, already sits at a perfect single-call score and stays close to it under averaging -- consistent with the aggregation recovering genuine stability rather than merely smoothing over disagreement.

At the individual-recommendation level, agreement is already close to ceiling: across the full corpus, a single call's hiring recommendation matches the majority verdict of five other independent calls on the same pair 98.5\% of the time (1,614/1,638 held-out comparisons over 273 pairs, six calls each). Ensemble averaging therefore has comparatively little left to recover at the recommendation field specifically, even as the continuous composite-score ranking above still benefits from it.

\begin{table}[H]
\caption{\texttt{overall\_recommendation} distribution and six-call inter-call reliability, full corpus (273 pairs).}
\label{tab_recommendation_reliability}
\centering
\footnotesize
\begin{tabular}{p{1.9in}p{1.2in}}
\hline
Category / Metric & Value\\
\hline
hire & 45 (16.5\%)\\
maybe & 177 (64.8\%)\\
no & 51 (18.7\%)\\
\hline
Raw observed agreement ($\bar{P}$) & 0.975\\
Expected agreement by chance ($P_e$) & 0.480\\
Fleiss' $\kappa$ & 0.952\\
\hline
\end{tabular}
\end{table}
Table~\ref{tab_recommendation_reliability} shows why the raw agreement figure alone is not the right reliability statistic here: \texttt{maybe} accounts for nearly two-thirds of consensus recommendations, so a high raw agreement rate is partly attributable to that base rate rather than to genuine six-call consistency. Fleiss' $\kappa$ corrects for exactly this by subtracting out the agreement expected from the category distribution alone; at 0.952 it still falls in the ``almost perfect'' range by the standard Landis-Koch scale, so the high raw agreement is not merely a base-rate artifact.

\subsection{Must-Have Gate: Diagnostic Validation}

% NEW
\subsection{Must-Have Gate: Diagnostic Validation}
```

- [ ] **Step 13: Cost/Latency subsection heading and intro — C3→C2, Stage 5→4, III-F→III-E**

```latex
% OLD
\subsection{Cost and Latency Efficiency (C3) and Comparison with the Prior Tournament-Based System}
This section reports the results of Stage 5's direct sort outperforming the iterative tournament (C3, Section III-F). Across the repeatability data (273 pairs $\times$ 3 repeats), a single Jev call takes a mean of 1.11s, for a total of 15.2 minutes of compute time across the full corpus -- call latency only, not skill/job profile extraction.

% NEW
\subsection{Cost and Latency Efficiency (C2) and Comparison with the Prior Tournament-Based System}
This section reports the results of Stage 4's direct sort outperforming the iterative tournament (C2, Section III-E). Across the repeatability data (273 pairs $\times$ 3 repeats), a single Jev call takes a mean of 1.11s, for a total of 15.2 minutes of compute time across the full corpus -- call latency only, not skill/job profile extraction.
```

- [ ] **Step 14: Comparison table — convergence and total-calls numbers reflect single-call production**

```latex
% OLD
Convergence & 0.957 & 0.982 (3-call average)\\
Total calls (corpus) & $\approx$2{,}061 (348 assess.\ + 1{,}713 tournament) & 819 (273$\times$3), 0 tournament\\

% NEW
Convergence & 0.957 & 0.964\\
Total calls (corpus) & $\approx$2{,}061 (348 assess.\ + 1{,}713 tournament) & 273 (273$\times$1), 0 tournament\\
```

- [ ] **Step 15: Comparison prose after the table — single-call, not three-call-averaged**

```latex
% OLD
Table~\ref{tab_comparison} shows this system is not a strict win on every axis: Faithfulness and per-answer confidence measure different constructs (groundedness vs.\ decision stability), so neither proves it is ``more accurate.'' But on the axis both systems share -- ranking-order stability -- nothing is traded away: three-call-averaged convergence now exceeds the prior tournament-based system's. It also wins on cost -- well under half the calls, no tournament infrastructure -- and offers a native, calibrated confidence the prior system had no way to provide. Its calls also parallelize, unlike the prior system's tournament, where each iteration waits on the last.

% NEW
Table~\ref{tab_comparison} shows this system is not a strict win on every axis: Faithfulness and per-answer confidence measure different constructs (groundedness vs.\ decision stability), so neither proves it is ``more accurate.'' But on the axis both systems share -- ranking-order stability -- nothing is traded away: single-call ranking convergence already exceeds the prior tournament-based system's. It also wins on cost -- well under half the calls, no tournament infrastructure -- and offers a native, calibrated confidence the prior system had no way to provide. Its calls also parallelize, unlike the prior system's tournament, where each iteration waits on the last.
```

- [ ] **Step 16: Conclusion — two adaptations, updated convergence numbers**

```latex
% OLD
This paper shows Jev, TypeSafe AI's first System One Model, outperforming the free-text assessment and iterative tournament-ranking stages of an existing pipeline, and validates three adaptations, presented stage by stage in Section III -- an empirical, large-sample test and extension of TypeSafe's own criteria-grounded question design guidance, multi-call ensemble averaging, and cost/latency efficiency from a single parallel pass outperforming the iterative tournament -- each motivated by a reliability problem observed in practice. The result is not a uniform improvement (Table~\ref{tab_comparison}): it trades all free-text interpretability for well under half the model calls, no tournament infrastructure, ranking convergence that now exceeds the prior tournament-based system's (0.982 vs.\ 0.957), and a native calibrated confidence the prior system never had.

% NEW
This paper shows Jev, TypeSafe AI's first System One Model, outperforming the free-text assessment and iterative tournament-ranking stages of an existing pipeline, and validates two adaptations, presented stage by stage in Section III -- an empirical, large-sample test and extension of TypeSafe's own criteria-grounded question design guidance, and cost/latency efficiency from a single parallel pass outperforming the iterative tournament -- each motivated by a reliability problem observed in practice. The result is not a uniform improvement (Table~\ref{tab_comparison}): it trades all free-text interpretability for well under half the model calls, no tournament infrastructure, ranking convergence that already exceeds the prior tournament-based system's (0.964 vs.\ 0.957), and a native calibrated confidence the prior system never had.
```

- [ ] **Step 17: Grep for any remaining dangling reference**

Run: `grep -n -i "multi.call\|tau_3\|3-call\|three-call\|ensemble\|\[C3\]\|Section III-G\|Section V-D" docs/paper2_jev.tex`

Expected: no output. If anything remains, it was missed by the steps above — fix it before committing (e.g. a stray "(C2)" left on the wrong subsection, or a "Section V-D"/"Section III-G" reference this plan's line numbers didn't anticipate).

- [ ] **Step 18: Commit**

```bash
git add -f docs/paper2_jev.tex
git commit -m "$(cat <<'EOF'
docs: remove multi-call ensemble averaging (C2) claim from paper 2

Paper now reports two contributions (criteria-grounded design,
cost/latency efficiency) instead of three, matching the code change
that dropped n_calls=3 ensembling from the production pipeline.
Convergence and total-call numbers now reflect single-call production
(0.964, 273 calls) rather than the retired 3-call ensemble (0.982, 819
calls); both numbers were already published in the paper's own
results, so no new experiment was run.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

(No `pdflatex` run, no `.pdf` committed — per Global Constraints.)

---

## Self-Review Notes

- **Spec coverage:** every bullet in `docs/superpowers/specs/2026-09-25-remove-multicall-ensemble-c2-design.md` maps to a task: assessment.py simplification → Task 1; script deletion + README → Task 2; every listed paper section → Task 3 (the plan additionally found and fixed two things the spec didn't call out by name but are direct consequences of the same change: the comparison table's "Total calls" row, which still said 819 (273×3), and the "Multi-Call Ensemble Averaging (C2)" results subsection's recommendation-reliability table/Fleiss'-kappa content, which was framed entirely as evidence for the retired contribution and is removed along with the rest of that subsection).
- **Placeholder scan:** none — every step has literal code/text, exact commands, and expected output.
- **Type/name consistency:** `generate_assessment` and `load_or_generate_assessment` signatures in Task 1's "Produces" match the bodies given in Step 3's Edits 4 and 6, and match how Task 1's test file (Step 1) calls them (no `n_calls` anywhere). No other task calls into Task 1's code.
