# Remove Recommendation/Qualifications/Certification (Core Pipeline, 1/5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the pipeline from asking Jev for a hiring recommendation (`overall_recommendation`, Choice), a minimum-qualifications judgment (`meets_min_qualifications`, Noul), or certification checks (`certification::*`, Noul) — the `Assessment` model and generation logic become Score-only (per-requirement, seniority, education).

**Architecture:** `Assessment` (Pydantic model) loses the four fields tied to these questions. `_build_questions` stops constructing the two baseline questions and the certification loop. `_answers_to_assessment` stops populating the removed fields. Ranking is unaffected — `composite_fit_score` was already computed only from Score-type fields.

**Tech Stack:** Python 3, pydantic, pytest, unittest.mock.

## Global Constraints

- `src/candidate_ranking/scoring/jev_client.py` is untouched — `JevQuestion`/`JevAnswer` stay fully generic (still support `noul`/`choice`/`score`); only the pipeline stops constructing noul/choice questions.
- `JDSkills.certifications` (Stage 1 extraction) is untouched — it becomes unused by `_build_questions` but stays extracted; removing it is a separate subsystem, not part of this sub-project.
- Do not touch `src/candidate_ranking/output/formatter.py`, `src/candidate_ranking/output/console_export.py`, any file under `console-web/`, or `scripts/run_jev_evaluation_study.py` / `scripts/analyze_jev_evaluation_study.py` — those are later sub-projects (2-4/5) and currently reference the fields removed here. They will raise `AttributeError` when actually exercised (reading a field `Assessment` no longer has) until their own sub-project lands. This is expected.
- Pydantic v2's default `extra` behavior is `"ignore"`: passing a removed field name as a constructor kwarg does **not** raise — it is silently dropped, and the resulting object simply doesn't have that attribute. Do not rely on construction-time errors to catch stale field usage; the regression test in Task 1 checks `Assessment.model_fields` directly instead.

---

### Task 1: Remove the fields from `Assessment` and `assessment.py`, update their own tests

**Files:**
- Modify: `src/candidate_ranking/models.py`
- Modify: `src/candidate_ranking/scoring/assessment.py`
- Modify: `tests/scoring/test_assessment.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `Assessment` (pydantic model) with fields `job_description_id`, `candidate_id`, `generated_by_model`, `requirement_scores: dict[str, float]`, `confidence: dict[str, float]`, `seniority_years_fit_score: float | None`, `education_fit_score: float | None`, `requirement_probabilities: dict[str, dict[str, float]]`, `seniority_probabilities: dict[str, float] | None`, `education_probabilities: dict[str, float] | None`, and computed `composite_fit_score: float`. No `overall_recommendation`, `meets_min_qualifications`, `certification_results`, or `recommendation_probabilities`. `_build_questions(jd_skills: JDSkills | None) -> list[JevQuestion]` now returns only Score-kind questions (per-requirement, seniority, education) — this is what Task 2's `test_pipeline.py` fixtures and later sub-projects must assume.

- [ ] **Step 1: Rewrite `tests/scoring/test_assessment.py`**

Replace the entire file with:

```python
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from candidate_ranking.models import Assessment, Candidate, JDSkills, JobDescription
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
    assert assessment.requirement_scores == {"Python": 100.0, "SQL": 25.0}
    assert assessment.composite_fit_score == 62.5  # mean(100, 25)
    assert assessment.confidence["requirement::Python"] == 0.9
    jev_client.evaluate.assert_called_once()
    state, questions = jev_client.evaluate.call_args.args
    assert "Backend Engineer" in state
    assert "I know Python." in state
    question_keys = {q.key for q in questions}
    assert question_keys == {"requirement::Python", "requirement::SQL"}


def test_generate_assessment_skips_requirement_scores_without_jd_skills():
    jev_client = Mock()
    jev_client.evaluate.return_value = []

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, jd_skills=None)

    assert assessment.requirement_scores == {}
    assert assessment.composite_fit_score == 0.0
    _, questions = jev_client.evaluate.call_args.args
    assert questions == []


def test_generate_assessment_retries_once_on_low_confidence_then_accepts():
    jev_client = Mock()
    jd_skills = JDSkills(
        job_description_id="jd-1", generated_by_model="qwen2.5:14b", technical_skills=["Python"],
        seniority_requirement="5+ years", seniority_min_years=5.0,
    )
    low_confidence_answers = [
        JevAnswer(key="requirement::Python", kind="score", value=4.0, confidence=0.9),
        JevAnswer(key="seniority_years", kind="score", value=3.0, confidence=0.3),
    ]
    high_confidence_answers = [
        JevAnswer(key="requirement::Python", kind="score", value=4.0, confidence=0.9),
        JevAnswer(key="seniority_years", kind="score", value=3.0, confidence=0.9),
    ]
    jev_client.evaluate.side_effect = [low_confidence_answers, high_confidence_answers]

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, jd_skills)

    assert jev_client.evaluate.call_count == 2
    assert assessment.confidence["seniority_years"] == 0.9
    second_state, _ = jev_client.evaluate.call_args_list[1].args
    assert "low-confidence" in second_state


def test_generate_assessment_wraps_jev_client_error():
    jev_client = Mock()
    jev_client.evaluate.side_effect = JevClientError("network down")

    with pytest.raises(AssessmentGenerationError, match="network down"):
        generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills())


def test_generate_assessment_preserves_probability_distributions():
    jev_client = Mock()
    jev_client.evaluate.return_value = [
        JevAnswer(key="requirement::Python", kind="score", value=4.0, confidence=0.9,
                   probabilities={"0": 0, "1": 0, "2": 0, "3": 0.1, "4": 0.9}),
    ]

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills())

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


def test_load_or_generate_assessment_cache_key_depends_on_jd_skills(tmp_path: Path):
    jev_client = Mock()
    jev_client.evaluate.return_value = _high_confidence_answers()

    load_or_generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, tmp_path, _jd_skills())
    assert jev_client.evaluate.call_count == 1

    load_or_generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, tmp_path, jd_skills=None)
    assert jev_client.evaluate.call_count == 2


def test_generate_assessment_builds_seniority_education_questions():
    jev_client = Mock()
    jev_client.evaluate.return_value = [
        JevAnswer(key="requirement::Python", kind="score", value=4.0, confidence=0.9),
        JevAnswer(key="seniority_years", kind="score", value=3.0, confidence=0.85),
        JevAnswer(key="education", kind="score", value=1.0, confidence=0.8),
    ]

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills_full())

    _, questions = jev_client.evaluate.call_args.args
    question_keys = {q.key for q in questions}
    assert "seniority_years" in question_keys
    assert "education" in question_keys
    assert not any(k.startswith("certification::") for k in question_keys)
    assert assessment.seniority_years_fit_score == 75.0
    assert assessment.education_fit_score == 25.0


def test_generate_assessment_omits_seniority_education_questions_when_not_stated():
    jev_client = Mock()
    jev_client.evaluate.return_value = [
        JevAnswer(key="requirement::Python", kind="score", value=4.0, confidence=0.9),
        JevAnswer(key="requirement::SQL", kind="score", value=1.0, confidence=0.8),
    ]

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills())

    _, questions = jev_client.evaluate.call_args.args
    question_keys = {q.key for q in questions}
    assert "seniority_years" not in question_keys
    assert "education" not in question_keys
    assert not any(k.startswith("certification::") for k in question_keys)
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

    for banned_phrase in ("supports that the candidate", "does not support that the candidate"):
        assert banned_phrase not in seniority_years_criteria
        assert banned_phrase not in education_criteria

    assert any(term in seniority_years_criteria for term in ("total experience", "years"))
    assert any(term in education_criteria for term in ("degree", "field", "credential"))


def test_build_questions_never_includes_recommendation_qualifications_certification():
    questions = _build_questions(
        JDSkills(
            job_description_id="jd-1", generated_by_model="qwen2.5:14b", technical_skills=["Python"],
            certifications=["PMP"], seniority_requirement="5+ years", seniority_min_years=5.0,
            education_requirement="Bachelor's degree",
        )
    )
    question_keys = {q.key for q in questions}
    assert "overall_recommendation" not in question_keys
    assert "meets_min_qualifications" not in question_keys
    assert not any(k.startswith("certification::") for k in question_keys)
    assert all(q.kind == "score" for q in questions)


def test_assessment_model_has_no_recommendation_qualifications_certification_fields():
    removed_fields = {
        "overall_recommendation", "meets_min_qualifications",
        "certification_results", "recommendation_probabilities",
    }
    assert removed_fields.isdisjoint(Assessment.model_fields)
```

This drops `test_generate_assessment_wraps_unusable_jev_response` (it tested a `KeyError`/`ValidationError` triggered by a missing `overall_recommendation`/`meets_min_qualifications` answer or an invalid literal value for `overall_recommendation` — with those fields gone, `_answers_to_assessment` no longer does any unguarded `by_key[...]` lookup and has no required-Literal field to violate, so this failure mode no longer exists) and `test_generate_assessment_builds_certification_seniority_education_questions` (replaced by `test_generate_assessment_builds_seniority_education_questions`, without the certification assertions). It adds two new regression tests: `test_build_questions_never_includes_recommendation_qualifications_certification` and `test_assessment_model_has_no_recommendation_qualifications_certification_fields`.

- [ ] **Step 2: Run the tests and confirm they fail against the current implementation**

Run: `uv run pytest tests/scoring/test_assessment.py -v`

Expected: several `FAIL`s. In particular `test_generate_assessment_maps_jev_answers_onto_assessment` fails on `assert question_keys == {"requirement::Python", "requirement::SQL"}` (current code still includes `overall_recommendation`/`meets_min_qualifications` in the question set), and `test_build_questions_never_includes_recommendation_qualifications_certification` / `test_assessment_model_has_no_recommendation_qualifications_certification_fields` fail because those fields/questions still exist.

- [ ] **Step 3: Rewrite `tests/test_models.py`**

Replace the entire file with:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from candidate_ranking.models import Assessment, JDSkills


def test_assessment_accepts_new_score_based_fields():
    assessment = Assessment(
        job_description_id="jd-1",
        candidate_id="cand-1",
        generated_by_model="typesafe/jev",
        requirement_scores={"python": 100.0, "sql": 50.0},
        confidence={"requirement::python": 0.9},
    )
    assert assessment.requirement_scores["python"] == 100.0


def test_assessment_composite_fit_score_is_mean_of_sub_scores():
    assessment = Assessment(
        job_description_id="jd-1",
        candidate_id="cand-1",
        generated_by_model="typesafe/jev",
        requirement_scores={"python": 80.0, "sql": 60.0},
        seniority_years_fit_score=100.0,
        education_fit_score=40.0,
    )
    assert assessment.composite_fit_score == (80.0 + 60.0 + 100.0 + 40.0) / 4


def test_assessment_composite_fit_score_ignores_missing_optional_scores():
    assessment = Assessment(
        job_description_id="jd-1",
        candidate_id="cand-1",
        generated_by_model="typesafe/jev",
        requirement_scores={"python": 80.0},
    )
    assert assessment.composite_fit_score == 80.0


def test_assessment_rejects_score_out_of_range():
    with pytest.raises(ValidationError):
        Assessment(
            job_description_id="jd-1",
            candidate_id="cand-1",
            generated_by_model="typesafe/jev",
            seniority_years_fit_score=150.0,
        )


def test_assessment_no_longer_has_free_text_fields():
    assert "strengths" not in Assessment.model_fields
    assert "weaknesses" not in Assessment.model_fields
    assert "reasoning" not in Assessment.model_fields
    assert "additional_skills" not in Assessment.model_fields


def test_assessment_no_longer_has_recommendation_qualifications_certification_fields():
    removed_fields = {
        "overall_recommendation", "meets_min_qualifications",
        "certification_results", "recommendation_probabilities",
    }
    assert removed_fields.isdisjoint(Assessment.model_fields)


def test_assessment_accepts_seniority_education_fields():
    assessment = Assessment(
        job_description_id="jd-1",
        candidate_id="cand-1",
        generated_by_model="typesafe/jev",
        seniority_years_fit_score=75.0,
        education_fit_score=None,
    )
    assert assessment.seniority_years_fit_score == 75.0
    assert assessment.education_fit_score is None


def test_assessment_seniority_and_education_default_empty():
    assessment = Assessment(
        job_description_id="jd-1",
        candidate_id="cand-1",
        generated_by_model="typesafe/jev",
    )
    assert assessment.seniority_years_fit_score is None
    assert assessment.education_fit_score is None
    assert assessment.composite_fit_score == 0.0


def test_jd_skills_accepts_certifications_seniority_education():
    jd_skills = JDSkills(
        job_description_id="jd-1",
        generated_by_model="qwen2.5:14b",
        technical_skills=["Python"],
        certifications=["AWS Certified Solutions Architect"],
        seniority_requirement="5+ years of backend development experience",
        education_requirement="Bachelor's degree in Computer Science or related field",
    )
    assert jd_skills.certifications == ["AWS Certified Solutions Architect"]
    assert jd_skills.seniority_requirement == "5+ years of backend development experience"


def test_jd_skills_certifications_seniority_education_default_empty():
    jd_skills = JDSkills(job_description_id="jd-1", generated_by_model="qwen2.5:14b", technical_skills=["Python"])
    assert jd_skills.certifications == []
    assert jd_skills.seniority_requirement is None
    assert jd_skills.education_requirement is None


def test_jd_skills_accepts_must_have_skills():
    jd_skills = JDSkills(
        job_description_id="jd-1",
        generated_by_model="qwen2.5:14b",
        technical_skills=["Python", "SQL"],
        must_have_skills=["Python"],
    )
    assert jd_skills.must_have_skills == ["Python"]


def test_jd_skills_must_have_skills_defaults_empty():
    jd_skills = JDSkills(job_description_id="jd-1", generated_by_model="qwen2.5:14b", technical_skills=["Python"])
    assert jd_skills.must_have_skills == []
```

`JDSkills`-related tests are unchanged from before — `JDSkills.certifications` is explicitly kept per this sub-project's scope.

- [ ] **Step 4: Run the models tests and confirm they fail against the current implementation**

Run: `uv run pytest tests/test_models.py -v`

Expected: `test_assessment_no_longer_has_recommendation_qualifications_certification_fields` fails (`Assessment.model_fields` still has `overall_recommendation` etc.); other rewritten tests that omit `overall_recommendation`/`meets_min_qualifications` from their `Assessment(...)` calls fail with a `pydantic_core.ValidationError` (`Field required`) since those two fields have no default in the current model.

- [ ] **Step 5: Update `src/candidate_ranking/models.py`**

```python
# OLD
class Assessment(BaseModel):
    job_description_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    generated_by_model: str = Field(min_length=1)
    overall_recommendation: Literal["hire", "maybe", "no"]
    meets_min_qualifications: bool
    requirement_scores: dict[str, float] = Field(default_factory=dict)
    confidence: dict[str, float] = Field(default_factory=dict)
    certification_results: dict[str, bool] = Field(default_factory=dict)
    seniority_years_fit_score: float | None = Field(default=None, ge=0, le=100)
    education_fit_score: float | None = Field(default=None, ge=0, le=100)
    # Full per-level/per-option probability distributions Jev returns alongside the collapsed
    # value/confidence above (see JevAnswer.probabilities) -- kept on Assessment itself, not just
    # the raw per-call JevAnswer list, so this detail survives into the standard pipeline output
    # (assessments.json) rather than being discarded whenever no separate raw-answer capture is
    # in play.
    recommendation_probabilities: dict[str, float] = Field(default_factory=dict)
    requirement_probabilities: dict[str, dict[str, float]] = Field(default_factory=dict)
    seniority_probabilities: dict[str, float] | None = None
    education_probabilities: dict[str, float] | None = None

    @computed_field
    @property
    def composite_fit_score(self) -> float:
        components = list(self.requirement_scores.values())
        if self.seniority_years_fit_score is not None:
            components.append(self.seniority_years_fit_score)
        if self.education_fit_score is not None:
            components.append(self.education_fit_score)
        return sum(components) / len(components) if components else 0.0

# NEW
class Assessment(BaseModel):
    job_description_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    generated_by_model: str = Field(min_length=1)
    requirement_scores: dict[str, float] = Field(default_factory=dict)
    confidence: dict[str, float] = Field(default_factory=dict)
    seniority_years_fit_score: float | None = Field(default=None, ge=0, le=100)
    education_fit_score: float | None = Field(default=None, ge=0, le=100)
    # Full per-level probability distributions Jev returns alongside the collapsed value/confidence
    # above (see JevAnswer.probabilities) -- kept on Assessment itself, not just the raw per-call
    # JevAnswer list, so this detail survives into the standard pipeline output (assessments.json)
    # rather than being discarded whenever no separate raw-answer capture is in play.
    requirement_probabilities: dict[str, dict[str, float]] = Field(default_factory=dict)
    seniority_probabilities: dict[str, float] | None = None
    education_probabilities: dict[str, float] | None = None

    @computed_field
    @property
    def composite_fit_score(self) -> float:
        components = list(self.requirement_scores.values())
        if self.seniority_years_fit_score is not None:
            components.append(self.seniority_years_fit_score)
        if self.education_fit_score is not None:
            components.append(self.education_fit_score)
        return sum(components) / len(components) if components else 0.0
```

(The `Literal` import at the top of `models.py` stays — it's still used by `Candidate.parse_status` and `TournamentResult.status`/`stop_reason`.)

- [ ] **Step 6: Update `src/candidate_ranking/scoring/assessment.py`**

Edit 1 — module constants:

```python
# OLD
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

# NEW
ASSESSMENT_SCOPE_VERSION = "jev-score-only-v1"

_REQUIREMENT_KEY_PREFIX = "requirement::"
_SENIORITY_YEARS_KEY = "seniority_years"
_EDUCATION_KEY = "education"
_RETRY_ON_LOW_CONFIDENCE_KEYS = (
    _SENIORITY_YEARS_KEY, _EDUCATION_KEY,
)
_CONFIDENCE_RETRY_THRESHOLD = 0.5
```

Edit 2 — drop `_certification_question_key`:

```python
# OLD
def _requirement_question_key(requirement: str) -> str:
    return f"{_REQUIREMENT_KEY_PREFIX}{requirement}"


def _certification_question_key(certification: str) -> str:
    return f"{_CERTIFICATION_KEY_PREFIX}{certification}"


def _seniority_years_instructions(seniority_requirement: str, min_years: float) -> str:

# NEW
def _requirement_question_key(requirement: str) -> str:
    return f"{_REQUIREMENT_KEY_PREFIX}{requirement}"


def _seniority_years_instructions(seniority_requirement: str, min_years: float) -> str:
```

Edit 3 — `_build_questions`:

```python
# OLD
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

# NEW
def _build_questions(jd_skills: JDSkills | None) -> list[JevQuestion]:
    questions: list[JevQuestion] = []
    for requirement in jd_skills.technical_skills if jd_skills else []:
        questions.append(
            JevQuestion(
                key=_requirement_question_key(requirement),
                kind="score",
                instructions=f"How well does the candidate's CV support the requirement '{requirement}'?",
                criteria=_REQUIREMENT_FIT_CRITERIA,
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
```

Edit 4 — `_answers_to_assessment`:

```python
# OLD
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

# NEW
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
    requirement_probabilities = {
        key[len(_REQUIREMENT_KEY_PREFIX):]: a.probabilities
        for key, a in by_key.items()
        if key.startswith(_REQUIREMENT_KEY_PREFIX) and a.probabilities
    }
    return Assessment(
        job_description_id=jd.id,
        candidate_id=candidate.id,
        generated_by_model=model_name,
        requirement_scores=requirement_scores,
        confidence=confidence,
        seniority_years_fit_score=(
            _score_to_percent(by_key[_SENIORITY_YEARS_KEY].value) if _SENIORITY_YEARS_KEY in by_key else None
        ),
        education_fit_score=_score_to_percent(by_key[_EDUCATION_KEY].value) if _EDUCATION_KEY in by_key else None,
        requirement_probabilities=requirement_probabilities,
        seniority_probabilities=(
            by_key[_SENIORITY_YEARS_KEY].probabilities if _SENIORITY_YEARS_KEY in by_key else None
        ),
        education_probabilities=by_key[_EDUCATION_KEY].probabilities if _EDUCATION_KEY in by_key else None,
    )
```

- [ ] **Step 7: Run both test files and confirm they pass**

Run: `uv run pytest tests/scoring/test_assessment.py tests/test_models.py -v`

Expected: all tests `PASS`.

- [ ] **Step 8: Commit**

```bash
git add src/candidate_ranking/models.py src/candidate_ranking/scoring/assessment.py \
  tests/scoring/test_assessment.py tests/test_models.py
git commit -m "$(cat <<'EOF'
refactor: remove recommendation/qualifications/certification from core pipeline

Assessment and assessment.py are now Score-only (per-requirement,
seniority, education) -- ranking is unaffected since composite_fit_score
was already computed only from Score fields. This is sub-project 1/5 of
retiring these fields (see docs/superpowers/specs/2026-09-25-remove-
recommendation-qualifications-certification-core-design.md); the output
layer, frontend, evaluation scripts, and paper still reference the
removed fields and will be fixed in their own follow-up sub-projects.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Contain the ripple effects in `tests/graphs/test_pipeline.py` and `tests/output/test_formatter.py`

**Files:**
- Modify: `tests/graphs/test_pipeline.py`
- Modify: `tests/output/test_formatter.py`

**Interfaces:**
- Consumes: `Assessment(...)` with Task 1's new field set (no `overall_recommendation`/`meets_min_qualifications` constructor kwargs — passing them is silently ignored by Pydantic's default `extra="ignore"`, not an error, but leaves the object without that attribute).
- Produces: nothing consumed by later tasks.

`src/candidate_ranking/graphs/pipeline.py` itself never reads the removed fields (verified: no match for `overall_recommendation`/`meets_min_qualifications` in that file), so most of `test_pipeline.py` needs only its `Assessment(...)` fixture calls fixed. But one test in that file (`test_rank_and_format_jd_writes_ranking_files`) calls `rank_and_format_jd`, which calls into `src/candidate_ranking/output/formatter.py` — a file this sub-project does not touch, and which still reads `assessment.overall_recommendation`/`assessment.meets_min_qualifications` (now-missing attributes) and will raise `AttributeError`. `tests/output/test_formatter.py` tests `formatter.py` directly and is entirely about the removed fields' presence in its output rows — every test in that file depends on `formatter.py`, unmodified here.

Both must be marked as **known, expected, temporary failures** (via `pytest.mark.xfail`) rather than left as unexplained red tests, with a reason pointing at the sub-project that resolves them. Do not modify `src/candidate_ranking/output/formatter.py` or `src/candidate_ranking/graphs/pipeline.py` in this task — only test files, per the Global Constraints.

- [ ] **Step 1: Fix `tests/graphs/test_pipeline.py`'s fixtures and mark the one formatter-dependent test `xfail`**

```python
# OLD
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from candidate_ranking.config import RunConfig
from candidate_ranking.graphs.pipeline import PipelineState, assessments_by_jd, build_pipeline_graph, rank_and_format_jd
from candidate_ranking.models import Assessment, JobDescription


def _ok_result(jd_id: str, candidate_id: str, score: float) -> dict:
    return {
        "jd_id": jd_id,
        "candidate_id": candidate_id,
        "status": "ok",
        "assessment": Assessment(
            job_description_id=jd_id, candidate_id=candidate_id, generated_by_model="typesafe/jev",
            overall_recommendation="hire", meets_min_qualifications=True,
            requirement_scores={"python": score},
        ),
        "error": None,
    }

# NEW
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from candidate_ranking.config import RunConfig
from candidate_ranking.graphs.pipeline import PipelineState, assessments_by_jd, build_pipeline_graph, rank_and_format_jd
from candidate_ranking.models import Assessment, JobDescription


def _ok_result(jd_id: str, candidate_id: str, score: float) -> dict:
    return {
        "jd_id": jd_id,
        "candidate_id": candidate_id,
        "status": "ok",
        "assessment": Assessment(
            job_description_id=jd_id, candidate_id=candidate_id, generated_by_model="typesafe/jev",
            requirement_scores={"python": score},
        ),
        "error": None,
    }
```

```python
# OLD
def test_rank_and_format_jd_writes_ranking_files(tmp_path: Path):
    jd = JobDescription(id="jd-1", title="Backend Engineer", raw_text="...", source_path="jd.pdf")
    assessments = {
        "cand-a": Assessment(
            job_description_id="jd-1", candidate_id="cand-a", generated_by_model="typesafe/jev",
            overall_recommendation="hire", meets_min_qualifications=True,
            requirement_scores={"python": 80.0},
        )
    }

    rank_and_format_jd(tmp_path, "run-1", jd, assessments)

    ranking = json.loads((tmp_path / "run-1" / "jd-1" / "ranking.json").read_text(encoding="utf-8"))
    assert ranking["rankings"][0]["candidate_id"] == "cand-a"

# NEW
@pytest.mark.xfail(
    reason="formatter.py still reads Assessment.overall_recommendation/meets_min_qualifications, "
    "removed in sub-project 1/5 of docs/superpowers/specs/2026-09-25-remove-recommendation-"
    "qualifications-certification-core-design.md -- fixed by sub-project 2/5 (output layer)",
    strict=False,
)
def test_rank_and_format_jd_writes_ranking_files(tmp_path: Path):
    jd = JobDescription(id="jd-1", title="Backend Engineer", raw_text="...", source_path="jd.pdf")
    assessments = {
        "cand-a": Assessment(
            job_description_id="jd-1", candidate_id="cand-a", generated_by_model="typesafe/jev",
            requirement_scores={"python": 80.0},
        )
    }

    rank_and_format_jd(tmp_path, "run-1", jd, assessments)

    ranking = json.loads((tmp_path / "run-1" / "jd-1" / "ranking.json").read_text(encoding="utf-8"))
    assert ranking["rankings"][0]["candidate_id"] == "cand-a"
```

Leave the rest of the file (`test_pipeline_state_has_no_tournament_results_key`, `test_assessments_by_jd_groups_ok_results_and_skips_failed`, `_failed_result`, any other tests) unchanged — `_failed_result` never constructs an `Assessment` with the removed fields, and `assessments_by_jd`/`PipelineState` don't touch them either.

- [ ] **Step 2: Mark all of `tests/output/test_formatter.py` `xfail`**

Every test in this file depends on `formatter.py` output rows containing `overall_recommendation`/`meets_min_qualifications` — add a module-level marker rather than touching each test individually:

```python
# OLD
from __future__ import annotations

import json
from pathlib import Path

from candidate_ranking.models import Assessment, JobDescription
from candidate_ranking.output.formatter import format_jd_ranking, write_jd_ranking

# NEW
from __future__ import annotations

import json
from pathlib import Path

import pytest

from candidate_ranking.models import Assessment, JobDescription
from candidate_ranking.output.formatter import format_jd_ranking, write_jd_ranking

pytestmark = pytest.mark.xfail(
    reason="formatter.py still reads Assessment.overall_recommendation/meets_min_qualifications, "
    "removed in sub-project 1/5 of docs/superpowers/specs/2026-09-25-remove-recommendation-"
    "qualifications-certification-core-design.md -- fixed by sub-project 2/5 (output layer)",
    strict=False,
)
```

Do not otherwise modify this file's test bodies or its `_assessment(...)` fixture in this task — sub-project 2/5 will rewrite them together with `formatter.py` itself.

- [ ] **Step 3: Run the full suite and confirm the expected state**

Run: `uv run pytest -v`

Expected: `tests/graphs/test_pipeline.py::test_rank_and_format_jd_writes_ranking_files` and every test in `tests/output/test_formatter.py` report `XFAIL` (not `FAIL`), everything else `PASS`. The overall run exits successfully (xfail does not fail a pytest session). If any test in `tests/output/test_formatter.py` unexpectedly reports `XPASS`, or any test outside these two files fails, stop and investigate before committing — that means this task's assumptions about what's still broken were wrong.

- [ ] **Step 4: Commit**

```bash
git add tests/graphs/test_pipeline.py tests/output/test_formatter.py
git commit -m "$(cat <<'EOF'
test: mark formatter-dependent tests xfail pending sub-project 2/5

formatter.py still reads the Assessment fields removed in sub-project
1/5 (recommendation/qualifications/certification); these tests fail
for that known, tracked reason until sub-project 2/5 (output layer)
lands. Fixture construction is updated to the new Assessment shape
everywhere it doesn't depend on formatter.py's own fix.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** models.py field removal → Task 1 Step 5; assessment.py question/answer-mapping/retry-keys/scope-version changes → Task 1 Step 6; `jev_client.py` and `JDSkills.certifications` explicitly untouched → Global Constraints + no task touches them; test coverage for the removal itself → Task 1 Steps 1 & 3 (plus two new regression tests per the spec's testing section). The spec's explicit note that other files will break until their own sub-project → Task 2 makes this an intentional, verified, tracked state (`xfail`) instead of silent red tests.
- **Placeholder scan:** none — every step has literal code, exact commands, and expected output.
- **Type/name consistency:** `Assessment`'s field set in Task 1 Step 5 matches exactly what Task 1's Steps 1 and 3 test files construct and assert on, and matches what Task 2's fixtures construct. `_build_questions` returning only score-kind `JevQuestion`s (Task 1 Step 6, Edit 3) matches Task 1's `test_build_questions_never_includes_recommendation_qualifications_certification`.
- **Discovered during planning, not in the original spec:** Pydantic v2's `extra="ignore"` default means old code passing the removed field names as constructor kwargs does not raise at construction time — it silently drops them. This is why Task 2 needed `xfail` markers (the breakage shows up as `AttributeError` deep in `formatter.py`, not as a constructor error at the fixture) rather than a simpler "these tests now raise `ValidationError`" fix. Called out explicitly in Global Constraints so neither task's implementer is surprised by it.
