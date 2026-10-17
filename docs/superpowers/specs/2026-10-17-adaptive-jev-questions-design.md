# Adaptive Jev assessment questions per job profile

## Motivation

`_build_jev_questions` in `src/candidate_ranking/scoring/assessment.py` builds the
set of questions sent to Jev for every candidate/job-profile pair. Today every
question's wording is a fixed Python string, identical across all job profiles
(JPs) except for the skill name being substituted in. Three `JobProfile` fields
that are extracted from the job posting — `education_requirement`,
`seniority_requirement`, `other_requirements` — are never turned into a
question at all, so Jev never assesses them.

Fixing that gap raised a broader question: which of the *existing* fixed
questions should become per-JP adaptive, and which shouldn't. The dividing
line adopted here: a question is **adaptive** when the thing it asks about is
defined by the job profile's content (the specific skill, the specific
education/seniority/other requirement) — the wording genuinely differs
per JP. A question is **deleted** when it describes the candidate in
isolation, independent of the target role — there's nothing role-specific to
adapt, and keeping it fixed doesn't serve this project's goal of job-fit
assessment.

## Scope

Per-dimension disposition, decided in conversation:

| Dimension | Disposition | Why |
|---|---|---|
| `must_have_skills` / `nice_to_have_skills` evidence-strength | **Adaptive** | The skill comes from the JP; instructions/criteria can be tailored to what "strong evidence" looks like for that specific skill |
| `total_relevant_years` | **Adaptive** | "Relevant" is defined relative to this role; calibration (what counts as junior/senior years) should reflect this JP's seniority |
| `education_requirement`, `seniority_requirement`, `other_requirements` | **Adaptive (new)** | Content itself is job-specific; no fixed template can cover it |
| `career_direction` (ascending/lateral/descending/unclear) | **Deleted** | Describes the candidate's own trajectory shape; doesn't reference JP content |
| `tenure_pattern` (stable/variable/concerning) | **Deleted** | Same — candidate history only |
| `average_tenure_months` | **Deleted** | Supporting metric for `tenure_pattern`, deleted with it |
| `most_recent_role_duration_months` | **Deleted** | Candidate history only |
| `has_multinational_experience` | **Deleted** | Binary fact about candidate history, not role-dependent |

Net effect: `CareerTrajectory` (the model and the `Assessment.career_trajectory`
field) is removed entirely. `total_relevant_years` stays as a top-level
`Assessment` field (confirmed used as a ranking tie-break sort key in
`active_pairwise_ranking.py`).

Out of scope for this change: `assessment_audit.py`, `fairness_audit.py`,
`qualification_reliability.py`, `ranking_stability.py` reporting — these keep
working against whatever fields remain on `Assessment`; no new report
sections are added for the adaptive requirement evaluations.

## Architecture

Job-profile generation becomes two LLM calls instead of one:

**Call 1 (unchanged)** — `JOB_PROFILE_PROMPT` / `_JobProfileOutput` in
`job_profile.py`: extracts `must_have_skills`, `nice_to_have_skills`,
`education_requirement`, `seniority_requirement`, `other_requirements` from
the raw job posting text. Unchanged, including the deterministic
`"Must Have Skills: ..."` line override that can replace the LLM's own
`must_have_skills` guess.

**Call 2 (new)** — runs after Call 1's result is finalized (i.e. after the
explicit-must-have override is applied), so the skill list it receives as
input is guaranteed correct. Given the finalized skill list and the three
requirement fields, it generates the full adaptive rubric:

- one `{instructions, strong, adequate, weak, none}` entry per skill (matched
  back by skill name, not list position, so a reordering or omission in the
  LLM's output can't silently misalign skill ↔ criteria)
- one `{instructions}` entry calibrating `total_relevant_years` to this role
- one entry per non-null `education_requirement` / `seniority_requirement`,
  and one per item in `other_requirements`, each
  `{source, requirement, criteria_meets, criteria_partially_meets, criteria_does_not_meet}`

This two-call split exists specifically to avoid the skill-list misalignment
bug described above; it is not a general "split for its own sake."

Both calls are cached per JP exactly like job-profile generation is cached
today (cache key derived from the formatted prompt + model name) — this is a
one-time cost per JP, not per candidate. No change to the number of Jev calls
per candidate; the richer rubric is just more questions in the same Jev
`evaluate()` payload.

## Data model (`models.py`)

Remove:
```python
class CareerTrajectory(BaseModel): ...   # removed entirely
```
and `Assessment.career_trajectory`.

Add:
```python
class SkillCriteria(BaseModel):
    skill: str
    instructions: str
    strong: str
    adequate: str
    weak: str
    none: str

class TotalYearsCriteria(BaseModel):
    instructions: str

class AdaptiveRequirement(BaseModel):
    source: Literal["education", "seniority", "other"]
    requirement: str
    criteria_meets: str
    criteria_partially_meets: str
    criteria_does_not_meet: str
```
on `JobProfile`:
```python
skill_criteria: list[SkillCriteria] = Field(default_factory=list)
total_years_criteria: TotalYearsCriteria | None = None
adaptive_requirements: list[AdaptiveRequirement] = Field(default_factory=list)
```

Add:
```python
class AdaptiveRequirementEvaluation(BaseModel):
    source: Literal["education", "seniority", "other"]
    requirement: str
    evidence: str
    match_level: Literal["meets", "partially_meets", "does_not_meet"]
    confidence: float = Field(ge=0.0, le=1.0)
```
on `Assessment`:
```python
adaptive_requirement_evaluations: list[AdaptiveRequirementEvaluation] = Field(default_factory=list)
```

`adaptive_requirement_evaluations` never affects `qualified` or
`disqualification_reason` — confirmed as a soft signal only. Qualification
stays derived purely from `must_have_evaluations`, unchanged.

## Generation (`job_profile.py`)

- New prompt + `_RubricOutput` structured-output schema for Call 2, taking
  the finalized `must_have_skills`, `nice_to_have_skills`,
  `education_requirement`, `seniority_requirement`, `other_requirements` as
  input.
- `generate_job_profile` runs Call 2 after building the final skill list
  (after the explicit-must-have override), and populates
  `skill_criteria` / `total_years_criteria` / `adaptive_requirements` on the
  returned `JobProfile`.
- Fallback: if Call 2's `skill_criteria` is missing an entry for some skill
  (name not found), fall back to the existing generic
  `_SKILL_EVIDENCE_CRITERIA` wording for that skill only, and log a warning.
  This constant is kept in the codebase as a safety net — it stops being the
  default path, but a model hiccup on one skill shouldn't fail the whole JP.
  Same fallback behavior for a missing `total_years_criteria` (fall back to
  the current fixed calibration text, logged).
- Caching: reuse the existing `load_or_generate_job_profile` cache
  machinery — the cache key is derived from the full formatted prompt, so
  adding Call 2's output to what's cached per JP falls out of the existing
  cache-key-from-prompt-content pattern. Cache entry stores the complete
  `JobProfile` including the new fields.

## Evaluation (`scoring/assessment.py`)

`_build_jev_questions(jp)` rewritten to read wording from `jp` instead of
Python string literals:

- Per must-have/nice-to-have skill: look up `jp.skill_criteria` by skill
  name (falling back as above); build the `choice` question from its
  `instructions` / `strong` / `adequate` / `weak` / `none`.
- `total_relevant_years`: `score` question using `jp.total_years_criteria`
  (or fallback) for `instructions`; the tick-label scale
  (`["0 years", ..., "5+ years"]`) stays fixed in code — it's a mechanical
  scale consumed by `_score_to_int`, not descriptive content.
- One `choice` question per `jp.adaptive_requirements` entry, key
  `adaptive_{i}_match_level`, criteria `{"meets": ..., "partially_meets":
  ..., "does_not_meet": ...}`.
- The five trajectory questions (`career_direction`, `tenure_pattern`,
  `average_tenure_months`, `most_recent_role_duration_months`,
  `has_multinational_experience`) are removed, not replaced.

`_answers_to_assessment` rewritten to:
- drop the `CareerTrajectory` construction block entirely
- build `adaptive_requirement_evaluations` by zipping `jp.adaptive_requirements`
  with the corresponding `adaptive_{i}_match_level` answers; evidence text
  via the existing `_extract_bm25_evidence(cv, req.requirement)` helper,
  same as skill evidence today.

## Downstream touch points

- **`scoring/render.py`** — `_render_candidate_evidence`: remove the
  trajectory line (`traj = a.career_trajectory`, `traj_text`, and its use in
  the returned string); add an "Other requirements:" block listing
  `adaptive_requirement_evaluations` as `[match_level] requirement (source)`,
  so this signal reaches the pairwise/listwise ranking Jev calls the same
  way skill evidence does today.
- **`scoring/active_pairwise_ranking.py`** — remove "career trajectory" from
  the two prompt locations that instruct Jev to weigh it as a ranking
  criterion (the numbered list around line 35, and the "Consider: ..." line
  around line 175). Without this, the ranking prompt would ask Jev to judge
  something `render.py` no longer gives it any evidence for.
- **`output/writer.py`** (line ~42 area) and **`output/console_export.py`**
  (lines ~151, ~160-162) — remove the `career_trajectory` /
  `has_multinational_experience` export fields. `total_relevant_years`
  export is unaffected.

## Testing

- `tests/scoring/test_job_profile.py` — add coverage for Call 2: given a stubbed/mocked chain returning a
  `_RubricOutput`, verify `skill_criteria` / `total_years_criteria` /
  `adaptive_requirements` land correctly on the returned `JobProfile`, and
  that skill-criteria matching is by name (test a case where the rubric's
  skill order differs from `must_have_skills` order).
- Fallback test: a skill present in `must_have_skills` but absent from
  `skill_criteria` falls back to `_SKILL_EVIDENCE_CRITERIA` wording, and
  logs a warning.
- `tests/scoring/test_assessment.py` — update to remove the five trajectory
  `JevAnswer` fixtures and their assertions; add coverage for
  `_build_jev_questions` emitting one `choice` question per
  `jp.adaptive_requirements` entry with the right key/criteria, and for
  `_answers_to_assessment` building `adaptive_requirement_evaluations`
  correctly, including confirming `qualified`/`disqualification_reason`
  are unaffected regardless of `match_level`.
- `tests/test_models.py` — remove `CareerTrajectory`-related tests; add
  round-trip tests for `AdaptiveRequirement` / `AdaptiveRequirementEvaluation`.
- `tests/output/test_console_export.py`, `tests/output/test_formatter.py` —
  remove assertions on deleted trajectory fields.
- `tests/scoring/test_active_pairwise_ranking.py` — no change needed to the
  `total_relevant_years` sort-key tests (field is kept); add/verify no
  remaining references to "career trajectory" in ranking prompt text.
- Render test (new or existing) — verify the "Other requirements:" block
  renders correctly, and renders as "(none)" when
  `adaptive_requirement_evaluations` is empty.

## Explicitly not changing

- `qualified` / `disqualification_reason` computation (still purely
  `must_have_evaluations`-driven).
- Number of Jev calls per candidate (still one `evaluate()` call; just a
  different question set inside the same payload).
- `assessment_audit.py`, `fairness_audit.py`, `qualification_reliability.py`,
  `ranking_stability.py` — no new reporting sections added for the new
  adaptive requirement evaluations.
