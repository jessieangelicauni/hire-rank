# Seniority/Education Score-Criteria Redesign — Design

Date: 2026-09-19

## Motivation

`meets_seniority_requirement` and `meets_education_requirement` are
currently Noul (true/false) questions answered in one shot per
(job, applicant) pair. A single bit cannot express the gradations that
actually occur: "4 years experience when 5 is required" collapses to
the same `false` as "0 years"; "10 years senior experience in the
wrong domain" collapses to the same ambiguous judgment as a domain
match; "Master's degree in an unrelated field" collapses to the same
signal as "no education mentioned at all."

Technical requirements already solve exactly this problem with a
Score question (5 ordinal levels, concrete evidence-based criteria per
level — see `_REQUIREMENT_FIT_CRITERIA` in `assessment.py`), validated
in the paper's criteria-grounded ablation (`r=0.747`, `p<0.001`). This
redesign applies the same, already-validated pattern to seniority and
education instead of inventing new machinery.

Certification is explicitly **not** changed: each required
certification is already its own independent Noul question
(`_certification_question_key`), so "2 of 3 required certs held"
already emerges from combining multiple binary answers — no
information is lost the way it is for the single holistic
seniority/education questions.

## Scope

**In scope:**
- `src/candidate_ranking/scoring/assessment.py` — question kind,
  criteria definitions, aggregation for seniority/education
- `src/candidate_ranking/models.py` — `Assessment` field types
- `src/candidate_ranking/output/console_export.py` — export the new
  score fields (currently `meets_seniority_requirement` /
  `meets_education_requirement` are **not exported to console-web at
  all**; this adds them for the first time, following the existing
  `requirement_scores` pattern)
- `console-web/src/data.ts`, `Leaderboard.tsx`, `Comparison.tsx` — new
  score display for seniority/education fit
- `scripts/analyze_jev_evaluation_study.py` — new Score-type ablation
  (concrete vs. bare-label) replacing the old Noul ablation
  (concrete vs. circular) for these two fields
- `docs/paper2_jev.tex` — Methodology (§III-A, §III-B), Table 1
  (`tab_score_ablation`), Results (§V-A/V-B), abstract

**Out of scope (unchanged):**
- Certification: stays per-cert Noul
- `meets_min_qualifications`: stays a single holistic Noul question,
  independent of seniority/education — no gating logic reads
  seniority/education answers to compute it, so nothing breaks
  downstream of this change
- Overall-fit Score, per-requirement Score, recommendation Choice:
  untouched

## Question & Criteria Changes

`assessment.py`, `_build_questions()`: both seniority and education
questions change `kind="noul"` → `kind="score"`, keeping their
existing per-job interpolated `instructions` (the specific stated
requirement text is already job-adaptive — that does not change),
replacing their `criteria` dict with a 5-level ordinal list in the
same shape as `_REQUIREMENT_FIT_CRITERIA`:

```python
_SENIORITY_FIT_CRITERIA = [
    "No relevant experience or years mentioned toward this requirement",
    "Some relevant experience but clearly short of the stated years/level, "
    "or relevant only in an adjacent domain (e.g., internship-level, or "
    "senior experience in an unrelated field)",
    "Close to but slightly under the stated years/level requirement, in a "
    "clearly related role",
    "Meets the stated years/level requirement in a directly matching role",
    "Exceeds the stated years/level requirement in a directly matching "
    "role, with demonstrated seniority (e.g., ownership scope, leadership)",
]

_EDUCATION_FIT_CRITERIA = [
    "No relevant education mentioned",
    "Education mentioned but neither the field nor the degree level "
    "matches the requirement",
    "Matches the requirement on field or degree level, but not both",
    "Matches the requirement's field and degree level",
    "Exceeds the requirement (higher degree level) in the same or a "
    "closely related field",
]
```

Both lists go through the same `_score_to_percent` mapping already
used for `_OVERALL_FIT_CRITERIA` / `_REQUIREMENT_FIT_CRITERIA`
(index × 25, since each list has 5 entries) — no new scaling logic.

## Model Changes

`models.py` — `Assessment`:

```python
meets_seniority_requirement: bool | None = None   # removed
meets_education_requirement: bool | None = None   # removed
```
replaced with:
```python
seniority_fit_score: float | None = Field(default=None, ge=0, le=100)
education_fit_score: float | None = Field(default=None, ge=0, le=100)
```

`None` when the job profile states no seniority/education requirement
(mirrors the current `None`-when-absent behavior, and the existing
`if jd_skills and jd_skills.seniority_requirement:` guard in
`_build_questions`).

## Aggregation

`assessment.py`, the multi-call combine step (around
`_aggregate_optional_bool` today): drop
`_aggregate_optional_bool(calls, "meets_seniority_requirement")` /
`..._education_requirement`. Replace with a new
`_aggregate_mean_optional(calls, field)` helper: since
seniority/education presence is determined by the job profile
(`jd_skills.seniority_requirement`/`education_requirement`), it is
identical across all `n` calls for a given assessment — either every
call has a value or every call has `None`. The helper returns
`statistics.mean(values)` when every call has a non-`None` value for
`field`, else `None`; it does not need to handle a mixed
present/absent case, since that cannot occur here.

## Output & UI

**`console_export.py`**: add `seniority_fit_score` and
`education_fit_score` to the per-assessment export dict (same
location as `requirement_scores`, line ~254) as new top-level numeric
fields — this is new surface area, not a rename, since these were
never exported before.

**`console-web/src/data.ts`**: add
`seniority_fit_score: number | null` and
`education_fit_score: number | null` to the `Assessment` type and its
default value.

**`Leaderboard.tsx`**: add a small score row to `ScoreSummary` (or a
new component alongside `RequirementScores`) displaying seniority/
education fit using the same bar/number treatment as
`RequirementScores`, only rendered when the value is non-null (job
profile stated that requirement).

**`Comparison.tsx`** (correction after inspecting the file): it does
**not** read `requirement_scores` at all — it only shows role-level
aggregates (`meanFitScore`, `meetsMinRate`, `hireRate`,
`rankingStability`) from `COMPARISON`. There is no existing
per-requirement display there to extend, so it is left untouched by
this change; adding a new aggregate seniority/education stat card
would be new scope beyond what was asked.

## Ablation Redesign (paper validation)

`scripts/analyze_jev_evaluation_study.py`: replace the seniority/
education rows of the criteria-design ablation. Old design compared
concrete vs. circular Noul criteria; new design mirrors the
Overall-fit/per-requirement Score ablation exactly — concrete
evidence-based criteria (above) vs. bare ordinal labels ("0", "25",
"50", "75", "100", no description), paired by (job, applicant),
Wilcoxon signed-rank + rank-biserial effect size, same statistical
machinery already implemented for the Score ablation.

Sample: **all eligible pairs** (every pair whose job profile states a
seniority/education requirement), not a subsample — this both
validates the new design and fixes the underpowered-test problem
observed for the old Seniority row (`n=66`, `p=0.134`, same effect
size `r=0.18` as the significant Education/Pooled rows). Cost: one
extra Jev call per eligible pair, same mechanism as the existing
Score ablation's extra call.

## Paper Updates (`docs/paper2_jev.tex`)

- §III-A (pipeline description): "...one Score per technical
  requirement, one Score per stated seniority/education requirement,
  and one Noul per stated certification" (certification is now the
  only remaining per-field Noul type besides min-qualifications).
- §III-B (Criteria-Grounded Question Design): the seniority/education
  paragraph (currently describing the circular-Noul-criteria fix) is
  rewritten to describe the same bare-label-to-concrete-evidence fix
  used for Score questions, since seniority/education are now Score
  questions.
- Table `tab_score_ablation`: the "Noul criteria (concrete vs.
  circular)" section (Seniority/Education/Pooled rows) is replaced
  with new Seniority/Education rows in the same row-shape as
  Overall-fit/Per-requirement (concrete vs. bare-label, full-sample
  n).
- Abstract and §V-A/V-B: all seniority/education numbers (currently
  0.828/0.839 pooled, `p=0.0063`) are replaced outright with the
  rerun's results — no old-design numbers retained anywhere in the
  paper (full replacement, not an added comparison).

## Testing Plan

- Unit tests for `_build_questions()`: seniority/education entries
  have `kind="score"` and the new criteria lists, only present when
  `jd_skills.seniority_requirement` / `education_requirement` is set.
- Unit tests for aggregation: `seniority_fit_score`/
  `education_fit_score` averaged correctly across calls; `None` when
  absent from all calls.
- Unit tests for `console_export.py`: new fields present in export
  output with correct values.
- Snapshot/render check for the new `Leaderboard.tsx`/`Comparison.tsx`
  score display (renders bar/number when present, omits when `null`).
- Re-run of the full evaluation study (all 346 shortlisted pairs) plus
  the new full-sample seniority/education ablation, feeding updated
  numbers into the paper.
