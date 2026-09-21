# Analytics Role Breakdown & Per-Applicant Multicall View

## Purpose

The Analytics page (`console-web/src/views/Comparison.tsx`) currently shows, per role, only `meanFitScore`, `meetsMinRate`, `hireRate`, and `rankingStability`. Every other per-applicant value already captured by the pipeline — seniority/education scores, per-requirement scores, and the recommendation split — is invisible at the analytics level, even though it is fully computed and already rendered one applicant at a time in the Leaderboard detail card.

Separately, the Leaderboard detail card shows only the final, averaged `composite_fit_score` for an applicant. The production pipeline calls Jev 3 independent times per applicant and averages the result, but discards the 3 raw calls — there is currently no way to see how much (or little) those 3 calls agreed.

This design adds both: a per-role breakdown in Analytics, and a per-applicant multicall view in the Leaderboard detail card.

## 1. Per-role breakdown in Analytics

### UI

Each row in the existing "By role" list becomes clickable. Clicking toggles an expanded panel beneath that row, showing:

- **Seniority (years) & Education** — average score across the role's shortlisted applicants, as two `StatCard`s (same visual style as the summary cards above). A score is omitted from the average, and the whole stat hidden, if no applicant in the role has that field (mirrors the existing `QualificationScores` null-handling in Leaderboard).
- **Recommendation breakdown** — count and percentage of `hire` / `maybe` / `no` among the role's shortlisted applicants, as a 3-segment bar (reusing `RateBar`-style rendering) plus counts.
- **Per-requirement average** — for each technical requirement key appearing in the role's assessments, the mean score across applicants, rendered as a bar list matching the visual style of `RequirementScores` in Leaderboard.tsx, sorted descending (strongest requirement first, consistent with the existing per-applicant view, which sorts `([, a], [, b]) => b - a`).

Only one role expands at a time (accordion behavior), to keep the page from growing unbounded.

### Data

No backend change. All values needed (`requirement_scores`, `seniority_years_fit_score`, `education_fit_score`, `overall_recommendation`) are already present per applicant in `data.assessments` (`console-web/src/data/real-data.json`), reachable today via `assessmentFor(applicant.id)`.

Add a pure function, e.g. `roleBreakdownFor(role: Role): RoleBreakdown` (in `Comparison.tsx` or a small new `console-web/src/lib/roleBreakdown.ts`), that:
1. Filters `APPLICANTS` to `roleId === role.id`.
2. Maps each to its `Assessment` via `assessmentFor`.
3. Computes: mean seniority score (over non-null), mean education score (over non-null), recommendation counts, and a `Record<string, number>` of mean score per requirement key (unioning keys across the role's applicants; a requirement missing for a given applicant is excluded from that requirement's mean, not treated as 0).

This is computed on demand when a role is expanded (or memoized per role) — not exported from Python, since the source data already covers every applicant shown.

## 2. Per-applicant multicall view in Leaderboard

### Decision: use the existing repeatability-study data, not a fresh pipeline run

The production assessment pipeline (`generate_assessment`, `n_calls=3`) already makes 3 independent Jev calls per applicant, but only the aggregate `Assessment` is persisted — the 3 raw calls are discarded (`src/candidate_ranking/scoring/assessment.py:290-318`, `src/candidate_ranking/output/run_output.py`).

Re-deriving the *exact* 3 calls behind each displayed score would require changing `generate_assessment` to persist raw calls and re-running the full production pipeline (259 pairs × 3 calls). Per the user's decision, this design instead reuses the separate repeatability study already on disk: `runs/<run_id>/evaluation/test_retest.json`, produced by `scripts/run_jev_evaluation_study.py`, which independently collected 3 single-call repeats for every pair in the current production run (`20260921-085111`, 259 pairs — the same run `console_export.py` already exports from).

This is a different set of 3 calls than the ones behind the displayed `composite_fit_score` (Jev's answers vary slightly call to call), so the UI must not imply they are the same calls. It is presented explicitly as a repeatability sample.

### Data (Python)

In `src/candidate_ranking/output/console_export.py`:
- Add a loader that reads `run_dir / "evaluation" / "test_retest.json"` if present (mirrors the existing `_load_shortlisting_audit` pattern) and indexes its records by `(jd_id, candidate_id)`.
- For each exported assessment row (`row_id = f"{cv_id}::{jd_id}"`), if a matching test-retest record exists, attach a new top-level export field `repeatSamples[row_id]`: a list of up to 3 `{compositeFitScore, overallRecommendation, meetsMinQualifications}` objects, taken from that record's `repeats`.
- If `test_retest.json` is absent (e.g. a run where the evaluation study wasn't run), `repeatSamples` is `{}` — no error, feature just doesn't render.
- Re-export via the existing `export_console_web_data(cfg, run_id)` call — no new Jev calls, since `test_retest.json` already exists on disk for the current production run.

### Data (TypeScript)

In `console-web/src/data.ts`:
- New type `RepeatSample = { compositeFitScore: number; overallRecommendation: Assessment['overall_recommendation']; meetsMinQualifications: boolean }`.
- New export `REPEAT_SAMPLES: Record<string, RepeatSample[]>` from `data.repeatSamples ?? {}`.
- New helper `repeatSamplesFor(applicantRowId: string): RepeatSample[]` returning `REPEAT_SAMPLES[applicantRowId] ?? []`.

### UI

In `Leaderboard.tsx`'s `DetailPanel`, below the existing `ScoreSummary`/`QualificationScores`/`RequirementScores`, add a new section (rendered only if `repeatSamplesFor(...)` is non-empty):

- Heading: "Repeatability sample (3 independent calls)".
- One row per sample: composite fit score + recommendation badge (reuse the existing recommendation color/label mapping).
- A small muted caption: this is a separate reliability sample, not the exact calls behind the score above, and its average may differ slightly.

## Testing

Both features are pure frontend + a Python export-time join over existing files — no model/API calls involved. Verify:
- `console_export.py`: re-export against run `20260921-085111` and confirm `repeatSamples` is populated for all 259 rows and shaped as expected.
- `npx tsc -b --noEmit` in `console-web/` passes.
- Manual check in the running dev server: expand a role in Analytics and confirm the breakdown numbers are sane (e.g. spot-check one role's per-requirement average against the Leaderboard detail cards for a couple of its applicants); open an applicant detail card and confirm the 3-sample section appears and the caption is present.
