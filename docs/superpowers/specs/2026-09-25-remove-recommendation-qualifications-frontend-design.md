# Remove Recommendation/Qualifications from Frontend (Sub-project 3/5)

## Purpose

Sub-projects 1/5 (core pipeline) and 2/5 (output layer) removed `overall_recommendation`, `meets_min_qualifications`, `certification_results`, and `recommendation_probabilities`, and stopped exporting `hireRate`, `meetsMinRate`, `overallRecommendation`, `meetsMinQualifications` in `console-web/src/data/real-data.json`. The console-web frontend still has TypeScript types and UI reading these fields; against fresh data they degrade to blank or misleading display (not crashes — sub-project 2's final review traced the full runtime chain and confirmed no hard crash path, only wrong/missing UI). This sub-project removes them from the frontend, per the design principle confirmed with the user: remove, don't replace with substitute metrics — the product intent is that HR only needs skill-fit scoring, not a UI padded out with other stats to fill the space.

## 1. `console-web/src/data.ts`

- `Assessment` interface: remove `overall_recommendation`, `meets_min_qualifications`.
- `RepeatSample` interface: remove `overallRecommendation`, `meetsMinQualifications` — leaves only `compositeFitScore`.
- `ComparisonRow` interface: remove `meetsMinRate`, `hireRate`.
- `EMPTY_ASSESSMENT`: remove the two corresponding default values.
- Tighten the type cast: `const data = realData as unknown as RealData;` → `const data = realData as RealData;`. Sub-project 2's final review flagged the `as unknown` double-cast as defeating structural type-checking, meaning `tsc` would not have caught this exact class of shape drift; dropping it makes `real-data.json`'s actual shape checked against the `RealData` interface going forward.

## 2. `console-web/src/lib/roleBreakdown.ts`

- `RoleBreakdown` interface: remove `recommendationCounts`.
- `roleBreakdownFor`: remove the `recommendationCounts` accumulation loop (the `for (const a of assessments) recommendationCounts[a.overall_recommendation] += 1;` block and its initialization).

## 3. `console-web/src/views/Leaderboard.tsx`

- `ScoreSummary`: remove the recommendation badge (the `<div>` rendering `RECOMMENDATION_LABEL[assessment.overall_recommendation]` with its accent/danger coloring) and the "✓/✕ Meets minimum qualifications" text block. What remains is the composite score display alone, unchanged.
- Remove the now-unused `RECOMMENDATION_LABEL` constant and the `recommendationColor`/`recommendationSoft` local variables in `ScoreSummary` that computed its styling.
- `RepeatabilitySample`: each sample row drops the recommendation column (the `<span>` rendering `RECOMMENDATION_LABEL[sample.overallRecommendation]` and its `color` computation) — each row shows only `Call N` and its composite score.

## 4. `console-web/src/views/Comparison.tsx`

- `metricFor`: drop `meetsMinRate`/`hireRate` from the returned object.
- Top stat-card row: remove the "Avg. meets-minimum rate" and "Avg. hire-recommendation rate" `StatCard`s and their `avgMeetsMinRate`/`avgHireRate`/`meetsMinRates`/`hireRates` computations — only "Avg. fit score across roles" remains.
- Per-role row: remove the two `RateBar` + percentage blocks for `meetsMinRate` and `hireRate` — the row keeps the fit-score text and the `τ` (ranking stability) block.
- `RoleBreakdownPanel`: remove the entire "Recommendation breakdown" section (the 3-segment hire/maybe/no bar and its count labels), and the now-unused `totalRecommendations` computation and `recommendationCounts` destructure.

## Out of scope

- No backend/Python files — sub-projects 1/2 already handled those.
- `scripts/run_jev_evaluation_study.py`'s `collect_test_retest` (still `AttributeError`s reading a live `Assessment.overall_recommendation` — sub-project 4/5).
- `docs/paper2_jev.tex` — sub-project 5/5.
- The committed `console-web/src/data/real-data.json` is not regenerated as part of this sub-project — it still has the old shape (from before sub-project 2) until someone re-runs the production pipeline. This sub-project's own manual dev-server verification should account for that (see Testing).

## Testing

This repo has no automated test runner for `console-web/` (no `*.test.tsx` files, no test script in `package.json`). Verification is:
- `npx tsc -b --noEmit` in `console-web/` passes with zero errors — meaningful now that `data.ts`'s cast no longer suppresses structural checking, so any remaining reference to a removed field anywhere in the frontend will surface as a compile error rather than a silent `undefined`.
- Manual check in the running dev server: open the Leaderboard for a role, select an applicant, confirm the detail panel shows the composite score with no recommendation badge or qualifications line, and (if repeatability samples exist for that applicant) confirm each sample row shows only its score. Open Analytics, confirm the top row shows a single fit-score card, confirm each role's row shows only fit score and τ, and expand a role's breakdown panel to confirm no "Recommendation breakdown" section appears.
- Since the committed `real-data.json` still has the old shape (see Out of scope), the dev server will initially still render the old UI elements (extra data present, but no longer read by the updated code, so this is safe to leave as-is for local verification) — for a true empty-state check (fields absent entirely), regenerate `real-data.json` via the production pipeline first, or verify against sub-project 2's own test fixtures / by temporarily removing the relevant keys from a local copy of the file.
