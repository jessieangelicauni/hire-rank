# Remove Recommendation/Qualifications from Evaluation Scripts (Sub-project 4/5)

## Purpose

Sub-project 1/5 made the production `Assessment` model Score-only (no `overall_recommendation`, `meets_min_qualifications`, `certification_results`, `recommendation_probabilities`). `scripts/run_jev_evaluation_study.py`'s `collect_test_retest` still reads `assessment.overall_recommendation`/`assessment.meets_min_qualifications` from a live `Assessment` object it builds via the real (now Score-only) `_build_questions`/`_answers_to_assessment` — this `AttributeError`s today. Beyond that direct crash, exploring the surrounding code surfaced two further consequences of the same upstream change, both requiring an explicit decision (made with the user during design, both resolved as "remove"):

1. `collect_ablation`'s "unaffected control" mechanism (asking `overall_recommendation`/`meets_min_qualifications` alongside the deliberately-varied Score questions, to prove the ablation's measured confidence effect is specific to Score criteria and not a general artifact) silently stops working — not a crash, but a methodologically dead check, since the "concrete" side of that comparison (`assessments.json`'s `confidence` dict, from production) no longer has those two keys at all. Decision: remove the mechanism, matching how `collect_seniority_education_ablation` already does direct concrete-vs-vague comparison with no control field — this makes all three ablations in this file follow the same pattern instead of one being an inconsistent outlier.
2. `analyze_test_retest`'s `recommendation_full_agreement_rate` (computed from the same now-removed per-repeat field) flows through `report.json` → `console_export.py`'s `_evaluation_summary` → `data.ts`'s `EvaluationSummary` → a "Recommendation agreement (repeat calls)" `StatCard` in `Comparison.tsx`'s Analytics page (sub-projects 2/5 and 3/5, already merged). Left alone, that card would render a permanent `—` referencing a retired concept. Decision: remove it end-to-end, touching the already-merged sub-project 2/3 files as a direct, necessary consequence of completing the removal.

## 1. `scripts/run_jev_evaluation_study.py`

- `_vague_build_questions`: remove the `overall_recommendation` (Choice) and `meets_min_qualifications` (Noul) `JevQuestion` entries from the initial list — the function returns only the per-technical-skill Score questions with `_VAGUE_SCORE_CRITERIA`, matching what production's `_build_questions` now does (Score-only). This also means the ablation calls stop paying for two now-pointless questions.
- `collect_test_retest`: remove the `"overall_recommendation": assessment.overall_recommendation,` and `"meets_min_qualifications": assessment.meets_min_qualifications,` lines from the `repeats_out.append({...})` dict — this is the fix for the actual crash. Each repeat's dict becomes `{"composite_fit_score", "requirement_scores", "latency_seconds"}`.

## 2. `scripts/analyze_jev_evaluation_study.py`

- Remove the `_UNAFFECTED_CONTROL_KEYS = ("overall_recommendation", "meets_min_qualifications")` constant.
- `analyze_ablation`: remove the `concrete_control`/`vague_control` accumulator lists, the `for key in _UNAFFECTED_CONTROL_KEYS: ...` loop that populated them, and the `"unaffected_control": _paired_bucket(concrete_control, vague_control, "unaffected_control_confidence")` entry from the returned dict. The function keeps computing and returning `requirement_questions` (the actual ablation result) and `vague_latency_seconds_mean` unchanged. `_paired_bucket` itself is unchanged (still used by `requirement_questions` and by `analyze_seniority_education_ablation`'s per-field buckets).
- `render_markdown`: remove the `if abl["unaffected_control"]["wilcoxon"]:` block (the "unaffected_control confidence" printed line) from the ablation section — the `requirement_questions` line stays.
- `analyze_test_retest`: remove the `recommendations = {r["overall_recommendation"] for r in repeats}` / `if len(recommendations) == 1: recommendation_full_agreement += 1` logic and the `recommendation_full_agreement = 0` initializer, and remove `"recommendation_full_agreement_rate": recommendation_full_agreement / n if n else None,` from the returned dict.
- `render_markdown`: remove the `f"- recommendation full agreement rate: {tr['recommendation_full_agreement_rate']:.1%}",` line from the test-retest section.

## 3. `src/candidate_ranking/output/console_export.py`

- `_evaluation_summary`: remove `"recommendationAgreementRate": test_retest.get("recommendation_full_agreement_rate"),` from the returned dict.

## 4. `console-web/src/data.ts`

- `EvaluationSummary` interface: remove `recommendationAgreementRate: number | null;`.

## 5. `console-web/src/views/Comparison.tsx`

- Remove the `"Recommendation agreement (repeat calls)"` `StatCard` from the "Reliability & validation" section (inside the `{EVALUATION_SUMMARY && (...)}` block) — the other three cards (mean ranking stability, composite-score stdev, requirement/composite coherence) stay, per the same "remove, don't replace" rule already applied in sub-project 3/5.

## Out of scope

- `docs/paper2_jev.tex` — sub-project 5/5. A grep confirmed the paper currently has no reference to "unaffected control" or "recommendation agreement/full agreement," so sub-project 5 has nothing to clean up from this specific change.
- `collect_seniority_education_ablation` and `analyze_seniority_education_ablation` — untouched; they never had an unaffected-control mechanism, and this design brings the technical-skill ablation in line with their existing pattern rather than changing them.
- No changes to `src/candidate_ranking/scoring/assessment.py` or `models.py` — sub-project 1 already finished that.

## Testing

- No test file currently covers `run_jev_evaluation_study.py` or `analyze_jev_evaluation_study.py` directly (confirmed: `tests/test_scripts_import.py` only import-smoke-tests every file under `scripts/`, added earlier this session specifically to catch import-time breakage like this). That test already exercises this exact class of bug (an import-time `AttributeError`/`ImportError` from a stale symbol) but `collect_test_retest`'s bug is inside a function body, only triggered when called — the import-smoke test alone would not have caught it. No new dedicated test file is added for these scripts as part of this design (matching the existing project convention that these evaluation scripts are exercised manually against a real run, not via `pytest`) — verification is a `python -c` sanity check plus reading the edited functions for correctness, not new automated tests.
- `tests/output/test_console_export.py`: update the fixture/assertions for `_evaluation_summary` (the `test_export_console_web_data_includes_evaluation_summary_when_report_exists` test's `report` fixture currently includes `"recommendation_full_agreement_rate": 0.935` and asserts `summary["recommendationAgreementRate"] == 0.935` — both need removing to match the new shape).
- Manual verification: `npx tsc -b --noEmit` in `console-web/` passes clean (as it did at the end of sub-project 3/5) after removing `recommendationAgreementRate` from `data.ts` and its usage in `Comparison.tsx`.
