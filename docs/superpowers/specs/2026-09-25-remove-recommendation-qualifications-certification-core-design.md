# Remove Recommendation/Qualifications/Certification from Core Pipeline (Sub-project 1/5)

## Purpose

Per user decision, the pipeline no longer asks Jev for a hiring recommendation (`overall_recommendation`, Choice), a minimum-qualifications judgment (`meets_min_qualifications`, Noul), or certification checks (`certification::*`, Noul). HR only needs skill-fit scoring. Ranking already sorts purely on `composite_fit_score`, which is computed only from Score-type fields (technical requirements, seniority, education) — removing these three fields does not change ranking behavior, only the assessment output's shape.

This is sub-project 1 of 5 in a larger removal (output layer, frontend, evaluation scripts, and the paper follow in later specs). This sub-project covers only the core assessment model and generation logic.

## 1. `src/candidate_ranking/models.py` — `Assessment`

Remove these fields: `overall_recommendation`, `meets_min_qualifications`, `certification_results`, `recommendation_probabilities`.

Remaining fields (unchanged): `job_description_id`, `candidate_id`, `generated_by_model`, `requirement_scores`, `confidence`, `seniority_years_fit_score`, `education_fit_score`, `requirement_probabilities`, `seniority_probabilities`, `education_probabilities`, and the computed `composite_fit_score` property (its implementation does not reference any of the removed fields and needs no change).

## 2. `src/candidate_ranking/scoring/assessment.py`

- `_build_questions`: remove the `overall_recommendation` and `meets_min_qualifications` `JevQuestion` entries from the initial list, and remove the `for certification in jd_skills.certifications ...` loop entirely. The function returns only per-requirement Score questions, plus seniority/education Score questions when applicable (unchanged).
- `_answers_to_assessment`: remove the `certification_results` dict comprehension, and remove the four removed-field assignments (`overall_recommendation=...`, `meets_min_qualifications=...`, `certification_results=...`, `recommendation_probabilities=...`) from the returned `Assessment(...)` call.
- `_RETRY_ON_LOW_CONFIDENCE_KEYS`: becomes `(_SENIORITY_YEARS_KEY, _EDUCATION_KEY)`.
- Remove the now-unused module-level names: `_RECOMMENDATION_KEY`, `_MIN_QUALIFICATIONS_KEY`, `_CERTIFICATION_KEY_PREFIX`, `_certification_question_key`.
- `ASSESSMENT_SCOPE_VERSION`: change from `"jev-score-recommendation-v1"` to `"jev-score-only-v1"` — this string is written into every run's `manifest.json` (`assessment_scope` field, via `cli.py`) and its old name explicitly claims "recommendation" is part of the scope, which is no longer true.

## 3. Out of scope (explicit boundaries)

- `src/candidate_ranking/scoring/jev_client.py` is untouched. `JevQuestion`/`JevAnswer` stay fully generic (still support `noul`/`choice`/`score` kinds) — this is a general Jev API client, not pipeline-specific logic; only the pipeline stops constructing noul/choice questions.
- `JDSkills.certifications` (the extracted list of a JD's required certifications, populated during Stage 1 skill extraction) is untouched. It becomes unused by `_build_questions` after this change, but removing it would mean touching the skill-extraction stage, a separate subsystem not covered by this decision. It stays extracted but unconsumed by assessment.
- `src/candidate_ranking/graphs/pipeline.py`: no change expected — it calls `load_or_generate_assessment(jd, candidate, jev_client, JEV_MODEL_NAME, cfg.cache_dir, jd_skills=jd_skills)`, a signature untouched by this sub-project.
- Output layer (`formatter.py`, `console_export.py`), frontend (`data.ts`, `Leaderboard.tsx`, `roleBreakdown.ts`), evaluation scripts (`run_jev_evaluation_study.py`, `analyze_jev_evaluation_study.py`), and the paper are each their own follow-up sub-project spec — none of those files are touched here. (Note: those files currently reference the fields removed here, e.g. `assessment.overall_recommendation` — they will break until their own sub-project lands. This is expected and sequenced deliberately; do not "fix" them opportunistically in this sub-project.)

## Testing

Update `tests/scoring/test_assessment.py`:
- `_high_confidence_answers()` and any other test fixture that builds a list of `JevAnswer` for `overall_recommendation`/`meets_min_qualifications` no longer needs those entries (the pipeline no longer asks for them, so a mock `JevClient` need not return them).
- Remove tests that specifically exercise the removed fields (e.g. certification results, recommendation retry behavior, recommendation probability preservation for the `overall_recommendation` key).
- Remaining tests (per-requirement scoring, seniority/education scoring, low-confidence retry on seniority/education, requirement/seniority/education probability preservation, cache behavior) keep working with `JevAnswer` lists containing only `requirement::*`/`seniority_years`/`education` score-kind entries.
- Add a regression test asserting `Assessment` has no `overall_recommendation`, `meets_min_qualifications`, `certification_results`, or `recommendation_probabilities` field (e.g. via `Assessment.model_fields`), so a future change can't silently reintroduce them without deliberate review.
