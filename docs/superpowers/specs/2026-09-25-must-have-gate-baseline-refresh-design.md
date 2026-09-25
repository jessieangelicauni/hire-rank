# Must-Have-Gate Baseline Number Refresh

## Purpose

`docs/paper2_jev.tex`'s "Must-Have Gate: Diagnostic Validation" subsection (Section V-B) still reports "excludes 44 of 303 candidates (14.5%)" — computed against an old 303-pair corpus, before the paper's numbers were refreshed to the current 273-pair run (`20260924-203510`). This was blocked on TypeSafe billing (402 Payment Required); billing is now resolved. This design refreshes that number against the current run, and separately strengthens the subsection's currently-vague claim ("validated using C1's per-requirement granularity") with a concrete figure from the diagnostic script that backs it.

No code changes are involved — this is a data-refresh task using two scripts that already exist and already do exactly what's needed.

## 1. Run the pipeline with the must-have gate disabled

```bash
uv run python -m candidate_ranking.cli run --min-must-have-matches 0
```

No `--run-id` is passed, so the CLI generates a fresh timestamped run directory (a different shortlist configuration cannot share a run id with the existing gated run). This automatically reuses `runs/_cache/` — both the LLM skill-extraction cache (`cv_skills.json`, `jd_skills.json`) and the Jev assessment cache (`runs/_cache/assessments/<jd_id>/<candidate_id>.json`, keyed independently of `run_id` or shortlist config) — so candidates already assessed under the current 273-pair gated run are not re-assessed. Only candidates newly admitted by dropping the must-have requirement get a fresh (single, per the current production pipeline) Jev call.

## 2. Compute the exclusion count

Sum `shortlist_sizes` (per-job-profile counts already present in `manifest.json`, as in the existing `runs/20260924-203510/manifest.json`) for the new gate-disabled run, and compare against the same sum for `runs/20260924-203510/manifest.json` (273, the current with-gate total). The difference and percentage become the refreshed "excludes N of \<baseline-total\> (X%)" figure.

## 3. Run the shortlisting-quality diagnostic on the new baseline

```bash
uv run python scripts/audit_shortlisting_quality.py --run-id <new-baseline-run-id>
```

Default flags (`--min-pool-median 30`, `--near-zero-threshold 15`, `--flag-fraction 0.7`, `--profile-flag-fraction 0.2`) are used as-is — this script and its defaults already exist and are unrelated to the must-have gate itself (it works from whatever `assessments.json` a run directory has). No new Jev calls: it reads the assessments the pipeline run in Step 1 already produced. Output: `runs/<new-baseline-run-id>/evaluation/shortlisting_audit.{json,md}`.

## 4. Update `docs/paper2_jev.tex` — Section V-B ("Must-Have Gate: Diagnostic Validation")

Two changes to this subsection, using the real numbers produced by Steps 2 and 3 (not estimated or placeholder values — the implementation step must read them from the actual script output before editing the paper):

- Replace "excludes 44 of 303 candidates (14.5%)" with the Step 2 result, phrased identically ("excludes N of \<baseline-total\> candidates (X%)").
- Replace the current vague closing clause "This gate is a pipeline design validated using C1's per-requirement granularity (Section III-F), not one of this paper's two architecture-level contributions." with a version citing the Step 3 result concretely — e.g. (exact wording to be finalized against the real numbers): "...validated by the diagnostic of Section III-F: Y% of the candidates admitted only by dropping the must-have requirement are flagged as scoring near-zero on their pool's informative requirements, not one of this paper's two architecture-level contributions."
- "39% were classified must-have" is untouched — that figure comes from Stage 1 skill classification, independent of the shortlisting threshold, and already reflects the current 273-pair run.

## Out of scope

- No changes to `src/`, `scripts/`, or any test file — both scripts used here are already correct and unmodified.
- The ranking-convergence pending item from the same prior blocked-on-billing note is not part of this design — that claim was removed from the paper entirely earlier in this project (a separate, already-completed decision), so there is nothing left to refresh for it.

## Verification

- The exclusion count and percentage written into the paper must be copy-checked against the actual sums from both runs' `manifest.json` files, not hand-computed or estimated.
- The diagnostic percentage written into the paper must be copied from the real `shortlisting_audit.json` output for the new baseline run, not estimated.
- After editing, grep `docs/paper2_jev.tex` for the old "44 of 303" and "14.5" strings to confirm no stale reference remains.
