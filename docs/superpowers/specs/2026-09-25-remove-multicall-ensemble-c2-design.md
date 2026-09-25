# Remove Multi-Call Ensemble Averaging (C2)

## Purpose

The production pipeline calls Jev 3 independent times per (job description, candidate) pair by default (`DEFAULT_N_CALLS = 3` in `src/candidate_ranking/scoring/assessment.py`) and averages the results. Combined with a separate research script (`scripts/validate_multi_call_averaging.py`) that runs 2 additional uncached trial batches of 3 repeats each per pair to validate this averaging, Jev token spend has been running high enough to trip TypeSafe billing limits (402 errors) during iteration.

Multi-call ensemble averaging is also the paper's second contribution (C2), documented in `docs/paper2_jev.tex` (abstract, Section III-E, comparison table, conclusion). The decision made in this design is to retire C2 entirely — both the production mechanism and the paper's claim about it — rather than optimize its cost. The paper keeps two contributions: C1 (criteria-grounded question design) and cost/latency efficiency (renumbered to C2).

This does not require any new Jev calls or re-running any experiment: the single-call convergence number the paper needs (0.964, vs. the prior tournament system's 0.957) is already stated in the current abstract text, computed from data already on disk.

## 1. Code: `src/candidate_ranking/scoring/assessment.py`

Remove the n_calls/ensembling mechanism entirely (not just default it to 1):

- `generate_assessment()` calls `_generate_single_assessment()` once and returns its `Assessment` directly. No `n_calls` parameter.
- Delete now-unreachable aggregation helpers: `_aggregate_recommendation`, `_aggregate_meets_min_qualifications`, `_aggregate_mean_dict`, `_aggregate_bool_dict`, `_aggregate_mean_optional`, `_aggregate_mean_nested_dict`, `_aggregate_mean_optional_dict`, and the `DEFAULT_N_CALLS` constant.
- `_assessment_cache_key()` and `load_or_generate_assessment()` drop the `n_calls` parameter (it no longer varies).
- Retry-on-low-confidence in `_generate_single_assessment` (the 2-attempt loop over `_RETRY_ON_LOW_CONFIDENCE_KEYS`) is unrelated to ensembling and stays as-is.
- `src/candidate_ranking/graphs/pipeline.py`'s call to `load_or_generate_assessment` drops accordingly (it doesn't pass `n_calls` explicitly today, so no change needed there beyond the signature update propagating).

## 2. Scripts

Delete:
- `scripts/validate_multi_call_averaging.py` — its sole purpose is computing $\tau_3$ (3-call average convergence) to compare against $\tau_1$; with no C2 claim, there is nothing left to validate.
- `scripts/generate_convergence_figure.py` — plots $\tau_1$ vs. $\tau_3$ from the script above's output; orphaned once that script is gone.

Keep unchanged:
- `collect_test_retest` in `scripts/run_jev_evaluation_study.py` — produces the repeatability data and $\tau_1$ (single-call convergence) used by C1/cost-efficiency evaluation. This is diagnostic reliability characterization, independent of whether production ensembles calls.
- `collect_ablation` in the same file (`n_calls=3` default) — a separate ablation experiment (vague vs. concrete question phrasing) supporting C1, not part of the C2 claim. Out of scope for this change.

## 3. Paper: `docs/paper2_jev.tex`

All edits are text-only (per standing instruction: never compile or commit a PDF here).

- **Abstract**: remove "multi-call ensemble averaging" from the three-mechanism list (now two); replace "that three-call ensemble averaging, run independently and in parallel, recovers ranking convergence from 0.964 to 0.982 -- exceeding the prior tournament-based system's 0.957" with a direct single-call claim: convergence of 0.964, exceeding the prior tournament system's 0.957.
- **Contribution list / intro**: remove the "Multi-Call Ensemble Averaging" bullet; renumber the remaining "C3" (cost/latency efficiency) to "C2" everywhere it's cited (inline citations, section headers, stage diagram labels).
- **Pipeline stage diagram**: remove the "Stage 4: Multi-Call Ensemble Aggregation [C2: Ensemble Averaging]" box; renumber subsequent stages.
- **Section III-E "Multi-Call Ensemble Averaging (C2)"**: delete the subsection, its Fig. `fig_convergence` reference, and caption.
- **Methodology paragraph** (the "three evaluation procedures" sentence): trim "ranking convergence (Kendall-$\tau$ ... from single calls and three-call averages)" to single-call only.
- **Must-have-gate paragraph**: verify it already speaks only of single-call convergence; fix if any stray three-call reference remains.
- **Comparison table** (`tab_comparison`): change the Convergence row from "0.957 & 0.982 (3-call average)" to "0.957 & 0.964".
- **Results/comparison prose**: replace "three-call-averaged convergence now exceeds the prior tournament-based system's" with single-call framing; the "well under half as many model calls" claim gets stronger (single call, not three), so it can stay or be strengthened.
- **Conclusion**: remove "multi-call ensemble averaging" from the contribution summary; change "(0.982 vs. 0.957)" to "(0.964 vs. 0.957)".
- **Future work**: check for any remaining C2/ensembling reference and remove.

A final grep pass over the whole file for `multi.call|tau_3|3-call|three-call|ensemble` confirms nothing is left dangling.

## Out of scope

- `console-web/src/data/real-data.json` and the "Repeatability sample" UI feature (`Leaderboard.tsx`) are unaffected — they read from `test_retest.json` via `collect_test_retest`, which is unchanged, and their label is a dynamic call count, not a hardcoded reference to production ensembling. No action needed.
- `collect_ablation`'s own `n_calls=3` default (separate C1-supporting experiment).

## Testing

- `tests/scoring/test_assessment.py` / `tests/scoring/test_jev_client.py`: remove tests exercising aggregation across multiple calls; add/adjust a test asserting `generate_assessment` invokes `JevClient.evaluate` exactly once (absent low-confidence retries) per call to `_generate_single_assessment`.
- Run the full existing test suite for `candidate_ranking.scoring` to confirm nothing else depends on the removed functions/parameters.
- Grep `docs/paper2_jev.tex` for `multi.call|tau_3|3-call|three-call|ensemble|Section III-E` after editing to confirm no dangling references.
- No live Jev calls or PDF compilation as part of verifying this change.
