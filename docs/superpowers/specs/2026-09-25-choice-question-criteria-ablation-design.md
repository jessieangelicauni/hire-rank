# Extend Criteria-Grounded Design (C1) to Choice Questions

## Purpose

Paper 2's C1 contribution (criteria-grounded question design) is currently validated only for Score questions (per-requirement, seniority, education — `tab_score_ablation` in `docs/paper2_jev.tex`). The paper's own "Future work" names Choice questions as an untested extension: "TypeSafe's own guidance is comparatively undeveloped" there. This design extends the existing criteria-design ablation to the pipeline's one Choice question (`overall_recommendation`), at full corpus scale (273 pairs), and reports it as a new row in the same table — strengthening C1's generality claim rather than adding a new contribution.

No new experimental infrastructure is needed: `scripts/run_jev_evaluation_study.py`'s `collect_ablation` already collects a `vague_confidence` dict covering *every* field in each vague call, including `overall_recommendation` — it has simply never been analyzed or reported, because `_vague_build_questions`'s `overall_recommendation` criteria is currently a verbatim copy of the concrete version (included only as necessary call context, not as an ablation target).

## 1. Bare-label Choice criteria

In `scripts/run_jev_evaluation_study.py`, `_vague_build_questions` (the function building each ablation call's "vague" question set), change the `overall_recommendation` question's criteria from a copy of the concrete descriptions to bare labels — the Choice-question analog of the existing bare-number Score baseline (`_VAGUE_SCORE_CRITERIA = ["0", "25", "50", "75", "100"]`):

```python
criteria={"hire": "Hire", "maybe": "Maybe", "no": "No"}
```

This is safe to change in place: Jev answers every question in a call independently against the same state (the paper's own Section II claim), so this change cannot affect the confidence of the per-requirement Score questions asked in the same vague call — it only changes what `overall_recommendation`'s own vague-side confidence measures.

## 2. Checkpointing for `collect_ablation`

`collect_ablation` currently writes its result list only after every sample pair completes (`main()` writes `ablation.json` once, at the end) — identical to the pattern that caused wasted Jev spend earlier in this project's history when a similar uncached, non-resumable loop (`validate_multi_call_averaging.py`, since deleted) was interrupted mid-run. Running this ablation at full corpus scale (273 pairs, versus the previous default of 40) meaningfully raises the chance of hitting exactly that failure mode.

Add the same lightweight resumable-checkpoint pattern used elsewhere in this project's evaluation scripts: a JSONL file, one line appended per completed pair (under a lock, since `collect_ablation` runs pairs concurrently via `ThreadPoolExecutor`), with already-checkpointed pairs skipped on restart.

- `collect_ablation` gains a required `checkpoint_path: Path` parameter.
- Before dispatching work: read any existing checkpoint file, build a `{(jd_id, candidate_id): record}` map of already-completed pairs, and only submit the remaining pairs to the executor.
- Inside `run_one`, after computing a pair's result, append it as one JSON line to `checkpoint_path` (lock-protected) before returning it.
- The function's return value is unchanged in shape: the full list of per-pair result dicts, in `sample_pairs` order, assembled from the completed map (pre-existing + newly computed).
- `main()` passes `out_dir / "ablation.checkpoint.jsonl"` as the checkpoint path (same `out_dir = run_dir / "evaluation"` directory already used for `ablation.json` and the other evaluation outputs). The checkpoint file is left on disk after a successful run (harmless, and lets a future re-run of the same corpus skip already-collected pairs for free).

## 3. Running the study — no new CLI flags needed

Both flags this study needs already exist:

```bash
uv run python scripts/run_jev_evaluation_study.py \
  --run-id <RUN_ID> --technical-ablation-only \
  --ablation-sample-size 273 --vague-n-calls 1
```

- `--ablation-sample-size 273` (existing flag, default 40): `main()` already does `ablation_size = min(ablation_sample_size, len(pairs))`, so this naturally becomes "all shortlisted pairs" without any code change.
- `--vague-n-calls 1` (existing flag, default 3): keeps the vague/bare side at exactly one call per pair, matching the concrete side's one sample (`cached["confidence"]`, itself a single production call now that the multi-call ensembling removal earlier in this project made production single-call). Before that removal, both sides were symmetric at 3 calls each; leaving `--vague-n-calls` at its old default of 3 against a now-single-call concrete baseline would silently reintroduce that asymmetry for *any* future ablation run (Score or Choice), not just this one — passing `1` here keeps both sides at one sample, consistent with the pipeline's current single-call reality. This changes nothing about the function's own default (still 3), only how this particular run invokes it.
- `--technical-ablation-only` (existing flag): skips the unrelated `collect_test_retest` step (repeatability/latency data, not part of this study).

Total new Jev spend: 273 calls (one per shortlisted pair, one call each) — comparable to one additional full-corpus production pass, not a multiplied-out re-run.

## 4. Analysis: report the new comparison

In `scripts/analyze_jev_evaluation_study.py`, alongside the existing per-requirement/seniority/education Wilcoxon signed-rank comparisons computed from `ablation.json`, add the same computation for the `overall_recommendation` key: paired concrete-vs-bare confidence per pair, Wilcoxon signed-rank test, mean confidence under each condition, effect size — identical statistical treatment to what's already done for the Score fields, just keyed on `overall_recommendation` instead of `requirement::*`/`seniority_years`/`education`. Reported in `report.json`/`report.md` as a new named entry alongside the existing ones.

## 5. Paper changes (`docs/paper2_jev.tex`)

- **Table `tab_score_ablation`** (Section V-A): add one row, `Recommendation (Choice) & 273 & <bare conf> / <concrete conf> & <p> (<r>)`, following the existing row format (`Question group & $n$ & Conf. (concrete/ordinal 0--100) & $p$ ($r$)`) — column header may need "ordinal 0--100" generalized to "bare" since this row's baseline isn't a number, e.g. `Conf. (concrete/bare)`.
- **Section V-A prose**: one or two added sentences noting criteria-grounded design is now validated for the Choice primitive too, not only Score — generalizing C1's claim across both primitives Jev exposes for a graded/categorical judgment (Noul remains untested, matching the paper's own limitation).
- **Conclusion / Future work**: remove "extending the same criteria-grounded principle to Choice questions, where TypeSafe's own guidance is comparatively undeveloped" from the future-work list (now done); Noul remains the one untested primitive, worth a brief mention if the future-work sentence needs rebalancing after the removal.
- No change to contribution numbering (C1/C2 stay as-is) — this is explicitly an extension of C1, not a new contribution, per the earlier decision in this project.

## Out of scope

- `collect_seniority_education_ablation`'s own separate `vague_n_calls` parameter and call sites — untouched by this design; if it's ever re-run at scale it would face the same symmetry question, but that's a separate decision for whoever runs it next.
- `collect_test_retest` / `--repeats` — an unrelated study (repeatability/latency), not touched here.
- Noul-question criteria-grounded validation — explicitly left as the paper's one remaining untested primitive; not part of this design.

## Testing

- `_vague_build_questions`: a test asserting the `overall_recommendation` question's `criteria` is now `{"hire": "Hire", "maybe": "Maybe", "no": "No"}`, not equal to the concrete version's criteria (regression guard against the two silently drifting back into sync, which would silently break the ablation's validity).
- `collect_ablation` checkpointing: a test with a fake `JevClient` verifying (a) a fresh run writes one checkpoint line per pair and returns all pairs' results in the original `sample_pairs` order; (b) a pre-populated checkpoint file causes those pairs to be skipped (`JevClient.evaluate` not called for them) while still being present in the returned results, sourced from the checkpoint.
- New analysis function in `analyze_jev_evaluation_study.py`: a test with synthetic paired concrete/bare confidence records asserting the Wilcoxon p-value and effect size are computed correctly for the `overall_recommendation` key, mirroring whatever test coverage already exists for the per-requirement/seniority/education computations.
- Manual verification: run the command in Section 3 against an existing run, confirm `report.json`/`report.md` contains a `overall_recommendation` entry with `n=273`, and confirm interrupting and resuming the run (e.g. Ctrl-C partway through, or simulated by pre-seeding a partial checkpoint file) does not re-call Jev for already-checkpointed pairs.
