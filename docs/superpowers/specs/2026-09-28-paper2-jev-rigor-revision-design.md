# Paper 2 (Jev) Rigor Revision: Ten Claim/Evidence Tightening Points

## Purpose

`docs/paper2_jev.tex` currently has a large, uncommitted rewrite in the working tree (adding the "Reliability Tests" third contribution, a TikZ pipeline diagram, and a setup-comparison table). That rewrite is accepted as-is and is the base this design builds on. The user identified ten places where the paper's claims outrun, misstate, or under-support what the evidence actually shows. Every number this design adds is pulled from the real run-273 evaluation artifacts already on disk (`runs/20260924-203510/evaluation/*.json`, produced by `scripts/analyze_jev_evaluation_study.py`) or computed directly from that data — none are estimated or invented.

No code changes are involved. This is a `.tex`-only editing task; the underlying analysis script and run data are already correct and are read, not modified. Per standing project convention, no `pdflatex` compile or PDF commit — `.tex` only.

## A. Soften three overreaching claims (no new numbers)

1. **"Closest prior published system" (lines 117, 177)** — `\cite{ref2}` is the authors' own unpublished manuscript (its own `\bibitem{ref2}` entry already says so). Change both table captions from "closest prior published system" to "closest prior system", dropping "published".

2. **Faithfulness vs. calibrated-confidence juxtaposition (Table `tab_comparison_results`, line 187 row "Confidence signal")** — the two numbers (0.880 vs. 0.894) sit in the same table row inviting a head-to-head read, even though body text elsewhere already says they're not comparable. Add an explicit table footnote (a `\footnotesize` note below the table, tied to the row via a marker like `$^{\dagger}$` on "Confidence signal") stating the two metrics measure different constructs (groundedness vs. decision stability) and are not on a shared scale.

3. **"Hallucination risk removed by design" (abstract line 34: "...removing free-text generation and its hallucination risk by design"; Conclusion line 219: "hallucination risk removed by design, since typed answers replace free text with nothing left to hallucinate")** — both instances overclaim: eliminating free text removes *free-text* hallucination and schema violations, not the possibility of an incorrect structured judgment. Reword both to scope the claim, e.g.: "removing the specific risk of free-text hallucination and schema violation by design (an individual structured judgment can still be wrong)." Match this qualifier in both the abstract and conclusion instances so they stay consistent with each other.

## B. Clarify call-count, latency-accounting, and generalization scope

4. **273 vs. 819 calls (Table `tab_comparison_results` line 186 "Total calls (corpus)"; Section V intro line 174)** — 273 is the production pipeline's actual per-corpus cost (one call per shortlisted pair); 819 (273×3) only exists because the Reliability Tests subsection repeats each call three times to measure repeat-to-repeat stability. Add a footnote to the "Total calls (corpus)" table row clarifying this, and adjust the sentence at line 174 ("Across 819 timed Jev calls...") to state explicitly that the 819 figure is the reliability-test measurement cost, not the deployed pipeline's cost (already 273 elsewhere in the same table).

5. **14.2 minutes is accumulated, not wall-clock (line 174: "...for a total of 14.2 minutes of compute time across the full corpus -- call latency only...")** — this sums 819 sequential call latencies (819 × 1.04s mean ≈ 14.2 min); the Methodology section already states Jev calls parallelize with no iterative dependency, so real wall-clock time for the corpus would be far lower if run concurrently. Add a clause making this explicit, e.g.: "...summed sequentially across all 819 calls; because the pipeline issues these calls in parallel (Section III), actual wall-clock time for the corpus is far lower than this accumulated figure."

6. **4.5× speedup is single-profile (abstract line 34: "...and roughly 4.5$\times$ lower per-candidate latency than the prior system..."; Conclusion line 219: "...roughly 4.5$\times$ lower per-candidate wall-clock latency...")** — this number comes from one live head-to-head run on the frontend-engineer profile only (already scoped correctly in the Results section's own prose), but the abstract and conclusion state it without that qualifier, reading as a corpus-wide finding. Add "on a single live head-to-head profile comparison" (or equivalent) to both instances.

## C. Report real statistics currently missing or misleading

7. **Jev model version (line 146 "Assessment model & Jev (jev-latest), TypeSafe AI")** — confirmed via `runs/20260924-203510/manifest.json` (`"jev_model": "jev-latest"`) and `src/candidate_ranking/scoring/jev_client.py` (`JEV_MODEL_ID = "jev-latest"`, and the response parser never reads back a resolved model identifier) that no more specific pinned version was ever requested or recorded — "jev-latest" is genuinely the only identifier that exists for this run. Add one sentence to the Experimental Setup section (near Table `tab_config`) noting this as a reproducibility limitation: TypeSafe's API accepts only the `jev-latest` alias, no versioned snapshot identifier is returned by the API or logged by the pipeline, so exact model reproducibility cannot be guaranteed if TypeSafe updates the alias's underlying model.

8. **Wilcoxon effect size restored + new 95% CI (Figure `fig_score_ablation` caption/prose, lines 163–166)** — the ablation table in the pre-rewrite version of this file reported rank-biserial $r$ per group (0.89 / 0.77 / 0.47 / 0.58); this was dropped when the table became a figure. Restore $r$ in the figure caption or the prose at line 166, and add a 95% bootstrap CI (10,000 resamples, computed directly from the raw paired per-question values in `runs/20260924-203510/evaluation/ablation.json` and `seniority_education_ablation.json`) for the mean confidence difference per group:

   | Group | $n$ | Mean confidence gain | $r$ | 95% CI |
   |---|---|---|---|---|
   | Per-requirement | 772 | 0.223 | 0.89 | [0.204, 0.241] |
   | Seniority | 40 | 0.254 | 0.77 | [0.163, 0.336] |
   | Education | 116 | 0.090 | 0.47 | [0.041, 0.137] |
   | Pooled (sen.+edu.) | 156 | 0.132 | 0.58 | [0.088, 0.176] |

   State the CI method explicitly in prose ("95% bootstrap CI, 10,000 resamples, of the mean confidence difference") so a reader can judge it — do not present it as a Wilcoxon-native interval, since it isn't one.

9. **Tautological correlation demoted, ICC(1,1) promoted (Reliability Tests subsection, lines 201–216, specifically the "Score consistency" paragraph at line 206)** — the current headline claim (Spearman $\rho=0.972$ between mean per-requirement score and overall fit score) is partly definitional, since the overall score is literally the mean of those same per-requirement scores plus seniority/education. `runs/20260924-203510/evaluation/report.json`'s `score_decomposition_diagnostic` already contains a genuine test-retest reliability statistic — ICC(1,1), computed from the three independent repeats' composite fit scores per candidate, corpus mean **0.998** (range 0.995–1.000 across the 10 job profiles) — that does not share this problem. Restructure the subsection to lead with ICC(1,1) as the primary reliability statistic (new sentence citing 0.998, range 0.995–1.000), and keep the Spearman 0.972 correlation only as a secondary/supporting note with an explicit caveat that it is partly definitional by construction (mean-of-components vs. the score computed from those same components), not independent evidence of reliability.

## D. New content: ethics/fairness/human-oversight

10. **Short paragraph in Conclusion (after the limitations sentence, before "Future work..." at line 221)** — 3–5 sentences, not a separate subsection (page budget is tight per prior numbers-reduction pass). Must cover: this is a screening aid, not an automated hire/reject decision, and needs human-in-the-loop review before any adverse action; the evaluation corpus is mostly synthetic resumes (already disclosed in Experimental Setup), so bias behavior on real-world resume populations is untested; and calibrated confidence is a statement about decision stability, not a fairness or non-discrimination guarantee.

## Out of scope

- No changes to `src/`, `scripts/`, or `runs/` — all statistics are read from existing, already-correct analysis outputs.
- No re-running of `run_jev_evaluation_study.py` or `analyze_jev_evaluation_study.py` — the existing `20260924-203510` run already contains everything needed for points 7–9.
- No `pdflatex` compile, no `docs/paper2_jev.pdf` commit — `.tex` edits only, per standing project convention.
- Point 6 in the user's original list (model version) is a documented limitation, not a fix — there is no more specific version to substitute.

## Verification

- After editing, grep `docs/paper2_jev.tex` for the removed/changed strings to confirm no stale text remains: `"published system"`, `"hallucination risk removed by design"`.
- The four CI values and the ICC(1,1) figure (0.998, range 0.995–1.000) must be copy-checked against this design doc's Section C numbers (themselves computed from `runs/20260924-203510/evaluation/ablation.json`, `seniority_education_ablation.json`, and `report.json`'s `score_decomposition_diagnostic`) — not re-estimated or retyped from memory during implementation.
- Confirm the abstract and conclusion's two "hallucination risk" instances and two "4.5×" instances use matching, consistent qualifying language after editing (per points 3 and 6).
- Confirm Table `tab_comparison_results`'s "Total calls" and "Confidence signal" rows each carry their new footnote, and that the footnote markers render (no orphaned `$^{\dagger}$` without a corresponding note).
