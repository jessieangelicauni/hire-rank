# Paper 2 (Jev) Rigor Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the ten claim/evidence-tightening edits from `docs/superpowers/specs/2026-09-28-paper2-jev-rigor-revision-design.md` to `docs/paper2_jev.tex`, replacing overreaching or ambiguous claims with precisely scoped ones backed by real numbers already computed from `runs/20260924-203510/evaluation/*.json`.

**Architecture:** This is a `.tex`-only prose-editing task — one file, no code, no build step. Each task is a self-contained edit to one paragraph, caption, or table in `docs/paper2_jev.tex`, ordered top-to-bottom through the document so no task's line range overlaps another's. There is no automated test suite for LaTeX prose; "verification" for each task is (a) a `grep` check that the old text is gone, and (b) a re-read of the edited paragraph to confirm it reads correctly and stays internally consistent with the rest of the paper.

**Tech Stack:** LaTeX (IEEEtran document class). No compilation step in this plan — per standing project convention, never run `pdflatex` or commit a `.pdf`; the user compiles `docs/paper2_jev.tex` themselves.

## Global Constraints

- Edit `docs/paper2_jev.tex` only. No changes to `src/`, `scripts/`, `runs/`, or any other file.
- Never run `pdflatex`; never create or modify `docs/paper2_jev.pdf`.
- Every number introduced must match exactly what is recorded in the design spec (Section C), itself sourced from `runs/20260924-203510/evaluation/ablation.json`, `seniority_education_ablation.json`, and `report.json`. Do not re-derive, round differently, or retype from memory.
- All edits use the `Edit` tool with exact `old_string`/`new_string` matches against the current file content — do not use line-number-based patching, since line numbers shift after each edit.
- Commit after each task with a `docs:` prefixed message, following this repo's existing convention (see `git log -- docs/paper2_jev.tex`).

---

### Task 1: Abstract — scope the hallucination-risk claim and the 4.5× speedup

**Files:**
- Modify: `docs/paper2_jev.tex` (abstract paragraph, currently line 34)

**Interfaces:** None (prose-only; no cross-task dependencies).

- [ ] **Step 1: Narrow "hallucination risk...by design"**

Edit `docs/paper2_jev.tex`:

old_string:
```
reports a calibrated confidence per answer, removing free-text generation and its hallucination risk by design.
```

new_string:
```
reports a calibrated confidence per answer, removing the specific risk of free-text hallucination and schema violation by design (an individual structured judgment can still be wrong).
```

- [ ] **Step 2: Scope the 4.5× latency claim to the single-profile comparison it comes from**

Edit `docs/paper2_jev.tex`:

old_string:
```
and roughly 4.5$\times$ lower per-candidate latency than the prior system, with output priced at zero by design
```

new_string:
```
and, on a single live head-to-head profile comparison, roughly 4.5$\times$ lower per-candidate latency than the prior system, with output priced at zero by design
```

- [ ] **Step 3: Verify**

Run: `grep -n "hallucination risk by design" docs/paper2_jev.tex`
Expected: no output (old abstract phrasing is gone).

Run: `grep -n "single live head-to-head profile comparison" docs/paper2_jev.tex`
Expected: one match, in the abstract.

Read the full abstract paragraph once to confirm it still reads as one coherent sentence (no dangling clauses from the edit).

- [ ] **Step 4: Commit**

```bash
git add docs/paper2_jev.tex
git commit -m "docs: scope abstract's hallucination-risk and 4.5x claims precisely"
```

---

### Task 2: Experimental Setup table — drop "published" from the prior-system caption

**Files:**
- Modify: `docs/paper2_jev.tex` (Table `tab_comparison_setup` caption, currently line 117)

**Interfaces:** None.

- [ ] **Step 1: Edit the caption**

old_string:
```
\caption{Experimental setup compared against the closest prior published system \cite{ref2}.}
```

new_string:
```
\caption{Experimental setup compared against the closest prior system \cite{ref2}.}
```

- [ ] **Step 2: Verify**

Run: `grep -n "published system" docs/paper2_jev.tex`
Expected: one remaining match (Table `tab_comparison_results`'s caption — fixed in Task 6), not this one.

- [ ] **Step 3: Commit**

```bash
git add docs/paper2_jev.tex
git commit -m "docs: fix 'closest prior published system' wording (setup table)"
```

---

### Task 3: Experimental Setup — add the Jev model-version reproducibility limitation

**Files:**
- Modify: `docs/paper2_jev.tex` (sentence following Table `tab_config`, currently line 155)

**Interfaces:** None.

- [ ] **Step 1: Append the limitation sentence**

old_string:
```
Table~\ref{tab_config} shows the full configuration: which models perform assessment and extraction, the corpus and shortlist sizes, the thresholds that determine who is shortlisted, and the five-level Score design used in every Jev call -- the same values used throughout both the direct-sort pipeline and the tournament baseline it is compared against.
```

new_string:
```
Table~\ref{tab_config} shows the full configuration: which models perform assessment and extraction, the corpus and shortlist sizes, the thresholds that determine who is shortlisted, and the five-level Score design used in every Jev call -- the same values used throughout both the direct-sort pipeline and the tournament baseline it is compared against. TypeSafe's API accepts only a \texttt{jev-latest} alias; it returns no versioned snapshot identifier in its response, and the pipeline does not log one, so exact model reproducibility cannot be guaranteed if TypeSafe updates the alias's underlying model.
```

- [ ] **Step 2: Verify**

Run: `grep -n "versioned snapshot identifier" docs/paper2_jev.tex`
Expected: one match.

- [ ] **Step 3: Commit**

```bash
git add docs/paper2_jev.tex
git commit -m "docs: disclose jev-latest model-version reproducibility limitation"
```

---

### Task 4: Criteria-Grounded ablation — restore effect size and add 95% CI

**Files:**
- Modify: `docs/paper2_jev.tex` (prose following Figure `fig_score_ablation`, currently line 166)

**Interfaces:** None. Numbers below are taken verbatim from the design spec (Section C, point 8), themselves computed from `runs/20260924-203510/evaluation/ablation.json` and `seniority_education_ablation.json` — do not recompute or re-round.

- [ ] **Step 1: Replace the ablation results sentence**

old_string:
```
Figure~\ref{fig_score_ablation} shows each question asked twice under the two criteria designs, with mean confidence under each and the Wilcoxon result: all differences are significant by Wilcoxon signed-rank test ($p<0.001$; $n=772$ per-requirement, $n=40$ seniority, $n=116$ education, $n=156$ pooled). The per-requirement questions show a large gain.
```

new_string:
```
Figure~\ref{fig_score_ablation} shows each question asked twice under the two criteria designs, with mean confidence under each. All differences are significant by Wilcoxon signed-rank test, with rank-biserial effect size $r$ and a 95\% bootstrap confidence interval (10,000 resamples) on the mean confidence gain: per-requirement ($n=772$, $r=0.89$, gain $0.223$, 95\% CI $[0.204, 0.241]$, $p<0.001$), seniority ($n=40$, $r=0.77$, gain $0.254$, 95\% CI $[0.163, 0.336]$, $p<0.001$), education ($n=116$, $r=0.47$, gain $0.090$, 95\% CI $[0.041, 0.137]$, $p<0.001$), and pooled seniority/education ($n=156$, $r=0.58$, gain $0.132$, 95\% CI $[0.088, 0.176]$, $p<0.001$). The per-requirement questions show a large gain.
```

- [ ] **Step 2: Verify**

Run: `grep -nF "rank-biserial effect size" docs/paper2_jev.tex`
Expected: one match, in the edited paragraph.

Read the full paragraph (now spanning the replaced sentence plus the unchanged seniority/education discussion after it) to confirm the flow still makes sense — the unchanged trailing sentences already discuss "large effect" and "medium-to-large effect" per group, which should now read as consistent with the restored $r$ values (0.89 per-requirement > 0.77 seniority > 0.58 pooled > 0.47 education).

- [ ] **Step 3: Commit**

```bash
git add docs/paper2_jev.tex
git commit -m "docs: restore Wilcoxon effect sizes and add 95% CIs to ablation results"
```

---

### Task 5: Cost/Latency intro — separate 273-call pipeline cost from 819-call reliability cost, and state wall-clock

**Files:**
- Modify: `docs/paper2_jev.tex` (paragraph opening "Cost and Latency Efficiency...", currently line 174)

**Interfaces:** None. The 8-worker and 4-worker concurrency figures come from `scripts/run_jev_evaluation_study.py`'s `--max-workers` default and `.env`'s `CANDIDATE_RANKING_OLLAMA_NUM_PARALLEL` — re-check both files at implementation time in case either value has changed since the design was written (see spec's Verification section).

- [ ] **Step 1: Confirm the concurrency values are still current**

Run: `grep -n "max-workers" scripts/run_jev_evaluation_study.py` — expected: `default=8`.
Run: `grep -n "CANDIDATE_RANKING_OLLAMA_NUM_PARALLEL" .env` — expected: `=4`.

If either value differs from 8 or 4, recompute the wall-clock figures in Step 2 as `(n_calls / workers) * 1.04` seconds, converted to minutes, before writing the sentence.

- [ ] **Step 2: Replace the paragraph**

old_string:
```
This section reports the results of the direct sort outperforming the iterative tournament. Across 819 timed Jev calls (273 pairs, 3 independent calls each), a single call takes a mean of 1.04s, for a total of 14.2 minutes of compute time across the full corpus -- call latency only, not skill/job profile extraction.
```

new_string:
```
This section reports the results of the direct sort outperforming the iterative tournament. The production pipeline issues 273 Jev calls, one per shortlisted pair; a separate reliability-test measurement (Section III-D) repeats each pair's call three times independently for 819 timed Jev calls total, to test repeat-to-repeat stability rather than to represent the pipeline's own per-corpus cost. A single call takes a mean of 1.04s. Summed sequentially, the 819 reliability-test calls take 14.2 minutes of compute time -- call latency only, not skill/job profile extraction; at the reliability-test script's concurrency (8 concurrent calls), this corresponds to roughly 1.8 minutes of wall-clock time for the corpus, and at the production pipeline's own concurrency (4 concurrent calls, Section III), the 273-call pipeline corresponds to roughly 1.2 minutes of wall-clock time.
```

- [ ] **Step 3: Verify**

Run: `grep -n "819 timed Jev calls (273 pairs" docs/paper2_jev.tex`
Expected: no output (old phrasing fully replaced).

Run: `grep -n "wall-clock time for the corpus" docs/paper2_jev.tex`
Expected: one match.

- [ ] **Step 4: Commit**

```bash
git add docs/paper2_jev.tex
git commit -m "docs: separate 273-call pipeline cost from 819-call reliability cost, add wall-clock"
```

---

### Task 6: Results table — fix caption, footnote the Confidence-signal and Total-calls rows, restate compute time

**Files:**
- Modify: `docs/paper2_jev.tex` (Table `tab_comparison_results`, currently lines 177–194)

**Interfaces:** None. Reuses the same 8-worker / 4-worker concurrency figures as Task 5 — if Task 5's Step 1 found different values, use the recomputed minutes here too instead of 1.2.

- [ ] **Step 1: Fix the caption**

old_string:
```
\caption{Results compared against the closest prior published system \cite{ref2}.}
```

new_string:
```
\caption{Results compared against the closest prior system \cite{ref2}.}
```

- [ ] **Step 2: Mark the Total-calls row for a footnote**

old_string:
```
Total calls (corpus) & $\approx$2{,}061 & 273, 0 tournament\\
```

new_string:
```
Total calls (corpus)$^{\ddagger}$ & $\approx$2{,}061 & 273, 0 tournament\\
```

- [ ] **Step 3: Mark the Confidence-signal row for a footnote**

old_string:
```
Confidence signal & Faithfulness 0.880 & Calibrated confidence 0.894\\
```

new_string:
```
Confidence signal$^{\dagger}$ & Faithfulness 0.880 & Calibrated confidence 0.894\\
```

- [ ] **Step 4: Restate the Total-compute-time row to distinguish sequential sum from wall-clock**

old_string:
```
Total compute time (corpus) & Not reported & 14.2 minutes\\
```

new_string:
```
Total compute time (corpus)$^{\ddagger}$ & Not reported & 14.2 min (819 calls, sequential); $\approx$1.2 min (273-call pipeline, wall-clock)\\
```

- [ ] **Step 5: Insert the footnote block below the table**

old_string:
```
\end{tabular}
\end{table}
Table~\ref{tab_comparison_results} shows this comparison
```

new_string:
```
\end{tabular}
\vspace{2pt}
{\footnotesize $^{\dagger}$Faithfulness and calibrated confidence measure different constructs (groundedness vs.\ decision stability) and are not on a shared scale; neither number should be read as ``higher accuracy.'' $^{\ddagger}$819 calls (273 pairs $\times$ 3 repeats) were used only for the repeat-to-repeat reliability-test measurement (Section III-D, V); the deployed pipeline itself issues 273 calls, one per shortlisted pair.}
\end{table}
Table~\ref{tab_comparison_results} shows this comparison
```

- [ ] **Step 6: Verify**

Run: `grep -n "published system" docs/paper2_jev.tex`
Expected: no output (both captions now fixed, Task 2 + this task).

Run: `grep -noF '$^{\dagger}$' docs/paper2_jev.tex`
Expected: 2 matches — one on the "Confidence signal" row, one at the start of the footnote text.

Run: `grep -noF '$^{\ddagger}$' docs/paper2_jev.tex`
Expected: 3 matches — the "Total calls" row, the "Total compute time" row, and the footnote text.

Read the table plus footnote block once to confirm the LaTeX braces balance (`\end{tabular}` / `\vspace` / footnotesize group / `\end{table}` in the right order) and that no stray `$` breaks math mode.

- [ ] **Step 7: Commit**

```bash
git add docs/paper2_jev.tex
git commit -m "docs: footnote non-comparable metrics and 273-vs-819 call counts in results table"
```

---

### Task 7: Reliability Tests — promote ICC(1,1), demote the tautological Spearman correlation

**Files:**
- Modify: `docs/paper2_jev.tex` ("Score consistency" paragraph, currently line 206)

**Interfaces:** None. ICC=0.998 (range 0.995–1.000) comes from `runs/20260924-203510/evaluation/report.json`'s `score_decomposition_diagnostic.per_job_profile[*].icc` — do not recompute.

- [ ] **Step 1: Replace the paragraph**

old_string:
```
\textbf{Score consistency.} The per-requirement scores and the derived overall score are not just averaged but consistent with each other: across all 273 pairs, the mean of a pair's per-requirement scores correlates with its overall fit score at Spearman $\rho=0.972$, confirming the ranking input reflects the same per-requirement evidence the score breakdown check inspects separately.
```

new_string:
```
\textbf{Test-retest reliability (ICC).} A one-way random-effects intraclass correlation, ICC(1,1), measures what fraction of total score variance across the three repeats is between-applicant signal rather than within-applicant repeat noise: computed per job profile from the three repeats' overall fit scores, the corpus mean is ICC$=0.998$ (range 0.995--1.000 across the 10 job profiles), indicating the per-requirement repeat noise described above contributes negligibly to the applicant-level overall score that ranking is based on. As a secondary, partly definitional check -- since the overall score is by construction the mean of the per-requirement, seniority, and education scores -- the mean of a pair's per-requirement scores also correlates with its overall fit score at Spearman $\rho=0.972$ across all 273 pairs, consistent with but not independent evidence for the ICC result above.
```

- [ ] **Step 2: Verify**

Run: `grep -n "ICC" docs/paper2_jev.tex`
Expected: at least one match, in the edited paragraph.

Run: `grep -n "Score consistency" docs/paper2_jev.tex`
Expected: no output (the bold lead-in was renamed to "Test-retest reliability (ICC)").

- [ ] **Step 3: Commit**

```bash
git add docs/paper2_jev.tex
git commit -m "docs: lead Reliability Tests with ICC(1,1), demote tautological correlation"
```

---

### Task 8: Conclusion — mirror the abstract's hallucination-risk and 4.5× scoping

**Files:**
- Modify: `docs/paper2_jev.tex` (Conclusion paragraph, currently line 219)

**Interfaces:** Must use the same qualifying phrases as Task 1 so the abstract and conclusion stay worded consistently (per spec Verification section).

- [ ] **Step 1: Narrow "hallucination risk removed by design"**

old_string:
```
this work's advantages are: hallucination risk removed by design, since typed answers replace free text with nothing left to hallucinate;
```

new_string:
```
this work's advantages are: the specific risk of free-text hallucination and schema violation removed by design, since typed answers replace free text with nothing left to hallucinate or fail schema validation (an individual structured judgment can still be wrong);
```

- [ ] **Step 2: Scope the 4.5× claim**

old_string:
```
with no tournament infrastructure to maintain and roughly 4.5$\times$ lower per-candidate wall-clock latency;
```

new_string:
```
with no tournament infrastructure to maintain and, on a single live head-to-head profile comparison, roughly 4.5$\times$ lower per-candidate wall-clock latency;
```

- [ ] **Step 3: Verify**

Run: `grep -n "hallucination risk removed by design" docs/paper2_jev.tex`
Expected: no output in the old unscoped form.

Run: `grep -c "single live head-to-head profile comparison" docs/paper2_jev.tex`
Expected: 2 (abstract from Task 1, conclusion here).

- [ ] **Step 4: Commit**

```bash
git add docs/paper2_jev.tex
git commit -m "docs: mirror abstract's hallucination-risk and 4.5x scoping in conclusion"
```

---

### Task 9: Conclusion — add the ethics/fairness/human-oversight paragraph

**Files:**
- Modify: `docs/paper2_jev.tex` (between the limitation sentence and "Future work...", currently lines 220–221)

**Interfaces:** None.

- [ ] **Step 1: Insert the paragraph**

old_string:
```
A limitation is that this work trades away the prior system's free-text interpretability -- a score and a confidence, not a narrative, and per-requirement scores only partly substitute.
Future work includes independently testing RLCD's calibration guarantee against ground-truth correctness rather than self-consistency alone, since calibration does not itself guarantee any individual prediction is correct; and testing whether the per-level independence property behind criteria-grounded design holds across TypeSafe's full documented 2--10 level range, not only the five used here.
```

new_string:
```
A limitation is that this work trades away the prior system's free-text interpretability -- a score and a confidence, not a narrative, and per-requirement scores only partly substitute.
This system is a screening aid, not an automated hire/reject decision: any adverse action should involve human review of the underlying score breakdown and resume evidence, not an unexamined rank cutoff. The evaluation corpus is overwhelmingly synthetic (Section IV), so this paper makes no claim about bias behavior on real-world resume populations, which can differ systematically by protected-class-correlated writing style, employment gaps, or institution names. Calibrated confidence (Section II) is a statement about decision stability -- whether the same judgment recurs on repetition -- not a fairness or non-discrimination guarantee, and a well-calibrated model can still be systematically biased against a protected group.
Future work includes independently testing RLCD's calibration guarantee against ground-truth correctness rather than self-consistency alone, since calibration does not itself guarantee any individual prediction is correct; and testing whether the per-level independence property behind criteria-grounded design holds across TypeSafe's full documented 2--10 level range, not only the five used here.
```

- [ ] **Step 2: Verify**

Run: `grep -n "screening aid" docs/paper2_jev.tex`
Expected: one match.

Read the full Conclusion section once end-to-end to confirm paragraph order and flow (limitation → ethics → future work).

- [ ] **Step 3: Commit**

```bash
git add docs/paper2_jev.tex
git commit -m "docs: add ethics/fairness/human-oversight paragraph to conclusion"
```

---

### Task 10: Full-document verification pass

**Files:** None modified — read-only checks against `docs/paper2_jev.tex`.

- [ ] **Step 1: Confirm every stale string is gone**

Run each and confirm the "Expected" result:
- `grep -n "published system" docs/paper2_jev.tex` → no output
- `grep -n "hallucination risk removed by design" docs/paper2_jev.tex` → no output
- `grep -n "hallucination risk by design" docs/paper2_jev.tex` → no output
- `grep -nF "819 timed Jev calls (273 pairs" docs/paper2_jev.tex` → no output
- `grep -nF "Score consistency." docs/paper2_jev.tex` → no output

- [ ] **Step 2: Confirm every new element is present exactly once (or twice, where the design calls for both abstract and conclusion)**

- `grep -c "single live head-to-head profile comparison" docs/paper2_jev.tex` → 2
- `grep -c "screening aid" docs/paper2_jev.tex` → 1
- `grep -noF 'ICC$=0.998$' docs/paper2_jev.tex` → 1 match
- `grep -c "versioned snapshot identifier" docs/paper2_jev.tex` → 1
- `grep -c "wall-clock time for the corpus" docs/paper2_jev.tex` → 1

- [ ] **Step 3: Read the full file top to bottom**

Confirm: no unbalanced `\begin`/`\end` pairs introduced by any edit, no orphaned `$^{\dagger}$`/`$^{\ddagger}$` markers without their footnote, and the abstract/conclusion read as coherent paragraphs after all edits land.

- [ ] **Step 4: No commit needed**

This task is verification-only (Tasks 1–9 already committed their own changes). If any check in Step 1 or Step 2 fails, fix the specific task's edit and amend that task's commit before moving on — do not leave a failing verification uncommitted-fixed.
