# Paper .tex Table III-to-Prose Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert Table III ("Position-robust tournament reliability") in `docs/bare_jrnl.tex` into inline prose, renumber the two tables after it, and apply light optional trims only if needed to stay within the 5-page body target.

**Architecture:** One direct edit to `docs/bare_jrnl.tex` (table removal + prose merge + two renumbering edits), verified by the existing measurement script; if margin is short, up to 3 pre-drafted, wording-only trims are applied one at a time with remeasurement after each, stopping as soon as margin is comfortable.

**Tech Stack:** `pdflatex`/`pdftotext` (already installed), `scripts/measure_paper_tex_body_pages.py` (already exists, unchanged).

## Global Constraints

- No table or figure other than Table III may be removed. Tables I, II, and the two renumbered tables (old IV→III, old V→IV) survive with unchanged content. All 4 figures survive unchanged.
- No number, claim, or citation may be altered — Table III's conversion must state all four of its values (1,713; 0; 0; 0) and its three metric names in prose; the Step 3 trims (if applied) are wording-only.
- Body must end on or before page 5 (`scripts/measure_paper_tex_body_pages.py` exit 0) when this task is done.
- Step 3's trims are applied only as needed — stop as soon as the measurement script reports page 5. Do not apply all 3 trims if fewer already achieve the target.
- If Step 3's three trims together are still insufficient, stop and report to the user — do not cut into III-D's or V-B's technical mechanism descriptions (explicitly out of scope per the design spec).

---

## Task 1: Convert Table III to prose, renumber, and trim only as needed

**Files:**
- Modify: `docs/bare_jrnl.tex`

**Interfaces:**
- Consumes: `scripts/measure_paper_tex_body_pages.py`, run as `python3 scripts/measure_paper_tex_body_pages.py` from the repo root — prints the body-end page and exits 1 if above 5, 0 otherwise. No changes to this script.

- [ ] **Step 1: Convert Table III to inline prose**

In `docs/bare_jrnl.tex`, find this exact block (currently present verbatim):

```
The position-robust tournament's reliability claim is backed by the following counts from the full run, verified directly against \texttt{repeats.json} and \texttt{warnings.json}:

\begin{table}[!t]
\caption{Position-robust tournament reliability.}
\label{tab_reliability}
\centering
\begin{tabular}{p{2.0in}p{1.1in}}
\hline
Metric & Value\\
\hline
Total tournament ranking calls (rank\_subset invocations) & 1713\\
Initial invalid outputs (failed schema/permutation validation on attempt 1) & 0\\
Successful retries & 0 (none needed)\\
Unrecovered failures (ListwiseRankingError, both attempts invalid) & 0\\
\hline
\end{tabular}
\end{table}

All 1,713 listwise ranking calls (Table III) completed without an unrecovered failure, so no comparison in this run had to be discarded.
```

Replace the entire block above with this single paragraph:

```
The position-robust tournament's reliability claim is backed by the following counts from the full run, verified directly against \texttt{repeats.json} and \texttt{warnings.json}: of 1,713 total tournament ranking calls (\texttt{rank\_subset} invocations), 0 produced an initial invalid output (failed schema/permutation validation on the first attempt), 0 needed a successful retry, and 0 ended in an unrecovered failure (\texttt{ListwiseRankingError}, both attempts invalid) --- so no comparison in this run had to be discarded.
```

- [ ] **Step 2: Renumber the two remaining hardcoded table references**

In `docs/bare_jrnl.tex`, make these 2 edits (each string is unique in the file after Step 1):

1. Old: `Of the 130 pairs with an initial contradiction (Table IV), 88 were fixed by the retry and 42 still contradicted afterward and were dropped;`
   New: `Of the 130 pairs with an initial contradiction (Table III), 88 were fixed by the retry and 42 still contradicted afterward and were dropped;`

2. Old: `Table V's rows are drawn from the limitations and methodology stated in [1] and the results reported in Sections V-A--V-B.`
   New: `Table IV's rows are drawn from the limitations and methodology stated in [1] and the results reported in Sections V-A--V-B.`

- [ ] **Step 3: Recompile and measure**

Run:
```bash
cd docs && pdflatex -interaction=nonstopmode bare_jrnl.tex && pdflatex -interaction=nonstopmode bare_jrnl.tex && grep -c "^!" bare_jrnl.log; cd ..
python3 scripts/measure_paper_tex_body_pages.py
```

Expected: `0` LaTeX errors. The measurement script's page number and exit code determine what happens next:

- **If it reports page 5, exit 0:** margin is already comfortable (this is the expected, most likely outcome — Table III's removal frees real space). Skip Step 4 entirely and go to Step 5.
- **If it reports page 6 or higher, exit 1:** proceed to Step 4, applying the trims one at a time.

- [ ] **Step 4: Apply light trims one at a time, only as needed (skip if Step 3 already hit page 5)**

Apply in this order. After **each individual trim**, recompile (two-pass `pdflatex`) and rerun `python3 scripts/measure_paper_tex_body_pages.py`. Stop applying further trims as soon as it reports page 5, exit 0.

**Trim 1** — In `docs/bare_jrnl.tex`, Section III-C ("Assessment Generation with Self-Correction"):

Old:
```
Generation is constrained against failure modes traced from the Faithfulness audit: no speculation, no merging two resume facts into one claim, no skill or tool absent from the resume, no invented purpose or use case, no hedging language (``implied by'', ``likely'') in place of a direct claim, and --- critically --- no weakness claiming a skill gap that contradicts the applicant's own already-extracted skills, including skills appearing only as a bare list item.
```
New:
```
Generation is constrained against failure modes traced from the Faithfulness audit: no speculation, no merging two resume facts into one claim, no skill or tool absent from the resume, no invented purpose or use case, no hedging language (``implied by'', ``likely''), and no weakness claiming a skill gap that contradicts the applicant's own already-extracted skills, including skills appearing only as a bare list item.
```

Recompile and remeasure. If page 5/exit 0, stop here (skip Trims 2 and 3).

**Trim 2** — In `docs/bare_jrnl.tex`, Section III-C, immediately following the sentence Trim 1 touched:

Old:
```
A bare-list-item skill with no descriptive context goes into additional\_skills instead of strengths or weaknesses, since it is still a skill the applicant has.
```
New:
```
A bare list item with no descriptive context instead goes into additional\_skills (not strengths or weaknesses) since it is still a skill the applicant has.
```

Recompile and remeasure. If page 5/exit 0, stop here (skip Trim 3).

**Trim 3** — In `docs/bare_jrnl.tex`, Section III-E ("Evaluation"):

Old:
```
Each job profile's final ranking averages applicant utilities across its successful stability repeats, which reduces the noise a single repeat's point estimate would otherwise carry into the ranking.
```
New:
```
Each job profile's final ranking averages applicant utilities across its successful stability repeats, reducing noise from any single repeat's point estimate.
```

Recompile and remeasure.

**If, after all 3 trims, the measurement script still reports above page 5:** STOP. Do not cut further into III-D or V-B's technical mechanism descriptions (out of scope per the design spec). Report this to the controller/user rather than improvising additional cuts.

- [ ] **Step 5: Verify figure/table/reference completeness**

Run:
```bash
cd docs && pdftotext bare_jrnl.pdf /tmp/table3_check.txt && cd ..
python3 -c "
with open('/tmp/table3_check.txt') as f:
    c = f.read()
import re
figs = len(re.findall(r'Fig\. \d+\.', c))
tabs = len(re.findall(r'^TABLE [IV]+', c, re.MULTILINE))
missing_refs = [n for n in range(1, 20) if f'[{n}]' not in c]
print('figure captions found (expect >=4):', figs)
print('table captions found (expect 4):', tabs)
print('missing reference markers (expect []):', missing_refs)
print('old \"Table III\" reliability caption gone (expect False):', 'POSITION-ROBUST TOURNAMENT RELIABILITY' in c.upper())
"
```
Expected: figure captions >= 4, table captions == **4** (down from 5 — Table III's caption is gone), missing reference markers == `[]`, and the old reliability-table caption text does NOT appear anywhere (confirms the table, not just its number, is really gone).

- [ ] **Step 6: Spot-check protected numbers**

Run:
```bash
python3 -c "
with open('/tmp/table3_check.txt') as f:
    c = f.read()
checks = ['1,713', '1268', '130', '88', '42', '0.00%', '0.880', '0.930', '0.979', '300']
for chk in checks:
    print(chk, '=>', 'OK' if chk in c else 'MISSING')
"
```
Expected: every line prints `OK`. Note `1,713` (with comma) — the new prose sentence uses the comma-formatted form, matching the rest of the paper's convention (e.g. Conclusion's "1,713 ranking calls").

- [ ] **Step 7: Commit**

```bash
git add docs/bare_jrnl.tex
git commit -m "$(cat <<'EOF'
Convert Table III to prose in bare_jrnl.tex, renumber tables IV/V

Per explicit user request: Table III's reliability counts (1,713 calls,
0 invalid outputs, 0 retries, 0 unrecovered failures) now appear as
inline prose instead of a table, folded into the sentence that already
introduced them. Table IV and V's hardcoded in-text ordinal references
renumbered to III and IV. Body confirmed still within the 5-page target.
EOF
)"
```
