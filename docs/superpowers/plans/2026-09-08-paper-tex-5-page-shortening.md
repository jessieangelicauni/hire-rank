# Paper .tex 5-Page Shortening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get `docs/bare_jrnl.tex`'s main body (Title through end of Conclusion) from its current 6 pages down to 5, without removing any figure, table, table row, or reference.

**Architecture:** A measurement script drives a two-phase edit: Phase 1 changes typography/layout only (table font size, figure widths) with zero content change; Phase 2 tightens prose wording (never a number, claim, or citation) in the specific sentences identified during design research. Every edit in this plan has already been applied to a scratch copy and verified by real `pdflatex` compiles — this is not a blind measure-and-guess loop, the exact combination that reaches page 5 is known and specified below.

**Tech Stack:** `pdflatex`, `pdftotext`/`pdfinfo`/`pdftoppm` (`texlive-latex-base`/`texlive-latex-recommended`, already installed), Python 3 stdlib only (no new dependencies).

## Global Constraints

- No figure, table, table row, or reference may be removed — verified after every task by counting 4 figures / 5 tables / 19 references present.
- The "Protected content" list in `docs/superpowers/specs/2026-09-08-paper-tex-5-page-shortening-design.md` must survive with unchanged values/meaning — wording may only be tightened, never a number, claim, or citation changed.
- `docs/` and `scripts/` are both blanket-gitignored (`.gitignore` lines `docs/` and `scripts/`), but this repo's established convention is to force-add files under them (`git add -f`) — every commit step below uses `-f`.
- `docs/bare_jrnl.tex` is currently **untracked** (confirmed via `git ls-files`) — the first commit in this plan is what brings it under version control.
- Two-pass `pdflatex` compiles are required before trusting any page-count measurement (settles floats and cross-references).

---

## Task 1: Measurement script

**Files:**
- Create: `scripts/measure_paper_tex_body_pages.py`

**Interfaces:**
- Produces: a standalone script, run as `python3 scripts/measure_paper_tex_body_pages.py` from the repo root. Prints the body-end page and exits 1 if it's above 5, exits 0 otherwise. No importable symbols are consumed by later tasks — later tasks just run it as a subprocess.

- [ ] **Step 1: Write the script**

Create `scripts/measure_paper_tex_body_pages.py`:

```python
"""Compiles docs/bare_jrnl.tex with pdflatex (twice, to settle floats and
cross-references) and reports which page "ACKNOWLEDGMENT" first appears on,
using it as the boundary for where the main body (Title through end of
Section VI. Conclusion) ends -- the paper's Acknowledgment section
immediately follows the Conclusion, with no other section between them.
Exits nonzero if that page is greater than 5, per
docs/superpowers/specs/2026-09-08-paper-tex-5-page-shortening-design.md's
target.

Run with: python3 scripts/measure_paper_tex_body_pages.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
TEX_PATH = DOCS_DIR / "bare_jrnl.tex"
BODY_PAGE_LIMIT = 5


def compile_to_pdf(tex_path: Path, work_dir: Path) -> Path:
    for source_name in ("bare_jrnl.tex", "IEEEtran.cls", "IEEEtran.bst"):
        shutil.copy(DOCS_DIR / source_name, work_dir / source_name)
    figures_link = work_dir / "figures"
    if not figures_link.exists():
        figures_link.symlink_to(DOCS_DIR / "figures")
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "bare_jrnl.tex"],
            cwd=work_dir,
            check=True,
            capture_output=True,
        )
    return work_dir / "bare_jrnl.pdf"


def find_body_end_page(pdf_path: Path) -> int:
    page_count = int(
        subprocess.run(
            ["pdfinfo", str(pdf_path)], check=True, capture_output=True, text=True
        ).stdout.split("Pages:")[1].split()[0]
    )
    for page in range(1, page_count + 1):
        text = subprocess.run(
            ["pdftotext", "-f", str(page), "-l", str(page), str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if "ACKNOWLEDGMENT" in text:
            return page
    raise ValueError(f"'ACKNOWLEDGMENT' not found in any of the {page_count} rendered pages")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = compile_to_pdf(TEX_PATH, Path(tmp_dir))
        body_end_page = find_body_end_page(pdf_path)

    print(f"Body (Title through Conclusion) ends on page {body_end_page} "
          f"(ACKNOWLEDGMENT starts there). Target: <= {BODY_PAGE_LIMIT}.")
    if body_end_page > BODY_PAGE_LIMIT:
        print(
            f"OVER TARGET by {body_end_page - BODY_PAGE_LIMIT} page(s). Do not silently "
            "cut a figure, a table row, or a reference to close the gap -- see "
            "docs/superpowers/specs/2026-09-08-paper-tex-5-page-shortening-design.md's "
            "Phase 3 / Open parameters."
        )
        sys.exit(1)
    print("Within target.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against the current, unmodified `bare_jrnl.tex` to confirm the baseline**

Run: `python3 scripts/measure_paper_tex_body_pages.py`

Expected output (exit code 1):
```
Body (Title through Conclusion) ends on page 6 (ACKNOWLEDGMENT starts there). Target: <= 5.
OVER TARGET by 1 page(s). Do not silently cut a figure, a table row, or a reference to close the gap -- see docs/superpowers/specs/2026-09-08-paper-tex-5-page-shortening-design.md's Phase 3 / Open parameters.
```

This confirms the script works and matches this plan's verified baseline (6 pages) before any edits are made.

- [ ] **Step 3: Commit**

```bash
git add -f scripts/measure_paper_tex_body_pages.py
git commit -m "$(cat <<'EOF'
Add script to measure bare_jrnl.tex's compiled body page count

Compiles with pdflatex and reports which page ACKNOWLEDGMENT first
appears on, as a proxy for where the main body ends. Confirms the
current baseline: body ends on page 6, one page over the 5-page target.
EOF
)"
```

---

## Task 2: Phase 1 — zero-content layout levers

**Files:**
- Modify: `docs/bare_jrnl.tex`

**Interfaces:**
- Consumes: `scripts/measure_paper_tex_body_pages.py` from Task 1, run as a subprocess.
- Produces: an updated `docs/bare_jrnl.tex` that compiles clean. This task alone does **not** reach the 5-page target (verified during design research: layout changes alone leave it at page 6) — that's expected; Task 3 finishes the job. Documenting this explicitly so it isn't mistaken for a failed task.

This task makes two kinds of change, neither touching any cell value, caption, or word of body prose: (a) shrink all 5 tables to `\footnotesize` (a standard IEEE convention for dense tables), and (b) shrink the 4 figures' rendered widths modestly (chosen to stay clearly legible — verified visually during design research).

- [ ] **Step 1: Set all 5 tables to `\footnotesize`**

Each table currently has this structure (`\label` differs per table, distinguishing the 5 edits below): `\centering` on its own line, immediately followed by `\begin{tabular}{...}` on the next line, with nothing in between. Add `\footnotesize` between them, once per table.

In `docs/bare_jrnl.tex`, make these 5 edits:

1. Table I (`tab_config`) — old:
```
\label{tab_config}
\centering
\begin{tabular}{p{1.4in}p{1.7in}}
```
new:
```
\label{tab_config}
\centering
\footnotesize
\begin{tabular}{p{1.4in}p{1.7in}}
```

2. Table II (`tab_faithfulness`) — old:
```
\label{tab_faithfulness}
\centering
\begin{tabular}{ll}
```
new:
```
\label{tab_faithfulness}
\centering
\footnotesize
\begin{tabular}{ll}
```

3. Table III (`tab_reliability`) — old:
```
\label{tab_reliability}
\centering
\begin{tabular}{p{2.0in}p{1.1in}}
```
new:
```
\label{tab_reliability}
\centering
\footnotesize
\begin{tabular}{p{2.0in}p{1.1in}}
```

4. Table IV (`tab_retry`) — old:
```
\label{tab_retry}
\centering
\begin{tabular}{p{2.0in}p{1.1in}}
```
new:
```
\label{tab_retry}
\centering
\footnotesize
\begin{tabular}{p{2.0in}p{1.1in}}
```

5. Table V (`tab_comparison`) — old:
```
\label{tab_comparison}
\centering
\begin{tabular}{p{1.5in}p{2.5in}p{2.5in}}
```
new:
```
\label{tab_comparison}
\centering
\footnotesize
\begin{tabular}{p{1.5in}p{2.5in}p{2.5in}}
```

- [ ] **Step 2: Shrink the 4 figures' widths**

In `docs/bare_jrnl.tex`, make these 4 edits (each `\includegraphics` line is unique by filename/current width, no extra anchor needed):

1. Fig. 1 (pipeline architecture, `figure*`) — old:
```
\includegraphics[width=\textwidth]{Overall Architecture of the Proposed Applicant Ranking Methodology.png}
```
new:
```
\includegraphics[width=0.8\textwidth]{Overall Architecture of the Proposed Applicant Ranking Methodology.png}
```

2. Fig. 2 (self-correction mechanism) — old:
```
\includegraphics[width=2.27in]{Hallucination-Aware Self-Correction Mechanism for Applicant Assessment.png}
```
new:
```
\includegraphics[width=2.0in]{Hallucination-Aware Self-Correction Mechanism for Applicant Assessment.png}
```

3. Fig. 3 (MC-KG tournament ranking) — old:
```
\includegraphics[width=3.15in]{MC-KG-Guided Tournament Ranking and Plackett-Luce Aggregation.png}
```
new:
```
\includegraphics[width=2.7in]{MC-KG-Guided Tournament Ranking and Plackett-Luce Aggregation.png}
```

4. Fig. 4 (Kendall-tau convergence chart) — old:
```
\includegraphics[width=3.45in]{table6_kendall_tau.png}
```
new:
```
\includegraphics[width=2.9in]{table6_kendall_tau.png}
```

- [ ] **Step 3: Recompile and measure**

Run: `python3 scripts/measure_paper_tex_body_pages.py`

Expected output (exit code 1 — still over target, this is expected; Phase 2 in Task 3 closes the remaining gap):
```
Body (Title through Conclusion) ends on page 6 (ACKNOWLEDGMENT starts there). Target: <= 5.
OVER TARGET by 1 page(s). ...
```

Also confirm the compile itself is clean (no LaTeX errors introduced by this task):

Run: `cd docs && pdflatex -interaction=nonstopmode bare_jrnl.tex && pdflatex -interaction=nonstopmode bare_jrnl.tex && grep -c "^!" bare_jrnl.log; cd ..`
Expected: `0` (zero lines starting with `!`, LaTeX's error marker).

- [ ] **Step 4: Visual spot-check the shrunk figures for legibility**

Run: `cd docs && pdftoppm -png -r 100 -f 3 -l 3 bare_jrnl.pdf /tmp/check_pg3 && cd ..`

Open `/tmp/check_pg3-3.png` and confirm Fig. 1's internal box labels and Fig. 2's diagram text are still clearly readable at normal PDF zoom (not just technically present). If not legible, reduce the width shrink less aggressively (e.g. `0.85\textwidth` instead of `0.8\textwidth`) and re-run Step 3.

- [ ] **Step 5: Commit**

```bash
git add -f docs/bare_jrnl.tex
git commit -m "$(cat <<'EOF'
Shrink bare_jrnl.tex's tables and figures for page-count headroom

Phase 1 of the 5-page shortening effort: all 5 tables to \footnotesize
(standard IEEE convention, no cell content changed) and modest width
reductions on all 4 figures (no image cropped, only rendered smaller).
Zero body text changed. Still one page over target on its own --
Phase 2 (prose tightening) closes the remainder.
EOF
)"
```

---

## Task 3: Phase 2 — prose tightening

**Files:**
- Modify: `docs/bare_jrnl.tex`

**Interfaces:**
- Consumes: `scripts/measure_paper_tex_body_pages.py` from Task 1.
- Produces: `docs/bare_jrnl.tex` compiling with body ending on page 5 — the plan's target. Every edit below was verified during design research (real `pdflatex` compile) to preserve every citation, every number in the "Protected content" list, and reach the 5-page target in combination with Task 2.

Every edit here follows the same rule: cut filler/redundant framing (mostly the recurring "As Fig./Table X shows" transition scaffolding), never cut a number, a claim, or a citation.

- [ ] **Step 1: Tighten Section VI's Conclusion, second paragraph**

This is the paragraph that currently overflows onto page 6 — it's the direct cause of the overrun, so it's tightened first. It contains the protected Limitations statement (synthetic-corpus scoping) and the three future-work directions; both survive with unchanged meaning, only the framing is tightened, per the spec's explicit allowance ("only reworded for length if needed").

Old:
```
This study's corpus consists of synthetic resumes over synthetic job profiles authored by the paper's own authors, not real hiring data; its claims are correspondingly scoped to pipeline-level system reliability --- self-correction, ranking robustness, and convergence --- rather than real-world hiring validity or fairness. Building on these results, future work will pursue three directions: 1) human-rater validation --- comparing this pipeline against independent human-expert rankings using the same protocol as [1]; 2) an independent judge model --- sourcing Faithfulness scoring from a separate model than the one used for generation, to rule out shared-model bias; and 3) demographic and fairness evaluation --- checking whether shortlisting, assessment, or ranking behavior varies systematically across candidate demographic groups. Taken together, these mechanisms advance a proven architecture --- the LLM listwise tournament with Plackett-Luce aggregation --- toward one whose failure modes on a real, locally-hosted model are documented, mitigated, and measured, with the future work above set to close the remaining validation gaps.
```

New:
```
This study's corpus is synthetic resumes over synthetic job profiles authored by the paper's own authors, not real hiring data; its claims are scoped to pipeline-level reliability --- self-correction, ranking robustness, and convergence --- not real-world hiring validity or fairness. Future work will pursue three directions: 1) human-rater validation against independent human-expert rankings, using the same protocol as [1]; 2) an independent judge model, sourcing Faithfulness scoring from a model separate from generation to rule out shared-model bias; and 3) demographic and fairness evaluation, checking whether shortlisting, assessment, or ranking behavior varies across candidate demographic groups. Together, these mechanisms advance a proven architecture --- the LLM listwise tournament with Plackett-Luce aggregation --- toward one whose failure modes on a real, locally-hosted model are documented, mitigated, and measured.
```

- [ ] **Step 2: Tighten the 9 "As Fig./Table X shows" transition sentences**

Each pair below is a distinct, unique match in the file. None changes a number, a citation, or a table/figure caption — only the connecting sentence around it.

1. Old: `As shown in Fig.~1, the pipeline runs LLM-based skill extraction and semantic shortlisting,`
   New: `The pipeline (Fig.~1) runs LLM-based skill extraction and semantic shortlisting,`

2. Old: `As Fig.~2 shows, each generated weakness is checked in code`
   New: `Each generated weakness is checked in code`

3. Old:
```
As shown in Fig.~3, Monte Carlo knowledge-gradient subset selection feeds the LLM listwise tournament, whose partial rankings are aggregated via a Plackett-Luce model to update global applicant utilities until a stopping criterion is met.
```
   New:
```
Monte Carlo knowledge-gradient subset selection (Fig.~3) feeds the LLM listwise tournament, aggregated via a Plackett-Luce model until a stopping criterion is met.
```

4. Old:
```
As Table I shows, these values were held fixed across all runs reported in this paper. The target-appearances, ceiling, and floor rows drive the adaptive iteration mechanism, which computes each job profile's iteration budget from its shortlist size rather than using a single fixed count.
```
   New:
```
These values were held fixed across all runs. The target-appearances, ceiling, and floor rows drive the adaptive iteration mechanism, computing each job profile's iteration budget from its shortlist size rather than a single fixed count.
```

5. Old: `As Fig.~4 shows, within-repeat convergence (Kendall-$\tau$) is uniformly high`
   New: `Within-repeat convergence (Kendall-$\tau$, Fig.~4) is uniformly high`

6. Old:
```
As Table II shows, the sample comprises 300 stratified items from the run's assessments, covering all 10 job profiles, with 300 of the 300 scored (0 excluded as evaluation failures). The run-wide mean is 0.880, following successive prompt refinements. The judge model is the same Qwen2.5-14B model used for generation, not an independent judge, adopted after a smaller 7B judge was found to misjudge clearly resume-grounded claims.
```
   New:
```
The sample comprises 300 stratified items from the run's assessments, covering all 10 job profiles, with 300 of the 300 scored (0 excluded as evaluation failures). The run-wide mean is 0.880. The judge model is the same Qwen2.5-14B model used for generation, not an independent judge, adopted after a smaller 7B judge misjudged clearly resume-grounded claims.
```

7. Old:
```
As Table II shows, per-role Faithfulness scores range from full-stack-engineer (lowest, 0.764) to devops-engineer (highest, 0.944), bracketing the run-wide mean of 0.880.
```
   New:
```
Per-role Faithfulness scores range from full-stack-engineer (lowest, 0.764) to devops-engineer (highest, 0.944), bracketing the run-wide mean of 0.880.
```

8. Old:
```
As shown in Table III, all 1,713 listwise ranking calls completed without an unrecovered failure, so no comparison in this run had to be discarded.
```
   New:
```
All 1,713 listwise ranking calls completed without an unrecovered failure, so no comparison in this run had to be discarded.
```

9. Old: `As Table IV shows, of the 130 pairs with an initial contradiction,`
   New: `Of the 130 pairs with an initial contradiction,`

10. Old:
```
As shown in Table V, rows are drawn from the limitations and methodology explicitly stated in [1] and the results reported in Sections V-A--V-B of this paper. The comparison is not one-directional: Yuksel et al. [1] report a genuine external validation this pipeline does not yet have --- agreement with independent human-expert ratings. What this pipeline offers instead is internal, mechanistic reliability evidence for hallucination mitigation and tournament robustness that [1] does not report at all. Both kinds of evidence are complementary.
```
    New:
```
Table V's rows are drawn from the limitations and methodology stated in [1] and the results reported in Sections V-A--V-B. The comparison is not one-directional: Yuksel et al. [1] report a genuine external validation this pipeline does not yet have --- agreement with independent human-expert ratings. This pipeline instead offers internal, mechanistic reliability evidence for hallucination mitigation and tournament robustness that [1] does not report. Both kinds of evidence are complementary.
```

(Note: 10 items listed — the "9" in the step title counts distinct source sentences; item 6/7 are two separate sentences both introduced by "As Table II shows" in the original, hence 10 edits total.)

- [ ] **Step 3: Tighten Literature Review II-A and II-B, and one Introduction sentence**

All 19 references stay cited; only secondary elaboration is cut, per the spec's Cut/Phase-2 rule. II-C (Yuksel et al. [1]) is untouched, as it's the paper's key comparative anchor.

1. Old: `Jahan et al. [5]'s zero-shot multi-agent consensus vote sees recall drop to 27.9\% from requiring agent agreement;`
   New: `Jahan et al. [5]'s zero-shot multi-agent consensus vote drops recall to 27.9\%;`

2. Old: `Gera et al. [8] report only a single-resume case study with no benchmark;`
   New: `Gera et al. [8] report only a single-resume case study;`

3. Old: `None validates hallucination mitigation with a verified retry loop against the applicant's own extracted skills, as this paper's constrained, retry-on-contradiction generation does.`
   New: `None validates hallucination mitigation with a verified retry loop against the applicant's own extracted skills, as this paper does.`

4. Old:
```
Synthesizing 141 such papers, Dasaklis et al. [15] find hallucination and cross-session score instability to be recurring, unresolved risks across the field and call for dedicated hallucination-tracking metrics --- precisely the mechanism this paper's retry-on-contradiction loop provides.
```
   New:
```
Synthesizing 141 such papers, Dasaklis et al. [15] find hallucination and score instability to be recurring, unresolved risks and call for dedicated hallucination-tracking metrics --- precisely what this paper's retry-on-contradiction loop provides.
```

5. Old:
```
Rosenberger et al. [13] embed resumes and ESCO job descriptions into a shared space and rank by cosine similarity, validated on only five resumes and ten HR experts by the authors' own admission.
```
   New:
```
Rosenberger et al. [13] embed resumes and ESCO job descriptions into a shared space and rank by cosine similarity, validated on only five resumes and ten HR experts.
```

6. Old: `neither compares candidates within a shared context as a listwise judgment would.`
   New: `neither compares candidates within a shared context.`

7. Old:
```
LLMs can extract skills from unstructured text, justify why an applicant fits a role, and compare several applicants in a single pass --- none of which older keyword or embedding-similarity systems do on their own.
```
   New:
```
LLMs can extract skills from unstructured text, justify why an applicant fits a role, and compare several applicants in a single pass --- none of which older keyword or embedding-similarity systems do.
```

- [ ] **Step 4: Recompile and measure**

Run: `python3 scripts/measure_paper_tex_body_pages.py`

Expected output (exit code 0 — target reached):
```
Body (Title through Conclusion) ends on page 5 (ACKNOWLEDGMENT starts there). Target: <= 5.
Within target.
```

If this doesn't show page 5 (e.g. a different pdflatex/font version on this machine reflows text slightly differently than the one used during design research), do not cut a figure, table row, or reference. Instead, re-open `docs/superpowers/specs/2026-09-08-paper-tex-5-page-shortening-design.md`'s Phase 3 (Table V's prose cells only) and apply it, remeasuring after.

- [ ] **Step 5: Confirm zero LaTeX errors**

Run: `cd docs && pdflatex -interaction=nonstopmode bare_jrnl.tex && pdflatex -interaction=nonstopmode bare_jrnl.tex && grep -c "^!" bare_jrnl.log; cd ..`
Expected: `0`

- [ ] **Step 6: Commit**

```bash
git add -f docs/bare_jrnl.tex
git commit -m "$(cat <<'EOF'
Tighten bare_jrnl.tex prose to close the remaining page-count gap

Phase 2 of the 5-page shortening effort: cuts the recurring "As Fig./
Table X shows" transition scaffolding, tightens Literature Review
II-A/II-B secondary elaboration (zero citations dropped, II-C
untouched), and reworks Section VI's second paragraph (the one that
was overflowing onto page 6) for length only -- the Limitations
scoping claim and all three future-work directions are unchanged in
meaning. Combined with the prior commit's layout changes, body now
ends on page 5.
EOF
)"
```

---

## Task 4: Final verification

**Files:**
- None modified (verification only; only touches files if a defect surfaces, in which case fix inline and re-run this task's steps before proceeding).

**Interfaces:**
- Consumes: `scripts/measure_paper_tex_body_pages.py` from Task 1; the "Protected content" list in `docs/superpowers/specs/2026-09-08-paper-tex-5-page-shortening-design.md`.

- [ ] **Step 1: Confirm figure/table/reference completeness**

Run:
```bash
cd docs && pdflatex -interaction=nonstopmode bare_jrnl.tex >/dev/null && pdflatex -interaction=nonstopmode bare_jrnl.tex >/dev/null && pdftotext bare_jrnl.pdf /tmp/final_check.txt && cd ..
python3 -c "
with open('/tmp/final_check.txt') as f:
    c = f.read()
import re
figs = len(re.findall(r'Fig\. \d+\.', c))
tabs = len(re.findall(r'^TABLE [IV]+', c, re.MULTILINE))
missing_refs = [n for n in range(1, 20) if f'[{n}]' not in c]
print('figure captions found (expect >=4):', figs)
print('table captions found (expect 5):', tabs)
print('missing reference markers (expect []):', missing_refs)
"
```
Expected: figure captions >= 4, table captions == 5, missing reference markers == `[]` (empty list — every one of [1]-[19] appears somewhere in the compiled text).

- [ ] **Step 2: Spot-check every "Protected content" item from the design spec**

Run:
```bash
python3 -c "
with open('/tmp/final_check.txt') as f:
    c = f.read()
checks = ['1713', '1268', '130', '88', '42', '0.00%', '0.880', '0.930', '0.979', '300', '125 / 30', '8 (drives adaptive']
for chk in checks:
    print(chk, '=>', 'OK' if chk in c else 'MISSING')
"
```
Expected: every line prints `OK`. If any prints `MISSING`, stop and investigate before proceeding — this means a protected number was altered or reflowed unexpectedly, and must be restored before this work is considered done.

- [ ] **Step 3: Visual page-by-page inspection**

Run: `cd docs && pdftoppm -png -r 100 bare_jrnl.pdf /tmp/final_page && cd ..`

Open each of `/tmp/final_page-1.png` through `/tmp/final_page-N.png` (N = total page count) and confirm: no overlapping text, no figure/table floating to a visually broken spot, no awkward table split across a column break, all 5 `\footnotesize` tables still legible, all 4 resized figures still legible. This is the same technique used throughout the prior `.docx` shortening work.

- [ ] **Step 4: Confirm the body-page-count script passes**

Run: `python3 scripts/measure_paper_tex_body_pages.py`
Expected: exit code 0, "Within target." (already confirmed in Task 3 Step 4, re-run here as the final gate).

No commit needed for this task unless Step 2 or Step 3 surfaces a defect that required a fix — in that case, make the fix, re-run all 4 steps, then commit with a message describing what was corrected.
