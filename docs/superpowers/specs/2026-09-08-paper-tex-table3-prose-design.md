# Paper .tex Table III-to-Prose Conversion — Design

## Purpose

Follow-on to `docs/superpowers/specs/2026-09-08-paper-tex-5-page-shortening-design.md`
(already implemented and merged — `docs/bare_jrnl.tex`'s body currently
ends on page 5). Per explicit user request in this session: convert Table
III ("Position-robust tournament reliability") into inline prose instead
of a table, and apply light additional prose-tightening — only as much as
needed, not a fixed target — to four specific subsections: III-C
("Assessment Generation with Self-Correction"), III-D ("Tournament
Ranking"), III-E ("Evaluation"), and V-B ("Generation Faithfulness and
Contradiction Audit"). The body must remain within 5 pages throughout.

Unlike the prior shortening pass (which explicitly kept all 5 tables),
this is a **narrow, explicit exception**: Table III specifically is
removed and its data folded into prose. No other table or figure is
touched, removed, or renumbered beyond the mechanical renumbering this
one removal causes.

## Protected content (must survive with unchanged meaning)

Same list as the prior spec, still binding, with one adjustment: Table
III's own values move from a table to prose but must still all appear,
unchanged:

- Table III's four values — now in prose: 1,713 total ranking calls, 0
  initial invalid outputs, 0 successful retries needed, 0 unrecovered
  failures — plus the metric names (`rank_subset` invocations,
  schema/permutation validation, `ListwiseRankingError`).
- Table II (per-role Faithfulness rows, "300 stratified / 300 scored"
  wording, 0.880 mean, "following successive prompt refinements").
- Table IV's values (1268/130/88/42/0.00%) — table survives, only its
  in-text ordinal reference changes from "Table IV" to "Table III".
- Table V's values (all comparison cells) — table survives, only its
  in-text ordinal reference changes from "Table V" to "Table IV".
- Fig. 2's caption and self-correction mechanism description.
- Section III's adaptive-iteration formula and Table I's values.
- Section VI's Limitations paragraph.
- All 4 figure captions, all 19 `\bibitem` entries.

## Non-goals

- No other table or figure removed — Tables I, II, IV (renumbered III),
  V (renumbered IV), and all 4 figures survive unchanged in content.
- No cuts to sections/subsections other than III-C, III-D, III-E, V-B.
- Light-touch tightening only: cut wording-only filler in those four
  subsections, and only as much as needed to stay within 5 pages after
  the Table III conversion — not a fixed compression percentage. If the
  Table III conversion alone secures comfortable margin, stop there.
- No number, claim, or citation touched in either the table conversion
  or the prose trims — the table conversion must state every one of
  Table III's four values and metric names, just in prose form.

## Approach

### Step 1 — Convert Table III to prose

Replace the table environment (`\label{tab_reliability}` through
`\end{table}`) and its lead-in/follow-up sentences with one consolidated
paragraph:

Old (lead-in sentence + table + follow-up sentence):
```
The position-robust tournament's reliability claim is backed by the
following counts from the full run, verified directly against
\texttt{repeats.json} and \texttt{warnings.json}:

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

All 1,713 listwise ranking calls (Table III) completed without an
unrecovered failure, so no comparison in this run had to be discarded.
```

New (one paragraph, no table):
```
The position-robust tournament's reliability claim is backed by the
following counts from the full run, verified directly against
\texttt{repeats.json} and \texttt{warnings.json}: of 1,713 total
tournament ranking calls (\texttt{rank\_subset} invocations), 0 produced
an initial invalid output (failed schema/permutation validation on the
first attempt), 0 needed a successful retry, and 0 ended in an
unrecovered failure (\texttt{ListwiseRankingError}, both attempts
invalid) --- so no comparison in this run had to be discarded.
```

### Step 2 — Renumber the two tables after it

LaTeX auto-numbers table *captions*; only hardcoded in-text ordinal
mentions need manual edits (verified via grep — exactly 2 remaining after
Step 1):

- `(Table IV)` → `(Table III)`, in the retry-outcomes sentence ("Of the
  130 pairs with an initial contradiction (Table IV), 88 were fixed...").
- `Table V's rows are drawn from...` → `Table IV's rows are drawn
  from...`.

### Step 3 — Light trims (apply only as needed)

After Steps 1–2, recompile and measure. If the body is already
comfortably within 5 pages, stop — do not apply any of the following.
Otherwise, apply in this order, remeasuring after each, stopping as soon
as margin is comfortable:

1. III-C: cut "in place of a direct claim" (redundant after "no hedging
   language (...)") and "critically" (emphasis filler), from the
   assessment-generation constraints sentence.
2. III-C: merge the two sentences about bare-list-item skills into one,
   removing the repeated "bare-list-item" phrasing — "including skills
   appearing only as a bare list item. A bare-list-item skill with no
   descriptive context goes into additional\_skills instead of strengths
   or weaknesses, since it is still a skill the applicant has." becomes
   "including a skill appearing only as a bare list item, which instead
   goes into additional\_skills (not strengths or weaknesses) since it is
   still a skill the applicant has."
3. III-E: tighten "which reduces the noise a single repeat's point
   estimate would otherwise carry into the ranking" to "reducing noise
   from any single repeat's point estimate."

III-D and V-B's remaining prose (beyond what Step 1/2 already touch) is
dense, technical description of the paper's core mechanisms — not
identified as containing further wording-only filler beyond what the
prior shortening pass already removed. If margin is still short after
all three trims above, this is a stopping point requiring a check-in
with the user rather than cutting into that technical description.

## Mechanics

- Edit `docs/bare_jrnl.tex` directly (git-tracked).
- Reuse `scripts/measure_paper_tex_body_pages.py` unchanged — same
  two-pass `pdflatex` + `pdftotext` measurement loop as the prior plan.
- "Comfortable margin" (Step 3's stopping condition) is operationalized
  as: the measurement script reports page 5 (exit 0), *and* the page
  containing "ACKNOWLEDGMENT" has at least some Conclusion-paragraph text
  before it (i.e., not immediately at the very top of the page with
  nothing else) — checked via the same `pdftotext` page-split technique
  used throughout this session, not a new heuristic.

## Verification

- `pdflatex` compiles clean (0 errors).
- Body ends on or before page 5 (`scripts/measure_paper_tex_body_pages.py`
  exits 0).
- Exactly 4 tables now (I, II, III [was IV], IV [was V]) with correct
  captions and auto-numbering; all 4 figures and all 19 references still
  present.
- Every "Protected content" item spot-checked against the final compiled
  PDF text, including Table III's four values now appearing as prose.
- No dangling "Table III" (old) or "Table V" (old) references remain
  anywhere in the document after renumbering — grep for both strings
  should only match the new, correct usages.
- Visual page-by-page inspection for layout defects, consistent with
  prior passes this session.

## Open parameters / risks

- Removing Table III's `table` float changes vertical space distribution
  on that page in a way that isn't perfectly predictable without
  compiling — the Mechanics section's measure-first, trim-only-if-needed
  approach handles this rather than assuming Step 3 is required.
- If Step 3's three trims together are insufficient, per Approach this
  stops for a user check-in rather than touching III-D's or V-B's
  technical mechanism descriptions, which were deliberately judged
  outside "light touch" scope in this design.
