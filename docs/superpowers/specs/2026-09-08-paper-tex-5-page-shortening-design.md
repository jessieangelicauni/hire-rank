# Paper .tex 5-Page Shortening — Design

## Purpose

`docs/bare_jrnl.tex` (the LaTeX port of the paper, produced by the prior
"paper-latex-port" work) currently compiles to 7 pages total, with the main
body (Title/Abstract through the end of Section VI. Conclusion) spanning
pages 1 through page 6 (Section VI's Conclusion and the Acknowledgment
heading both land on page 6; References then continue through page 7).
The target is **5 pages for the main body only**; References and
Acknowledgment are exempt and may run onto extra pages.

This reverses the LaTeX-port design's explicit non-goal ("not re-shortening
the paper to hit 5 pages in LaTeX ... whatever page count results is
accepted") per the user's direct request in this session. Unlike the
earlier `.docx` shortening pass, no figure, table, table row, or reference
may be removed here — the user explicitly asked to "persist ... all figure
and table."

## Protected content (must survive with unchanged meaning)

Verified directly against the current `bare_jrnl.tex` text (some values
have moved since the `.docx`-era spec was written, e.g. Table IV's counts
are now 1268/130/88/42/0.00%, not the older 1224/26/0.82% figures — the
current `.tex` is authoritative):

- Table III (1,713 tournament ranking calls, 0 failures at every stage).
- Table IV (1268 weaknesses checked, 130 first-attempt contradictions, 88
  fixed by retry, 42 dropped, 0/1268 = 0.00% residual).
- Table V (all comparison cells, both columns).
- Fig. 2's caption and the self-correction mechanism description in
  Section III (the "Assessment Generation with Self-Correction" prose).
- Section III's adaptive-iteration formula (the `clamp(...)` display
  equation) and Table I's configuration values.
- Table II's per-role Faithfulness rows, the "300 stratified items ... 300
  of the 300 scored (0 excluded)" wording, and the run-wide mean of 0.880.
- Section VI's Limitations paragraph (synthetic-corpus scoping, no
  demographic/fairness claim).
- All 4 figure captions and all 5 table captions, unchanged.
- All 19 `\bibitem` entries.

## Non-goals

- Not removing any of the 4 figures or 5 tables, any table row, or any of
  the 19 references — a hard constraint, stricter than the `.docx` design's
  "check with the user first" contingency tier.
- Not touching References or the Acknowledgment section — they may run
  onto extra pages beyond the 5-page body target.
- Not modifying `docs/candidate-ranking-paper-new.docx`, any of its
  shortening scripts, or any pipeline code/run data — this is a `.tex`-only
  editorial and layout pass.
- Not renumbering figures or tables (none are removed, so this doesn't
  arise).
- Not installing new LaTeX packages beyond what `bare_jrnl.tex` already
  loads, unless a chosen fix genuinely requires one.

## Approach

### Phase 1 — Zero-content layout/typography levers (try first)

Reconnaissance: rendering the current PDF and inspecting page-by-page text
density found page 3 unusually sparse (2,269 characters vs. ~5,000–6,000 on
neighboring pages) — `Fig. 1` (`figure*`, full `\textwidth`, placement
`[!t]`) is starving that page of body text. This suggests some of the
1-page gap may be closeable without cutting any words.

Actions, in order, remeasuring after each:

1. Set all 5 tables to `\small` (a standard, widely-used IEEE convention
   for dense tables; changes only font size, not any cell content or
   value).
2. Adjust Fig. 1 and Fig. 2's placement specifiers and/or width to relieve
   the page-3 imbalance (e.g. trying `[t]` vs `[!t]`, modest width
   reduction within normal IEEE double-column figure sizing norms).
3. Confirm the standard two-pass `pdflatex` compile (already the LaTeX-port
   design's own convention) is what's being measured, so floats and
   cross-references have fully settled before any page count is trusted.

### Phase 2 — Light prose-tightening (only for the gap remaining after Phase 1)

Same rule as the `.docx` shortening work's Cut 3: cut filler and redundant
framing; never cut a number, a claim, or a citation. If a sentence carries
a unique fact, it stays, just tighter.

Targets, in this order, remeasuring after each and stopping as soon as the
body reaches page 5:

1. Introduction's motivating paragraphs.
2. Literature Review II-A and II-B (II-C is Yuksel et al. [1], the paper's
   key comparative anchor, and stays closest to its current level of
   detail, consistent with the `.docx` work's treatment of it).
3. The recurring "As Fig./Table X shows, ..." transition sentences in
   Methodology and Results.
4. Conclusion.

### Phase 3 — Contingency (only if Phases 1–2 are insufficient)

Tighten Table V's prose-heavy comparison cells (shorter phrasing, same
content) — the sole table content touched by this work, and only as a last
resort. No other table, row, figure, or reference is touched under any
circumstance; if even this doesn't close the gap, stop and check with the
user rather than removing evidence to hit the target.

## Mechanics

- Edit `docs/bare_jrnl.tex` directly. It's git-tracked, so git history is
  the backup — no separate `-backup.tex` file, unlike the `.docx` work's
  convention (which existed because `.docx` is a binary, undiffable
  format).
- New script `scripts/measure_paper_tex_body_pages.py`, mirroring the
  existing `scripts/measure_paper_body_pages.py`: runs `pdflatex
  docs/bare_jrnl.tex` twice (from `docs/`, to settle floats/cross-refs),
  then `pdftotext` on the resulting PDF, splits on form-feed into pages,
  finds the first page whose text contains "ACKNOWLEDGMENT" (case-
  sensitive match on the actual section heading, not a substring like
  "preferences" — verified during this design's reconnaissance that a
  naive case-insensitive substring search false-positives on that word),
  and reports/exits nonzero if that page is above 5.
- Iterate phase by phase in the order above, remeasuring after each
  change, stopping as soon as the target is hit.

## Verification

- `pdflatex` compiles clean (0 errors); benign warnings (e.g. "Underfull
  \hbox") are acceptable unless they indicate lost content.
- Body (Title through end of Conclusion) ends on or before page 5, per
  `scripts/measure_paper_tex_body_pages.py`.
- All 4 figures, all 5 tables (with correct captions), and all 19
  references still present — reusing the LaTeX-port design's own
  content-completeness check (`pdftotext` + citation marker scan).
- Every item in "Protected content" above spot-checked against the final
  compiled PDF text for unchanged meaning and values.
- Visual page-by-page inspection of the rendered PDF for layout defects
  (overlapping text, badly floated figures/tables, awkward splits)
  introduced by the Phase 1 layout changes.

## Open parameters / risks

- LaTeX's two-column float placement is somewhat non-deterministic and
  iterative — Phase 1's figure placement adjustments may need a few trial
  recompiles to land well, the same "measure then iterate" caveat the
  `.docx` work already established for its own medium.
- `\small` tables (Phase 1, action 1) is a standard, low-risk convention,
  but should still be visually checked for readability at typical PDF
  viewing zoom, not just for page count.
- If Phases 1–3 together don't close the gap without touching a figure,
  table row, or reference, this is a stopping point requiring user
  input, not a silent overrun of the "persist all figure and table"
  constraint.
