# ICAIDES 2026 Deck Redesign — Design Spec

## Goal

Replace the current 26-slide `docs/presentation/icaides2026_jev.pptx` (built by
`docs/presentation/build_pptx.py`) and its speaker notes (`docs/presentation/script.md`)
with a 14-slide deck that:

- Mirrors the structure of `docs/paper2_jev.tex` section-by-section (no slides for
  content that doesn't exist in the paper).
- Uses a "data-forward minimal" visual style: one core idea per slide, numeric
  results shown as a large hero number.
- Reads as professional/Q1-Scopus-appropriate: white background only, no solid
  black fills anywhere (no filled icon illustrations, no inverted white-on-black
  boxes, no filled progress bars), pure black outlines and text on white.

## Visual System

- **Background:** white on every slide, no exceptions (title and closing slide
  included).
- **Color:** black text/lines on white only. No accent color.
- **Fills:** none. Every shape is either plain text or a shape with a thin black
  outline (`line.width ≈ 1.25–1.75pt`) and transparent/white fill. This removes
  all of the current icon-drawing functions (`add_bullseye_icon`, `add_bolt_icon`,
  `add_loop_icon`, `add_gear_icon`, `add_warning_icon`, `add_doc_icon`,
  `add_funnel_icon`, `add_pair_icon`, `add_stack_icon`, `add_briefcase_icon`,
  `add_single_bar_icon`) and every `fill=BLACK` call (comparison-box header,
  "current stage" highlight box, proportion-bar fill).
- **Emphasis without fill:** where the old deck used an inverted black box to mark
  a "current" or "selected" state (e.g. mini-pipeline stage highlight, chosen
  Score-primitive option), the new deck uses **bold text + an underline rule**
  under that item instead.
- **Font:** Arial throughout (title bold, body regular) — safe, professional,
  universally available in PowerPoint. Set explicitly via `font.name` (the old
  file never set this, so it fell back to the theme default).
- **Layout rule:** one idea per slide.
  - Numeric-result slides: one hero number (large, bold, centered) + one caption
    line underneath explaining what it is.
  - Concept/methodology slides: a title + 2–4 short lines of body text (no long
    bullet lists), optionally a simple outline diagram (boxes + thin arrows,
    no fill, no decoration).
- **Diagrams kept, redrawn outline-only:** the 5-stage pipeline diagram and the
  old-system-vs-new-system comparison keep their existing box+arrow layout
  logic from `build_pptx.py` (positions/geometry can be reused), but every box
  becomes an outline-only box and every "current/highlighted" fill becomes
  bold+underline text instead.

## Content Cuts (relative to the current 26-slide deck)

These are intentionally dropped or folded into another slide, since they don't
map to a distinct paper section/finding, per user approval:

- "Ingestion & Shortlisting" deep-dive slide → folded into one line on the
  Architecture slide (Slide 6); not one of the paper's three contributions.
- "Score Breakdown Test" deep-dive slide → dropped. It is explicitly not one of
  the paper's three contributions (`paper2_jev.tex` line 118, line 188). It still
  appears as a plain label in the Architecture diagram (Slide 6), unexplained,
  matching how little space the paper gives it outside its own subsection.
- "Must-Have Gate, Justified" (32/305, 10.5%) result slide → dropped as a
  dedicated slide. This is a real paper result but is a validation of the
  must-have gate, not one of the three contributions; cutting it keeps the
  deck strictly to the three-contributions narrative the paper's abstract
  promises. (If the user wants it back in later, it fits as a footnote on
  Slide 9 Experimental Setup, next to the 46.1% must-have figure.)
- "Not a Strict Win on Every Axis" honesty-check slide → folded into Slide 12
  (Calibration & Stability) as its caption line, since it's a caveat on the
  same two numbers (0.894 vs 0.880), not a separate finding.
- "Cost: What We Did and Didn't Measure" honesty-check slide → folded into
  Slide 11 (Efficiency) as its caption line, same reasoning.
- Corpus-at-a-glance and one-variable-fixed slides (old Slides 15–16) → merged
  into a single Slide 9, since both describe the same Experimental Setup
  section of the paper.
- Key Takeaways and Limitations & Future Work (old Slides 24–25) → merged into
  a single Slide 13, both being the paper's Conclusion section.

## Slide-by-Slide Content (14 slides)

1. **Title**
   - "Reliable Structured-Decision Applicant Ranking with a Non-Autoregressive System One Model"
   - Authors: Jessie Angelica, Hasanul Fahmi Zuhri, Rusdianto Roestam, Muhammad Arief
   - One line: "Can a model that only fills in structured answers rank applicants better than one that writes free text?"

2. **Motivation & Gap** (paper §I, paras 1–2)
   - Free-text assessment risks hallucination and gives no native confidence signal.
   - Jev (TypeSafe, Sep 2026) is the first "System One Model" — no independent evaluation existed before this paper.

3. **Objective** (paper §I, para 3 / Abstract)
   - Three contributions: criteria-grounded question design; cost/latency efficiency from one parallel pass vs. an iterative tournament; repeat-to-repeat reliability testing.

4. **Background: What Is a System One Model** (paper §II)
   - Non-autoregressive: answers typed questions in one parallel pass, no generated tokens.
   - Three typed primitives: Noul (yes/no + confidence), Choice (multiple-choice), Score (ordinal scale). This paper uses Score only.

5. **Background: Why It Works** (paper §II)
   - Each Score level is judged independently, with no view of neighboring levels — a bare ordinal number carries no signal, a concrete situation does.
   - Trained via RLCD (calibration-direct reward), not RLHF or RLVR — calibration ≠ correctness.

6. **Architecture** (paper §III)
   - Outline-only 5-box pipeline diagram: Extraction → Shortlisting → Structured Assessment via Jev (parallel) → Direct Sort → Score Breakdown Test (plain label, unexplained).
   - One line: steps 1–2 unchanged from the prior system; contributions live in steps 3–4.

7. **Contribution 1: Criteria-Grounded Question Design** (paper §III)
   - Each Score level written as a concrete situation instead of a bare number.

8. **Contribution 2: Direct Sort Ranking** (paper §III)
   - Old: iterative Plackett-Luce tournament, round-by-round. New: one score per candidate, sort directly, fully parallel.

9. **Experimental Setup** (paper §IV)
   - 273 shortlisted (job, applicant) pairs · 10 job profiles · 501 resumes (500 synthetic + 1 real) · 3 independent repeats per pair.
   - One variable changed between systems: the assessment step. 46.1% of extracted required skills classified must-have.

10. **Result: Confidence Gain** (paper §V)
    - Hero number: **0.672 → 0.894** (per-requirement)
    - Caption: also 0.635 → 0.766 for pooled seniority/education confidence.

11. **Result: Efficiency** (paper §V)
    - Hero number: **273 vs. ~2,061** model calls
    - Caption: ~4.5× lower per-candidate latency; output priced at zero by design, not billed per token (vendor claim on price, not independently measured — one clause, not a separate slide).

12. **Result: Calibration & Stability** (paper §V)
    - Hero number: **0.965 vs. ≈0.957** (Kendall's τ, ranking stability across 3 repeats)
    - Caption: native calibrated confidence 0.894 vs. prior system's Faithfulness 0.880 — different constructs, not a claim of higher accuracy.

13. **Conclusion** (paper §VI)
    - Key takeaways (3 numbers above) + limitations in one slide: calibration ≠ correctness (next: test against ground truth); only a 5-level scale tested (Jev supports up to 10); single pipeline (independent replication would strengthen the case).

14. **Thank You**
    - Contact + "Questions & Discussion."

## Files Touched

- `docs/presentation/build_pptx.py` — rewritten: strip all fill/icon helpers,
  add outline-box/hero-stat/pipeline helpers per the Visual System above,
  rebuild all 14 slides.
- `docs/presentation/script.md` — rewritten: 14 slides, same simple/basic
  English register already established earlier in this deck's history, one
  paragraph of speaker notes per slide.
- `docs/presentation/icaides2026_jev.pptx` — regenerated by running the
  rewritten `build_pptx.py`.
- `docs/presentation/generate_charts.py` and any PNGs it produces
  (`confidence_chart.png`, `model_calls_chart.png`, `latency_chart.png`,
  `stability_chart.png`) — no longer referenced once results become hero-stat
  text slides instead of `add_picture_slide` chart images; left in place on
  disk (not deleted) since removing files isn't necessary to satisfy this
  request, but no longer called from `build_pptx.py`.

## Open Item Resolved During Review

Font choice was proposed as Arial (professional, universally available); user
did not object, so this spec locks it in as final rather than leaving it as a
follow-up question.
