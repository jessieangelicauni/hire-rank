# ICAIDES 2026 — Speaker Script
**Reliable Structured-Decision Applicant Ranking with a Non-Autoregressive System One Model**
14 slides · ~15 minutes · white background, black text, Calibri, one dark-slate accent color for rules/bars/key numbers · mirrors `docs/paper2_jev.tex` section by section.

Every slide below lists: **Title**, **Content** (what's on the slide), **Visual**, and the **Speaker Script** (what to say, ~25–45 sec/slide). Scripts are written the way you'd actually talk — casual, conversational, contractions and all — with natural transitions between slides. Every one still carries both the *why* (the reason) and the *how* (the mechanism or method); it's just delivered like a talk, not read like an abstract. All of it is sourced from the paper's own stated rationale, never invented.

**Fixed terminology, used identically everywhere in this deck (no synonyms substituted):**
- **applicant** — the person being ranked (never "candidate")
- **concrete situational criteria** — full term; **concrete criteria** — short form, used only after the full term appears once
- **bare ordinal-number scale** — the baseline being compared against (never "bare number," "bare label," "0–100 number")
- **Score** (capitalized) — Jev's ordinal-answer primitive; lowercase **score** — a numeric value (e.g., "overall score")
- **the tournament** — the prior system's iterative ranking method; **the prior system** — the whole system from `docs/paper1.tex`
- **RLCD** (Reinforcement Learning for Calibrated Decisions), **RLHF** (human-preference reward), **RLVR** (verifiable-outcome reward) — spelled out in full on first use, acronym only after
- **ICC(1,1)** — spelled out as "intraclass correlation" on first use

---

## Slide 1 — Title
**Content:** Paper title · authors · affiliations · ICAIDES 2026, 12–13 Nov 2026, Universitas Tarumanagara (hybrid) · one-line framing question.
**Visual:** Left-aligned title block with a short accent rule marking the top; conference/venue line in accent color above the title; framing question in italics beneath a thin divider rule.
**Script:** "Good [morning/afternoon], everyone. I'm Jessie Angelica — this is joint work with Hasanul Fahmi Zuhri, Rusdianto Roestam, and Muhammad Arief. So here's the question we set out to answer: can a model that only fills in structured answers rank job applicants as well as an LLM that writes full paragraphs — but at a fraction of the cost? That's what we're here to talk about."

## Section I — Introduction

### Slide 2 — Motivation & Gap
**Content:** Two-column compare — *Free-text assessment* (writes a paragraph verdict, can assert what the resume doesn't support, no built-in confidence signal) vs. *The gap* (TypeSafe released Jev, the first "System One Model," Sept 2026; no independent evaluation existed before this paper).
**Visual:** Two-column bullet list (accent-colored square markers) separated by a thin vertical rule — no boxes.
**Script:** "So, most AI hiring tools today read a resume and write you a paragraph about the applicant. Problem is, that paragraph gets generated word by word, and nothing's actually checking it against the resume as it goes — so it can just say a strength is there when it isn't, and there's no confidence score to warn you. Then in September 2026, TypeSafe releases Jev, the first 'System One Model.' Sounds great — except every claim about how reliable it is comes from TypeSafe itself. Nobody outside the company had tested it yet. That's the gap we're filling."

### Slide 3 — Objective & Three Contributions
**Content:** Three equal boxes: (1) Criteria-Grounded Question Design, (2) Cost & Latency Efficiency, (3) Repeat-to-Repeat Reliability. Caption: "Jev is TypeSafe's model, not ours — our job is deploying it reliably."
**Visual:** Three large accent-colored numerals (01/02/03), each with a short rule and a title/description beneath — no boxes.
**Script:** "Now, to be clear — Jev is TypeSafe's model, not ours. What we're contributing is making it actually work in a real hiring pipeline, and that's three parts. First, better question design — we write every Score level out as concrete situational criteria, because TypeSafe's own guidance had never been tested at scale. Second, we cut cost and latency by swapping the tournament for a direct sort. And third, reliability — we repeat every call three times and check the agreement, because a single call has no ensemble backing it up if something goes wrong."

## Section II — Literature Review / Background

### Slide 4 — What Is a System One Model
**Content:** Non-autoregressive — answers typed questions in one parallel pass, no generated tokens. Three output primitives: Noul (yes/no), Choice (multiple-choice), Score (ordinal scale — used throughout this paper).
**Visual:** Three columns, each a name + short rule + description; Score marked in accent color with a heavier rule as the one this paper uses.
**Script:** "Quick bit of background. Normal LLMs write text one token at a time — and that's exactly how they end up drifting into things that aren't true. Jev skips that entirely: it doesn't write a single word. It answers typed questions in one parallel pass, and it only ever answers in three fixed shapes — yes/no, multiple-choice, or an ordinal scale — so it can never come back broken or in the wrong format. We use that ordinal scale, called Score, for everything in this paper."

### Slide 5 — Why It Works: Criteria Grounding & Calibration
**Content:** Left: a bare ordinal-number label ("50") judged alone, no neighboring levels to compare against, no evidence to check. Right: a level written as concrete situational criteria ("used substantively in a real role") the model can match against resume text. Below: trained via RLCD — rewarded for calibration, not a human-preference reward (RLHF) or a verifiable-outcome reward (RLVR). Calibration ≠ correctness.
**Visual:** Two label/example pairs side by side with an accent arrow between them, plus one caption line below a divider rule.
**Script:** "Here's the trick, and it's a simple one. Jev judges each level alone, with nothing else to compare it to. A number like '50' gives it nothing to check. But 'used substantively in a real role' gives it resume text to check against. Jev's training follows the same idea: it's trained with RLCD, Reinforcement Learning for Calibrated Decisions, which rewards it for being calibrated. That's different from RLHF, human-preference reward — the usual way chatbots are tuned to sound helpful — and different from RLVR, verifiable-outcome reward, used when there's a checkable right answer, like in math or code. That second one, RLVR, is known to make models overconfident even as they get more accurate. Calibration isn't the same as being correct, but it's what makes our confidence numbers actually mean something."

## Section III — Methodology

### Slide 6 — Pipeline Architecture
**Content:** Five-stage vertical flow: Extraction & Skill Classification → Shortlisting → Structured Assessment via Jev (parallel calls, each also repeated 3× for reliability) → Direct Sort → Final Ranking. Caption: steps 1, 2, and 5 unchanged from the prior system; all three contributions live in steps 3–4 — Contribution 1 and 3 in the Jev call itself, Contribution 2 in the sort.
**Visual:** Vertical numbered sequence (circled 1–5) linked by a thin connecting rule — a simplified, native redraw of the paper's Figure 1 (TikZ pipeline diagram), without boxes.
**Script:** "Here's the pipeline, end to end. Extraction and shortlisting — we left those alone, because neither one generates free text, so there's no hallucination risk to fix there in the first place. Everything changes at the assessment step, which is where the risk actually lived. Jev runs once per applicant, fully in parallel — that's contribution one. That same call also gets repeated three times independently, to test reliability — that's contribution three, and I'll get to those results shortly. Then instead of a tournament, we just sort directly by score — that's contribution two."

### Slide 7 — Contribution 1: Criteria-Grounded Question Design
**Content:** Two-column compare — *Bare ordinal-number scale* (0/25/50/75/100, no evidence to check) vs. *Concrete situational criteria* ("only listed" vs. "used in a real role," gives Jev resume evidence). Footer previews the result: per-requirement confidence 0.672 → 0.894.
**Visual:** Same two-column rule-divided bullet template as Slide 2, with a result line below a divider rule.
**Script:** "So concretely, what does this look like? Instead of a bare ordinal-number scale, every Score level gets written out as concrete situational criteria — 'only listed as a skill' versus 'used with clear responsibility.' Why bother? Because Jev judges each level completely on its own, so a bare label gives it nothing to work with — but concrete situational criteria gives it something real to check against the resume."

### Slide 8 — Contribution 2: Direct Sort Ranking
**Content:** Two-column compare — *The Tournament* (Monte Carlo knowledge-gradient + Bayesian Plackett-Luce, round-by-round, each iteration waits on the last) vs. *Direct Sort* (one score per applicant, fully parallel, no ranking process). Footer previews the result: 273 vs. ≈2,061 calls.
**Visual:** Same two-column rule-divided bullet template.
**Script:** "The old system ranked applicants through a tournament — round after round, and each round has to wait on the result of the last one. That's just how it works; it can't all happen at once. Jev doesn't have that problem: every applicant gets scored completely independently, so there's nothing to wait on. Which means we can skip the tournament entirely and just sort by score."

## Section IV — Experimental Setup

### Slide 9 — Experimental Setup
**Content:** Four stat tiles: 273 shortlisted (job, applicant) pairs · 10 job profiles · 500 resumes (EraMatch v3.0 synthetic corpus) · 3× independent repeats per pair. Footer: one variable changed between systems (the assessment step); 46.1% of extracted requirements classified must-have; must-have gate excludes 32/305 applicants (10.5%) a total-only baseline would have shortlisted.
**Visual:** Four-column metric row — large accent-colored number, short rule, small label — separated by thin vertical rules, no boxes.
**Script:** "Here's what we actually tested: 273 shortlisted pairs, across 10 job profiles, drawn from a pool of 500 resumes — applicants get shortlisted by cosine similarity against each job's required skills. Now, only one thing changed between the two systems we're comparing — how applicants get scored — everything else stayed exactly the same, so any difference we see really comes from that one change. And we ran each pair three separate times, since a single-call system has no ensemble to fall back on if one answer's off."

## Section V — Results and Discussion

### Slide 10 — Result: Criteria-Grounded Design Raises Confidence
**Content:** Simple two-bar comparison: per-requirement confidence 0.672 (bare ordinal-number scale) → 0.894 (concrete criteria); pooled seniority/education 0.635 → 0.766. Stat line: Wilcoxon signed-rank, all differences p<0.001; per-requirement r=0.89, 95% CI [0.204, 0.241] on the gain (n=772).
**Visual:** Thin horizontal bars on a baseline rule (not solid blocks), accent fill, value labeled at the bar's end — a simplified native redraw of the paper's Figure 2.
**Script:** "Okay, first result. Using concrete criteria instead of a bare scale bumped per-requirement confidence from 0.672 up to 0.894. Same story for seniority and education — 0.635 up to 0.766. And these aren't small, noisy differences — Wilcoxon test, p under 0.001, confidence interval well clear of zero. This is actually the first time TypeSafe's own guidance has been tested at this scale, and it holds up across every question type, not just one."

### Slide 11 — Result: Cost & Latency Efficiency
**Content:** 273 vs. ≈2,061 total model calls (well under half). Per-applicant end-to-end latency, single live head-to-head check: 16.5s (this pipeline) vs. 74.3s (prior system). Output priced at zero by design — no tokens generated. Footer caveat: Faithfulness (0.880, prior) and calibrated confidence (0.894, this work) are not on a directly comparable scale.
**Visual:** Left: large accent number for call count. Right: two thin bars for latency (16.5s vs. 74.3s), separated by a vertical rule — no invented multiplier, raw measured numbers only.
**Script:** "Now the efficiency story. 273 calls versus roughly 2,061 for the old system — direct sort just needs one call per applicant, the tournament needed a lot of comparison rounds. Latency-wise: 16.5 seconds versus 74.3, mostly because Jev's calls run in parallel and the tournament's can't. One honest catch — the table also compares confidence signals, 0.880 Faithfulness versus our 0.894 — but those measure really different things, so this isn't us claiming to be 'more accurate.'"

### Slide 12 — Result: Reliability — Repeat-to-Repeat Stability
**Content:** ICC(1,1) = 0.998 (range 0.995–1.000 across 10 job profiles) — almost all score variance across repeats is real applicant signal, not noise. Ranking stability: Kendall's τ = 0.965 (this work, range 0.943–1.000) vs. ≈0.957 (prior system, range 0.93–0.98) — at least matching, not a clear win.
**Visual:** One large accent hero number for ICC(1,1)=0.998 with a short rule and caption; two thin bars beneath for the τ comparison — mirrors the paper's Figure 3 (per-job-profile ranking stability).
**Script:** "Last result: can we trust a single call, with no backup or double-check? We tested this by repeating every call three times. Here's the key idea — we don't just check if the parts add up to the total, since the total is made from those same parts, and that wouldn't really tell us much. Instead, we compare the three repeats to each other directly. That's called ICC, intraclass correlation, and it comes out at 0.998 — almost perfect. That tells us the differences we see between applicants are real, not just noise. The final rankings hold up too: Kendall's tau of 0.965, matching the old system's 0.957."

## Section VI — Conclusion

### Slide 13 — Conclusion
**Content:** Three "outperforms" tiles (cost: 273 vs. ≈2,061 calls · latency: 16.5s vs. 74.3s · hallucination/schema risk removed by design). One "matches, not wins" line (ranking stability, τ=0.965 vs. ≈0.957). Limitations & scope box: calibration ≠ correctness (test against ground truth next); synthetic-resume corpus only, no fairness claim on real hiring data; screening aid, not an automated hire/reject decision — human review required.
**Visual:** Three-column metric row for "outperforms," a rule-separated single line for "matches," and an accent-marked bullet list for limitations — all rule-divided sections, no boxes.
**Script:** "So, to wrap up. This pipeline wins on cost and latency, clearly — well under half the calls, 16.5 seconds versus 74.3 — and it removes hallucination risk by design, since there's simply no free text left to hallucinate. On ranking stability, it's a tie, not a win, and we're saying that plainly rather than overselling it. Three honest limits before I finish: calibration isn't correctness, so testing against real ground truth is next. Our corpus is synthetic, so we're not making any fairness claims on real hiring data. And this is a screening aid, not an automated decision-maker — a human still needs to review the evidence."

### Slide 14 — Thank You
**Content:** "Thank You" · "Questions & Discussion" · presenter contact line.
**Visual:** Left-aligned closing block, mirrors Slide 1's typography and accent rule.
**Script:** "Thank you so much — happy to take any questions."
