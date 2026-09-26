# ICAIDES 2026 — Speaker Script
**Reliable Structured-Decision Applicant Ranking with a Non-Autoregressive System One Model**
14 slides. Simple language, straight to the point, one idea per slide —
mirrors the paper section by section. Every sentence explains what
something is, why it matters, or how it works. No filler.

---

**Slide 1 — Title**
Good [morning/afternoon]. I'm Jessie Angelica, with Hasanul Fahmi Zuhri, Rusdianto Roestam, and Muhammad Arief. Our question: can an AI model that only picks structured answers — not free text — rank job applicants better than models that write paragraphs?

## Section I — Introduction

**Slide 2 — Motivation & Gap**
Most AI hiring tools read a resume and write a paragraph about the candidate. The problem: the model can write things the resume does not support. This is called hallucination, and there is no confidence score to catch it. In September 2026, TypeSafe released Jev, the first "System One Model." At that point, no one outside TypeSafe had tested it independently — that is the gap this paper fills.

**Slide 3 — Objective**
Jev is TypeSafe's model, not ours. Our job is to make it work reliably in a real hiring pipeline. Three contributions: better questions for it to answer, a faster and cheaper pipeline, and proof its answers stay stable when repeated.

## Section II — Literature Review

**Slide 4 — What Is a System One Model**
Normal AI models write text one word at a time, which is why they can drift into saying things that are not true. Jev does not write text at all — it answers typed questions in one parallel pass, each with a confidence number. It only answers in three formats: yes/no, multiple-choice, or a rating scale. This paper uses the rating scale for everything.

**Slide 5 — Why It Works**
Jev judges each point on its rating scale on its own, without seeing the other numbers — so a bare number like "50" gives it nothing to check. Describe that same point as a real situation instead, and Jev has resume evidence to check it against. Jev is trained differently too: rewarded for making its confidence match reality, not for matching human preference or getting a checkable answer right. That is called calibration — it is not the same as being correct.

## Section III — Methodology

**Slide 6 — Architecture**
Here is the full pipeline. Extraction and shortlisting are unchanged from our earlier system. Then Jev runs once per candidate, in parallel — that is contribution one. Then we sort directly by score instead of running a tournament — that is contribution two. A score breakdown test runs offline afterward, just to double-check the shortlisting filter.

**Slide 7 — Criteria-Grounded Question Design**
Our first contribution. Instead of asking Jev to rate a match from 0 to 100, we describe each rating level as a real situation — "this skill is only listed" versus "this skill was actually used in a real job." Jev judges each level alone, so it needs something real to check.

**Slide 8 — Direct Sort Ranking**
Our second contribution. The old tournament ranked candidates round by round, each round waiting for the last one to finish. Jev gives each candidate one score directly, so we just sort by score — every candidate's score can be computed at the same time.

## Section IV — Experimental Setup

**Slide 9 — Experimental Setup**
Here is what we tested: 273 shortlisted candidate-job pairs, across 10 job postings, from a pool of 501 resumes — 500 synthetic plus one real one. We ran each pair through Jev three separate times to check consistency. We changed exactly one thing between the two systems: how candidates get scored. Everything else, including which skills count as must-have, stayed fixed.

## Section V — Results and Discussion

**Slide 10 — Result: Confidence Gain**
Writing each level as a real situation, instead of a bare number, raised confidence from 0.672 to 0.894 for individual requirements, and from 0.635 to 0.766 for experience and education. Both are big, reliable improvements.

**Slide 11 — Result: Efficiency**
We used less than half the model calls — 273, versus about 2,061 for the old system — and about 4.5 times lower latency per candidate. Cost is priced at zero by design, not billed per token — that's TypeSafe's claim on price, we did not measure it ourselves.

**Slide 12 — Result: Calibration & Stability**
Repeating each call three times gave matching rankings 96.5% of the time — Kendall's tau of 0.965, versus about 0.957 for the old system. And Jev's confidence, 0.894, holds up well against the old system's Faithfulness score of 0.880 — though these measure different things, so this isn't a claim that Jev is "more accurate."

## Section VI — Conclusion

**Slide 13 — Conclusion**
Three things to remember: confidence went from 0.672 to 0.894; the pipeline needs less than half the calls and runs about 4.5 times faster; and repeated rankings matched 96.5% of the time. Three honest gaps: calibration isn't the same as correctness, so the next step is checking against real known-correct answers; we only tested a 5-level scale, and Jev supports up to 10; and this is one hiring pipeline — testing it elsewhere would make the case stronger.

**Slide 14 — Thank You**
Thank you.
