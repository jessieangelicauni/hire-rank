# Jev vs. Paper 1: Comparison Notes

Date: 2026-09-18

Paper 1 ("Hallucination-Aware Self-Correction for LLM-Based Applicant
Assessment", `docs/paper1.tex`) documents the free-text LLM + tournament
pipeline this repository used before the Jev migration. This document
compares that system against the Jev-based replacement, using Paper 1's
own already-published results as the baseline rather than re-running the
old pipeline (see "On head-to-head reproduction" below for why).

## 1. Architecture

| Dimension | Paper 1 (prior system) | Jev (current system) |
|---|---|---|
| Core mechanism | Free-text LLM generation (Qwen2.5-14B-Instruct, Ollama) + active listwise tournament + Plackett-Luce aggregation | Single-call structured decision model (Noul / Choice / Score) |
| Output shape | Prose: strengths, weaknesses, additional skills | Structured: overall_fit_score, overall_recommendation, meets_min_qualifications, per-requirement scores, confidence |
| Hallucination handling | Live contradiction-check-and-retry loop (lexical + embedding heuristic against the applicant's own extracted skills) | Not applicable by construction -- output space is predefined, so a schema violation is impossible rather than statistically reduced |
| Applicant-identifier robustness | Positional tokens + schema-constrained decoding + retry (needed because free-text identifiers were unreliable with a 14B local model) | Not applicable -- one call per candidate, no multi-candidate identifier passing |
| Ranking | Iterative: MC-KG subset sampling -> LLM listwise judgment -> Plackett-Luce refit, repeated until convergence or budget exhausted | Direct: sort by `overall_fit_score` from a single call per candidate |
| Calls per (job, applicant) pair | 1 assessment call + this pair's share of N tournament rounds (iterative, shared across the job's shortlist) | 1 call (all questions answered in parallel within it) |
| Generation model | Qwen2.5-14B-Instruct, 4-bit, self-hosted via Ollama | `jev-latest`, hosted, TypeSafe AI |

## 2. Metrics (not directly comparable -- see caveat below)

| Metric | Paper 1 | Jev |
|---|---|---|
| Mean Faithfulness (strengths, LLM-judge groundedness) | **0.880** (n=300 stratified items, 10 job profiles) | N/A -- no free text to audit |
| Kendall-tau convergence (rank stability across tournament iterations) | **0.957** mean, range 0.930 (frontend-engineer, n=7) -- 0.979 (data-engineer, n=77) | N/A -- no iterative ranking process |
| Test-retest reliability (repeat-call stability) | Not measured | overall_fit_score stdev mean=0.80 (0-100 scale) across 338 pairs x 3 repeats |
| Internal coherence | Not measured directly (contradiction audit is the closest analog) | Spearman rho=0.667 (mean requirement score vs. overall score, p=6.7e-45) |
| Generation failure rate | 0/1,713 tournament ranking calls invalid; 0/1,268 residual weakness contradictions after retry | 0/338 assessment failures in the full-corpus run (`runs/20260918-072318/summary.json`) |
| Corpus | 500 resumes, 10 job profiles -> 348 shortlisted pairs (237 distinct applicants) | 501 CVs, 10 job profiles -> 338 shortlisted pairs |
| Human-expert validation | Not performed (explicitly named as future work) | Not performed |

**Caveat, stated plainly:** Faithfulness audits whether a *textual claim* is
entailed by the resume; Jev's reliability/coherence metrics audit whether a
*structured decision* is stable and internally consistent. These are
different constructs measuring different failure modes, not two readings of
the same underlying quantity. Neither number can be used to claim one
system is "more accurate" than the other -- that comparison would require
independent human-expert ranking, which neither system currently has.

## 3. A cross-system finding: full-stack-engineer is the weak point in both

Paper 1's per-role Faithfulness scores rank **full-stack-engineer lowest**
(0.764, vs. a 0.880 run-wide mean). Independently, the Jev evaluation on
this repo's current corpus also found full-stack-engineer among the
weakest roles: mean `overall_fit_score` 34.2 and 0% of its shortlisted
candidates meeting minimum qualifications (see the shortlisting
investigation in this session -- shortlisted candidates matched generic
skills like GitHub Actions/Kubernetes/AWS but scored zero on React/Next.js/
PostgreSQL, the role's actual core skills).

Two independently-built systems, scoring the same job profile against
highly similar corpora, converge on the same role being the weakest.
This is evidence the shortlisting stage (`scoring/skills.py`,
`shortlist_candidates`, unchanged since before Paper 1) is letting through
poorly-matched candidates for full-stack-engineer specifically --
independent of which assessment system consumes its output. Worth stating
in any write-up: this is a shared pipeline weakness, not evidence for or
against either assessment approach.

## 4. Draft comparison paragraph (for reuse in a paper)

> Unlike the retry-based self-correction mechanism in prior work, which
> mitigates hallucination in free-text claims through a live
> contradiction-check-and-regenerate loop, the Jev-based approach
> eliminates the hallucination failure mode by construction: output is
> constrained to a predefined answer space (Noul/Choice/Score), making a
> schema violation mathematically impossible rather than statistically
> reduced. This trades away free-text explanatory capacity
> (strengths/weaknesses narratives) for guaranteed type safety and a
> reduction from one assessment call plus a shared iterative tournament
> budget to a single call per candidate. Both systems share an unaddressed
> validation gap: neither has been evaluated against independent
> human-expert judgment. Notably, both systems -- despite differing
> architectures and being evaluated independently -- identify the same job
> profile (full-stack-engineer) as the weakest-performing case, suggesting
> the shortlisting stage shared by both pipelines, rather than either
> assessment mechanism, is the limiting factor for that role.

## 5. On head-to-head reproduction

We did not re-run the old Qwen2.5-14B + tournament pipeline for a live
side-by-side benchmark. Doing so would require reconstructing the removed
free-text assessment chain (`ASSESSMENT_GENERATION_PROMPT`,
`_GeneratedAssessment`, `build_assessment_chain`) as a standalone module,
since `models.Assessment` no longer carries the `strengths`/`weaknesses`
fields `ranking/tournament.py`'s `_render_candidates_text` needs -- a
non-trivial resurrection of a whole removed subsystem, not a quick
re-run. Paper 1's published Faithfulness and Kendall-tau figures are the
authors' own rigorously-obtained results (not a vendor claim), so they are
used directly as the baseline instead, with the non-comparability caveat
above stated explicitly wherever these numbers are cited together.

## Sources

- Paper 1: `docs/paper1.tex`, Sections IV-V (Experimental Setup, Results and Discussion)
- Jev evaluation: `runs/20260918-072318/evaluation/report.{json,md}`, produced by `scripts/run_jev_evaluation_study.py` + `scripts/analyze_jev_evaluation_study.py`
- Jev full-corpus run: `runs/20260918-072318/summary.json`
- Shortlisting investigation: this session's conversation (2026-09-18), corroborated by `runs/20260918-072318/{data-engineer,full-stack-engineer,java-developer,frontend-engineer}/assessments.json`
