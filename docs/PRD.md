# PRD — Jev-Based Applicant Screening & Ranking Tool

**Status:** Living document
**Last updated:** 2026-10-24

## 1. Purpose

Use Jev (TypeSafe AI's "System One Model") to **screen and rank real job applicants
against real job postings**. The pipeline is both a practical screening tool and
the empirical basis for an active IEEE-journal-style research paper
(`docs/main.tex`, "Reliable Structured-Decision Applicant Ranking with a
Non-Autoregressive System One Model") — §7's prior note that the research-paper
framing had been "retired" is **no longer accurate** and is corrected below (§2c).

For each job profile, candidates clearing a local pre-filter (§2b) get assessed
by Jev against a rubric extracted from that job's own posting text. Ranking is
primarily a transparent, per-criterion evidence score computed from Jev's own
answers (§2b); Jev's pairwise judgment (active tournament, Plackett-Luce) only
refines ordering among candidates that score sits too close to call.

## 2. What changed to get here

The project started from a research pipeline built around **synthetic** job
postings with a categorized must-have/nice-to-have skill split, an
embedding-based shortlist gate, a two-call job-profile extraction (skills +
rubric), a separate total-years-of-experience question, and a
reliability/audit evaluation harness. Over this rework:

- **Synthetic → real job postings.** The 10 job profiles are now real postings
  sourced from `job-profile/clean_jobs.csv` (a 2025 Kaggle LinkedIn Data Jobs
  dataset, 1,048 postings), matched by keyword to a target role category.
- **Must-have/nice-to-have skills → one `technical_skills` list.** No more
  hard gate requiring every must-have skill to match before a candidate is
  even assessed.
- **Shortlisting removed, then reintroduced (§2b) in a simpler form.**
- **Job-profile extraction simplified to one LLM call.** `job_profile.py`
  runs one rubric-generation call (Qwen3-14B, `reasoning=False`), focused
  exclusively on technical skills/tools/technologies.
- **4-level → 3-level evidence strength.** `strong` / `partial` / `none`.
- **`total_years_criteria` / `total_relevant_years` removed.** No separate
  years-of-experience question or field.
- **`id`/`title` redundancy removed.** `JobProfile` has only `id` (a slug).
- **Model: `qwen2.5:14b-instruct-q4_K_M` → `qwen3:14b` with `reasoning=False`**
  for both rubric and CV-profile extraction, for throughput.

## 2b. Fairness & ranking rework (2026-10-22)

Driven by a review of ranking fairness given Jev is the sole judge (see §5).

- **Probabilistic criterion scoring.** `scoring/criterion_score.py`:
  `RequirementEvaluation.probabilities` captures Jev's full
  strong/partial/none distribution; `expected_strength()` computes
  `Σ P(level) × weight(level)` instead of collapsing to one label.
- **Tiered ranking: transparent score primary, Jev pairwise only for ties.**
  `scoring/active_pairwise_ranking.py::rank_active_pairwise`: (1) computes
  `criterion_score` per candidate; (2) groups candidates into contiguous
  "tie clusters" via anchor-based grouping within `tie_epsilon`; (3) runs
  the active pairwise Jev tournament (Plackett-Luce) **only inside a tie
  cluster**; (4) folds the resulting utility back in as a bounded nudge
  (±`tie_epsilon`/4) on top of `criterion_score` — enough to reorder within
  a cluster, never enough to cross into a neighboring one.
- **Noise floor grounded in measurement, not guesswork.**
  `JEV_PROBABILITY_NOISE_FLOOR = 0.05`, measured empirically: the **same**
  question asked **3 times** (not 5 — an earlier draft of this document and
  of the paper both had this wrong; corrected 2026-10-24) on the same
  CV/criterion produced a probability spread up to 0.05 on an ambiguous
  case. `tie_epsilon` defaults to `max(1/len(evaluation_criteria),
  JEV_PROBABILITY_NOISE_FLOOR)`.
- **Self-ensemble capability, off by default.** `ensemble_repeats=1`
  (`CANDIDATE_RANKING_ENSEMBLE_REPEATS` to raise it).
- **Evidence-order (cosmetic) bias found and fixed.** Jev's pairwise verdict
  flipped on 1/16 sampled pairs (6.25%) purely from reordering identical
  evidence. Fix: evidence always rendered in canonical (alphabetical) order,
  plus an explicit prompt instruction that order carries no priority.
  Re-verified at full coverage (190 pairs): 1.6% residual.
- **Shortlist gate v2.** `scoring/shortlist.py::passes_shortlist_gate`: a
  candidate needs ≥5 (or all, if fewer than 5 exist) of the job's
  `technical_skills` matched — literal mention, a small implied-skill alias
  table (Django→Python, Spring Boot→Java, React→JavaScript,
  Kubernetes→Docker), or embedding cosine similarity
  (`all-MiniLM-L6-v2`) — before Jev assesses them at all for that job.

## 2c. 2026-10-24 session: full corpus run, evidence-grounding rewrite, observability, and paper reconciliation

This session ran the pipeline against the **full real corpus for the first
time** (previous status was "not run") and made several further changes,
all now reflected in both the code and `docs/main.tex`.

- **Shortlist embedding threshold raised 0.4 → 0.55.** Calibrated against
  more real examples than the original handful; this document's and the
  paper's earlier references to 0.4 were stale and have been corrected.
- **Evidence grounding rewritten from LLM-extraction to BM25 + literal +
  cosine fallback.** The prior design asked an LLM to extract a
  per-skill "evidence" excerpt from each CV as a separate, cached step —
  prone to going stale relative to the rubric and responsible for a 56.3%
  empty-snippet rate. `scoring/assessment.py::_find_evidence_sentence` now
  grounds every snippet directly in the candidate's own CV text: split into
  sentences (shared splitter with the shortlist gate, so a dotted
  identifier like `Node.js` isn't torn apart), then (1) a per-word literal
  match across all of a requirement's words, joining up to 3 matching
  sentences ranked by BM25 when more than one matches, or (2) cosine
  similarity (same model/threshold as the shortlist gate) if no word
  matches at all. Eliminates an entire class of cache-invalidation bugs,
  since nothing is cached independently of the CV text and the rubric.
- **`max_iterations` config parameter and `CONVERGENCE_SLACK_ITERATIONS`
  removed.** Both were added mid-session by a concurrent process, then
  explicitly reverted by user decision after discussion: the iteration
  budget for the pairwise tournament is simply `2 * min_iter`
  (`min_iter = ceil(n / subset_size)`) with no fixed per-run cap and no
  slack constant. `CANDIDATE_RANKING_PAIRWISE_MAX_ITERATIONS` no longer
  exists as an env var.
- **`tie_epsilon`'s third, dynamic `job_noise_estimate` term removed.**
  Same story: added mid-session, then explicitly reverted. `tie_epsilon`
  is back to the original 2-term `max(1/n_criteria, 0.05)` — no per-job
  dynamic widening based on that job's own measured ambiguity.
- **Langfuse cost tracing added**, gated on `LANGFUSE_PUBLIC_KEY` being set
  (`scoring/jev_client.py::_tracing_enabled`/`_traced_generation`). Every
  Jev call is traced with input-token count and resolved cost. As a direct
  consequence, **the `model` field in Jev's response (the resolved
  snapshot version, e.g. `jev-1.14.0`) is now actually captured** —
  correcting an earlier claim (in this document and in the paper) that the
  client doesn't log it and so can't retroactively confirm which snapshot
  answered.
- **Job profile corpus changed from 10 to 10, with different members.**
  `frontend-engineer` (a React Developer posting that never cleared ≥5
  matched skills against this corpus — 0 candidates shortlisted, a dead
  job profile) was replaced, after two more failed candidates
  (`Senior C# Developer`/.NET, `Junior Frontend Developer`/Angular — both
  also 0 shortlisted) and one redundant-with-`data-engineer` candidate
  (`Principal Data Engineer`, withdrawn per user request to avoid two
  "Data Engineer"-themed profiles), with `Senior Software Engineer,
  Systems Infrastructure` (LinkedIn) → `systems-infrastructure-engineer`
  (33 shortlisted). `data-scientist-engineer` was renamed to
  `data-scientist` (pure rename; its cached rubric and 9 assessments were
  migrated to the new id rather than re-spent against Jev, since nothing
  about the job's content changed).
- **Full-corpus run completed**, multiple times, as the job-profile list
  above was iterated on. Final shortlist-gate pass counts (out of 301
  candidates each): see `docs/main.tex` Table `tab_shortlist` — 294 of
  3,010 possible (job, candidate) pairs pass (9.8%), a 90.2% reduction in
  paid assessment calls.
- **Reliability pilot re-run under the current code.** Full-ranking
  repeatability (3 independent full-tournament re-runs, Kendall's τ) was
  checked at full coverage on two job profiles: `devops-engineer` (86
  candidates, τ=1.000, 150 Jev calls — **measured, kept in local
  artifacts, but by explicit user decision not reported in the paper**,
  to avoid framing job-profile selection by "most candidates" as the
  justification) and `data-engineer` (66 candidates, τ=0.981, 143 Jev
  calls, tournament converged within budget in every repeat — **this is
  the one reported in `docs/main.tex`**).
- **Total measured Jev spend to date: $0.10** (1,484 calls, 2,383,177
  input tconfig.okens, $0.042/M input, zero-cost output), read directly from
  Langfuse rather than estimated — now stated in the paper's Table
  `tab_config`.
- **Evidence-snippet auditability gap re-measured** against the current
  (BM25-based) grounding and the current corpus: 645 of 2,411
  strong/partial evaluations have an empty snippet (26.8%), 98.1% of the
  filled ones from an exact word match. (The 20.5%/97.4% figures from an
  earlier measurement, both in this document's history and in the paper,
  are superseded.)
- **`main.tex` recovery incident.** The paper's source file was
  accidentally overwritten on disk with a single stray line of unrelated
  text (editor mishap, not a code or git issue). Recovered losslessly from
  two independent sources — this session's own transcript history and the
  user's own last-saved copy (`docs/main (1).tex`) — which agreed byte-for-
  byte. No content was lost; flagged here only so a future reader isn't
  confused by `main (1).tex` sitting alongside `main.tex` in `docs/`.

## 3. Current architecture

```
job-profile/*.txt  ──┐
                      ├─> ingestion (jp.py / cv.py)
cv/*.pdf          ──┘
                      │
                      v
         job_profile.py: build_rubric_chain (Qwen3-14B, reasoning=False)
         ──> JobProfile.evaluation_criteria + technical_skills
                      │
                      v
         shortlist.py: passes_shortlist_gate — local, zero Jev cost.
         Min 5 (or all, if fewer exist) technical_skills matched (literal /
         implied-skill alias / embedding cosine ≥0.55, all-MiniLM-L6-v2)
         per candidate per JP. Below the bar → no assessment for that JP.
                      │
                      v (candidates that cleared the gate)
         assessment.py: _build_jev_questions → Jev (one call per
         candidate × job profile, all criteria bundled, traced via
         Langfuse if LANGFUSE_PUBLIC_KEY is set) → RequirementEvaluation
         (strong/partial/none + probabilities); evidence snippet grounded
         directly in CV text via _find_evidence_sentence (BM25 + literal
         + cosine fallback, §2c) — not cached separately, not LLM-extracted
                      │
                      v
         criterion_score.py: expected_strength (probability-weighted) per
         criterion → criterion_score (primary, auditable ranking signal)
                      │
                      v
         active_pairwise_ranking.py: tie-cluster candidates within
         tie_epsilon = max(1/n_criteria, 0.05); Jev pairwise tournament
         (active variance-based subset selection + Plackett-Luce,
         iteration budget = 2 * ceil(n/subset_size), no fixed cap) runs
         ONLY inside a tie cluster, folded back as a ±tie_epsilon/4 nudge
         on criterion_score — not the primary signal
                      │
                      v
         runs/<run_id>/<jp_id>/{ranking.json,ranking.md}
         + console-web/src/data/real-data.json (dashboard export)
```

Single CLI command: `uv run python -m candidate_ranking.cli run [--run-id ID]`.
Scope controls: `CANDIDATE_RANKING_MAX_JPS`, `CANDIDATE_RANKING_MAX_CANDIDATES`,
`CANDIDATE_RANKING_SHORTLIST_MIN_SKILLS` (default 5),
`CANDIDATE_RANKING_ENSEMBLE_REPEATS` (default 1) (env vars; scope vars unset = full corpus).
Optional: `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_HOST` for cost tracing.

**`console-web`** (the dashboard app) is **present and active** —
`output/console_export.py` writes `console-web/src/data/real-data.json` on
every run; the 2026-10-22 note below (§4 history) that it had been deleted
is stale.

## 4. Current status (as of 2026-10-24)

| Item | Status |
|---|---|
| Schema simplification (must-have removal, 3-level evidence, no total-years, id-only) | **Done** |
| Real job postings (10 profiles from `clean_jobs.csv`) | **Done** — composition has changed twice since 2026-10-22; see §2c for the current list |
| Job-profile rubric extraction (all 10 JPs) | **Done** — cached in `runs/_cache/job_profile.json` |
| Model: `qwen3:14b`, `reasoning=False` | **Done** |
| Probabilistic criterion scoring (§2b) | **Done** |
| Tiered ranking: criterion_score primary, Jev pairwise for ties only (§2b) | **Done** — exercised on real multi-candidate tie clusters in the full-corpus run (§2c), not just unit tests |
| Jev probability noise floor (§2b) | **Done** — measured at 0.05 from 3 repeated calls (corrected from an earlier "5") |
| Cosmetic/evidence-order bias (§2b) | **Found and fixed** — 6.25%→1.6% at full coverage (190 pairs) |
| Shortlist gate v2 | **Done, run at full scale** — threshold raised to 0.55 (from 0.4); 294/3,010 pairs pass (9.8%) |
| Evidence grounding | **Rewritten** (§2c) — BM25 + literal + cosine fallback, directly off CV text; 26.8% of strong/partial labels still have an empty snippet |
| Full-corpus run (10 job profiles × 301 CVs) | **Done** — multiple times, as the job-profile list was iterated on; latest run id `20261024-142003` |
| Full-ranking repeatability (Kendall's τ, 3 re-runs) | **Done** on two job profiles — `devops-engineer` (τ=1.000, not in paper) and `data-engineer` (τ=0.981, in paper) |
| Langfuse cost tracing | **Done** — $0.10 total spend to date, captured per-call including resolved model version (`jev-1.14.0`) |
| Research paper (`docs/main.tex`) | **Active, primary deliverable** — not retired; reconciled against all of the above as of 2026-10-24 |
| `README.md` | Broadly accurate on architecture; does not yet mention Langfuse tracing or the current job-profile list |
| `console-web` | **Present, active**, not deleted |
| Fairness mitigations (PII redaction, evidence-length capping) | **Still deferred** |
| Golden-set human calibration of Jev's rubric judgment | **Not done** — no ground-truth/hiring-outcome validation exists, only internal consistency checks (paper Section IV, Limitations) |
| Divergent-profile tie detection | **Explicitly deferred** — unchanged |

## 5. Known risks (no mitigation beyond what's listed)

- **Single judge, no cross-check.** All assessment and ranking decisions come
  from Jev alone. Position bias (hash-based A/B shuffle) and evidence-order
  bias (§2b, measured and fixed) are mitigated, and the pairwise tournament's
  blast radius is bounded to within a tie cluster, but there is still no
  independent second model/judge.
- **No PII/bias-signal redaction.** Candidate names, contact info, and
  graduation years go to Jev as-is. Still the single highest-leverage
  fairness gap open.
- **No ground-truth / human-hiring-outcome validation.** Everything measured
  (§2b, §2c, and `docs/main.tex` Section IV) is about *consistency*, not
  *accuracy*. A golden-set calibration was proposed and not implemented.
- **Shortlist gate false negatives.** The embedding threshold (now 0.55) is
  calibrated against real examples but not a systematic ground-truth sample.
  A false negative here means a candidate is silently never assessed by Jev
  at all for that job, with no later opportunity to be caught.
- **Evidence-snippet auditability gap.** 26.8% of strong/partial evaluations
  have no verbatim CV excerpt behind them — a reviewer cannot audit why that
  label was given, though `criterion_score` itself never used the snippet as
  an input.
- **Divergent-profile ties.** Plackett-Luce (and the tie-cluster mechanism
  built on it) assumes a single latent "fit" scale; two candidates strong in
  different individual criteria can still be forced into one pairwise
  "winner." Explicitly deferred as not worth the added complexity right now.
- **Full-ranking repeatability is not a general guarantee.** τ=1.000 on one
  job profile (`devops-engineer`) and τ=0.981 on another (`data-engineer`)
  were both measured at full coverage, but only two of ten job profiles have
  been checked this way, and a perfect τ on one profile does not predict the
  result on another (see `docs/main.tex` Limitations).

## 6. Scope of remaining work

1. Close the evidence-snippet auditability gap (26.8% empty) without
   touching the ranking signal itself (`criterion_score` doesn't consume it).
2. Extend full-ranking repeatability and the other three reliability checks
   (tiering-validity, repeat-call, cosmetic-order) to the remaining eight
   job profiles — `scripts/run_reliability_suite.py` already supports a
   single-job-profile filter specifically to make this cheap to do.
3. Update `README.md` to mention Langfuse tracing and the current
   job-profile list.
4. Golden-set human calibration of Jev's rubric judgment — proposed,
   not scheduled.
5. Revisit PII redaction / evidence-length capping — still the largest
   open fairness gap.
6. Revisit divergent-profile tie detection if real tie-cluster data shows
   it happening often enough to matter.
7. Decide whether to re-run the shortlist-embedding-threshold calibration
   (0.55) against a larger, more systematic sample than the handful of
   real examples it was tuned on so far.

## 7. Out of scope

- `candidate.skills` (a separate LLM skill-extraction call per candidate) and
  `must_have_skills`/`nice_to_have_skills` on `JobProfile` — these stay
  removed per explicit user confirmation, even though shortlisting itself
  was reintroduced (§2b) in a different, simpler form that doesn't depend
  on either.
- A second, independent LLM judge for cross-checking Jev's assessments —
  not planned; the mitigation strategy instead bounds Jev's own blast
  radius (tiered ranking, §2b) and measures its own consistency (§2c,
  paper Section IV) rather than adding a second model.

**Correction to this document's own history:** an earlier version of §1/§7
stated that the research-paper framing had been "retired" and that reviving
it "would mean rebuilding ... the paper documents from git history." That
was true as of 2026-10-22 but is no longer the case — the paper
(`docs/main.tex`) has been the subject of continuous, active work since,
and is reconciled against the current codebase as of this update (§2c).
