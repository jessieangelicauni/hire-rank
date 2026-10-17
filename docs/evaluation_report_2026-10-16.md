# Evaluation Report — Candidate Ranking Pipeline

**Date:** 2026-10-16
**Run evaluated:** `20261016-064409` (151 assessments, 0 failures, 10/10 JDs with candidates)
**Model:** `qwen2.5:14b-instruct-q4_K_M` via local Ollama
**Scope:** ranking stability (Kendall-tau + rank-biased overlap), qualification-gate
test-retest reliability (Cohen's kappa), latency, compute cost, and **four**
reliability/correctness bugs found and fixed during this investigation.

**On ground truth:** this system has none — no recruiter-labeled "correct" ranking
or "correct" qualify/disqualify decision exists to score against. Every metric
below is therefore a **self-consistency** measure (does the same judge reach the
same conclusion given the same input, run after run?), not an accuracy measure.
High self-consistency is necessary but not sufficient for correctness — a model
can be confidently, consistently wrong. It is, however, the only thing measurable
without human labels, and low self-consistency is sufficient on its own to show a
result can't be trusted, regardless of how plausible any single run looks.

---

## 1. Summary

The pipeline's final run produced assessments for all 10 job descriptions with zero
generation failures. Digging into ranking quality surfaced **four** real bugs,
all now fixed and verified:

1. **Stale assessment cache pollution** (§3.1) — `evaluate-stability` blindly loaded
   every assessment ever cached for a JD, including entries scored against an old,
   since-edited job profile. Mixing one stale, inconsistent candidate into a ranking
   subset confused the LLM badly enough to collapse `it-security-engineer`'s
   stability from τ=0.543 to effectively random.
2. **No escape from an unrankable subset** (§3.2) — when the LLM failed to produce
   a valid permutation for a subset, the active-learning loop re-selected the
   *exact same* subset next iteration, so a single hard subset could burn the
   entire 30-iteration budget with zero progress.
3. **Disqualified candidates were never excluded from ranking** (§3.3) — the
   qualification gate's `qualified` boolean (computed from its own
   non-compensatory must-have rule) was never read anywhere downstream. On the
   evaluated run, candidates who explicitly **failed a must-have requirement**
   ranked as high as **#8 out of 24** for `full-stack-engineer` — above multiple
   candidates who actually qualified. This is a correctness bug independent of
   ranking stability, and fixing it was the single biggest stability lever
   measured — until §3.4 surpassed it.
4. **Small candidate pools settled for a single LLM observation** (§3.4) — any
   JD with ≤10 qualified candidates (half of them, after fixing #3) took a
   shortcut: one BM25 warm-start plus exactly one LLM ranking call, no
   refinement. With nothing to average a `temperature=0.3` sample against, this
   was the noisiest path in the whole algorithm — it's what made
   `devops-engineer` the single worst JD (τ=0.619) right after fixing #3.

Mean ranking stability (Kendall-tau) across all 10 JDs moved
**0.810 → 0.863–0.867 → 0.918 → 0.939** across the four fixes; mean RBO reached
**0.915**. The previously-worst JD (`it-security-engineer`) went from
**0.543 (unstable) → 0.978 (stable)**, `full-stack-engineer` — the JD with the
worst disqualification contamination (13 of 24 candidates, 54%) — reached
**0.986**, and `devops-engineer` recovered from its post-fix-#3 low of 0.619 to
**0.810** once it stopped settling for one LLM call.

Two evaluation metrics beyond Kendall-tau were added in this session, both chosen
specifically because they don't require ground truth (§2, §4.1, §4.1c):
**rank-biased overlap (RBO)**, which fixes Kendall-tau's blind spot for top-of-list
agreement, and **qualification-gate test-retest reliability (Cohen's kappa)**,
which checks the underlying assessment the ranking is built on, not just the final
sort order. The qualification gate came back **near-perfect (κ=0.865)** —
genuinely reassuring, since it means the ranking-stability numbers above are
measuring real ranking noise, not noise inherited from an unreliable assessment
stage underneath it.

---

## 2. Methodology

- **Ranking stability**: `uv run python -m candidate_ranking.cli evaluate-stability`.
  For each JD, the active-listwise ranking algorithm (BM25 warm-start + iterative
  LLM subset ranking + Plackett-Luce fit) is run **twice** independently on the same
  cached assessments, and the two resulting utility rankings are compared with:
  - **Kendall's τ** — agreement across *every* pair of ranks, equally weighted.
    τ=1.0 means perfect agreement; τ<0.75 is classified unstable.
  - **Rank-biased overlap (RBO)**, Webber/Moffat/Zobel 2010 — agreement weighted
    toward the *top* of the list (geometric decay, `p=0.9`). Added in this session
    because τ has a real blind spot here: two runs that agree completely on the
    top 5 but shuffle the bottom half of a 24-candidate pool score about the same
    under τ as two runs that *disagree* on the top 5 but agree on the tail — not
    what matters for a shortlist a recruiter will only read the top of.
- **Qualification-gate reliability**: `uv run python -m candidate_ranking.cli
  evaluate-reliability`. Samples (JD, candidate) pairs from the cache, re-runs the
  qualification gate `--repeats` times (default 3) per pair with **caching bypassed
  entirely** — every repeat is a fresh LLM call — and measures:
  - Raw agreement rate and **Cohen's kappa** on the `qualified` boolean (kappa
    corrects for the agreement you'd expect purely from how often the model says
    "qualified" overall, so a model that qualifies 95% of candidates doesn't score
    high just by chance).
  - Exact-match agreement rate on the per-requirement verdict
    (MET/PARTIALLY_MET/NOT_MET) across repeats.
  This is the standard inter-rater-reliability approach to evaluating an LLM judge
  when no ground-truth labels exist: treat the model as a rater and check it
  against *itself* under identical conditions, since there is no second rater and
  no answer key to check it against instead.
- **Latency**: each LLM call type was timed directly against the real pipeline
  chains (`build_job_profile_chain`, `build_skill_extraction_chain`,
  `build_qualification_chain`, `_rank_subset`) with the model already warm, single
  representative calls per type, local Ollama server.
- **Cost**: fully local inference (Ollama) — $0 API cost. "Cost" reported as
  wall-clock compute time, derived from observed per-call latency × call counts at
  `CANDIDATE_RANKING_OLLAMA_NUM_PARALLEL=4`.

---

## 3. Root-cause investigations

### 3.1 Stale assessment cache pollution

**Symptom:** `it-security-engineer`'s active ranking failed validation on nearly
every attempt, exhausting all 30 iterations in both stability runs, producing
τ=0.543 — the only JD in the "unstable" band.

**Investigation:** reproducing the exact failing subset showed one candidate's
assessment (`cv-00217`) had been scored against the *original* must-have list
(`Wireshark, Nmap, Splunk`) from before this session's JD edit to
`Metasploit, Python`. `evaluation/ranking_stability.py`'s `load_cached_assessments`
reads every `*.json` file under `runs/_cache/assessments/<jd_id>/` with no check
against the JD's current job profile — a leftover file from before an edit stays
in the pool forever. A controlled test confirmed it: removing the one stale file
and reproducing the subset gave a **valid permutation 5/5 trials at both
subset_size=8 and subset_size=10** — subset size was never the problem.

A repo-wide scan found a second instance (`frontend-engineer/cv-00370.json`,
scored against an even older, pre-fix must-have list). Both stale files removed.

**Fix** (`src/candidate_ranking/evaluation/ranking_stability.py`):
- `load_cached_assessments` now loads the current job-profile cache
  (`load_job_profiles`) and skips any assessment whose evaluated must-have
  requirement set doesn't match the JD's current `must_have_skills` — logged as a
  warning, not silently dropped.
- New tests: `test_load_cached_assessments_skips_entries_scored_against_a_stale_profile`.

### 3.2 Active-ranking loop could not escape an unrankable subset

**Symptom:** independently of §3.1, `frontend-engineer`'s second stability run (with
entirely clean data) got stuck for the full 30 iterations, returning the identical
invalid permutation (`[1, 5, 7, 8, 9, 10, 2, 3, 5, 6]` — "5" twice, "4" missing)
every single attempt.

**Root cause:** `_select_uncertain_subset` deterministically picks the top-K
highest-variance candidates. When `_rank_subset` fails validation, the utility
vector `u` and its variances are *unchanged* (nothing was appended to `rankings`),
so the next iteration recomputes the identical top-K subset and sends the LLM the
byte-identical prompt. For a subset the 14B quantized model is confidently wrong
about, this becomes an infinite loop that burns the whole iteration budget on one
combination while every other candidate's uncertainty goes unexamined.

**Fix** (`src/candidate_ranking/scoring/active_listwise_ranking.py`):
- `_select_uncertain_subset` now takes an `avoid_signature` parameter; if the
  freshly-selected top-K subset matches the one that just failed, it swaps the
  lowest-variance member for the next-best candidate not already in the subset —
  guaranteeing the next LLM call sees a genuinely different prompt.
- The main loop tracks `last_failed_signature` and passes it through.
- New tests: `test_select_uncertain_subset_avoids_repeating_a_failed_combination`,
  `test_select_uncertain_subset_unaffected_when_signature_differs`,
  `test_active_loop_does_not_get_stuck_on_one_unrankable_subset`.

**Verified impact:** re-running `evaluate-stability` with both fixes produced
**zero validation failures across all 10 JDs** (down from several per run
previously) — the loop now reliably explores different subsets instead of
getting stuck.

### 3.3 Disqualified candidates were ranked alongside qualified ones

**Symptom:** none visible in any metric in §4.1 directly — this was found by
going back to first principles and asking "does the ranking output actually
respect the qualification gate's own verdict?", not by chasing a stability
number. It shows up as unusually *high variance* in exactly the JDs with the
most disqualified candidates mixed in (see below), but nothing flags it as a
distinct problem unless you look at the ranked output directly.

**Investigation:** `grep -rn "qualified" src/candidate_ranking/` turned up
*zero* matches in `graphs/pipeline.py` or `scoring/active_listwise_ranking.py` —
the field is computed by the qualification gate (`llm_assessment.py`,
following an explicit "NON-COMPENSATORY RULE: if any MUST_HAVE verdict is
NOT_MET → candidate is disqualified immediately" in its own prompt), stored on
every `Assessment`, and then never read again. Checking the actual final-run
ranking output confirmed it concretely:

```
rank  candidate_id    qualified?
   7  cv-00380        qualified       utility=0.621
   8  cv-00331        DISQUALIFIED    utility=0.225
   9  cv-00607        DISQUALIFIED    utility=0.219
  10  cv-00548        DISQUALIFIED    utility=0.189
  11  cv-00371        qualified       utility=0.104
```
(`full-stack-engineer`, run `20261016-064409` — `runs/20261016-064409/full-stack-engineer/ranking.json`)

Three candidates who explicitly failed a must-have requirement outrank a
candidate who met every requirement. Checked disqualification rate across all
10 JDs — it's substantial everywhere, not a one-off:

| JD | Disqualified / Total |
|---|---|
| java-developer | 12/16 (75%) |
| full-stack-engineer | 13/24 (54%) |
| cloud-engineer | 7/13 (54%) |
| backend-engineer | 5/10 (50%) |
| it-security-engineer | 6/14 (43%) |
| data-scientist-engineer | 5/15 (33%) |
| ui-ux-designer | 8/26 (31%) |
| frontend-engineer | 5/17 (29%) |
| data-engineer | 2/8 (25%) |
| devops-engineer | 1/7 (14%) |

**Fix** (`src/candidate_ranking/scoring/active_listwise_ranking.py`):
`rank_active_listwise` now partitions candidates into qualified/disqualified
before ranking. The LLM-based active loop (BM25 warm-start + iterative subset
ranking + Plackett-Luce) runs **only on qualified candidates** — disqualified
candidates no longer consume any ranking LLM calls at all, since spending
budget comparing people who already failed a must-have isn't worthwhile.
Disqualified candidates are ordered among themselves by BM25 relevance (free)
and placed as a block with utility strictly below the lowest qualified
candidate's, so a disqualified candidate can never outrank a qualified one
regardless of how the LLM would have rated their evidence. When every
candidate is qualified, or every candidate is disqualified, there's nothing to
separate and the algorithm runs exactly as before.

New tests: `test_disqualified_candidates_never_outrank_a_qualified_one`,
`test_disqualified_candidates_keep_a_relative_order_among_themselves`,
`test_all_disqualified_pool_still_returns_every_candidate`. Also added a
`DISQUALIFIED` flag to `ranking.md`'s human-readable output
(`output/writer.py`) — the structured `ranking.json` already carried
`qualified` per entry, but the markdown report a recruiter actually reads did
not surface it at all.

**Verified impact on real data:** re-ranked `full-stack-engineer`'s cached
assessments directly — every one of the 11 qualified candidates now ranks
above all 13 disqualified ones (confirmed programmatically, not just
read off the table). See §4.1 for the stability-metric impact, which was the
largest single lever among the first three fixes — until §3.4.

### 3.4 Small candidate pools never got more than one LLM ranking call

**Symptom:** flagged as a follow-up, not yet fixed, in the previous version of
this report — `devops-engineer` swung from τ≈1.0 to **τ=0.619** (the worst of
any JD at the time) right after §3.3 shrank its qualified pool from 7 to 6.

**Root cause:** `rank_active_listwise` had a shortcut for `n ≤ subset_size`:
one `_rank_subset` call on the whole pool, no iterative refinement, return
immediately. That shortcut predates this session (original design,
`docs/superpowers/specs/2026-10-15-active-listwise-ranking-design.md` §4: "N ≤
subset_size → Single LLM call with all candidates, skip loop") and made sense
when it was written — fewer candidates, less work. But §2 and §3.3 both
establish that a single `temperature=0.3` LLM call is inherently noisy by
design (the non-zero temperature exists specifically so repeat calls *can*
disagree, which is what makes Plackett-Luce aggregation meaningful at all) —
and the shortcut gave small pools exactly one such call to settle the whole
ranking, with nothing to aggregate it against. After §3.3 started correctly
excluding disqualified candidates, roughly half the JDs' qualified pools
dropped to ≤10 and started hitting this path.

**Fix** (`src/candidate_ranking/scoring/active_listwise_ranking.py`): removed
the shortcut. `_select_uncertain_subset` already handles `subset_size ≥ n`
correctly — it naturally selects the whole pool every iteration — so small
pools now flow through the exact same convergence-seeking loop as large ones,
getting repeated independent LLM rankings aggregated via Plackett-Luce instead
of one. `min_iter = ceil(n/subset_size) = 1` for these pools, so they can still
stop after a single iteration *if* that single ranking's delta is already
below `min_delta_u` — the fix doesn't force extra calls on genuinely easy
cases, it just stops forcing the loop to accept the first call unconditionally
regardless of whether it looked stable.

New tests: `test_small_pool_gets_more_than_one_independent_llm_ranking`
(replaces the old `test_small_pool_one_llm_call`, which was asserting the
behavior being removed).

**Verified impact on real data:** re-ran `devops-engineer`'s cached assessments
through `rank_active_listwise` twice — τ=0.619 → **0.905** in an isolated
check, and 0.810 in the full `evaluate-stability` run reported in §4.1 (run
to run variance between even these two checks is itself expected, per §2/§4.1
finding 3 — this fix reduces variance, it doesn't eliminate the underlying
temperature-driven noise).

---

## 4. Results

### 4.1 Ranking stability — Kendall-tau and RBO, across all four fixes

Five real, independent `evaluate-stability` invocations: **before** any fix
(stale cache + stuck-subset bugs both present), **+1&2** (both applied, run
twice — "A"/"B" — to show temperature-driven run-to-run variance before §3.3
existed), **+1&2&3** (disqualified-candidate exclusion added), and **+all 4**
(small-pool shortcut removed too — the final state).

| JD | Candidates (final, qualified) | τ before | τ +1&2 (A) | τ +1&2 (B) | τ +1&2&3 | τ **+all 4** | RBO **+all 4** |
|---|---|---|---|---|---|---|---|
| backend-engineer | 10 (5) | 1.000 | 1.000 | 0.911 | 0.867 | **1.000** | 1.000 |
| cloud-engineer | 13 (6) | 0.821 | 0.872 | 0.897 | 1.000 | 1.000 | 1.000 |
| data-engineer | 8 (6) | 0.929 | 0.929 | 0.929 | 1.000 | 0.857 | 0.889 |
| data-scientist-engineer | 15 (10) | 0.905 | 0.848 | 0.867 | 0.924 | 0.962 | 0.920 |
| **devops-engineer** | 7 (6) | 0.714 | 1.000 | 0.810 | **0.619** | **0.810** | 0.757 |
| frontend-engineer | 17 (12) | 0.804 | 0.838 | 0.912 | 0.912 | 0.912 | 0.917 |
| full-stack-engineer | 24 (11) | 0.870 | 0.746 | 0.761 | 0.957 | **0.986** | 0.985 |
| it-security-engineer | 14 (8) | **0.543** | 0.846 | 0.912 | 1.000 | 0.978 | 0.942 |
| java-developer | 16 (4) | 0.767 | 0.850 | 0.933 | 0.996 | 1.000 | 1.000 |
| ui-ux-designer | 26 (18) | 0.748 | 0.705 | 0.742 | 0.908 | 0.883 | 0.739 |
| **Mean** | | **0.810** | **0.863** | **0.867** | **0.918** | **0.939** | **0.915** |

Each column is one real, independent run (raw JSON for the final column:
`runs/_cache/stability_report.json`). Bold marks `devops-engineer`'s dip after
§3.3 and recovery after §3.4 — the clearest single thread through the whole
investigation.

Interpretation: τ or RBO ≥ 0.90 stable · 0.75–0.90 moderate · < 0.75 unstable.

**Four findings, not one:**

1. **Bugs 1 & 2 fixed the noisiest early cases** — `it-security-engineer` went
   from unstable (0.543) to moderate (0.85–0.91) once the stale candidate
   stopped poisoning its subsets.
2. **Bug 3 was the biggest single lever on its own** — `full-stack-engineer`
   (54% disqualified) and `ui-ux-designer` (31% disqualified) jumped from the
   unstable/borderline band past 0.90 once disqualified candidates stopped
   being compared against qualified ones in the same LLM call.
3. **Bug 3 had a side effect that bug 4 then fixed**: shrinking pools by
   removing disqualified candidates pushed several JDs onto the single-call
   path, and `devops-engineer` (now only 6 qualified) took the brunt of it —
   dropping to the worst score in the table (τ=0.619). Removing that shortcut
   (§3.4) recovered it to 0.810 and lifted several of the newly-small pools
   (`backend-engineer` 0.867→1.000, `cloud-engineer` stayed at 1.000,
   `data-engineer` dipped to 0.857 — see finding 4). Mean τ: 0.918 → 0.939.
4. **Run-to-run variance from temperature=0.3 is real and never fully goes
   away** — compare "+1&2 (A)" vs "(B)" (`backend-engineer` 1.000→0.911,
   `devops-engineer` 1.000→0.810) and note that even in the final column,
   `data-engineer` (now getting the *full* iterative loop) still landed at a
   middling 0.857/0.889. This is intentional
   (`docs/superpowers/specs/2026-10-14-llm-ranking-design.md`: temperature=0
   would make repeat runs identical and the Plackett-Luce aggregation
   pointless), not a bug — but it means **no single stability run should be
   read as "the" stability number** for any one JD, before or after every fix
   in this report.

### 4.1c Qualification-gate test-retest reliability (Cohen's kappa)

Unlike §4.1, this evaluates the **assessment stage**, not the ranking stage — the
layer everything else is built on. 20 (JD, candidate) pairs sampled across all 10
JDs, qualification gate re-run 3× fresh per pair (cache bypassed, real LLM calls,
`CANDIDATE_RANKING_OLLAMA_NUM_PARALLEL` not used here — sequential):

| Metric | Value | Reading |
|---|---|---|
| Raw `qualified` agreement rate | 0.933 | 93% of repeat pairs reached the same qualify/disqualify call |
| **Cohen's kappa** (`qualified`) | **0.865** | **Near-perfect** (>0.80) — agreement well beyond what the model's own qualify/disqualify base rate would predict by chance |
| Per-requirement verdict agreement | 0.944 | MET/PARTIALLY_MET/NOT_MET matched 94% of the time across repeats, per shared requirement |

This is good news for the pipeline as a whole: it means §4.1's ranking-stability
numbers are measuring genuine *ranking* noise on top of a reliable assessment
layer, not noise that was already baked in one level down. Raw agreement (93%)
alone would have been hard to interpret — a model that calls 90% of candidates
"qualified" would hit ~82% raw agreement by chance alone — kappa is what confirms
this is real, non-trivial consistency.

Reproduce with: `uv run python -m candidate_ranking.cli evaluate-reliability
--sample-per-jd 2 --repeats 3` (full report: `runs/_cache/reliability_report.json`).

### 4.2 Latency per call type (warm model, local Ollama)

| Call type | Latency |
|---|---|
| Job profile extraction (per JD) | 3.40s |
| CV skill extraction (per candidate) | 7.37s |
| Qualification gate assessment (per JD×candidate pair) | 2.95s |
| Active ranking subset (per iteration, n=8–10) | 1.18s |

### 4.3 Cost

Fully local inference — **$0 API cost**. Compute time for the evaluated run
(501 candidates × 10 JDs, `CANDIDATE_RANKING_OLLAMA_NUM_PARALLEL=4`):

| Stage | Calls | Wall-clock (÷4 parallel) |
|---|---|---|
| CV skill extraction | 501 | ≈ 15 min |
| Job profile extraction | 10 | negligible (parallel) |
| Qualification gate | 151 | ≈ 2 min |
| Active ranking | variable (10 JDs × up to 30 iterations) | the dominant remaining cost for large pools |
| **Total observed** | | **≈ 20–25 min** |

---

## 5. Recommendations

1. **Done**: stale-assessment filtering (§3.1), stuck-subset avoidance (§3.2),
   excluding disqualified candidates from ranking (§3.3), and removing the
   small-pool single-call shortcut (§3.4) — all four shipped and test-covered
   in this session. §3.3 is the one to prioritize re-verifying first in any
   future change to the qualification gate or ranking stage, since it's both
   the most user-visible (final ranking output) and had the largest standalone
   stability impact; §3.4 is the one to re-check if `CANDIDATE_RANKING_SUBSET_SIZE`
   or typical shortlist sizes change materially, since it changes which JDs
   fall into the now-iterative small-pool path.
2. **Done**: rank-biased overlap (§4.1) and qualification-gate test-retest reliability (§4.1c) — implemented, tested, wired into `evaluate-stability` and the new `evaluate-reliability` CLI subcommand respectively.
3. **Consider**: raise `CANDIDATE_RANKING_MAX_ITERATIONS` for JDs with large shortlists if tighter ranking stability is needed there — cheap per the latency numbers above. Note from §4.1 finding 4 that even a JD now getting the full iterative treatment (`data-engineer`) landed at a middling 0.857/0.889 in the final run — more iterations reduces but does not eliminate temperature-driven variance.
4. **Consider**: a periodic cleanup pass (or a `--prune-stale-assessments` flag) to delete assessment cache files that no longer match any current job profile, rather than relying solely on read-time filtering — keeps `runs/_cache/assessments/` from growing unboundedly across repeated JD edits.
5. **Consider**: widen `evaluate-reliability`'s default sample (`--sample-per-jd`, currently 2) before trusting the kappa=0.865 figure as representative — 20 pairs is enough to be reassuring, not enough to be a tight confidence interval. The flag exists specifically so this is cheap to redo at a larger N.
6. **Note for future runs**: §4.1's run-to-run variance (finding 4) means a single `evaluate-stability` invocation is a noisy estimate of its own metric, at every pool size. If ranking stability becomes a release gate, run it 2–3× and look at the spread, not one number.
