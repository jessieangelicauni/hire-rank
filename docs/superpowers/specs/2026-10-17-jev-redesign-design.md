# Jev-Backed Qualification Gate & Pairwise Ranking — Design

**Date:** 2026-10-17
**Status:** Design agreed in conversation; not yet implemented. One HR policy
question remains open (§4).
**Builds on:** `docs/evaluation_report_2026-10-16.md` (4 bugs found/fixed —
already committed, see commits `9e225dc`, `13d32ab`, `0d43ea5`, `bd49a80`).

## 0. Why this exists

A follow-up audit of the 2026-10-16 evaluation report surfaced 4 further
issues in `src/candidate_ranking/scoring/llm_assessment.py` and
`active_listwise_ranking.py` (§1). Investigating fix #2 (the qualification
gate's `qualified` field isn't verified against its own `must_have_evaluations`)
led to a much bigger question — should the qualification gate even be asking
a general-purpose LLM for this at all, given this exact classification
problem used to be solved by **Jev**, TypeSafe AI's typed "System One Model"
API, before it was removed from this repo 2 days ago
(`f0296f2`, "replaced by llm_assessment.py")?

The conversation that produced this doc worked through: is Jev worth
bringing back (yes, cost accepted) · for the whole qualification gate or just
one field (whole gate, to stay at 1 call) · what about fields Jev
structurally can't produce (free-text explanations — solved with a second,
free, local Ollama call) · should the question set be fixed or adaptive per
job description (adaptive) · and finally, should the **ranking** stage also
move to Jev as a pairwise tournament (yes, reusing the existing
active-learning loop rather than a new algorithm).

## 1. Four bug fixes (independent of the Jev decision)

Found by auditing `llm_assessment.py` / `active_listwise_ranking.py` beyond
the 4 bugs in the 2026-10-16 report:

1. **Cache-key staleness on prompt edits** — `_assessment_cache_key` hashes
   jd/candidate/profile/model but not the prompt content, unlike
   `job_profile.py`/`cv.py` which correctly hash the rendered prompt.
   *Resolved by the Jev migration itself* (§3): the new cache key hashes the
   literal rendered question set, which changes automatically whenever
   question-building code changes — no manual version string needed, same
   pattern as `job_profile.py`.
2. **`qualified` trusted verbatim from the LLM**, never recomputed from the
   must-have verdicts in the same response — the model can (and per κ=0.865
   in the eval report, sometimes does) contradict itself. *Fix, still
   needed under Jev:* compute `qualified = all(verdict != "NOT_MET" for
   must-have evals)` in code; never trust a separately-reported boolean.
3. **Stuck-subset retry is a no-op when `subset_size >= n`** (the common
   case for small qualified pools after the 2026-10-16 fixes) —
   `_select_uncertain_subset`'s swap-based retry has no candidate outside
   the subset to swap in, so a failed retry resends an identical prompt.
   **Still needed, and more important** under the pairwise design (§5),
   since pairwise fixes `subset_size=2` and this edge case (no 3rd candidate
   to swap in) becomes far more common at small `n`.
   *Fix:* when no swap candidate exists, rotate the presentation order by
   the existing `iteration` loop counter instead of leaving it unchanged.
4. **Bare `except Exception` in `generate_assessment`'s retry loop**
   (`llm_assessment.py:226`) masks real code bugs as LLM flakiness,
   swallowing the original traceback — `job_profile.py`'s sibling function
   already does this correctly (`except GenerationError`). *Fix:* narrow to
   the specific domain error type (also applies to the new Jev client error
   type once introduced, and to `cli.py:320`).

## 2. Qualification gate → Jev (hybrid)

**Why Jev at all:** evidence-strength classification is exactly Jev's
design case — typed answers (Noul/Choice/Score) make an invalid value
structurally impossible, eliminating the self-contradiction risk in bug #2
above at the source, not just patching it in code.

**Why the *whole* gate, not just one field:** Jev and Ollama are different
APIs; splitting fields across both means 2 calls per candidate instead of 1.
Moving the whole gate keeps it at 1 (paid) call per candidate — accepted
cost: ~151 paid calls per run at this corpus size, recurring on every future
run, not a one-time cost.

**Structural fields → Jev** (`choice`/`score`/`noul`, one call per
candidate, question set built dynamically per `JobProfile` — see §3):
- Per must-have / nice-to-have skill: `evidence_strength` as `choice`
  (`strong`/`adequate`/`weak`/`none` — `none` added so Jev can express "no
  evidence at all", which the original 3-value scale couldn't).
  `verdict` is **not** asked — computed in code:
  `{strong,adequate→MET; weak→PARTIALLY_MET; none→NOT_MET}`. This also
  resolves the separate "verdict vs evidence_strength can silently
  disagree" risk noticed during design (same bug class as #2, one level
  deeper) — it disappears because verdict is no longer a second, independent
  model decision.
- `qualified` — computed in code from must-have verdicts (§1.2), never
  asked.
- `total_relevant_years` → `score`.
- `career_trajectory` → `direction`/`tenure_pattern` as `choice`,
  `average_tenure_months`/`most_recent_role_duration_months` as `score`,
  `has_multinational_experience` as `noul`.
- `red_flags` → **redesigned** as 5 fixed flag types (`employment_gap`,
  `skill_mismatch`, `regression`, `job_hopping`, `other`), each contributing
  3 fixed questions: `has_<type>` (`noul`), `<type>_severity` (`choice`),
  `<type>_duration_months` (`score`). Jev cannot express a free-form,
  variable-length list — this is the structural workaround.

**Free-text fields → second, free, local Ollama call** (new, small prompt;
no equivalent in the deleted pre-Jev `assessment.py`): `evidence` (CV quote
per requirement), `strongest_achievements`, `notable_signals`,
`disqualification_reason`, red flag `detail`, career-trajectory `evidence`
and `multinational_evidence`. Jev's answer types (bool/enum/float) cannot
produce free text at all — this isn't a design choice, it's a hard API
limitation. 2 calls/candidate total, but only the Jev one is paid; the
Ollama one is the same local/free inference already used for skill/JD
extraction.

**Caching:** hash the literal rendered question payload (Jev side) / prompt
messages (Ollama side) + model name — the already-proven pattern from
`job_profile.py`/`cv.py`, and in fact what the pre-removal `assessment.py`
already did for Jev. No manual version string required.

**GPU impact:** Jev is a hosted API (`api.typesafe.ai`) — zero local GPU use.
Net effect of this migration is *less* local GPU dependency than today, not
more: only skill/JD extraction and the new free-text companion call stay
local.

## 3. Adaptive question design (per `JobProfile`)

The question set is **built dynamically per job profile**, not a static
template — count and content vary per JD:

| Source | Cardinality | Question kind |
|---|---|---|
| `must_have_skills` | 1 per skill (varies per JD) | `choice` |
| `nice_to_have_skills` | 1 per skill (varies per JD) | `choice` |
| `education_requirement` | 0 or 1 (only if JD sets it) | `choice`: `meets` / `equivalent_experience` / `does_not_meet` |
| `seniority_requirement` | 0 or 1 (only if JD sets it) | `choice`: `meets_or_exceeds` / `below` |
| `other_requirements` | 1 per item (varies per JD) | `noul` |
| `total_relevant_years` | always 1 | `score` |
| career trajectory | always 5 (fixed) | mixed |
| red flags | always 15 (5 types × 3) | mixed |

**Gap this closes:** `JobProfile.education_requirement`,
`seniority_requirement`, and `other_requirements` already exist on the model
(`models.py`) but are **currently never read** by the qualification gate —
only skills factor into `qualified` today. A candidate failing an explicit
degree or seniority requirement can currently still qualify. This redesign
is the first time these fields get consulted at all.

Shared instruction scaffold for all skill `choice` questions (STRONG /
ADEQUATE / WEAK / NONE evidence rules, "re-read the whole CV before
answering NONE — skills are often implied") carries over unchanged from the
current `QUALIFICATION_GATE_PROMPT`.

## 4. OPEN DECISION — not yet answered

**Should failing education / seniority / `other_requirements` be
non-compensatory (auto-disqualifying, like a failed must-have skill), or
advisory-only (shown in the report, doesn't block qualification)?**

Recommendation given during design: **advisory-only**, because
`JobProfile` has no signal distinguishing a JD's hard "required" wording
from a soft "preferred" one for these fields (skills already have this via
the explicit `Must Have Skills:` line parsed by `job_profile.py`'s
`_parse_explicit_must_have`; education/seniority/other_requirements don't).
Auto-disqualifying on an ambiguous signal risks rejecting strong candidates
over a technicality. **This still needs the user's explicit sign-off before
implementation starts.**

## 5. Ranking → pairwise tournament via Jev (replaces active-listwise's LLM backend)

**Key realization:** pairwise doesn't need a new ranking algorithm. The
existing active-learning loop in `active_listwise_ranking.py` — BM25
warm-start, `_select_uncertain_subset` (variance-driven selection),
`_rank_subset`, Plackett-Luce fit (`ranking/plackett_luce.py`), convergence
check — is already fully generic in subset size. Plackett-Luce itself
reduces to Bradley-Terry at size 2, so pairwise comparisons are just the
`subset_size=2` special case of the exact same math. **What's actually new**
is only the comparator: `_rank_subset` currently calls an Ollama chain;
the pairwise-via-Jev version calls Jev instead, with a `choice` question
("which candidate is the stronger fit: A or B?").

**Why adaptive pair selection over full round-robin:** full round-robin for
`n` candidates is `n(n-1)/2` comparisons — 276 for a 24-candidate JD, versus
roughly 30–40 under the adaptive loop (reusing the same iteration budget
math as today: `min_iter = ceil(n/subset_size)`, now with `subset_size=2`).
Decided to keep the adaptive loop and not go full round-robin, trading a
larger `max_iterations` for continued sample efficiency rather than letting
call count explode quadratically. Jev being cheap changes the economics of
*more iterations*, not of comparing *every* pair.

**Known trade-off, accepted but worth remembering:** one listwise call over
8–10 candidates implies far more pairwise preference information (~28
pairs' worth) than one pairwise call (1 pair). At equal iteration budgets,
pairwise-via-active-learning will converge more slowly / noisily than the
current listwise approach. The plan to compensate is a materially higher
`max_iterations`, affordable specifically because Jev's marginal cost per
call is low — this is a cost-for-signal trade, not a free upgrade.

**Bug #3 (§1.3) is directly load-bearing here**, not optional — at
`subset_size=2`, the "no swap candidate available" edge case (whenever a
JD's remaining uncertain pool is itself ≤2) comes up far more often than in
the current 8–10-sized-subset world.

## 6. Build order

1. **§1** bug fixes (#3, #4) — small, no dependency on anything else.
2. **§2–§4** qualification gate → Jev hybrid — needed before §5, since
   ranking partitions on `Assessment.qualified`, which must be correct
   first. Blocked on §4's open decision.
3. **§5** pairwise ranking via Jev — most experimental, built last, on top
   of a corrected `Assessment`.

## 7. Files expected to change

`src/candidate_ranking/scoring/jev_client.py` (restore from `f0296f2^`,
unchanged — generic, model-shape-agnostic) · `llm_assessment.py` (rewrite:
adaptive Jev question builder + separate free-text Ollama prompt, merge into
one `Assessment`) · `active_listwise_ranking.py` (§1.3 fix; new Jev-backed
pair comparator) · `config.py` (+`jev_api_key`, re-added) · `.env.example`
(+`CANDIDATE_RANKING_JEV_API_KEY` placeholder) · `graphs/pipeline.py` /
`cli.py` (construct both an Ollama chain and a Jev client) · tests: restore
`test_jev_client.py`, rewrite `test_llm_assessment.py` /
`test_active_listwise_ranking.py` for the new schema and backend.
