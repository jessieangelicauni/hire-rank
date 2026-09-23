# Automated, Accuracy-Guarded Criteria Optimization for Score-Type Questions — Design

Date: 2026-09-23

## Motivation

Criteria-Grounded Question Design (CGQD) as it exists in the pipeline today
(`_REQUIREMENT_FIT_CRITERIA` in `assessment.py`) was hand-written once and
validated after the fact with a paired ablation (concrete vs. bare-label,
`docs/paper2_jev.tex` §III-B/V-A). TypeSafe's own documentation
(`docs.typesafe.ai/primitives/score`) gives only static, human-facing
heuristics for writing criteria (concrete situations, one dimension per
question, dedicated levels for rare extreme cases, structured `what`/
`examples` fields for boundary cases) — it prescribes no method for
automatically discovering or improving a criteria set, and its own
"unspecified elements" list admits domain-specific tuning (level count,
example construction) is left to the user. No public source on Jev
(TypeSafe's docs, its release blog, or third-party posts) describes an
automated process for this — all of it stops at "here is how a human
should write good criteria."

A second gap, which the same documentation flags explicitly: *"Don't
assume higher confidence proves better description quality."* Any
automated search that optimizes criteria purely for Jev-reported
confidence risks Goodharting — finding wording that makes Jev sound sure
without being right.

This design closes both gaps with an automated criteria-design optimizer:
a round-based propose/evaluate loop, backed by the codebase's existing
LangChain pattern (not a new framework), guarded against confidence-only
Goodharting by an independent-model (Claude) accuracy check, and refreshed
each round by mining the lowest-confidence cases from the last round
(a standard, well-established technique — not an adversarial agent, and
not claimed as novel on its own).

**What this claims as novel:** the automated optimization method itself —
propose, evaluate against real Jev calls, validate against an independent
accuracy anchor, repeat — for a design problem TypeSafe's own documentation
only addresses with static human-facing heuristics.

**What this does not claim:** that concrete-vs-bare-label criteria design
(CGQD itself) is this project's invention (it is TypeSafe's, cited as
`ref23`), and that the hard-case-mining step is an adversarial "red-team"
technique (it is confidence-based example selection, a well-known method,
included here because it is useful, not because it is new).

## Scope

**In scope:**
- The per-technical-requirement Score question only
  (`_REQUIREMENT_FIT_CRITERIA` and its use in `_build_questions` /
  `_answers_to_assessment` in `src/candidate_ranking/scoring/assessment.py`).
- New library code for the optimization loop, metric, and hard-case miner.
- New driver scripts for running the optimization and the final
  baseline-vs-optimized comparison.
- New paper subsection and results table reporting the comparison.

**Out of scope:**
- Seniority and education criteria — unchanged.
- Must-have/nice-to-have classification, shortlisting, ensemble
  aggregation, ranking — untouched.
- DSPy, or any LLM-orchestration framework not already a project
  dependency — the proposer is built on the existing LangChain pattern
  (`jd_skills.py`), not a new library, because it needs nothing DSPy would
  provide beyond what `ChatPromptTemplate` + `with_structured_output` +
  `invoke_and_validate` already do in this codebase.
- Synthetic CV-text perturbation for hard-case generation — a possible
  future extension, not required for the core result; the hard-case miner
  selects from real, already-collected low-confidence data only.

## Components

### 1. `src/candidate_ranking/evaluation/criteria_optimization.py` (new)

- **`evaluate_criteria(criteria, sample, jds_by_id, candidates_by_id, jev_client) -> list[CriteriaEvalRecord]`**:
  builds `JevQuestion`s with the candidate criteria substituted for
  `_REQUIREMENT_FIT_CRITERIA`, calls `JevClient.evaluate` on each triple in
  `sample`, and returns per-triple `(score, confidence)`.
- **`compute_metric(records, proxy_labels) -> float`**: combines mean Jev
  confidence with agreement against proxy labels. Confidence is only
  credited on triples where Jev's picked level matches (or is within 1 of)
  the proxy label; a confidently-wrong triple scores at or near zero rather
  than rewarding the confidence. This is the direct mitigation for the
  Goodharting risk in Motivation.
- **`select_hard_triples(records, k) -> list[RequirementTriple]`** (the
  hard-case miner): after each round, selects the K lowest-confidence
  triples under the current best candidate from the validation split and
  folds them into the next round's hard-case pool (oversampled). This is
  free — confidence is already returned by every Jev call — so no
  synthetic perturbation is needed. Named and documented as confidence-based
  hard-case mining, not as an adversarial agent.
- **`run_optimization(...) -> tuple[list[str], list[OptimizationRound]]`**:
  the round loop — seed from the hand-written baseline, propose, evaluate,
  keep the best, refresh hard cases, stop on a call budget or no
  improvement over K rounds.

### 2. `src/candidate_ranking/evaluation/proxy_labeler.py` (new)

- **`ProxyLabelClient(anthropic_client, cache_path, model)`** with
  `.label(triple, jd, candidate) -> int` (0-4): calls Claude (model
  configurable via `RunConfig.proxy_label_model`) with the same rubric
  description Jev receives, asking it to pick the best-fitting level
  directly from the job/CV text. Used as the independent-model accuracy
  anchor — independent of Qwen (the proposer) to avoid self-grading bias.
  Cached by `(job_id, candidate_id, requirement)` content hash, since
  labels don't change across optimization rounds.

### 3. `src/candidate_ranking/evaluation/criteria_proposer.py` (new)

- Built on the existing LangChain pattern from
  `src/candidate_ranking/scoring/jd_skills.py`, not DSPy:
  `PROPOSE_CRITERIA_PROMPT` (`ChatPromptTemplate`),
  `build_criteria_proposer_chain(llm: BaseChatModel) -> Runnable`
  (`PROMPT | llm.with_structured_output(_ProposedCriteria)`), and
  `propose_criteria(current_criteria, hard_case_summaries, chain) -> list[str]`,
  which calls `invoke_and_validate` (`candidate_ranking.generation`) and
  raises `GenerationError` if the proposed criteria fail shape validation.
  The `llm` is `ChatOllama` pointed at the same `cfg.ollama_model` already
  used for skill extraction — no new model dependency for this role.

### 4. `scripts/optimize_criteria.py` (new)

CLI driver mirroring `run_jev_evaluation_study.py`'s corpus-loading
pattern. Splits the existing ~800 per-requirement triples (already
collected for the current ablation) into 60% optimize-train / 20%
validation (hard-case mining source) / 20% held-out test (untouched until
the final comparison). Runs the optimizer loop under a fixed
`--max-jev-calls` budget, writes the winning criteria set and full
round-by-round metric history to `runs/_cache/criteria_optimization/`.

### 5. `scripts/evaluate_criteria_comparison.py` (new)

Final comparison on the held-out test split only: hand-written baseline
(`_REQUIREMENT_FIT_CRITERIA`) vs. optimized criteria, reporting mean
confidence and accuracy-vs-proxy-label for both, Wilcoxon signed-rank
paired by triple — same statistical presentation already used for
`tab_score_ablation`.

## Algorithm

1. Seed criteria = current `_REQUIREMENT_FIT_CRITERIA`.
2. Proposer (LangChain chain over `ChatOllama`) generates a candidate
   criteria set, given the current best criteria and the current
   hard-case pool.
3. Candidate scored by `compute_metric()` on optimize-train, with the
   current hard-case pool oversampled.
4. Best candidate retained; loop continues.
5. Every round: re-mine lowest-confidence validation triples under the
   current best candidate via `select_hard_triples`, refresh the pool.
6. Stop at the call budget, or on no validation-metric improvement over K
   consecutive rounds.
7. Evaluate best candidate and baseline on held-out test; report via
   `evaluate_criteria_comparison.py`.

## Error Handling

- Jev API errors: reuse `JevClientError`, matching existing script
  conventions (`run_jev_evaluation_study.py` does not add extra retry
  around `jev_client.evaluate` either — errors propagate and the script
  is rerun).
- Proposer LLM failures: `invoke_and_validate` already raises
  `GenerationError` on schema/validation failure; not caught specially
  here, consistent with how `generate_jd_skills` handles it (a bounded
  retry loop) — this task does the same: retry once, then propagate.
- Hard call-budget cap bounds API cost regardless of convergence.
- Proxy labels are cached by content hash (mirroring `_assessment_cache_key`)
  so re-runs during development don't re-spend Claude API budget.

## Testing Plan

- Unit tests for `compute_metric()`: a confidently-wrong triple must score
  lower than an equally-confident, correct triple.
- Unit tests for `select_hard_triples`'s K-lowest-confidence selection.
- Unit test asserting any proposed criteria set has exactly 5 entries.
- Integration test with a stub `JevClient` and stub proposer chain
  exercising the full loop with no real network calls.
- Smoke test for `evaluate_criteria_comparison.py` on a tiny fixture
  sample.

## Paper Updates (`docs/paper2_jev.tex`)

- New results subsection: "Automated, Accuracy-Guarded Criteria
  Optimization," describing the propose/evaluate loop, the
  confidence+accuracy metric, and the confidence-based hard-case mining,
  and reporting the held-out comparison (confidence and
  accuracy-vs-proxy-label, optimized vs. hand-written, Wilcoxon
  signed-rank).
- Explicit, stated limitation: the accuracy comparison is against an
  independent LLM's judgment (proxy label), not human-annotated ground
  truth. This is stated plainly rather than glossed over — it mirrors
  the exact reference-model-bias caveat the paper's own Literature
  Review already holds TypeSafe's vendor evals to.
- Explicit non-claim, stated in the contribution's own wording: this
  section does not claim to have invented criteria-grounded design
  (attributed to `ref23` throughout), and does not describe the hard-case
  miner as adversarial red-teaming.

## Open Risk Worth Flagging

The proxy-label accuracy check is the load-bearing piece that keeps this
design from just being a fancier confidence-maximizer. If Claude and
Qwen2.5-14B share correlated blind spots on some class of resumes, the
accuracy guardrail could still miss confidently-wrong criteria in that
class. This is a real limitation to state in the paper, not something the
design fully closes.
