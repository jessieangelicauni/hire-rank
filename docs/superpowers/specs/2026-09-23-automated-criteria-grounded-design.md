# Automated, Adversarially-Validated Criteria-Grounded Question Design — Design

Date: 2026-09-23

## Motivation

Criteria-Grounded Question Design (CGQD) as it exists in the pipeline today
(`_REQUIREMENT_FIT_CRITERIA`, `_SENIORITY_YEARS_FIT_CRITERIA`,
`_EDUCATION_FIT_CRITERIA` in `assessment.py`) was hand-written once and
validated after the fact with a paired ablation (concrete vs. bare-label,
`docs/paper2_jev.tex` §III-B/V-A). TypeSafe's own documentation
(`docs.typesafe.ai/primitives/score`) gives only static, human-facing
heuristics for writing criteria (concrete situations, one dimension per
question, dedicated levels for rare extreme cases, structured `what`/
`examples` fields for boundary cases) — it prescribes no method for
automatically discovering or improving a criteria set, and its own
"unspecified elements" list admits domain-specific tuning (level count,
example construction) is left to the user.

A second gap, which the same documentation flags explicitly: *"Don't
assume higher confidence proves better description quality."* Any
automated search that optimizes criteria purely for Jev-reported
confidence risks Goodharting — finding wording that makes Jev sound sure
without being right.

This design closes both gaps: an automated criteria-design optimizer
(built on DSPy's instruction-optimization machinery, previously validated
only for autoregressive LM prompts, adapted here to a non-autoregressive
System One Model's Score criteria) guarded against confidence-only
Goodharting by an independent-model accuracy check and a red-team critic
that actively surfaces boundary-ambiguous cases.

## Scope

**In scope:**
- The per-technical-requirement Score question only
  (`_REQUIREMENT_FIT_CRITERIA` and its use in `_build_questions` /
  `_answers_to_assessment` in `src/candidate_ranking/scoring/assessment.py`).
- New library code for the optimization loop, metric, and red-team critic.
- New driver scripts for running the optimization and the final
  baseline-vs-optimized comparison.
- New paper subsection and results table reporting the comparison.

**Out of scope:**
- Seniority and education criteria — unchanged.
- Must-have/nice-to-have classification, shortlisting, ensemble
  aggregation, ranking — untouched.
- Synthetic CV-text perturbation by the red-team critic is a stretch
  goal (see Red-Team Critic below), not required for the core result.

## Components

### 1. `src/candidate_ranking/evaluation/criteria_optimization.py` (new)

- **Proposer**: a DSPy module (`dspy.COPRO` or `dspy.MIPROv2`) backed by
  Qwen2.5-14B-Instruct (already the pipeline's extraction model — no new
  model dependency for this role). Given the current best criteria set
  (seeded from `_REQUIREMENT_FIT_CRITERIA`) plus the current round's
  hard-case pool, proposes a revised 5-level criteria list in the same
  shape as `_REQUIREMENT_FIT_CRITERIA`.
- **`evaluate_criteria(criteria, sample, jev_client) -> CriteriaEvalResult`**:
  builds `JevQuestion`s with the candidate criteria substituted for
  `_REQUIREMENT_FIT_CRITERIA`, calls `JevClient.evaluate` on `sample`, and
  returns per-triple `(score, confidence)`.
- **`proxy_label(job, candidate, requirement) -> int`** (1-5): calls
  Claude Opus 5 via API with the same rubric description Jev receives,
  asking it to pick the best-fitting level directly from the job/CV text.
  Used as the independent-model, pseudo-ground-truth accuracy anchor —
  independent of Qwen (proposer/critic) to avoid self-grading bias.
  Cached by `(job_id, candidate_id, requirement)` content hash, same
  pattern as `_assessment_cache_key`, since labels don't change across
  optimization rounds.
- **`metric(criteria, sample) -> float`**: combines mean Jev confidence
  with agreement against proxy labels. Confidence is only credited on
  triples where Jev's picked level matches (or is within 1 of) the proxy
  label; a confidently-wrong triple scores at or near zero rather than
  rewarding the confidence. This is the direct mitigation for the
  Goodharting risk in Motivation.
- **`RedTeamCritic`**: after each round, selects the K lowest-confidence
  triples under the current best candidate from the validation split and
  folds them into the next round's hard-case pool (oversampled). This is
  free — confidence is already returned by every Jev call — so no
  synthetic perturbation is needed for v1. **Stretch goal**: an
  LLM-based perturbation step that lightly edits a requirement-relevant
  CV sentence to sit deliberately between two levels, accepted only if
  `proxy_label` itself finds the perturbed pair genuinely ambiguous
  (i.e., the perturbation must survive an independent-model plausibility
  check, not just move Jev's confidence).

### 2. `scripts/optimize_criteria.py` (new)

CLI driver mirroring `run_jev_evaluation_study.py`'s corpus-loading
(`load_corpus`, `load_run_pairs`). Splits the existing 798 per-requirement
triples (already collected for the current ablation) into:
- 60% optimize-train (metric evaluation during search)
- 20% validation (red-team hard-case mining source, DSPy's internal
  validation split)
- 20% held-out test (untouched until the final comparison)

Runs the optimizer loop under a fixed `--max-jev-calls` budget, writes the
winning criteria set and full round-by-round metric history to
`runs/_cache/criteria_optimization/`.

### 3. `scripts/evaluate_criteria_comparison.py` (new)

Final comparison on the held-out test split only: hand-written baseline
(`_REQUIREMENT_FIT_CRITERIA`) vs. DSPy-optimized criteria, reporting mean
confidence and accuracy-vs-proxy-label for both, Wilcoxon signed-rank
paired by triple — same statistical presentation already used for
`tab_score_ablation`.

## Algorithm

1. Seed criteria = current `_REQUIREMENT_FIT_CRITERIA` (also serves as a
   DSPy few-shot anchor, not just a starting point to discard).
2. Proposer generates N candidate criteria sets per round.
3. Each candidate scored by `metric()` on optimize-train, with the
   current hard-case pool oversampled.
4. Best candidate(s) retained; DSPy continues its standard optimizer
   iteration.
5. Every round: re-mine lowest-confidence validation triples under the
   current best candidate via `RedTeamCritic`, refresh the hard-case pool.
6. Stop at the call budget, or on no validation-metric improvement over K
   consecutive rounds.
7. Evaluate best candidate and baseline on held-out test; report via
   `evaluate_criteria_comparison.py`.

## Error Handling

- Jev API errors: reuse `JevClientError` and the existing retry pattern
  in `assessment.py`.
- Proposer LM failures: DSPy's own retry/backoff; a failed candidate is
  logged and skipped, not fatal to the round.
- Hard call-budget cap bounds API cost regardless of convergence.
- Proxy labels and per-criteria-set Jev evaluations are cached by content
  hash (mirroring `_assessment_cache_key`) so re-runs during development
  don't re-spend budget.

## Testing Plan

- Unit tests for `metric()`: a confidently-wrong triple must score lower
  than an equally-confident, correct triple.
- Unit tests for `RedTeamCritic`'s K-lowest-confidence selection.
- Unit test asserting any proposed criteria set has exactly 5 entries
  (matching the existing `_SCORE_MAX_INDEX` assumption asserted elsewhere
  in `assessment.py`).
- Integration test with a stub `JevClient` and stub proposer LM exercising
  the full loop with no real network calls.
- Smoke test for `evaluate_criteria_comparison.py` on a tiny fixture
  sample.

## Paper Updates (`docs/paper2_jev.tex`)

- New results subsection: "Automated, Adversarially-Validated
  Criteria-Grounded Question Design," describing the DSPy proposer, the
  confidence+accuracy metric, and the red-team critic, and reporting the
  held-out comparison (confidence and accuracy-vs-proxy-label,
  optimized vs. hand-written, Wilcoxon signed-rank).
- Explicit, stated limitation: the accuracy comparison is against an
  independent LLM's judgment (proxy label), not human-annotated ground
  truth. This is stated plainly rather than glossed over — it mirrors
  the exact reference-model-bias caveat the paper's own Literature
  Review already holds TypeSafe's vendor evals to (§II, citing the
  blog's admission that using "GPT-6 Astra and Fable 5.1 as the
  reference answer...biases answers towards OpenAI and Anthropic's
  models").

## Open Risk Worth Flagging

The proxy-label accuracy check is the load-bearing piece that keeps this
design from just being a fancier confidence-maximizer. If Claude Opus 5
and Qwen2.5-14B share correlated blind spots on some class of resumes,
the accuracy guardrail could still miss confidently-wrong criteria in
that class. This is a real limitation to state in the paper, not
something the design fully closes.
