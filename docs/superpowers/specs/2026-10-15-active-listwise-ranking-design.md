# Active Listwise Ranking — Design Spec

**Date:** 2026-10-15
**Status:** Approved — ready for implementation
**Reference:** Yuksel et al. (2026), "Agentic AI for Human Resources: LLM-Driven Candidate Assessment", EACL 2026.

---

## 1. Goal

Replace `rank_sliding_window_ensemble` in the pipeline with an active learning loop that selects the most informative candidate subsets at each iteration. More efficient at scale (N > 50) than fixed sliding window because comparisons are targeted at the highest-uncertainty region rather than sweeping candidates in positional order.

---

## 2. Motivation

Sliding window has a fundamental limitation: it compares candidates only within positional windows of the initial BM25 sort. For N=200 with window=15 and step=7, a candidate at position #16 never shares a window with candidates #1–#15 in the forward pass. This produces locally consistent but globally uncertain rankings.

Active learning with variance-based subset selection fixes this: the Plackett–Luce posterior identifies which candidates have the most uncertain utility estimates, and the next comparison is targeted precisely at those candidates. This is more sample-efficient for large pools.

---

## 3. Algorithm

```
WARM START (0 LLM calls)
  ordered = BM25 sort of candidates against JD
  rankings = [ordered]
  u = fit_utilities(rankings, item_ids, prior_variance)

COMPUTE ITERATION BOUNDS
  min_iter = ceil(N / subset_size)          # at least one full coverage sweep
  max_iter = max(40, 2 × min_iter)

ACTIVE LOOP (i = 1 .. max_iter)
  variances = diag(laplace_covariance(u, rankings, prior_variance))
  subset    = top-K item_ids by variance
  new_order = LLM rank(subset)              # lean prompt, ranking only
  if LLM failed: log warning, continue     # skip — do not crash
  rankings.append(new_order)
  u_new  = fit_utilities(rankings, item_ids, prior_variance)
  delta_u = ||u_new - u|| / N
  if delta_u < min_delta_u AND i >= min_iter: break
  u = u_new

RETURN {cid: float(u[i]) for i, cid in enumerate(item_ids)}
```

---

## 4. Edge Cases

| Case | Handling |
|------|----------|
| N ≤ 1 | Return `{cid: 1.0}` immediately, no LLM call |
| N ≤ subset_size | Single LLM call with all candidates, skip loop |
| LLM fails on subset | Log warning, skip iteration — do not append to rankings |
| Optimizer does not converge | Log warning (already in `plackett_luce.py`), use `result.x` |
| Loop exhausts max_iter without converging | Return final utilities, log warning with final delta_u value |

---

## 5. Lean Prompt

The active loop uses a stripped-down prompt — no `candidate_evaluations`, ranking only. This reduces output tokens and prevents the LLM from "overthinking" during tournament rounds, as recommended by Yuksel et al.

```python
ACTIVE_RANKING_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior hiring panel ranking candidates for a specific role.\n\n"
     "Rank ALL candidates from strongest to weakest fit for this role.\n\n"
     "RANKING CRITERIA (in priority order):\n"
     "1. Relevance — experience directly matching daily responsibilities\n"
     "2. Evidence quality — concrete achievements over vague responsibilities\n"
     "3. Career trajectory — ascending scope and seniority\n"
     "4. Recency — recent hands-on experience\n"
     "5. Nice-to-have coverage — breadth of preferred skills as tiebreaker\n\n"
     "Output ONLY the ranking permutation. No explanations. No ties. "
     "ranking must contain every candidate number exactly once."),
    ("human",
     "ROLE: {jd_title}\n\nJOB SUMMARY: {jd_summary}\n\nCANDIDATES:\n{candidates_text}"),
])
```

Output model: `ranking: list[int]` only (no `candidate_evaluations`).

---

## 6. Parameters & Config

New fields in `RunConfig`:

```python
ranking_subset_size: int = 8          # K candidates per active loop query
ranking_max_iterations: int = 40      # hard cap (scaled up if N is large)
ranking_min_delta_u: float = 0.01     # normalized convergence threshold
```

`ranking_pl_prior_variance: float = 1.0` — already exists, retained unchanged.
`ranking_ensemble_seeds` — **removed** (no longer relevant).

New env vars:
```
CANDIDATE_RANKING_SUBSET_SIZE=8
CANDIDATE_RANKING_MAX_ITERATIONS=40
CANDIDATE_RANKING_MIN_DELTA_U=0.01
```

---

## 7. File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| **Create** | `src/candidate_ranking/scoring/active_listwise_ranking.py` | `rank_active_listwise`, `ACTIVE_RANKING_PROMPT`, `_select_uncertain_subset`, `_active_ranking_model` |
| **Create** | `tests/scoring/test_active_listwise_ranking.py` | 9 unit tests |
| **Modify** | `src/candidate_ranking/config.py` | Add 3 new fields, remove `ranking_ensemble_seeds` |
| **Modify** | `tests/test_config.py` | Update ensemble seed test → new fields |
| **Modify** | `src/candidate_ranking/graphs/pipeline.py` | Swap import + call to `rank_active_listwise` |
| **Modify** | `.env.example` | Add 3 new env vars, remove `CANDIDATE_RANKING_ENSEMBLE_SEEDS` |
| **Keep** | `src/candidate_ranking/scoring/sliding_window_ranking.py` | Unchanged — still used by `evaluate-stability` |
| **Keep** | `src/candidate_ranking/ranking/plackett_luce.py` | Unchanged — `fit_utilities` + `laplace_covariance` already there |

---

## 8. Function Signature

```python
def rank_active_listwise(
    jd: JobDescription,
    assessments: dict[str, Assessment],
    build_chain: Callable[[type[BaseModel]], Runnable],
    subset_size: int = 8,
    max_iterations: int = 40,
    min_delta_u: float = 0.01,
    pl_prior_variance: float = 1.0,
) -> dict[str, float]:
    """Active listwise ranking via variance-based subset selection + Plackett-Luce.

    Warm-starts from BM25 sort (0 LLM calls), then iteratively queries the LLM
    on the most uncertain candidate subsets until convergence or budget exhaustion.
    Higher utility = better fit.
    """
```

---

## 9. Tests

File: `tests/scoring/test_active_listwise_ranking.py`

```
test_single_candidate_no_chain_call          — N=1, return immediately
test_small_pool_single_call                  — N ≤ subset_size, exactly 1 LLM call
test_returns_all_candidate_ids               — output keys == input keys
test_converges_before_max_iterations         — deterministic mock chain → stops early
test_respects_max_iterations                 — noisy mock → stops at max
test_respects_min_iterations                 — fast-converging mock → runs min_iter times
test_llm_failure_skips_iteration             — chain raises GenerationError → no crash
test_output_types                            — all values are float
test_stronger_candidate_gets_higher_utility  — candidate with more MET evals ranks higher
```

Mock strategy: `MagicMock` chain returning deterministic rankings based on `years_relevant` in assessment fixture (same pattern as existing `test_sliding_window_ranking.py`).

---

## 10. Cost Profile

For N=100 qualified candidates, subset_size=8:
- Warm start: 0 LLM calls
- min_iter = ceil(100/8) = 13
- Typical convergence: 15–25 iterations (based on Yuksel et al. Figure 2)
- **Total: ~15–25 LLM calls vs 26 for current sliding window**

For N=200:
- min_iter = 25, max_iter = 50
- Typical: 30–40 iterations
- **Total: ~30–40 LLM calls vs 54 for current sliding window**

Active learning is more efficient AND globally consistent.
