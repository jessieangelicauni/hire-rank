# Ensemble Sliding Window Ranking — Design Spec

**Date:** 2026-10-15
**Status:** Approved — ready for implementation

---

## 1. Goal

Replace the single-run sliding window ranker with a K-run ensemble that aggregates multiple independent orderings via Plackett-Luce (P-L) posterior. This directly addresses 5 known weaknesses of the single-run approach.

---

## 2. Weaknesses Solved

| Weakness | How Solved |
|---|---|
| No uncertainty quantification | P-L posterior gives principled utility scale (Laplace covariance available if needed) |
| Transitivity not guaranteed | P-L aggregation across K runs smooths pairwise inconsistencies |
| Distant candidates never directly compared | Different seeds → different window compositions → more pairs covered across runs |
| Sensitive to initial sort | K different starting orders reduce sensitivity to any single initialization |
| Kendall-τ not yet measured | Seeds[0] and seeds[1] of ensemble can be reused directly for τ evaluation |

---

## 3. Algorithm

```
for seed in seeds (default: 42, 137, 271):
    utilities = rank_sliding_window(jd, assessments, build_chain,
                                    window_size, step, rng_seed=seed)
    ordering  = sorted(item_ids, key=utilities.__getitem__, reverse=True)
    rankings.append(ordering)

pl_utilities = fit_utilities(rankings, item_ids, prior_variance)
return {cid: float(pl_utilities[i]) for i, cid in enumerate(item_ids)}
```

**Cost:** 3 × 26 = 78 LLM calls for N=100 (vs Paper 1's 125 — 38% cheaper, same robustness).

---

## 4. New Function

```python
def rank_sliding_window_ensemble(
    jd: JobDescription,
    assessments: dict[str, Assessment],
    build_chain: Callable[[type[BaseModel]], Runnable],
    window_size: int = 15,
    step: int = 7,
    seeds: tuple[int, ...] = (42, 137, 271),
    pl_prior_variance: float = 1.0,
) -> dict[str, float]:
```

Added to `src/candidate_ranking/scoring/sliding_window_ranking.py` (same file, closely related).
Returns same type as `rank_sliding_window` — backward compatible interface.

---

## 5. Config Changes

Add to `RunConfig`:
```python
ranking_ensemble_seeds: tuple[int, ...] = (42, 137, 271)
ranking_pl_prior_variance: float = 1.0
```

Add corresponding `_ENV_OVERRIDES` entries and `.env.example` lines.

---

## 6. Pipeline Change

In `pipeline.py` `rank_and_format_jd_node`, replace:
```python
utilities = rank_sliding_window(
    jd, qualified, build_sliding_window_chain,
    window_size=cfg.sliding_window_size,
    step=cfg.sliding_window_step,
)
```
With:
```python
utilities = rank_sliding_window_ensemble(
    jd, qualified, build_sliding_window_chain,
    window_size=cfg.sliding_window_size,
    step=cfg.sliding_window_step,
    seeds=cfg.ranking_ensemble_seeds,
    pl_prior_variance=cfg.ranking_pl_prior_variance,
)
```

---

## 7. Dependencies

- `fit_utilities` from `candidate_ranking.ranking.plackett_luce` — already exists, no new deps
- `rank_sliding_window` from same file — already exists

---

## 8. Key Design Decisions

**Single-run preserved:** `rank_sliding_window` stays in the codebase — used by Kendall-τ evaluation and available as a faster fallback.

**Same return type:** `dict[str, float]` — no breaking changes to pipeline or CLI.

**seeds configurable via env var:** Production can override to 2 seeds for 2× cost savings when budget is tight.

**No variance surfaced in pipeline:** YAGNI — P-L variance is available via `laplace_covariance()` if needed later for UI display or confidence thresholds. Not added now.
