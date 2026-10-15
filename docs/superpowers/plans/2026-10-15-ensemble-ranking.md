# Ensemble Sliding Window Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace single-run sliding window ranking with a K-run ensemble that aggregates orderings via Plackett-Luce posterior, resolving 5 known weaknesses of the single-run approach.

**Architecture:** Run `rank_sliding_window` K times with different random seeds (default: 42, 137, 271), collect each run's best-to-worst ordering, then call `fit_utilities` from `plackett_luce.py` to compute MAP utility estimates. The new function `rank_sliding_window_ensemble` returns the same `dict[str, float]` interface as the existing single-run function — backward compatible with all callers.

**Tech Stack:** Python 3.12, numpy, scipy (L-BFGS-B via `plackett_luce.fit_utilities`), existing `rank_sliding_window` — no new dependencies.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/candidate_ranking/scoring/sliding_window_ranking.py` | Add `rank_sliding_window_ensemble` + import `fit_utilities` |
| Modify | `tests/scoring/test_sliding_window_ranking.py` | 4 new tests for ensemble function |
| Modify | `src/candidate_ranking/config.py` | Add `ranking_ensemble_seeds`, `ranking_pl_prior_variance`, `_parse_seeds`, env overrides |
| Modify | `tests/test_config.py` | 3 new tests for ensemble config fields |
| Modify | `src/candidate_ranking/graphs/pipeline.py` | Swap import + call to use ensemble |
| Modify | `.env.example` | Document 2 new env vars |

---

## Task 1: Add `rank_sliding_window_ensemble` function

**Files:**
- Modify: `src/candidate_ranking/scoring/sliding_window_ranking.py`
- Test: `tests/scoring/test_sliding_window_ranking.py`

- [ ] **Step 1: Add 4 failing tests to the test file**

Open `tests/scoring/test_sliding_window_ranking.py`. Add the following 4 tests at the end of the file. Also update the import at line 9–17 to include `rank_sliding_window_ensemble`:

```python
from candidate_ranking.scoring.sliding_window_ranking import (
    SLIDING_WINDOW_RANKING_PROMPT,
    _CandidateEvaluation,
    _sliding_window_model,
    build_sliding_window_chain_factory,
    _initial_sort,
    _rank_window,
    rank_sliding_window,
    rank_sliding_window_ensemble,
)
```

Then append the 4 tests:

```python
def test_ensemble_returns_dict_with_all_candidates():
    jd = _jd()
    assessments = {f"cand-{i}": _assessment(f"cand-{i}") for i in range(5)}
    result = rank_sliding_window_ensemble(jd, assessments, _identity_chain_factory(), window_size=3, step=2)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(assessments.keys())
    assert all(isinstance(v, float) for v in result.values())


def test_ensemble_single_candidate_returns_without_chain_call():
    jd = _jd()
    assessments = {"cand-a": _assessment("cand-a")}
    build_chain = MagicMock()
    result = rank_sliding_window_ensemble(jd, assessments, build_chain)
    assert result == {"cand-a": 1.0}
    build_chain.assert_not_called()


def test_ensemble_single_seed_ordering_matches_single_run():
    jd = _jd()
    assessments = {f"cand-{i}": _assessment(f"cand-{i}", strong_met=1, years=i + 1) for i in range(5)}
    chain_factory = _identity_chain_factory()
    single_utils = rank_sliding_window(jd, assessments, chain_factory, window_size=3, step=2, rng_seed=42)
    ensemble_utils = rank_sliding_window_ensemble(jd, assessments, _identity_chain_factory(), window_size=3, step=2, seeds=(42,))
    item_ids = list(assessments)
    single_order = sorted(item_ids, key=single_utils.__getitem__, reverse=True)
    ensemble_order = sorted(item_ids, key=ensemble_utils.__getitem__, reverse=True)
    assert single_order == ensemble_order


def test_ensemble_deterministic_with_same_seeds():
    jd = _jd()
    assessments = {f"cand-{i}": _assessment(f"cand-{i}", strong_met=1, years=i + 1) for i in range(5)}
    result_a = rank_sliding_window_ensemble(jd, assessments, _identity_chain_factory(), window_size=3, step=2, seeds=(42, 137))
    result_b = rank_sliding_window_ensemble(jd, assessments, _identity_chain_factory(), window_size=3, step=2, seeds=(42, 137))
    assert result_a == result_b
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/scoring/test_sliding_window_ranking.py::test_ensemble_returns_dict_with_all_candidates tests/scoring/test_sliding_window_ranking.py::test_ensemble_single_candidate_returns_without_chain_call tests/scoring/test_sliding_window_ranking.py::test_ensemble_single_seed_ordering_matches_single_run tests/scoring/test_sliding_window_ranking.py::test_ensemble_deterministic_with_same_seeds -v
```

Expected: FAIL with `ImportError: cannot import name 'rank_sliding_window_ensemble'`

- [ ] **Step 3: Add `rank_sliding_window_ensemble` to `sliding_window_ranking.py`**

In `src/candidate_ranking/scoring/sliding_window_ranking.py`:

**3a.** Add import at the top of the file, after the existing imports (after line 18 `from candidate_ranking.scoring.llm_assessment import ...`):

```python
from candidate_ranking.ranking.plackett_luce import fit_utilities
```

**3b.** Append the function at the bottom of the file (after `rank_sliding_window`):

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
    """K-run ensemble with Plackett-Luce aggregation. Higher utility = better fit."""
    item_ids = list(assessments)
    if len(item_ids) <= 1:
        return {cid: 1.0 for cid in item_ids}

    rankings: list[list[str]] = []
    for seed in seeds:
        utilities = rank_sliding_window(jd, assessments, build_chain, window_size, step, rng_seed=seed)
        ordering = sorted(item_ids, key=utilities.__getitem__, reverse=True)
        rankings.append(ordering)

    pl_utilities = fit_utilities(rankings, item_ids, pl_prior_variance)
    return {cid: float(pl_utilities[i]) for i, cid in enumerate(item_ids)}
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/scoring/test_sliding_window_ranking.py -v
```

Expected: All tests pass (including the 4 new ones and all 11 existing ones — 15 total).

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/scoring/sliding_window_ranking.py tests/scoring/test_sliding_window_ranking.py
git commit -m "feat: add rank_sliding_window_ensemble with Plackett-Luce aggregation"
```

---

## Task 2: Add ensemble config fields

**Files:**
- Modify: `src/candidate_ranking/config.py`
- Modify: `tests/test_config.py`
- Modify: `.env.example`

- [ ] **Step 1: Write 3 failing tests in `tests/test_config.py`**

Append to the end of the file:

```python
def test_full_config_has_ensemble_defaults():
    cfg = RunConfig.full(Path("/tmp/project"))
    assert cfg.ranking_ensemble_seeds == (42, 137, 271)
    assert cfg.ranking_pl_prior_variance == 1.0


def test_env_override_ensemble_seeds(monkeypatch):
    monkeypatch.setenv("CANDIDATE_RANKING_ENSEMBLE_SEEDS", "1,2,3")
    cfg = apply_env_overrides(RunConfig.full(Path("/tmp/project")))
    assert cfg.ranking_ensemble_seeds == (1, 2, 3)


def test_env_override_pl_prior_variance(monkeypatch):
    monkeypatch.setenv("CANDIDATE_RANKING_RANKING_PL_PRIOR_VARIANCE", "2.0")
    cfg = apply_env_overrides(RunConfig.full(Path("/tmp/project")))
    assert cfg.ranking_pl_prior_variance == 2.0
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_config.py::test_full_config_has_ensemble_defaults tests/test_config.py::test_env_override_ensemble_seeds tests/test_config.py::test_env_override_pl_prior_variance -v
```

Expected: FAIL with `AttributeError: 'RunConfig' object has no attribute 'ranking_ensemble_seeds'`

- [ ] **Step 3: Update `src/candidate_ranking/config.py`**

**3a.** Add 2 new fields to `RunConfig` (after `sliding_window_step: int = 7` on line 27):

```python
    ranking_ensemble_seeds: tuple[int, ...] = (42, 137, 271)
    ranking_pl_prior_variance: float = 1.0
```

**3b.** Add `_parse_seeds` helper function **above** `_ENV_OVERRIDES` (after the `RunConfig` class, before line 46):

```python
def _parse_seeds(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())
```

**3c.** Add 2 entries to `_ENV_OVERRIDES` (after the `CANDIDATE_RANKING_SLIDING_WINDOW_STEP` entry on line 62):

```python
    "CANDIDATE_RANKING_ENSEMBLE_SEEDS": ("ranking_ensemble_seeds", _parse_seeds),
    "CANDIDATE_RANKING_RANKING_PL_PRIOR_VARIANCE": ("ranking_pl_prior_variance", float),
```

- [ ] **Step 4: Update `.env.example`**

After the line `CANDIDATE_RANKING_SLIDING_WINDOW_STEP=7`, add:

```
CANDIDATE_RANKING_ENSEMBLE_SEEDS=42,137,271
CANDIDATE_RANKING_RANKING_PL_PRIOR_VARIANCE=1.0
```

- [ ] **Step 5: Run config tests to verify they pass**

```
pytest tests/test_config.py -v
```

Expected: All 9 tests pass (6 existing + 3 new).

- [ ] **Step 6: Commit**

```bash
git add src/candidate_ranking/config.py tests/test_config.py .env.example
git commit -m "feat: add ranking_ensemble_seeds and ranking_pl_prior_variance to RunConfig"
```

---

## Task 3: Wire ensemble into pipeline

**Files:**
- Modify: `src/candidate_ranking/graphs/pipeline.py`

No new tests needed — `tests/graphs/test_pipeline.py` mocks `build_sliding_window_chain` via `MagicMock()` and does not call into `rank_sliding_window` directly, so it remains valid after this change.

- [ ] **Step 1: Update the import in `pipeline.py`**

In `src/candidate_ranking/graphs/pipeline.py`, change line 26 from:

```python
from candidate_ranking.scoring.sliding_window_ranking import rank_sliding_window
```

to:

```python
from candidate_ranking.scoring.sliding_window_ranking import rank_sliding_window_ensemble
```

- [ ] **Step 2: Update the call in `rank_and_format_jd_node`**

In `pipeline.py`, replace lines 172–178 (the `rank_sliding_window(...)` call) with:

```python
        utilities = rank_sliding_window_ensemble(
            jd,
            qualified,
            build_sliding_window_chain,
            window_size=cfg.sliding_window_size,
            step=cfg.sliding_window_step,
            seeds=cfg.ranking_ensemble_seeds,
            pl_prior_variance=cfg.ranking_pl_prior_variance,
        )
```

- [ ] **Step 3: Run the full test suite**

```
pytest -v
```

Expected: All tests pass. Pay attention to:
- `tests/graphs/test_pipeline.py` — no failures expected since it uses MagicMock for the chain
- `tests/scoring/test_sliding_window_ranking.py` — all 15 tests pass
- `tests/test_config.py` — all 9 tests pass

- [ ] **Step 4: Commit**

```bash
git add src/candidate_ranking/graphs/pipeline.py
git commit -m "feat: wire rank_sliding_window_ensemble into pipeline — ensemble replaces single-run ranking"
```
