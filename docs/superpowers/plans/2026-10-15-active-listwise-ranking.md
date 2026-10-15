# Active Listwise Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `rank_sliding_window_ensemble` in the pipeline with an active learning loop that queries the LLM only on the most uncertain candidate subsets, converging faster and with global consistency.

**Architecture:** BM25 warm start (0 LLM calls) → Plackett-Luce prior → iterative variance-based subset selection via `laplace_covariance` → convergence check. `rank_active_listwise` takes `llm: BaseChatModel` directly — no chain factory wrapper needed.

**Tech Stack:** `rank_bm25`, `numpy`, `pydantic.create_model`, `langchain_core`, existing `fit_utilities` + `laplace_covariance` from `plackett_luce.py`.

---

## File Map

| Action | Path |
|--------|------|
| Create | `src/candidate_ranking/scoring/active_listwise_ranking.py` |
| Create | `tests/scoring/test_active_listwise_ranking.py` |
| Modify | `src/candidate_ranking/config.py` |
| Modify | `tests/test_config.py` |
| Modify | `src/candidate_ranking/graphs/pipeline.py` |
| Modify | `src/candidate_ranking/cli.py` |
| Modify | `tests/graphs/test_pipeline.py` |
| Modify | `.env.example` |

---

### Task 1: Active listwise ranking module (TDD)

**Files:**
- Create: `tests/scoring/test_active_listwise_ranking.py`
- Create: `src/candidate_ranking/scoring/active_listwise_ranking.py`

- [ ] **Step 1: Write 5 failing tests**

```python
# tests/scoring/test_active_listwise_ranking.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

from candidate_ranking.models import Assessment, JobDescription, RequirementEvaluation
from candidate_ranking.scoring.active_listwise_ranking import rank_active_listwise

MODULE = "candidate_ranking.scoring.active_listwise_ranking"


def _jd() -> JobDescription:
    return JobDescription(
        id="jd-t", title="Engineer",
        raw_text="Python Django REST API",
        source_path="/tmp/jd.txt",
    )


def _assessment(cid: str, years: int = 3) -> Assessment:
    return Assessment(
        job_description_id="jd-t", candidate_id=cid, generated_by_model="test",
        must_have_evaluations=[RequirementEvaluation(
            requirement="Python", verdict="MET", evidence="used it",
            evidence_strength="strong", recency="current", years_demonstrated=years,
        )],
        total_relevant_years=years, qualified=True,
    )


def test_single_candidate_returns_immediately():
    result = rank_active_listwise(_jd(), {"a": _assessment("a")}, MagicMock())
    assert result == {"a": 1.0}


def test_small_pool_one_llm_call():
    assessments = {f"c{i}": _assessment(f"c{i}") for i in range(5)}
    call_count = 0

    def mock_rank(jd, subset_ids, asmt, llm):
        nonlocal call_count
        call_count += 1
        return list(subset_ids)

    with patch(f"{MODULE}._rank_subset", side_effect=mock_rank):
        result = rank_active_listwise(_jd(), assessments, MagicMock(), subset_size=8)

    assert call_count == 1
    assert set(result) == set(assessments)


def test_returns_float_utilities_for_all_candidates():
    assessments = {f"c{i}": _assessment(f"c{i}", years=i + 1) for i in range(10)}

    def mock_rank(jd, subset_ids, asmt, llm):
        return list(subset_ids)

    with patch(f"{MODULE}._rank_subset", side_effect=mock_rank):
        result = rank_active_listwise(_jd(), assessments, MagicMock(), subset_size=4)

    assert set(result) == set(assessments)
    assert all(isinstance(v, float) for v in result.values())


def test_llm_failure_does_not_crash():
    assessments = {f"c{i}": _assessment(f"c{i}") for i in range(10)}

    with patch(f"{MODULE}._rank_subset", return_value=None):
        result = rank_active_listwise(_jd(), assessments, MagicMock(), subset_size=4, max_iterations=3)

    assert set(result) == set(assessments)


def test_consistent_ranking_orders_candidates():
    assessments = {f"c{i}": _assessment(f"c{i}", years=i + 1) for i in range(10)}

    def mock_rank(jd, subset_ids, asmt, llm):
        return sorted(subset_ids, key=lambda cid: asmt[cid].total_relevant_years, reverse=True)

    with patch(f"{MODULE}._rank_subset", side_effect=mock_rank):
        result = rank_active_listwise(
            _jd(), assessments, MagicMock(),
            subset_size=5, max_iterations=20, min_delta_u=0.01,
        )

    assert result["c9"] > result["c0"]
```

- [ ] **Step 2: Run to verify tests fail**

```
pytest tests/scoring/test_active_listwise_ranking.py -v
```
Expected: ImportError (module doesn't exist yet)

- [ ] **Step 3: Implement the module**

```python
# src/candidate_ranking/scoring/active_listwise_ranking.py
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, create_model

from candidate_ranking.models import Assessment, JobDescription
from candidate_ranking.ranking.plackett_luce import fit_utilities, laplace_covariance
from candidate_ranking.scoring.render import _extract_jd_summary, _render_evidence_for_ranking
from candidate_ranking.utils import GenerationError, invoke_and_validate

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

ACTIVE_RANKING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior hiring panel ranking candidates for a specific role.\n\n"
        "Rank ALL candidates from strongest to weakest fit for this role.\n\n"
        "RANKING CRITERIA (in priority order):\n"
        "1. Relevance — experience directly matching daily responsibilities\n"
        "2. Evidence quality — concrete achievements over vague responsibilities\n"
        "3. Career trajectory — ascending scope and seniority\n"
        "4. Recency — recent hands-on experience\n"
        "5. Nice-to-have coverage — breadth of preferred skills as tiebreaker\n\n"
        "Output ONLY the ranking permutation. No explanations. No ties. "
        "ranking must contain every candidate number exactly once.",
    ),
    ("human", "ROLE: {jd_title}\n\nJOB SUMMARY: {jd_summary}\n\nCANDIDATES:\n{candidates_text}"),
])


def _active_ranking_model(size: int) -> type[BaseModel]:
    return create_model(
        "_ActiveRankingResult",
        ranking=(list[int], Field(min_length=size, max_length=size)),
    )


def _bm25_sort(jd: JobDescription, assessments: dict[str, Assessment]) -> list[str]:
    from rank_bm25 import BM25Okapi

    cids = list(assessments)
    corpus = []
    for cid in cids:
        a = assessments[cid]
        parts = [f"{ev.requirement} {ev.evidence}" for ev in a.must_have_evaluations]
        parts += [f"{ev.requirement} {ev.evidence}" for ev in a.nice_to_have_evaluations if ev.verdict in ("MET", "PARTIALLY_MET")]
        parts += a.strongest_achievements
        corpus.append(" ".join(parts).lower().split())
    scores = BM25Okapi(corpus).get_scores(jd.raw_text.lower().split())
    return [cid for cid, _ in sorted(zip(cids, scores), key=lambda x: -x[1])]


def _select_uncertain_subset(variances: np.ndarray, item_ids: list[str], subset_size: int) -> list[str]:
    return [item_ids[i] for i in np.argsort(variances)[::-1][:subset_size]]


def _rank_subset(
    jd: JobDescription,
    subset_ids: list[str],
    assessments: dict[str, Assessment],
    llm: "BaseChatModel",
) -> list[str] | None:
    size = len(subset_ids)
    token_to_id = {i + 1: cid for i, cid in enumerate(subset_ids)}
    model = _active_ranking_model(size)
    chain = ACTIVE_RANKING_PROMPT | llm.with_structured_output(model)

    def validate(result: BaseModel) -> str | None:
        r = result.ranking  # type: ignore[attr-defined]
        if set(r) != set(token_to_id) or len(r) != size:
            return f"ranking must be a permutation of {sorted(token_to_id)}, got {r}"
        return None

    try:
        result = invoke_and_validate(
            chain,
            {
                "jd_title": jd.title,
                "jd_summary": _extract_jd_summary(jd),
                "candidates_text": _render_evidence_for_ranking(subset_ids, assessments),
            },
            model, validate,
            log_context=f"Active ranking {jd.id} (size={size})",
        )
    except GenerationError as exc:
        logger.warning("Active ranking subset failed for %s: %s — skipping", jd.id, exc)
        return None

    return [token_to_id[t] for t in result.ranking]  # type: ignore[attr-defined]


def rank_active_listwise(
    jd: JobDescription,
    assessments: dict[str, Assessment],
    llm: "BaseChatModel",
    subset_size: int = 8,
    max_iterations: int = 40,
    min_delta_u: float = 0.01,
    pl_prior_variance: float = 1.0,
) -> dict[str, float]:
    """Active listwise ranking via variance-based subset selection + Plackett-Luce.

    Warm-starts from BM25 sort (0 LLM calls), then queries the LLM on the most
    uncertain candidate subsets until convergence or budget exhaustion.
    Higher utility = better fit.
    """
    item_ids = list(assessments)
    n = len(item_ids)

    if n <= 1:
        return {cid: 1.0 for cid in item_ids}

    rankings: list[list[str]] = [_bm25_sort(jd, assessments)]
    u = fit_utilities(rankings, item_ids, pl_prior_variance)

    if n <= subset_size:
        ordering = _rank_subset(jd, item_ids, assessments, llm)
        if ordering is not None:
            rankings.append(ordering)
            u = fit_utilities(rankings, item_ids, pl_prior_variance)
        return {cid: float(u[i]) for i, cid in enumerate(item_ids)}

    min_iter = math.ceil(n / subset_size)
    actual_max_iter = max(max_iterations, 2 * min_iter)

    for iteration in range(1, actual_max_iter + 1):
        variances = np.diag(laplace_covariance(u, rankings, item_ids, pl_prior_variance))
        subset_ids = _select_uncertain_subset(variances, item_ids, subset_size)

        ordering = _rank_subset(jd, subset_ids, assessments, llm)
        if ordering is None:
            continue

        rankings.append(ordering)
        u_new = fit_utilities(rankings, item_ids, pl_prior_variance)
        delta_u = float(np.linalg.norm(u_new - u) / n)
        u = u_new

        if delta_u < min_delta_u and iteration >= min_iter:
            logger.info("Active ranking for %s converged at iteration %d (delta_u=%.4f)", jd.id, iteration, delta_u)
            break
    else:
        logger.warning("Active ranking for %s did not converge after %d iterations", jd.id, actual_max_iter)

    return {cid: float(u[i]) for i, cid in enumerate(item_ids)}
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/scoring/test_active_listwise_ranking.py -v
```
Expected: 5 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add tests/scoring/test_active_listwise_ranking.py src/candidate_ranking/scoring/active_listwise_ranking.py
git commit -m "feat: add rank_active_listwise with BM25 warm start and P-L variance sampling"
```

---

### Task 2: Update config — remove ensemble seeds, add active ranking params

**Files:**
- Modify: `src/candidate_ranking/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Update test_config.py — replace ensemble seed tests with active ranking tests**

In `tests/test_config.py`, remove these two tests:
```python
def test_full_config_has_ensemble_defaults():
    ...

def test_env_override_ensemble_seeds(monkeypatch):
    ...
```

Add these in their place:
```python
def test_full_config_has_active_ranking_defaults():
    cfg = RunConfig.full(Path("/tmp/project"))
    assert cfg.ranking_subset_size == 8
    assert cfg.ranking_max_iterations == 40
    assert cfg.ranking_min_delta_u == 0.01
    assert cfg.ranking_pl_prior_variance == 1.0
    assert not hasattr(cfg, "ranking_ensemble_seeds")


def test_env_override_ranking_subset_size(monkeypatch):
    monkeypatch.setenv("CANDIDATE_RANKING_SUBSET_SIZE", "6")
    cfg = apply_env_overrides(RunConfig.full(Path("/tmp/project")))
    assert cfg.ranking_subset_size == 6


def test_env_override_ranking_max_iterations(monkeypatch):
    monkeypatch.setenv("CANDIDATE_RANKING_MAX_ITERATIONS", "60")
    cfg = apply_env_overrides(RunConfig.full(Path("/tmp/project")))
    assert cfg.ranking_max_iterations == 60


def test_env_override_ranking_min_delta_u(monkeypatch):
    monkeypatch.setenv("CANDIDATE_RANKING_MIN_DELTA_U", "0.005")
    cfg = apply_env_overrides(RunConfig.full(Path("/tmp/project")))
    assert cfg.ranking_min_delta_u == 0.005
```

- [ ] **Step 2: Run to verify tests fail**

```
pytest tests/test_config.py -v
```
Expected: `test_full_config_has_active_ranking_defaults` FAIL (AttributeError: ranking_subset_size)

- [ ] **Step 3: Update config.py**

In `src/candidate_ranking/config.py`:

Replace the `ranking_ensemble_seeds` field and `_parse_seeds` function with the 3 new fields:

**In `RunConfig` dataclass**, replace:
```python
    ranking_ensemble_seeds: tuple[int, ...] = (42,)
    ranking_pl_prior_variance: float = 1.0
```
With:
```python
    ranking_pl_prior_variance: float = 1.0
    ranking_subset_size: int = 8
    ranking_max_iterations: int = 40
    ranking_min_delta_u: float = 0.01
```

**Delete the `_parse_seeds` function** (lines 46-47):
```python
def _parse_seeds(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())
```

**In `_ENV_OVERRIDES`**, replace:
```python
    "CANDIDATE_RANKING_ENSEMBLE_SEEDS": ("ranking_ensemble_seeds", _parse_seeds),
    "CANDIDATE_RANKING_RANKING_PL_PRIOR_VARIANCE": ("ranking_pl_prior_variance", float),
```
With:
```python
    "CANDIDATE_RANKING_RANKING_PL_PRIOR_VARIANCE": ("ranking_pl_prior_variance", float),
    "CANDIDATE_RANKING_SUBSET_SIZE": ("ranking_subset_size", int),
    "CANDIDATE_RANKING_MAX_ITERATIONS": ("ranking_max_iterations", int),
    "CANDIDATE_RANKING_MIN_DELTA_U": ("ranking_min_delta_u", float),
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_config.py -v
```
Expected: all tests PASSED

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/config.py tests/test_config.py
git commit -m "feat: replace ranking_ensemble_seeds with active ranking config params"
```

---

### Task 3: Wire active listwise into pipeline, cli, and env example

**Files:**
- Modify: `src/candidate_ranking/graphs/pipeline.py`
- Modify: `src/candidate_ranking/cli.py`
- Modify: `tests/graphs/test_pipeline.py`
- Modify: `.env.example`

- [ ] **Step 1: Update pipeline.py**

In `src/candidate_ranking/graphs/pipeline.py`:

Replace import:
```python
from candidate_ranking.scoring.sliding_window_ranking import rank_sliding_window_ensemble
```
With:
```python
from candidate_ranking.scoring.active_listwise_ranking import rank_active_listwise
```

In `build_pipeline_graph` signature, rename parameter:
```python
    build_sliding_window_chain: Callable[[type[BaseModel]], Runnable],
```
→
```python
    ranking_llm: "BaseChatModel",
```

Also add the import at the top:
```python
from typing import TYPE_CHECKING
...
if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
```

In `rank_and_format_jd_node`, replace the `rank_sliding_window_ensemble` call:
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
With:
```python
        utilities = rank_active_listwise(
            jd,
            qualified,
            ranking_llm,
            subset_size=cfg.ranking_subset_size,
            max_iterations=cfg.ranking_max_iterations,
            min_delta_u=cfg.ranking_min_delta_u,
            pl_prior_variance=cfg.ranking_pl_prior_variance,
        )
```

Also remove the now-unused `Callable` and `Runnable` imports if they are no longer used elsewhere in the file. Check first — `Callable` is used in other nodes, keep it. Remove only `from pydantic import BaseModel` if no longer used (it's in the function signature of `build_sliding_window_chain` — once that's gone, check if `BaseModel` appears anywhere else in the file).

- [ ] **Step 2: Update test_pipeline.py**

In `tests/graphs/test_pipeline.py`, in `test_build_pipeline_graph_has_requirements_and_no_jev_nodes`, rename:
```python
        build_sliding_window_chain=MagicMock(),
```
→
```python
        ranking_llm=MagicMock(),
```

- [ ] **Step 3: Update cli.py**

In `src/candidate_ranking/cli.py`, in the `run()` function:

Remove the line:
```python
    build_sliding_window_chain = build_sliding_window_chain_factory(llm_ranking)
```

Change in the `build_pipeline_graph(...)` call:
```python
        build_sliding_window_chain=build_sliding_window_chain,
```
→
```python
        ranking_llm=llm_ranking,
```

> **Note:** Keep `from candidate_ranking.scoring.sliding_window_ranking import build_sliding_window_chain_factory` — it is still used in the `evaluate-stability` branch of `main()`.

- [ ] **Step 4: Update .env.example**

Replace:
```
CANDIDATE_RANKING_SLIDING_WINDOW_SIZE=15
CANDIDATE_RANKING_SLIDING_WINDOW_STEP=7
CANDIDATE_RANKING_ENSEMBLE_SEEDS=42
CANDIDATE_RANKING_RANKING_PL_PRIOR_VARIANCE=1.0
```
With:
```
CANDIDATE_RANKING_SLIDING_WINDOW_SIZE=15
CANDIDATE_RANKING_SLIDING_WINDOW_STEP=7
CANDIDATE_RANKING_RANKING_PL_PRIOR_VARIANCE=1.0
CANDIDATE_RANKING_SUBSET_SIZE=8
CANDIDATE_RANKING_MAX_ITERATIONS=40
CANDIDATE_RANKING_MIN_DELTA_U=0.01
```

- [ ] **Step 5: Run full test suite**

```
pytest --tb=short -q
```
Expected: all tests PASSED (no references to `ranking_ensemble_seeds` or `build_sliding_window_chain` remaining)

- [ ] **Step 6: Commit**

```bash
git add src/candidate_ranking/graphs/pipeline.py src/candidate_ranking/cli.py tests/graphs/test_pipeline.py .env.example
git commit -m "feat: wire rank_active_listwise into pipeline, replace sliding window ensemble"
```
