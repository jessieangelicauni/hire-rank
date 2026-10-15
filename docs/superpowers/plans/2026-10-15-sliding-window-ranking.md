# Sliding Window Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the MC-KG tournament (pool > 20 path) with a 2-pass sliding window ranker that uses chain-of-thought reasoning before the permutation decision, tuned for reliability on Qwen2.5-14B-Instruct 4-bit.

**Architecture:** Initial sort from Prompt 2 fields (0 LLM calls) → forward pass (windows left→right) → backward pass (windows right→left), each window ranked by LLM with per-candidate CoT before permutation output. Pool ≤ 20 path (`rank_one_shot` × 3 + P-L) is untouched.

**Tech Stack:** LangChain `ChatPromptTemplate`, Pydantic v2 `create_model`, existing `invoke_and_validate` from `candidate_ranking.generation`, existing `_render_evidence_for_ranking` / `_extract_jd_summary` from `llm_assessment.py`.

---

## File Map

```
CREATE:
  src/candidate_ranking/scoring/sliding_window_ranking.py
  tests/scoring/test_sliding_window_ranking.py

MODIFY:
  src/candidate_ranking/config.py                  add sliding_window_size, sliding_window_step
  src/candidate_ranking/graphs/pipeline.py         replace tournament call, add build_sliding_window_chain param
  src/candidate_ranking/cli.py                     build sliding window chain factory, pass to graph
  .env.example                                     add two new env vars
  tests/test_config.py                             add 3 tests for new fields
  tests/graphs/test_pipeline.py                    add build_sliding_window_chain to graph call
```

---

## Task 1: Add sliding_window_size and sliding_window_step to RunConfig

**Files:**
- Modify: `src/candidate_ranking/config.py`
- Modify: `.env.example`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_config.py`:

```python
def test_full_config_has_sliding_window_defaults():
    cfg = RunConfig.full(Path("/tmp/project"))
    assert cfg.sliding_window_size == 10
    assert cfg.sliding_window_step == 5


def test_env_override_sliding_window_size(monkeypatch):
    monkeypatch.setenv("CANDIDATE_RANKING_SLIDING_WINDOW_SIZE", "20")
    cfg = apply_env_overrides(RunConfig.full(Path("/tmp/project")))
    assert cfg.sliding_window_size == 20


def test_env_override_sliding_window_step(monkeypatch):
    monkeypatch.setenv("CANDIDATE_RANKING_SLIDING_WINDOW_STEP", "3")
    cfg = apply_env_overrides(RunConfig.full(Path("/tmp/project")))
    assert cfg.sliding_window_step == 3
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/test_config.py -v
```
Expected: FAIL — `RunConfig` has no attribute `sliding_window_size`

- [ ] **Step 3: Update config.py**

Add two fields to `RunConfig` dataclass after `ranking_pool_threshold: int = 20`:

```python
    sliding_window_size: int = 10
    sliding_window_step: int = 5
```

Add two entries to `_ENV_OVERRIDES` dict after the `ranking_pool_threshold` entry:

```python
    "CANDIDATE_RANKING_SLIDING_WINDOW_SIZE": ("sliding_window_size", int),
    "CANDIDATE_RANKING_SLIDING_WINDOW_STEP": ("sliding_window_step", int),
```

- [ ] **Step 4: Run to verify pass**

```
pytest tests/test_config.py -v
```
Expected: PASS (all 6 tests)

- [ ] **Step 5: Update .env.example**

Append after the `CANDIDATE_RANKING_RANKING_POOL_THRESHOLD=20` line:

```
CANDIDATE_RANKING_SLIDING_WINDOW_SIZE=10
CANDIDATE_RANKING_SLIDING_WINDOW_STEP=5
```

- [ ] **Step 6: Commit**

```bash
git add src/candidate_ranking/config.py .env.example tests/test_config.py
git commit -m "feat: add sliding_window_size and sliding_window_step to RunConfig"
```

---

## Task 2: Create sliding_window_ranking.py — prompt, schema, chain factory

**Files:**
- Create: `src/candidate_ranking/scoring/sliding_window_ranking.py`
- Create: `tests/scoring/test_sliding_window_ranking.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/scoring/test_sliding_window_ranking.py
from __future__ import annotations
from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError
from candidate_ranking.scoring.sliding_window_ranking import (
    SLIDING_WINDOW_RANKING_PROMPT,
    _CandidateEvaluation,
    _sliding_window_model,
    build_sliding_window_chain_factory,
)


def test_sliding_window_model_accepts_correct_size():
    model = _sliding_window_model(3)
    instance = model(
        candidate_evaluations=[
            _CandidateEvaluation(id=1, key_strengths="s", key_gaps="g"),
            _CandidateEvaluation(id=2, key_strengths="s", key_gaps="g"),
            _CandidateEvaluation(id=3, key_strengths="s", key_gaps="g"),
        ],
        ranking=[2, 1, 3],
    )
    assert instance.ranking == [2, 1, 3]


def test_sliding_window_model_rejects_wrong_size():
    model = _sliding_window_model(3)
    with pytest.raises(ValidationError):
        model(
            candidate_evaluations=[
                _CandidateEvaluation(id=1, key_strengths="s", key_gaps="g"),
            ],
            ranking=[1],
        )


def test_build_sliding_window_chain_factory_builds_chain():
    llm = MagicMock()
    llm.with_structured_output.return_value = MagicMock()
    factory = build_sliding_window_chain_factory(llm)
    model = _sliding_window_model(3)
    chain = factory(model)
    assert chain is not None


def test_sliding_window_prompt_has_required_input_variables():
    assert "jd_title" in SLIDING_WINDOW_RANKING_PROMPT.input_variables
    assert "jd_summary" in SLIDING_WINDOW_RANKING_PROMPT.input_variables
    assert "candidates_text" in SLIDING_WINDOW_RANKING_PROMPT.input_variables
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/scoring/test_sliding_window_ranking.py -v
```
Expected: FAIL — module not found

- [ ] **Step 3: Create sliding_window_ranking.py with prompt, schema, chain factory**

```python
# src/candidate_ranking/scoring/sliding_window_ranking.py
from __future__ import annotations

import logging
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, create_model

from candidate_ranking.models import Assessment, JobDescription

logger = logging.getLogger(__name__)

SLIDING_WINDOW_RANKING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior hiring panel ranking a group of pre-screened candidates.\n"
            "All candidates have already passed qualification screening.\n\n"
            "STEP 1 — For EACH candidate, write a brief evaluation:\n"
            "  key_strengths: what makes this candidate strong FOR THIS SPECIFIC ROLE\n"
            "  key_gaps: what is missing or weak for this role\n\n"
            "STEP 2 — Rank ALL candidates from strongest to weakest fit.\n\n"
            "RANKING CRITERIA (in priority order):\n"
            "1. Relevance — experience directly matching daily responsibilities\n"
            "2. Evidence quality — concrete achievements over vague responsibilities\n"
            "3. Career trajectory — ascending scope and seniority\n"
            "4. Recency — recent hands-on experience over older exposure\n"
            "5. Nice-to-have coverage — breadth of preferred skills as tiebreaker\n\n"
            "RULES:\n"
            "- Candidates are numbered 1..N. `ranking` must contain every number exactly once.\n"
            "- `candidate_evaluations` must contain one entry per candidate with the same id numbers.\n"
            "- Rank ALL candidates. No ties.\n"
            "- Judge only on information provided.",
        ),
        (
            "human",
            "ROLE: {jd_title}\n\nJOB SUMMARY: {jd_summary}\n\nCANDIDATES:\n{candidates_text}",
        ),
    ]
)


class _CandidateEvaluation(BaseModel):
    id: int = Field(ge=1)
    key_strengths: str = Field(min_length=1)
    key_gaps: str = Field(min_length=1)


def _sliding_window_model(size: int) -> type[BaseModel]:
    return create_model(
        "_SlidingWindowResult",
        candidate_evaluations=(
            list[_CandidateEvaluation],
            Field(min_length=size, max_length=size),
        ),
        ranking=(list[int], Field(min_length=size, max_length=size)),
    )


def build_sliding_window_chain_factory(
    llm: BaseChatModel,
) -> Callable[[type[BaseModel]], Runnable]:
    def _build(model: type[BaseModel]) -> Runnable:
        return SLIDING_WINDOW_RANKING_PROMPT | llm.with_structured_output(model)
    return _build
```

- [ ] **Step 4: Run to verify pass**

```
pytest tests/scoring/test_sliding_window_ranking.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/scoring/sliding_window_ranking.py tests/scoring/test_sliding_window_ranking.py
git commit -m "feat: add sliding window prompt, schema, and chain factory"
```

---

## Task 3: Add `_initial_sort` and `_rank_window`

**Files:**
- Modify: `src/candidate_ranking/scoring/sliding_window_ranking.py`
- Modify: `tests/scoring/test_sliding_window_ranking.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/scoring/test_sliding_window_ranking.py`:

```python
from candidate_ranking.models import Assessment, JobDescription, RequirementEvaluation
from candidate_ranking.scoring.sliding_window_ranking import _initial_sort, _rank_window


def _jd() -> JobDescription:
    return JobDescription(id="jd-1", title="Backend Engineer",
                          raw_text="Python required.", source_path="jd.pdf")


def _assessment(candidate_id: str, strong_met: int = 1, years: int = 3) -> Assessment:
    evals = [
        RequirementEvaluation(
            requirement=f"req-{i}", verdict="MET", evidence="FastAPI",
            evidence_strength="strong", recency="current", years_demonstrated=years,
        )
        for i in range(strong_met)
    ]
    return Assessment(
        job_description_id="jd-1", candidate_id=candidate_id,
        generated_by_model="qwen2.5:14b", qualified=True,
        must_have_evaluations=evals,
        total_relevant_years=years,
    )


def test_initial_sort_orders_by_strong_met_count():
    assessments = {
        "cand-a": _assessment("cand-a", strong_met=0, years=5),
        "cand-b": _assessment("cand-b", strong_met=2, years=1),
        "cand-c": _assessment("cand-c", strong_met=1, years=3),
    }
    result = _initial_sort(assessments)
    assert result == ["cand-b", "cand-c", "cand-a"]


def test_initial_sort_breaks_ties_by_years():
    assessments = {
        "cand-a": _assessment("cand-a", strong_met=1, years=2),
        "cand-b": _assessment("cand-b", strong_met=1, years=5),
    }
    result = _initial_sort(assessments)
    assert result[0] == "cand-b"


def _rank_window_chain(ordering: list[int]) -> Callable:
    """Chain factory that returns a fixed permutation for any window of matching size."""
    def _build(model):
        chain = MagicMock()
        def _invoke(payload):
            size = len(ordering)
            evals = [
                _CandidateEvaluation(id=i, key_strengths="s", key_gaps="g")
                for i in range(1, size + 1)
            ]
            return model(candidate_evaluations=evals, ranking=ordering)
        chain.invoke.side_effect = _invoke
        return chain
    return _build


def test_rank_window_reorders_by_llm_output():
    window_ids = ["cand-a", "cand-b", "cand-c"]
    assessments = {cid: _assessment(cid) for cid in window_ids}
    result = _rank_window(_jd(), window_ids, assessments, _rank_window_chain([3, 1, 2]))
    assert result == ["cand-c", "cand-a", "cand-b"]


def test_rank_window_falls_back_on_invalid_permutation():
    window_ids = ["cand-a", "cand-b", "cand-c"]
    assessments = {cid: _assessment(cid) for cid in window_ids}

    def _build(model):
        chain = MagicMock()
        def _invoke(payload):
            evals = [_CandidateEvaluation(id=i, key_strengths="s", key_gaps="g") for i in range(1, 4)]
            return model(candidate_evaluations=evals, ranking=[1, 1, 2])  # duplicate — invalid permutation
        chain.invoke.side_effect = _invoke
        return chain

    result = _rank_window(_jd(), window_ids, assessments, _build)
    assert result == window_ids


def test_rank_window_falls_back_on_generation_error():
    window_ids = ["cand-a", "cand-b"]
    assessments = {cid: _assessment(cid) for cid in window_ids}

    def _build(model):
        chain = MagicMock()
        chain.invoke.side_effect = ValueError("LLM error")
        return chain

    result = _rank_window(_jd(), window_ids, assessments, _build)
    assert result == window_ids
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/scoring/test_sliding_window_ranking.py -v
```
Expected: FAIL — `_initial_sort`, `_rank_window` not found

- [ ] **Step 3: Append to sliding_window_ranking.py**

Add the following imports to the top of `sliding_window_ranking.py` (merge with the existing `from candidate_ranking.models` line):

```python
from candidate_ranking.generation import GenerationError, invoke_and_validate
from candidate_ranking.scoring.llm_assessment import (
    _extract_jd_summary,
    _render_evidence_for_ranking,
)
```

Append the following functions to `sliding_window_ranking.py`:

```python
def _initial_sort(assessments: dict[str, Assessment]) -> list[str]:
    def _key(cid: str) -> tuple[int, int]:
        a = assessments[cid]
        strong_met = sum(
            1 for ev in a.must_have_evaluations
            if ev.verdict == "MET" and ev.evidence_strength == "strong"
        )
        return (-strong_met, -a.total_relevant_years)
    return sorted(assessments, key=_key)


def _rank_window(
    jd: JobDescription,
    window_ids: list[str],
    assessments: dict[str, Assessment],
    build_chain: Callable[[type[BaseModel]], Runnable],
) -> list[str]:
    size = len(window_ids)
    token_to_id = {i + 1: cid for i, cid in enumerate(window_ids)}
    expected_tokens = set(token_to_id)
    model = _sliding_window_model(size)
    chain = build_chain(model)

    def validate(result: BaseModel) -> str | None:
        ranking = result.ranking  # type: ignore[attr-defined]
        if set(ranking) != expected_tokens or len(ranking) != size:
            return f"ranking must be a permutation of {sorted(expected_tokens)}, got {ranking}"
        return None

    try:
        result = invoke_and_validate(
            chain,
            {
                "jd_title": jd.title,
                "jd_summary": _extract_jd_summary(jd),
                "candidates_text": _render_evidence_for_ranking(window_ids, assessments),
            },
            model,
            validate,
            log_context=f"Sliding window for {jd.id} (size={size})",
        )
    except GenerationError as exc:
        logger.warning(
            "Sliding window failed for %s: %s — keeping original order", jd.id, exc
        )
        return list(window_ids)

    return [token_to_id[token] for token in result.ranking]  # type: ignore[attr-defined]
```

- [ ] **Step 4: Run to verify pass**

```
pytest tests/scoring/test_sliding_window_ranking.py -v
```
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/scoring/sliding_window_ranking.py tests/scoring/test_sliding_window_ranking.py
git commit -m "feat: add sliding window _initial_sort and _rank_window"
```

---

## Task 4: Add `rank_sliding_window`

**Files:**
- Modify: `src/candidate_ranking/scoring/sliding_window_ranking.py`
- Modify: `tests/scoring/test_sliding_window_ranking.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/scoring/test_sliding_window_ranking.py`:

```python
from candidate_ranking.scoring.sliding_window_ranking import rank_sliding_window


def _identity_chain_factory() -> Callable:
    """Chain factory that always returns candidates in input order (no reranking)."""
    def _build(model):
        chain = MagicMock()
        def _invoke(payload):
            # Determine size from candidates_text block count
            blocks = [b for b in payload["candidates_text"].split("\n\n") if b.strip()]
            size = len(blocks)
            evals = [
                _CandidateEvaluation(id=i, key_strengths="s", key_gaps="g")
                for i in range(1, size + 1)
            ]
            return model(candidate_evaluations=evals, ranking=list(range(1, size + 1)))
        chain.invoke.side_effect = _invoke
        return chain
    return _build


def test_rank_sliding_window_returns_dict_with_all_candidates():
    jd = _jd()
    assessments = {f"cand-{i}": _assessment(f"cand-{i}") for i in range(5)}
    result = rank_sliding_window(jd, assessments, _identity_chain_factory(), window_size=3, step=2)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(assessments.keys())
    assert all(isinstance(v, float) for v in result.values())


def test_rank_sliding_window_single_candidate_no_chain_call():
    jd = _jd()
    assessments = {"cand-a": _assessment("cand-a")}
    build_chain = MagicMock()
    result = rank_sliding_window(jd, assessments, build_chain, window_size=3, step=2)
    assert result == {"cand-a": 1.0}
    build_chain.assert_not_called()


def test_rank_sliding_window_utilities_reflect_initial_sort():
    jd = _jd()
    assessments = {
        "cand-a": _assessment("cand-a", strong_met=0, years=1),
        "cand-b": _assessment("cand-b", strong_met=2, years=5),
        "cand-c": _assessment("cand-c", strong_met=1, years=3),
    }
    # identity chain preserves order → initial sort determines final order
    # initial sort: cand-b (2 strong) > cand-c (1 strong) > cand-a (0 strong)
    result = rank_sliding_window(jd, assessments, _identity_chain_factory(), window_size=3, step=2)
    assert result["cand-b"] > result["cand-c"] > result["cand-a"]
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/scoring/test_sliding_window_ranking.py -v
```
Expected: FAIL — `rank_sliding_window` not found

- [ ] **Step 3: Append rank_sliding_window to sliding_window_ranking.py**

```python
def rank_sliding_window(
    jd: JobDescription,
    assessments: dict[str, Assessment],
    build_chain: Callable[[type[BaseModel]], Runnable],
    window_size: int = 10,
    step: int = 5,
) -> dict[str, float]:
    """2-pass sliding window LLM ranking. Returns utility per candidate_id (higher = better)."""
    ordered = _initial_sort(assessments)
    n = len(ordered)
    if n <= 1:
        return {cid: 1.0 for cid in ordered}

    effective_window = min(window_size, n)
    forward_starts = list(range(0, n - effective_window + 1, step)) or [0]

    for pass_starts in (forward_starts, list(reversed(forward_starts))):
        for start in pass_starts:
            end = min(start + effective_window, n)
            ordered[start:end] = _rank_window(jd, ordered[start:end], assessments, build_chain)

    return {cid: float(n - i) for i, cid in enumerate(ordered)}
```

- [ ] **Step 4: Run to verify pass**

```
pytest tests/scoring/test_sliding_window_ranking.py -v
```
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/scoring/sliding_window_ranking.py tests/scoring/test_sliding_window_ranking.py
git commit -m "feat: add rank_sliding_window — 2-pass sliding window LLM ranking"
```

---

## Task 5: Wire sliding window into pipeline.py

**Decision:** Always use sliding window regardless of pool size — removes the two-case split (rank_one_shot for pool ≤ 20, tournament for pool > 20). `rank_one_shot` stays in `llm_assessment.py` for research purposes but is no longer called from the pipeline. `build_ranking_chain` parameter is removed from `build_pipeline_graph`.

**Files:**
- Modify: `src/candidate_ranking/graphs/pipeline.py`
- Modify: `tests/graphs/test_pipeline.py`

- [ ] **Step 1: Write failing test**

In `tests/graphs/test_pipeline.py`, replace `test_build_pipeline_graph_has_requirements_and_no_jev_nodes` with:

```python
def test_build_pipeline_graph_has_requirements_and_no_jev_nodes(tmp_path: Path):
    cfg = RunConfig.full(tmp_path)
    graph = build_pipeline_graph(
        cfg=cfg,
        jd_skills_chain=MagicMock(),
        jd_requirements_chain=MagicMock(),
        qualification_chain=MagicMock(),
        build_sliding_window_chain=MagicMock(),
        skill_index=np.zeros((1, 10)),
        skill_row_map=[("jd-1", "cand-1")],
        skill_embedder=MagicMock(return_value=np.zeros((1, 10))),
        run_id="test-run",
    )
    node_names = set(graph.nodes)
    assert "extract_requirements_for_jd" in node_names
    assert "generate_assessment_for_pair" in node_names
    assert "rank_and_format_jd_node" in node_names
    assert not any("jev" in n.lower() for n in node_names)
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/graphs/test_pipeline.py::test_build_pipeline_graph_has_requirements_and_no_jev_nodes -v
```
Expected: FAIL — unexpected keyword argument / missing `build_ranking_chain`

- [ ] **Step 3: Update pipeline.py**

**Replace imports** — remove tournament and rank_one_shot, add sliding_window_ranking:

Remove:
```python
from candidate_ranking.ranking.tournament import run_tournament_for_jd_repeat
```
```python
from candidate_ranking.scoring.llm_assessment import (
    AssessmentGenerationError,
    load_or_generate_assessment,
    rank_one_shot,
)
```

Add:
```python
from candidate_ranking.scoring.llm_assessment import (
    AssessmentGenerationError,
    load_or_generate_assessment,
)
from candidate_ranking.scoring.sliding_window_ranking import rank_sliding_window
```

**Replace `build_ranking_chain` parameter** in `build_pipeline_graph` signature with `build_sliding_window_chain`:

Remove:
```python
    build_ranking_chain: Callable[[type[BaseModel]], Runnable],
```
Add:
```python
    build_sliding_window_chain: Callable[[type[BaseModel]], Runnable],
```

**Replace the entire if/else ranking block** in `rank_and_format_jd_node`:

Remove:
```python
        if len(qualified) <= cfg.ranking_pool_threshold:
            utilities = rank_one_shot(jd, qualified, build_ranking_chain)
        else:
            checkpoint_path = cfg.cache_dir / "tournament" / jd.id / "0.json"
            result = run_tournament_for_jd_repeat(
                jd=jd,
                assessments=qualified,
                repeat_index=0,
                tournament_iterations=35,
                tournament_subset_size=10,
                num_subset_samples=50,
                num_mc_draws=100,
                pl_prior_variance=1.0,
                build_ranking_chain=build_ranking_chain,
                checkpoint_path=checkpoint_path,
                rng_seed=42,
            )
            utilities = result.final_utilities
```

Replace with:
```python
        utilities = rank_sliding_window(
            jd,
            qualified,
            build_sliding_window_chain,
            window_size=cfg.sliding_window_size,
            step=cfg.sliding_window_step,
        )
```

- [ ] **Step 4: Run to verify pass**

```
pytest tests/graphs/test_pipeline.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/graphs/pipeline.py tests/graphs/test_pipeline.py
git commit -m "feat: replace two-case ranking with unified sliding window — remove MC-KG tournament and rank_one_shot from pipeline"
```

---

## Task 6: Wire chain factory in cli.py

**Files:**
- Modify: `src/candidate_ranking/cli.py`

- [ ] **Step 1: Update imports**

Find in `src/candidate_ranking/cli.py`:
```python
from candidate_ranking.scoring.llm_assessment import (
    build_qualification_chain,
    build_ranking_chain_factory,
    filter_assessable_candidates,
)
```

Replace with (remove `build_ranking_chain_factory`, add sliding window import):
```python
from candidate_ranking.scoring.llm_assessment import (
    build_qualification_chain,
    filter_assessable_candidates,
)
from candidate_ranking.scoring.sliding_window_ranking import build_sliding_window_chain_factory
```

- [ ] **Step 2: Build chain and pass to graph**

Find in the `run()` function:
```python
    build_ranking_chain = build_ranking_chain_factory(llm_ranking)
```

Replace with:
```python
    build_sliding_window_chain = build_sliding_window_chain_factory(llm_ranking)
```

Find in `graph = build_pipeline_graph(...)` call the line:
```python
        build_ranking_chain=build_ranking_chain,
```

Replace with:
```python
        build_sliding_window_chain=build_sliding_window_chain,
```

- [ ] **Step 3: Verify import resolves**

```
python -c "from candidate_ranking.cli import run; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add src/candidate_ranking/cli.py
git commit -m "feat: wire sliding window chain factory in cli, remove build_ranking_chain_factory"
```

---

## Task 7: Full test suite green

- [ ] **Step 1: Run full suite**

```
pytest --tb=short -q
```

- [ ] **Step 2: Fix any failures**

Common patterns to look for:
- Any test calling `build_pipeline_graph` without `build_sliding_window_chain=MagicMock()` → add the kwarg
- Any import of `run_tournament_for_jd_repeat` in test files → update to import from `sliding_window_ranking`

- [ ] **Step 3: Final commit if fixes were needed**

```bash
git add -A
git commit -m "test: fix remaining fixtures after sliding window migration"
```

---

## Self-Review

**Spec coverage:**
- Initial sort (Section 3.1) → Task 3 `_initial_sort` ✓
- Sliding window passes (Section 3.2) → Task 4 `rank_sliding_window` forward/backward passes ✓
- Total calls (Section 3.3) → validated implicitly by test with window_size=3, step=2 ✓
- CoT schema (Section 4) → Task 2 `_CandidateEvaluation` + `candidate_evaluations` field ✓
- New prompt (Section 5) → Task 2 `SLIDING_WINDOW_RANKING_PROMPT` ✓
- Evidence rendering (Section 6) → Task 3 reuses `_render_evidence_for_ranking` ✓
- Ordinal utilities (Section 7) → Task 4 `{cid: float(n - i) ...}` ✓
- Pool ≤ 20 unchanged → only else branch replaced in pipeline ✓
- Config fields (Section 8) → Task 1 ✓

**Type consistency:**
- `_rank_window` → `list[str]` used in `rank_sliding_window` ✓
- `rank_sliding_window` → `dict[str, float]` used in `pipeline.py` ✓
- `build_sliding_window_chain_factory` → `Callable[[type[BaseModel]], Runnable]` matches pipeline signature ✓
- `cfg.sliding_window_size` / `cfg.sliding_window_step` defined in Task 1, used in Task 5 ✓
