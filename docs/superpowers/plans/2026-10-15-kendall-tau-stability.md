# Kendall-τ Ranking Stability Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `evaluate-stability` CLI command that measures how consistently the sliding window ranker produces the same ordering when started from different initial candidate orderings.

**Architecture:** Add `rng_seed` parameter to `rank_sliding_window` (shuffles initial order when set, production default is None = unchanged). New `evaluation/ranking_stability.py` module computes Kendall-τ between two independent ranking runs. CLI subcommand loads cached assessments and reports per-JD τ + aggregate.

**Tech Stack:** `scipy.stats.kendalltau` (already a dependency), Python `random.Random`, existing `rank_sliding_window` from `sliding_window_ranking.py`, existing `Assessment` Pydantic model.

**Prerequisite:** This plan assumes `sliding_window_ranking.py` with `rank_sliding_window` already exists (from plan `2026-10-15-sliding-window-ranking.md`). Implement that plan first.

---

## File Map

```
MODIFY:
  src/candidate_ranking/scoring/sliding_window_ranking.py   add rng_seed param
  src/candidate_ranking/cli.py                              add evaluate-stability subcommand
  tests/scoring/test_sliding_window_ranking.py              add rng_seed tests

CREATE:
  src/candidate_ranking/evaluation/__init__.py
  src/candidate_ranking/evaluation/ranking_stability.py
  tests/evaluation/__init__.py
  tests/evaluation/test_ranking_stability.py
```

---

## Task 1: Add rng_seed to rank_sliding_window

**Files:**
- Modify: `src/candidate_ranking/scoring/sliding_window_ranking.py`
- Modify: `tests/scoring/test_sliding_window_ranking.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/scoring/test_sliding_window_ranking.py`:

```python
def test_rank_sliding_window_rng_seed_same_seed_reproducible():
    jd = _jd()
    assessments = {f"cand-{i}": _assessment(f"cand-{i}", strong_met=1, years=i + 1) for i in range(5)}
    build_chain = _identity_chain_factory()
    result_a = rank_sliding_window(jd, assessments, build_chain, window_size=3, step=2, rng_seed=42)
    result_b = rank_sliding_window(jd, assessments, build_chain, window_size=3, step=2, rng_seed=42)
    assert result_a == result_b


def test_rank_sliding_window_rng_seed_none_unchanged():
    jd = _jd()
    assessments = {f"cand-{i}": _assessment(f"cand-{i}", strong_met=1, years=i + 1) for i in range(5)}
    build_chain = _identity_chain_factory()
    result_default = rank_sliding_window(jd, assessments, build_chain, window_size=3, step=2)
    result_none = rank_sliding_window(jd, assessments, build_chain, window_size=3, step=2, rng_seed=None)
    assert result_default == result_none


def test_rank_sliding_window_different_seeds_produce_different_order():
    jd = _jd()
    # All same strong_met → initial sort by years → [cand-0, cand-1, ..., cand-4]
    # Different seeds shuffle this differently → different final utilities with identity chain
    assessments = {f"cand-{i}": _assessment(f"cand-{i}", strong_met=1, years=1) for i in range(5)}
    build_chain = _identity_chain_factory()
    result_42 = rank_sliding_window(jd, assessments, build_chain, window_size=5, step=3, rng_seed=42)
    result_137 = rank_sliding_window(jd, assessments, build_chain, window_size=5, step=3, rng_seed=137)
    assert result_42 != result_137
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/scoring/test_sliding_window_ranking.py -k "rng_seed" -v
```
Expected: FAIL — `rank_sliding_window() got an unexpected keyword argument 'rng_seed'`

- [ ] **Step 3: Update sliding_window_ranking.py**

Add `import random` at the top of `src/candidate_ranking/scoring/sliding_window_ranking.py` (after `from __future__ import annotations`):

```python
import random
```

Update `rank_sliding_window` signature and body — find the current function definition and replace:

```python
def rank_sliding_window(
    jd: JobDescription,
    assessments: dict[str, Assessment],
    build_chain: Callable[[type[BaseModel]], Runnable],
    window_size: int = 15,
    step: int = 7,
    rng_seed: int | None = None,
) -> dict[str, float]:
    """2-pass sliding window LLM ranking. Returns utility per candidate_id (higher = better)."""
    ordered = _initial_sort(assessments)
    if rng_seed is not None:
        random.Random(rng_seed).shuffle(ordered)
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
git commit -m "feat: add rng_seed param to rank_sliding_window for stability evaluation"
```

---

## Task 2: Create evaluation module — kendall_tau + StabilityReport

**Files:**
- Create: `src/candidate_ranking/evaluation/__init__.py`
- Create: `src/candidate_ranking/evaluation/ranking_stability.py`
- Create: `tests/evaluation/__init__.py`
- Create: `tests/evaluation/test_ranking_stability.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/evaluation/test_ranking_stability.py
from __future__ import annotations
import pytest
from candidate_ranking.evaluation.ranking_stability import (
    StabilityReport,
    kendall_tau,
)


def test_kendall_tau_identical_rankings_returns_one():
    utils = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert kendall_tau(utils, utils) == pytest.approx(1.0)


def test_kendall_tau_reversed_rankings_returns_minus_one():
    a = {"a": 3.0, "b": 2.0, "c": 1.0}
    b = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert kendall_tau(a, b) == pytest.approx(-1.0)


def test_kendall_tau_partial_agreement():
    a = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}
    b = {"a": 4.0, "b": 3.0, "c": 1.0, "d": 2.0}
    tau = kendall_tau(a, b)
    assert -1.0 <= tau <= 1.0
    assert tau > 0.0  # mostly agree


def test_stability_report_fields():
    report = StabilityReport(
        per_jd={"jd-1": 0.95, "jd-2": 0.88},
        mean_tau=0.915,
        min_tau=0.88,
        min_tau_jd="jd-2",
    )
    assert report.mean_tau == 0.915
    assert report.min_tau_jd == "jd-2"
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/evaluation/test_ranking_stability.py -v
```
Expected: FAIL — module not found

- [ ] **Step 3: Create evaluation/__init__.py**

Create empty file: `src/candidate_ranking/evaluation/__init__.py`

```python
```

Create empty file: `tests/evaluation/__init__.py`

```python
```

- [ ] **Step 4: Create ranking_stability.py with kendall_tau + StabilityReport**

```python
# src/candidate_ranking/evaluation/ranking_stability.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from candidate_ranking.models import Assessment, JobDescription
from candidate_ranking.scoring.sliding_window_ranking import rank_sliding_window

logger = logging.getLogger(__name__)


@dataclass
class StabilityReport:
    per_jd: dict[str, float]
    mean_tau: float
    min_tau: float
    min_tau_jd: str


def kendall_tau(
    utilities_a: dict[str, float],
    utilities_b: dict[str, float],
) -> float:
    """Kendall-τ between two utility dicts over shared candidate_ids. Returns value in [-1, 1]."""
    from scipy.stats import kendalltau as _kendalltau

    cids = sorted(utilities_a)
    a = [utilities_a[c] for c in cids]
    b = [utilities_b[c] for c in cids]
    tau, _ = _kendalltau(a, b)
    return float(tau)
```

- [ ] **Step 5: Run to verify pass**

```
pytest tests/evaluation/test_ranking_stability.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/candidate_ranking/evaluation/ tests/evaluation/
git commit -m "feat: add evaluation module with kendall_tau and StabilityReport"
```

---

## Task 3: Add evaluate_jd_stability + run_stability_report

**Files:**
- Modify: `src/candidate_ranking/evaluation/ranking_stability.py`
- Modify: `tests/evaluation/test_ranking_stability.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/evaluation/test_ranking_stability.py`:

```python
from unittest.mock import MagicMock
from candidate_ranking.models import Assessment, JobDescription, RequirementEvaluation
from candidate_ranking.scoring.sliding_window_ranking import _CandidateEvaluation
from candidate_ranking.evaluation.ranking_stability import (
    evaluate_jd_stability,
    run_stability_report,
)


def _jd() -> JobDescription:
    return JobDescription(id="jd-1", title="Backend Engineer",
                          raw_text="Python required.", source_path="jd.pdf")


def _assessment(candidate_id: str, years: int = 3) -> Assessment:
    return Assessment(
        job_description_id="jd-1", candidate_id=candidate_id,
        generated_by_model="qwen2.5:14b", qualified=True,
        must_have_evaluations=[
            RequirementEvaluation(
                requirement="Python", verdict="MET", evidence="FastAPI",
                evidence_strength="strong", recency="current", years_demonstrated=years,
            )
        ],
        total_relevant_years=years,
    )


def _identity_chain_factory():
    def _build(model):
        chain = MagicMock()
        def _invoke(payload):
            blocks = [b for b in payload["candidates_text"].split("\n\n") if b.strip()]
            size = len(blocks)
            evals = [_CandidateEvaluation(id=i, key_strengths="s", key_gaps="g")
                     for i in range(1, size + 1)]
            return model(candidate_evaluations=evals, ranking=list(range(1, size + 1)))
        chain.invoke.side_effect = _invoke
        return chain
    return _build


def test_evaluate_jd_stability_returns_float_between_minus_one_and_one():
    jd = _jd()
    assessments = {f"cand-{i}": _assessment(f"cand-{i}", years=i + 1) for i in range(6)}
    tau = evaluate_jd_stability(jd, assessments, _identity_chain_factory(), window_size=3, step=2)
    assert isinstance(tau, float)
    assert -1.0 <= tau <= 1.0


def test_run_stability_report_aggregates_correctly():
    jd1 = _jd()
    jd2 = JobDescription(id="jd-2", title="Frontend", raw_text="React.", source_path="jd2.pdf")
    assessments1 = {f"cand-{i}": _assessment(f"cand-{i}", years=i + 1) for i in range(4)}
    assessments2 = {f"cand-{i}": Assessment(
        job_description_id="jd-2", candidate_id=f"cand-{i}",
        generated_by_model="q", qualified=True,
        must_have_evaluations=[RequirementEvaluation(
            requirement="React", verdict="MET", evidence="Built UI",
            evidence_strength="strong", recency="current", years_demonstrated=i + 1,
        )],
        total_relevant_years=i + 1,
    ) for i in range(4)}
    jd_assessments = {"jd-1": (jd1, assessments1), "jd-2": (jd2, assessments2)}
    report = run_stability_report(jd_assessments, _identity_chain_factory(), window_size=3, step=2)
    assert set(report.per_jd.keys()) == {"jd-1", "jd-2"}
    assert report.min_tau_jd in {"jd-1", "jd-2"}
    assert report.mean_tau == pytest.approx(sum(report.per_jd.values()) / 2)


def test_run_stability_report_skips_jd_with_single_candidate():
    jd = _jd()
    jd_assessments = {"jd-1": (jd, {"cand-a": _assessment("cand-a")})}
    with pytest.raises(ValueError, match="No JDs"):
        run_stability_report(jd_assessments, _identity_chain_factory())
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/evaluation/test_ranking_stability.py -v
```
Expected: FAIL — `evaluate_jd_stability`, `run_stability_report` not found

- [ ] **Step 3: Append to ranking_stability.py**

```python
def evaluate_jd_stability(
    jd: JobDescription,
    assessments: dict[str, Assessment],
    build_chain: Callable[[type[BaseModel]], Runnable],
    window_size: int = 15,
    step: int = 7,
    seeds: tuple[int, int] = (42, 137),
) -> float:
    """Run ranking twice with different seeds, return Kendall-τ."""
    run_a = rank_sliding_window(jd, assessments, build_chain, window_size, step, rng_seed=seeds[0])
    run_b = rank_sliding_window(jd, assessments, build_chain, window_size, step, rng_seed=seeds[1])
    return kendall_tau(run_a, run_b)


def run_stability_report(
    jd_assessments: dict[str, tuple[JobDescription, dict[str, Assessment]]],
    build_chain: Callable[[type[BaseModel]], Runnable],
    window_size: int = 15,
    step: int = 7,
    seeds: tuple[int, int] = (42, 137),
) -> StabilityReport:
    """Run stability evaluation for all JDs. Returns per-JD τ + aggregate."""
    per_jd: dict[str, float] = {}
    for jd_id, (jd, assessments) in jd_assessments.items():
        qualified = {cid: a for cid, a in assessments.items() if a.qualified}
        if len(qualified) < 2:
            logger.info("Skipping JD %s: fewer than 2 qualified candidates", jd_id)
            continue
        per_jd[jd_id] = evaluate_jd_stability(jd, qualified, build_chain, window_size, step, seeds)

    if not per_jd:
        raise ValueError("No JDs with ≥ 2 qualified candidates found")

    min_jd = min(per_jd, key=per_jd.__getitem__)
    return StabilityReport(
        per_jd=per_jd,
        mean_tau=sum(per_jd.values()) / len(per_jd),
        min_tau=per_jd[min_jd],
        min_tau_jd=min_jd,
    )
```

Also add `from langchain_core.runnables import Runnable` to imports at top of file (already imported via sliding_window_ranking — but needed here explicitly):

The full import block at the top of `ranking_stability.py` should be:

```python
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from langchain_core.runnables import Runnable
from pydantic import BaseModel

from candidate_ranking.models import Assessment, JobDescription
from candidate_ranking.scoring.sliding_window_ranking import rank_sliding_window
```

- [ ] **Step 4: Run to verify pass**

```
pytest tests/evaluation/test_ranking_stability.py -v
```
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/evaluation/ranking_stability.py tests/evaluation/test_ranking_stability.py
git commit -m "feat: add evaluate_jd_stability and run_stability_report"
```

---

## Task 4: Add load_cached_assessments + wire CLI subcommand

**Files:**
- Modify: `src/candidate_ranking/evaluation/ranking_stability.py`
- Modify: `src/candidate_ranking/cli.py`
- Modify: `tests/evaluation/test_ranking_stability.py`

- [ ] **Step 1: Write failing test for load_cached_assessments**

Append to `tests/evaluation/test_ranking_stability.py`:

```python
from candidate_ranking.evaluation.ranking_stability import load_cached_assessments


def test_load_cached_assessments_reads_by_jd(tmp_path: Path):
    from pathlib import Path
    assessments_dir = tmp_path / "assessments"
    jd_dir = assessments_dir / "jd-1"
    jd_dir.mkdir(parents=True)
    a = _assessment("cand-a")
    (jd_dir / "cand-a.json").write_text(a.model_dump_json(), encoding="utf-8")
    result = load_cached_assessments(tmp_path)
    assert "jd-1" in result
    assert "cand-a" in result["jd-1"]
    assert result["jd-1"]["cand-a"].candidate_id == "cand-a"


def test_load_cached_assessments_returns_empty_for_missing_dir(tmp_path: Path):
    result = load_cached_assessments(tmp_path)
    assert result == {}
```

- [ ] **Step 2: Run to verify fail**

```
pytest tests/evaluation/test_ranking_stability.py::test_load_cached_assessments_reads_by_jd -v
```
Expected: FAIL — `load_cached_assessments` not found

- [ ] **Step 3: Add load_cached_assessments to ranking_stability.py**

Append to `src/candidate_ranking/evaluation/ranking_stability.py`:

```python
def load_cached_assessments(cache_dir: Path) -> dict[str, dict[str, Assessment]]:
    """Load all cached assessments from disk, grouped by jd_id."""
    assessments_dir = cache_dir / "assessments"
    if not assessments_dir.exists():
        return {}
    result: dict[str, dict[str, Assessment]] = {}
    for jd_dir in assessments_dir.iterdir():
        if not jd_dir.is_dir():
            continue
        for candidate_file in jd_dir.glob("*.json"):
            try:
                assessment = Assessment.model_validate_json(
                    candidate_file.read_text(encoding="utf-8")
                )
                result.setdefault(jd_dir.name, {})[assessment.candidate_id] = assessment
            except Exception as exc:
                logger.warning("Skipping %s: %s", candidate_file, exc)
    return result
```

- [ ] **Step 4: Run to verify load_cached_assessments tests pass**

```
pytest tests/evaluation/test_ranking_stability.py -v
```
Expected: PASS (all tests)

- [ ] **Step 5: Add evaluate-stability subcommand to cli.py**

Find in `src/candidate_ranking/cli.py` the imports block and add:

```python
from candidate_ranking.evaluation.ranking_stability import (
    load_cached_assessments,
    run_stability_report,
)
```

Find the `run_parser = subparsers.add_parser("run")` block in `main()`. After all `run_parser.add_argument(...)` lines, add the new subparser:

```python
    stability_parser = subparsers.add_parser("evaluate-stability")
    stability_parser.add_argument("--window-size", type=int, default=None)
    stability_parser.add_argument("--step", type=int, default=None)
    stability_parser.add_argument("--seeds", type=int, nargs=2, default=[42, 137])
```

Find the `if args.command == "run":` block and add after it:

```python
    elif args.command == "evaluate-stability":
        cfg = RunConfig.full(PROJECT_ROOT)
        cfg = apply_env_overrides(cfg)

        window_size = args.window_size or cfg.sliding_window_size
        step = args.step or cfg.sliding_window_step
        seeds = tuple(args.seeds)

        llm_ranking = ChatOllama(
            model=cfg.ollama_model, base_url=cfg.ollama_base_url,
            temperature=cfg.ranking_temperature, num_ctx=cfg.ollama_num_ctx,
        )
        from candidate_ranking.scoring.sliding_window_ranking import build_sliding_window_chain_factory
        build_chain = build_sliding_window_chain_factory(llm_ranking)

        jds = load_job_descriptions(cfg.jd_dir)
        jd_map = {jd.id: jd for jd in jds}
        cached = load_cached_assessments(cfg.cache_dir)

        jd_assessments = {
            jd_id: (jd_map[jd_id], assessments)
            for jd_id, assessments in cached.items()
            if jd_id in jd_map
        }
        if not jd_assessments:
            print("No cached assessments found. Run the pipeline first.", file=sys.stderr)
            sys.exit(1)

        report = run_stability_report(jd_assessments, build_chain, window_size, step, seeds)

        header = f"window_size={window_size}  step={step}  seeds={seeds}"
        sep = "─" * 56
        print(f"\nRanking Stability Report\n{header}\n{sep}")
        print(f"{'JD':<35} {'Candidates':>10}  {'τ':>6}")
        for jd_id, tau in sorted(report.per_jd.items()):
            n = len([a for a in cached[jd_id].values() if a.qualified])
            print(f"{jd_id:<35} {n:>10}  {tau:>6.3f}")
        print(sep)
        print(f"Mean τ : {report.mean_tau:.3f}")
        print(f"Min  τ : {report.min_tau:.3f}  ({report.min_tau_jd})")
        print(sep)
        print("Interpretation: τ ≥ 0.90 = stable, 0.75–0.90 = moderate, < 0.75 = unstable\n")

        import json as _json
        report_path = cfg.cache_dir / "stability_report.json"
        report_path.write_text(
            _json.dumps({"window_size": window_size, "step": step, "seeds": list(seeds),
                         "per_jd": report.per_jd, "mean_tau": report.mean_tau,
                         "min_tau": report.min_tau, "min_tau_jd": report.min_tau_jd},
                        indent=2),
            encoding="utf-8",
        )
        print(f"Saved to {report_path}")
```

- [ ] **Step 6: Verify import resolves**

```
python -c "from candidate_ranking.cli import main; print('ok')"
```
Expected: `ok`

- [ ] **Step 7: Commit**

```bash
git add src/candidate_ranking/evaluation/ranking_stability.py src/candidate_ranking/cli.py tests/evaluation/test_ranking_stability.py
git commit -m "feat: add load_cached_assessments and evaluate-stability CLI subcommand"
```

---

## Task 5: Full test suite green

- [ ] **Step 1: Run full suite**

```
pytest --tb=short -q
```

- [ ] **Step 2: Fix any failures**

Common patterns:
- Import errors from `evaluation` package → check `__init__.py` files exist
- `rank_sliding_window` signature mismatch → ensure `rng_seed` default is `None` (backward-compatible)

- [ ] **Step 3: Final commit if needed**

```bash
git add -A
git commit -m "test: fix remaining fixtures after stability evaluation addition"
```

---

## Self-Review

**Spec coverage:**
- Section 4 (rng_seed) → Task 1 ✓
- Section 5.1 (StabilityReport) → Task 2 ✓
- Section 5.2 (kendall_tau) → Task 2 ✓
- Section 5.3 (evaluate_jd_stability) → Task 3 ✓
- Section 5.4 (run_stability_report) → Task 3 ✓
- Section 6 (CLI subcommand + output format) → Task 4 ✓
- Section 7 (load_cached_assessments) → Task 4 ✓

**Type consistency:**
- `kendall_tau(dict[str, float], dict[str, float]) -> float` used in `evaluate_jd_stability` ✓
- `evaluate_jd_stability` returns `float`, used in `run_stability_report` per_jd dict ✓
- `run_stability_report` returns `StabilityReport`, used in CLI ✓
- `load_cached_assessments(Path) -> dict[str, dict[str, Assessment]]` used in CLI ✓
- `rng_seed: int | None = None` in `rank_sliding_window` — default None preserves production behavior ✓
