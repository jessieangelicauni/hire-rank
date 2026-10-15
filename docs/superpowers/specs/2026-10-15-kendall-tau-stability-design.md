# Kendall-τ Ranking Stability Evaluation — Design Spec

**Date:** 2026-10-15
**Status:** Draft — pending implementation

---

## 1. Goal

Add an on-demand stability evaluation command that measures how consistently the sliding window ranker produces the same ordering when given different initial candidate orderings. This validates window size choices and detects ranking instability before it reaches production.

---

## 2. Why Stability Evaluation

The sliding window ranker is deterministic given a fixed initial sort. To measure stability, we introduce variation by shuffling the initial order (controlled by `rng_seed`) and comparing two independent ranking runs.

A high Kendall-τ (≥ 0.90) means: regardless of starting order, the algorithm converges to the same ranking — the algorithm is reliable. A low τ means the ranking is sensitive to initialization and may not be trustworthy.

Secondary use: ablation study for window size selection. Run with window=10, 15, 20 and compare τ + failure rate to find the optimal window size empirically.

Paper 1 (MC-KG tournament) achieves τ = 0.93–0.98. This evaluation lets us verify the sliding window matches or exceeds that.

---

## 3. Architecture

```
MODIFY:
  src/candidate_ranking/scoring/sliding_window_ranking.py
    → add rng_seed: int | None = None to rank_sliding_window
    → when rng_seed is set, shuffle ordered list after _initial_sort

CREATE:
  src/candidate_ranking/evaluation/__init__.py
  src/candidate_ranking/evaluation/ranking_stability.py
  tests/evaluation/test_ranking_stability.py

MODIFY:
  src/candidate_ranking/cli.py
    → add evaluate-stability subcommand
```

---

## 4. rank_sliding_window Modification

Add `rng_seed: int | None = None` parameter. When `None` (production default), behaviour is unchanged. When set, shuffle the initially sorted list before the sliding window passes:

```python
def rank_sliding_window(
    jd: JobDescription,
    assessments: dict[str, Assessment],
    build_chain: Callable[[type[BaseModel]], Runnable],
    window_size: int = 15,
    step: int = 7,
    rng_seed: int | None = None,
) -> dict[str, float]:
    ordered = _initial_sort(assessments)
    if rng_seed is not None:
        random.Random(rng_seed).shuffle(ordered)
    n = len(ordered)
    if n <= 1:
        return {cid: 1.0 for cid in ordered}
    ...  # rest unchanged
```

This is the only change to `sliding_window_ranking.py`. The production pipeline always calls with `rng_seed=None` (default), so there is zero impact on production behaviour.

---

## 5. evaluation/ranking_stability.py

### 5.1 StabilityReport

```python
from dataclasses import dataclass

@dataclass
class StabilityReport:
    per_jd: dict[str, float]   # jd_id → Kendall-τ
    mean_tau: float
    min_tau: float
    min_tau_jd: str            # JD with lowest τ
```

### 5.2 kendall_tau

```python
def kendall_tau(
    utilities_a: dict[str, float],
    utilities_b: dict[str, float],
) -> float:
    """Kendall-τ between two utility dicts. Returns value in [-1, 1]."""
    from scipy.stats import kendalltau as scipy_kendall
    cids = sorted(utilities_a)
    a = [utilities_a[c] for c in cids]
    b = [utilities_b[c] for c in cids]
    tau, _ = scipy_kendall(a, b)
    return float(tau)
```

### 5.3 evaluate_jd_stability

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
```

### 5.4 run_stability_report

```python
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
            continue
        per_jd[jd_id] = evaluate_jd_stability(jd, qualified, build_chain, window_size, step, seeds)

    if not per_jd:
        raise ValueError("No JDs with ≥ 2 qualified candidates found in cache")

    min_jd = min(per_jd, key=per_jd.__getitem__)
    return StabilityReport(
        per_jd=per_jd,
        mean_tau=sum(per_jd.values()) / len(per_jd),
        min_tau=per_jd[min_jd],
        min_tau_jd=min_jd,
    )
```

---

## 6. CLI Subcommand

```bash
python -m candidate_ranking.cli evaluate-stability
python -m candidate_ranking.cli evaluate-stability --window-size 20 --step 10
python -m candidate_ranking.cli evaluate-stability --seeds 42 137
```

**Data loading:** reads directly from existing cache — no pipeline re-run needed:
- Assessments: `{cache_dir}/assessments/jd-X/cand-Y.json`
- JDs: loaded from `cfg.jd_dir`

**Output (stdout):**

```
Ranking Stability Report
window_size=15  step=7  seeds=(42, 137)
────────────────────────────────────────
JD                          Candidates  τ
backend-engineer            23          0.964
fullstack-engineer          41          0.931
devops-engineer             18          0.978
────────────────────────────────────────
Mean τ : 0.962
Min  τ : 0.931  (fullstack-engineer)
────────────────────────────────────────
Interpretation: τ ≥ 0.90 = stable, 0.75–0.90 = moderate, < 0.75 = unstable
```

Also saved to `{cache_dir}/stability_report.json`.

---

## 7. Assessment Loading for CLI

```python
def load_cached_assessments(cache_dir: Path) -> dict[str, dict[str, Assessment]]:
    """Load all cached assessments grouped by jd_id."""
    assessments_dir = cache_dir / "assessments"
    result: dict[str, dict[str, Assessment]] = {}
    for jd_dir in assessments_dir.iterdir():
        if not jd_dir.is_dir():
            continue
        for candidate_file in jd_dir.glob("*.json"):
            assessment = Assessment.model_validate_json(candidate_file.read_text())
            result.setdefault(jd_dir.name, {})[assessment.candidate_id] = assessment
    return result
```

---

## 8. Key Design Decisions

**rng_seed=None is the production default.**
Adding `rng_seed` to `rank_sliding_window` does not change any existing behaviour. The production pipeline never passes a seed, so the ranking path is identical to before.

**Two seeds, not three.**
Paper 1 uses stability between two runs. Two seeds give τ directly. Three runs would give three pairwise τ values requiring averaging — more complex for no additional insight.

**scipy.stats.kendalltau.**
scipy is already a transitive dependency (numpy, possibly scipy for P-L fitting). No new dependency added.

**No pipeline modification.**
`evaluate-stability` is a completely separate CLI subcommand. It reads from cache, runs 2× ranking per JD, reports. The main pipeline `run` command is untouched.

**Seeds (42, 137).**
Fixed defaults provide reproducible reports. Both are configurable via `--seeds` for ablation studies (e.g., testing multiple seed pairs to confirm stability estimate is seed-independent).
