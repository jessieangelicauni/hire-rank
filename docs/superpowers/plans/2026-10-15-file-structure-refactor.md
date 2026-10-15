# File Structure Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce file count by merging tiny related files and extracting misplaced helpers to clean up cross-module coupling.

**Architecture:** Three independent refactors — extract rendering helpers out of `llm_assessment.py` into `scoring/render.py`, merge `generation.py`+`json_cache.py` into `utils.py`, and merge `output/formatter.py`+`output/run_output.py` into `output/writer.py`. Each refactor creates one file, updates all import paths, deletes the old file(s), and verifies tests still pass.

**Tech Stack:** Python 3.12, pytest — pure refactor, no new logic.

---

## File Map

| Action | Path | Notes |
|--------|------|-------|
| Create | `src/candidate_ranking/scoring/render.py` | Houses `_extract_jd_summary` + `_render_evidence_for_ranking` |
| Delete | `_extract_jd_summary` + `_render_evidence_for_ranking` from `scoring/llm_assessment.py:283-311` | Moved to `render.py` |
| Create | `src/candidate_ranking/utils.py` | Merges `generation.py` + `json_cache.py` |
| Delete | `src/candidate_ranking/generation.py` | Merged into `utils.py` |
| Delete | `src/candidate_ranking/json_cache.py` | Merged into `utils.py` |
| Create | `src/candidate_ranking/output/writer.py` | Merges `formatter.py` + `run_output.py` |
| Delete | `src/candidate_ranking/output/formatter.py` | Merged into `writer.py` |
| Delete | `src/candidate_ranking/output/run_output.py` | Merged into `writer.py` |
| Modify | `src/candidate_ranking/scoring/sliding_window_ranking.py` | Import from `render` not `llm_assessment` |
| Modify | `src/candidate_ranking/scoring/llm_assessment.py` | Import from `utils` not `generation` |
| Modify | `src/candidate_ranking/scoring/jd_requirements.py` | Import from `utils` |
| Modify | `src/candidate_ranking/scoring/jd_skills.py` | Import from `utils` |
| Modify | `src/candidate_ranking/ingestion/cv.py` | Import from `utils` |
| Modify | `src/candidate_ranking/graphs/pipeline.py` | Import from `output.writer` |
| Modify | `src/candidate_ranking/cli.py` | Import from `output.writer` |
| Modify | `tests/output/test_formatter.py` | Import from `output.writer` |

---

## Task 1: Extract rendering helpers to `scoring/render.py`

**Problem:** `sliding_window_ranking.py` imports `_extract_jd_summary` and `_render_evidence_for_ranking` from `llm_assessment.py` — a ranking module depending on an assessment module is wrong coupling.

**Files:**
- Create: `src/candidate_ranking/scoring/render.py`
- Modify: `src/candidate_ranking/scoring/sliding_window_ranking.py:13-18`
- Modify: `src/candidate_ranking/scoring/llm_assessment.py:283-311`

- [ ] **Step 1: Create `src/candidate_ranking/scoring/render.py`**

```python
from __future__ import annotations

from candidate_ranking.models import Assessment, JobDescription


def _extract_jd_summary(jd: JobDescription) -> str:
    words = jd.raw_text.split()
    return f"{jd.title}\n\n" + " ".join(words[:200])


def _render_evidence_for_ranking(candidate_ids: list[str], assessments: dict[str, Assessment]) -> str:
    blocks = []
    for index, candidate_id in enumerate(candidate_ids, start=1):
        a = assessments[candidate_id]
        must_text = "\n".join(
            f"    [{ev.verdict}] {ev.requirement} ({ev.evidence_strength}, {ev.years_demonstrated}yr): {ev.evidence}"
            for ev in a.must_have_evaluations
        ) or "    (none)"
        met_nice = [ev for ev in a.nice_to_have_evaluations if ev.verdict in ("MET", "PARTIALLY_MET")]
        nice_text = ", ".join(f"{ev.requirement} [{ev.verdict}]" for ev in met_nice) or "(none)"
        ach_text = "\n".join(f"    - {ach}" for ach in a.strongest_achievements) or "    (none)"
        traj = a.career_trajectory
        traj_text = (
            f"{traj.direction}, avg {traj.average_tenure_months}mo/role"
            + (", multinational exp." if traj.has_multinational_experience else "")
        ) if traj else "unknown"
        blocks.append(
            f"{index}:\n"
            f"  Must-have evaluations:\n{must_text}\n"
            f"  Nice-to-have met: {nice_text}\n"
            f"  Achievements:\n{ach_text}\n"
            f"  Trajectory: {traj_text} | Relevant years: {a.total_relevant_years}"
        )
    return "\n\n".join(blocks)
```

- [ ] **Step 2: Update import in `sliding_window_ranking.py`**

Change lines 14-18 from:
```python
from candidate_ranking.generation import GenerationError, invoke_and_validate
from candidate_ranking.models import Assessment, JobDescription
from candidate_ranking.scoring.llm_assessment import (
    _extract_jd_summary,
    _render_evidence_for_ranking,
)
```
To:
```python
from candidate_ranking.generation import GenerationError, invoke_and_validate
from candidate_ranking.models import Assessment, JobDescription
from candidate_ranking.scoring.render import _extract_jd_summary, _render_evidence_for_ranking
```

- [ ] **Step 3: Remove the two functions from `llm_assessment.py`**

Delete lines 282-314 (the blank line before `_extract_jd_summary`, both functions, and the trailing blank lines):
```python
# DELETE this entire block from llm_assessment.py:

def _extract_jd_summary(jd: JobDescription) -> str:
    words = jd.raw_text.split()
    return f"{jd.title}\n\n" + " ".join(words[:200])


def _render_evidence_for_ranking(candidate_ids: list[str], assessments: dict[str, Assessment]) -> str:
    ...
```

After deletion, `llm_assessment.py` should end at `filter_assessable_candidates` (around line 280).

- [ ] **Step 4: Run tests**

```
pytest -v
```

Expected: All 80 tests pass. If any test fails with ImportError, check the import path change in Step 2.

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/scoring/render.py src/candidate_ranking/scoring/sliding_window_ranking.py src/candidate_ranking/scoring/llm_assessment.py
git commit -m "refactor: extract rendering helpers to scoring/render.py — remove coupling from sliding_window to llm_assessment"
```

---

## Task 2: Merge `generation.py` + `json_cache.py` → `utils.py`

**Problem:** Two tiny files (40 + 28 lines) that are always imported together have no reason to be separate.

**Files:**
- Create: `src/candidate_ranking/utils.py`
- Delete: `src/candidate_ranking/generation.py`
- Delete: `src/candidate_ranking/json_cache.py`
- Modify: `src/candidate_ranking/scoring/sliding_window_ranking.py`
- Modify: `src/candidate_ranking/scoring/llm_assessment.py`
- Modify: `src/candidate_ranking/scoring/jd_requirements.py`
- Modify: `src/candidate_ranking/scoring/jd_skills.py`
- Modify: `src/candidate_ranking/ingestion/cv.py`

- [ ] **Step 1: Create `src/candidate_ranking/utils.py`**

```python
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Callable, TypeVar

from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

T = TypeVar("T")


class GenerationError(Exception):
    pass


def invoke_and_validate(
    chain: Runnable,
    payload: dict[str, Any],
    expected_type: type[T],
    validate: Callable[[T], str | None],
    log_context: str,
) -> T:
    try:
        result = chain.invoke(payload)
    except ValueError as exc:
        logger.warning("%s failed schema validation: %s", log_context, exc)
        raise GenerationError(f"{log_context} failed: {exc}") from exc

    if not isinstance(result, expected_type):
        error = ValueError(f"chain returned {type(result).__name__}, expected {expected_type.__name__}")
        logger.warning("%s returned unexpected type %s", log_context, type(result).__name__)
        raise GenerationError(f"{log_context} failed: {error}") from error

    validation_error = validate(result)
    if validation_error is not None:
        logger.warning("%s failed validation: %s", log_context, validation_error)
        raise GenerationError(f"{log_context} failed: {validation_error}")

    return result


def load_json_cache(cache_path: Path, cache_label: str) -> dict:
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Discarding unreadable %s %s: %s", cache_label, cache_path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_json_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(f"{cache_path.suffix}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    tmp.replace(cache_path)
```

- [ ] **Step 2: Update imports in all 5 consumer files**

**`src/candidate_ranking/scoring/sliding_window_ranking.py`** — change line:
```python
# FROM:
from candidate_ranking.generation import GenerationError, invoke_and_validate
# TO:
from candidate_ranking.utils import GenerationError, invoke_and_validate
```

**`src/candidate_ranking/scoring/llm_assessment.py`** — change line:
```python
# FROM:
from candidate_ranking.generation import invoke_and_validate
# TO:
from candidate_ranking.utils import invoke_and_validate
```

**`src/candidate_ranking/scoring/jd_requirements.py`** — change lines:
```python
# FROM:
from candidate_ranking.generation import GenerationError, invoke_and_validate
from candidate_ranking.json_cache import load_json_cache, save_json_cache
# TO:
from candidate_ranking.utils import GenerationError, invoke_and_validate, load_json_cache, save_json_cache
```

**`src/candidate_ranking/scoring/jd_skills.py`** — change lines:
```python
# FROM:
from candidate_ranking.generation import GenerationError, invoke_and_validate
from candidate_ranking.json_cache import load_json_cache, save_json_cache
# TO:
from candidate_ranking.utils import GenerationError, invoke_and_validate, load_json_cache, save_json_cache
```

**`src/candidate_ranking/ingestion/cv.py`** — change lines:
```python
# FROM:
from candidate_ranking.generation import GenerationError, invoke_and_validate
from candidate_ranking.json_cache import load_json_cache, save_json_cache
# TO:
from candidate_ranking.utils import GenerationError, invoke_and_validate, load_json_cache, save_json_cache
```

- [ ] **Step 3: Delete the two old files**

```bash
rm src/candidate_ranking/generation.py
rm src/candidate_ranking/json_cache.py
```

- [ ] **Step 4: Run tests**

```
pytest -v
```

Expected: All 80 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/utils.py src/candidate_ranking/generation.py src/candidate_ranking/json_cache.py src/candidate_ranking/scoring/sliding_window_ranking.py src/candidate_ranking/scoring/llm_assessment.py src/candidate_ranking/scoring/jd_requirements.py src/candidate_ranking/scoring/jd_skills.py src/candidate_ranking/ingestion/cv.py
git commit -m "refactor: merge generation.py + json_cache.py into utils.py"
```

---

## Task 3: Merge `output/formatter.py` + `output/run_output.py` → `output/writer.py`

**Problem:** Two small output files (46 + 62 lines) both write to the `runs/` directory. No architectural reason to keep them separate.

**Files:**
- Create: `src/candidate_ranking/output/writer.py`
- Delete: `src/candidate_ranking/output/formatter.py`
- Delete: `src/candidate_ranking/output/run_output.py`
- Modify: `src/candidate_ranking/graphs/pipeline.py`
- Modify: `src/candidate_ranking/cli.py`
- Modify: `tests/output/test_formatter.py`

- [ ] **Step 1: Create `src/candidate_ranking/output/writer.py`**

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict

from candidate_ranking.models import Assessment, JobDescription


class AssessmentResultRecord(TypedDict):
    jd_id: str
    candidate_id: str
    status: Literal["ok", "failed"]
    assessment: Assessment | None
    error: str | None


def format_jd_ranking(
    jd: JobDescription,
    assessments: dict[str, Assessment],
    utilities: dict[str, float],
) -> tuple[str, dict]:
    ranked = sorted(utilities, key=utilities.__getitem__, reverse=True)

    lines = [
        f"# Ranking: {jd.title} ({jd.id})",
        "",
        "> Ranked by Plackett-Luce utility over LLM listwise comparisons.",
        "",
    ]
    rows = []
    for rank, candidate_id in enumerate(ranked, start=1):
        a = assessments.get(candidate_id)
        utility = utilities[candidate_id]
        lines.append(f"{rank}. **{candidate_id}** (utility={utility:.4f})")
        rows.append({
            "rank": rank,
            "candidate_id": candidate_id,
            "utility": utility,
            "qualified": a.qualified if a else None,
            "total_relevant_years": a.total_relevant_years if a else None,
        })

    return "\n".join(lines) + "\n", {"job_description_id": jd.id, "rankings": rows}


def write_jd_ranking(
    run_dir: Path,
    jd: JobDescription,
    assessments: dict[str, Assessment],
    utilities: dict[str, float],
) -> None:
    markdown_text, json_payload = format_jd_ranking(jd, assessments, utilities)
    jd_dir = run_dir / jd.id
    jd_dir.mkdir(parents=True, exist_ok=True)
    (jd_dir / "ranking.md").write_text(markdown_text, encoding="utf-8")
    (jd_dir / "ranking.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")


def write_run_output(
    runs_dir: Path,
    run_id: str,
    manifest: dict,
    assessment_results: list[AssessmentResultRecord],
) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    failures = [
        {"jd_id": r["jd_id"], "candidate_id": r["candidate_id"], "error": r["error"]}
        for r in assessment_results
        if r["status"] == "failed"
    ]
    (run_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")

    assessments_by_jd: dict[str, dict[str, dict]] = {}
    for r in assessment_results:
        if r["status"] != "ok":
            continue
        assessments_by_jd.setdefault(r["jd_id"], {})[r["candidate_id"]] = r["assessment"].model_dump()

    for jd_id, candidate_assessments in assessments_by_jd.items():
        jd_dir = run_dir / jd_id
        jd_dir.mkdir(parents=True, exist_ok=True)
        (jd_dir / "assessments.json").write_text(json.dumps(candidate_assessments, indent=2), encoding="utf-8")

    succeeded = sum(1 for r in assessment_results if r["status"] == "ok")
    failed = sum(1 for r in assessment_results if r["status"] == "failed")

    per_jd: dict[str, dict[str, int]] = {}
    for r in assessment_results:
        bucket = per_jd.setdefault(r["jd_id"], {"succeeded": 0, "failed": 0})
        bucket["succeeded" if r["status"] == "ok" else "failed"] += 1

    summary = {"succeeded": succeeded, "failed": failed, "per_jd": per_jd}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return run_dir
```

- [ ] **Step 2: Update imports in 3 consumer files**

**`src/candidate_ranking/graphs/pipeline.py`** — change line:
```python
# FROM:
from candidate_ranking.output.formatter import write_jd_ranking
# TO:
from candidate_ranking.output.writer import write_jd_ranking
```

**`src/candidate_ranking/cli.py`** — change line:
```python
# FROM:
from candidate_ranking.output.run_output import write_run_output
# TO:
from candidate_ranking.output.writer import write_run_output
```

**`tests/output/test_formatter.py`** — change line:
```python
# FROM:
from candidate_ranking.output.formatter import format_jd_ranking, write_jd_ranking
# TO:
from candidate_ranking.output.writer import format_jd_ranking, write_jd_ranking
```

- [ ] **Step 3: Delete the two old files**

```bash
rm src/candidate_ranking/output/formatter.py
rm src/candidate_ranking/output/run_output.py
```

- [ ] **Step 4: Run tests**

```
pytest -v
```

Expected: All 80 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/output/writer.py src/candidate_ranking/output/formatter.py src/candidate_ranking/output/run_output.py src/candidate_ranking/graphs/pipeline.py src/candidate_ranking/cli.py tests/output/test_formatter.py
git commit -m "refactor: merge output/formatter.py + output/run_output.py into output/writer.py"
```
