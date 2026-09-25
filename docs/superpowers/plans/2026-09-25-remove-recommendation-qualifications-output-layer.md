# Remove Recommendation/Qualifications from Output Layer (Sub-project 2/5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `src/candidate_ranking/output/formatter.py` and `src/candidate_ranking/output/console_export.py` from reading `overall_recommendation`/`meets_min_qualifications` (removed from `Assessment` in sub-project 1/5), and remove the sub-project-1 `xfail` markers that were tracking this breakage.

**Architecture:** Both files are purely subtractive edits — stop reading/writing the two retired fields wherever they appear, no new behavior. `console_export.py`'s exported JSON shape drops `hireRate`/`meetsMinRate`/`overallRecommendation`/`meetsMinQualifications` entirely, which sub-project 3/5 (frontend) will pick up.

**Tech Stack:** Python 3, pydantic, pytest.

## Global Constraints

- Do not touch any file under `console-web/` — sub-project 3/5's job, triggered by this sub-project's changed `real-data.json` shape.
- Do not touch `scripts/run_jev_evaluation_study.py` (its `collect_test_retest` still reads `assessment.overall_recommendation` from a live `Assessment` object and will `AttributeError` independent of this sub-project — that's sub-project 4/5's job).
- `console_export.py`'s `_evaluation_summary` function is NOT touched — it reads `report.json`'s `recommendation_full_agreement_rate` via `.get()`, which already degrades gracefully (returns `None`) once that key eventually disappears from a future `report.json`; nothing to fix here.
- Existing `test_retest.json` files on disk still carry `overall_recommendation`/`meets_min_qualifications` per repeat (sub-project 4/5 hasn't touched their producer yet) — `_repeat_samples_by_row` simply stops reading those two keys; an unused extra key in the source dict is harmless, not an error.

---

### Task 1: Fix `formatter.py` and remove its sub-project-1 `xfail` markers

**Files:**
- Modify: `src/candidate_ranking/output/formatter.py`
- Modify: `tests/output/test_formatter.py`
- Modify: `tests/graphs/test_pipeline.py`

**Interfaces:**
- Consumes: nothing from other tasks in this plan.
- Produces: `format_jd_ranking(jd, assessments) -> tuple[str, dict]` whose markdown line and `rows` dicts no longer contain `overall_recommendation`/`meets_min_qualifications` — nothing later in this plan depends on this signature (Task 2 is independent).

- [ ] **Step 1: Rewrite `tests/output/test_formatter.py`**

Replace the entire file with:

```python
from __future__ import annotations

import json
from pathlib import Path

from candidate_ranking.models import Assessment, JobDescription
from candidate_ranking.output.formatter import format_jd_ranking, write_jd_ranking


def _jd() -> JobDescription:
    return JobDescription(id="jd-1", title="Backend Engineer", raw_text="...", source_path="jd.pdf")


def _assessment(candidate_id: str, score: float) -> Assessment:
    return Assessment(
        job_description_id="jd-1",
        candidate_id=candidate_id,
        generated_by_model="typesafe/jev",
        requirement_scores={"Python": score},
        confidence={"requirement::Python": 0.9},
    )


def test_format_jd_ranking_sorts_descending_by_score():
    assessments = {
        "cand-a": _assessment("cand-a", 40.0),
        "cand-b": _assessment("cand-b", 90.0),
        "cand-c": _assessment("cand-c", 65.0),
    }

    markdown_text, json_payload = format_jd_ranking(_jd(), assessments)

    rows = json_payload["rankings"]
    assert [row["candidate_id"] for row in rows] == ["cand-b", "cand-c", "cand-a"]
    assert [row["rank"] for row in rows] == [1, 2, 3]
    assert rows[0]["composite_fit_score"] == 90.0
    assert rows[0]["requirement_scores"] == {"Python": 90.0}
    assert "overall_recommendation" not in rows[0]
    assert "meets_min_qualifications" not in rows[0]
    assert json_payload["job_description_id"] == "jd-1"
    assert "cand-b" in markdown_text
    assert markdown_text.index("cand-b") < markdown_text.index("cand-c") < markdown_text.index("cand-a")


def test_write_jd_ranking_writes_markdown_and_json(tmp_path: Path):
    assessments = {"cand-a": _assessment("cand-a", 55.0)}

    write_jd_ranking(tmp_path, _jd(), assessments)

    assert (tmp_path / "jd-1" / "ranking.md").exists()
    json_payload = json.loads((tmp_path / "jd-1" / "ranking.json").read_text(encoding="utf-8"))
    assert json_payload["rankings"][0]["candidate_id"] == "cand-a"
```

This drops the sub-project-1 `pytestmark = pytest.mark.xfail(...)` line and the now-unneeded `import pytest`, and simplifies `_assessment(...)` to no longer take `recommendation`/`meets_min` parameters.

- [ ] **Step 2: Remove the `xfail` decorator in `tests/graphs/test_pipeline.py`**

```python
# OLD
@pytest.mark.xfail(
    reason="formatter.py still reads Assessment.overall_recommendation/meets_min_qualifications, "
    "removed in sub-project 1/5 of docs/superpowers/specs/2026-09-25-remove-recommendation-"
    "qualifications-certification-core-design.md -- fixed by sub-project 2/5 (output layer)",
    strict=False,
)
def test_rank_and_format_jd_writes_ranking_files(tmp_path: Path):

# NEW
def test_rank_and_format_jd_writes_ranking_files(tmp_path: Path):
```

Its `Assessment(...)` fixture was already fixed in sub-project 1 (no `overall_recommendation`/`meets_min_qualifications` kwargs) — only the decorator needs removing. Leave everything else in this file untouched. If, after Step 2, `import pytest` in this file is no longer used by any other test, leave the import in place regardless — do not remove it in this task (out of scope; a lint concern, not a correctness one, and this file may still need `pytest` for other reasons not visible from this task alone).

- [ ] **Step 3: Run both test files and confirm they currently fail**

Run: `uv run pytest tests/output/test_formatter.py tests/graphs/test_pipeline.py -v`

Expected: `tests/output/test_formatter.py`'s two tests `FAIL` with `AttributeError: 'Assessment' object has no attribute 'overall_recommendation'` (raised from inside `format_jd_ranking`, called by the test — current `formatter.py` still reads it). `tests/graphs/test_pipeline.py::test_rank_and_format_jd_writes_ranking_files` also `FAIL`s the same way (no longer `XFAIL`, since the decorator is gone) — this is expected RED, not a problem.

- [ ] **Step 4: Update `src/candidate_ranking/output/formatter.py`**

```python
# OLD
def format_jd_ranking(jd: JobDescription, assessments: dict[str, Assessment]) -> tuple[str, dict]:
    ranked = sorted(assessments, key=lambda cid: assessments[cid].composite_fit_score, reverse=True)

    lines = [
        f"# Ranking: {jd.title} ({jd.id})",
        "",
        "> Scores, recommendations, and per-requirement fit are produced directly by Jev.",
        "",
    ]
    rows = []
    for rank, candidate_id in enumerate(ranked, start=1):
        assessment = assessments[candidate_id]
        lines.append(
            f"{rank}. **{candidate_id}** (score={assessment.composite_fit_score:.1f}, "
            f"{assessment.overall_recommendation}) — meets_min_qualifications="
            f"{assessment.meets_min_qualifications}"
        )
        rows.append(
            {
                "rank": rank,
                "candidate_id": candidate_id,
                "composite_fit_score": assessment.composite_fit_score,
                "overall_recommendation": assessment.overall_recommendation,
                "meets_min_qualifications": assessment.meets_min_qualifications,
                "requirement_scores": assessment.requirement_scores,
            }
        )

    markdown_text = "\n".join(lines) + "\n"
    json_payload = {"job_description_id": jd.id, "rankings": rows}
    return markdown_text, json_payload

# NEW
def format_jd_ranking(jd: JobDescription, assessments: dict[str, Assessment]) -> tuple[str, dict]:
    ranked = sorted(assessments, key=lambda cid: assessments[cid].composite_fit_score, reverse=True)

    lines = [
        f"# Ranking: {jd.title} ({jd.id})",
        "",
        "> Scores and per-requirement fit are produced directly by Jev.",
        "",
    ]
    rows = []
    for rank, candidate_id in enumerate(ranked, start=1):
        assessment = assessments[candidate_id]
        lines.append(f"{rank}. **{candidate_id}** (score={assessment.composite_fit_score:.1f})")
        rows.append(
            {
                "rank": rank,
                "candidate_id": candidate_id,
                "composite_fit_score": assessment.composite_fit_score,
                "requirement_scores": assessment.requirement_scores,
            }
        )

    markdown_text = "\n".join(lines) + "\n"
    json_payload = {"job_description_id": jd.id, "rankings": rows}
    return markdown_text, json_payload
```

- [ ] **Step 5: Run both test files and confirm they pass**

Run: `uv run pytest tests/output/test_formatter.py tests/graphs/test_pipeline.py -v`

Expected: all tests `PASS` (no `XFAIL`, no `FAIL`).

- [ ] **Step 6: Commit**

```bash
git add src/candidate_ranking/output/formatter.py tests/output/test_formatter.py tests/graphs/test_pipeline.py
git commit -m "$(cat <<'EOF'
fix: stop formatter.py reading removed Assessment fields

formatter.py no longer reads overall_recommendation/meets_min_qualifications
(removed from Assessment in sub-project 1/5); removes the xfail markers
sub-project 1/5 left tracking this breakage. Sub-project 2/5 of
docs/superpowers/specs/2026-09-25-remove-recommendation-qualifications-
output-layer-design.md.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Fix `console_export.py`

**Files:**
- Modify: `src/candidate_ranking/output/console_export.py`
- Modify: `tests/output/test_console_export.py`

**Interfaces:**
- Consumes: nothing from Task 1 (independent file).
- Produces: `export_console_web_data(cfg, run_id, output_path)` whose written JSON's `comparison[jd_id]` no longer has `meetsMinRate`/`hireRate`, whose `assessments[row_id]` no longer has `overall_recommendation`/`meets_min_qualifications`, and whose `repeatSamples[row_id]` entries no longer have `overallRecommendation`/`meetsMinQualifications`. This is what sub-project 3/5 (frontend) must assume when it updates `console-web/src/data.ts` and the views that read this file.

- [ ] **Step 1: Update `tests/output/test_console_export.py`'s fixture and assertions**

Edit 1 — the `run_with_jev_ranking` fixture's `ranking` and `assessments` dicts drop the two fields (simulating what a fresh sub-project-1 pipeline run now actually produces):

```python
# OLD
    ranking = {
        "job_description_id": "jd-1",
        "rankings": [
            {
                "rank": 1, "candidate_id": "cand-a", "composite_fit_score": 87.5,
                "overall_recommendation": "hire", "meets_min_qualifications": True,
                "requirement_scores": {"Python": 100.0},
            }
        ],
    }
    (run_dir / "jd-1" / "ranking.json").write_text(json.dumps(ranking), encoding="utf-8")

    assessments = {
        "cand-a": {
            "job_description_id": "jd-1", "candidate_id": "cand-a", "generated_by_model": "typesafe/jev",
            "composite_fit_score": 87.5, "overall_recommendation": "hire", "meets_min_qualifications": True,
            "requirement_scores": {"Python": 100.0}, "confidence": {"overall_recommendation": 0.9},
            "seniority_years_fit_score": 75.0, "education_fit_score": None,
        }
    }
    (run_dir / "jd-1" / "assessments.json").write_text(json.dumps(assessments), encoding="utf-8")

# NEW
    ranking = {
        "job_description_id": "jd-1",
        "rankings": [
            {
                "rank": 1, "candidate_id": "cand-a", "composite_fit_score": 87.5,
                "requirement_scores": {"Python": 100.0},
            }
        ],
    }
    (run_dir / "jd-1" / "ranking.json").write_text(json.dumps(ranking), encoding="utf-8")

    assessments = {
        "cand-a": {
            "job_description_id": "jd-1", "candidate_id": "cand-a", "generated_by_model": "typesafe/jev",
            "composite_fit_score": 87.5,
            "requirement_scores": {"Python": 100.0}, "confidence": {"requirement::Python": 0.9},
            "seniority_years_fit_score": 75.0, "education_fit_score": None,
        }
    }
    (run_dir / "jd-1" / "assessments.json").write_text(json.dumps(assessments), encoding="utf-8")
```

Edit 2 — `test_export_console_web_data_maps_jev_scores`'s assertions:

```python
# OLD
    assert assessment["composite_fit_score"] == 87.5
    assert assessment["overall_recommendation"] == "hire"
    assert assessment["requirement_scores"] == {"Python": 100.0}
    assert assessment["seniority_years_fit_score"] == 75.0
    assert assessment["education_fit_score"] is None
    assert "strengths" not in assessment
    comparison = data["comparison"]["jd-1"]
    assert comparison["meanFitScore"] == 87.5
    assert comparison["meetsMinRate"] == 1.0
    assert comparison["hireRate"] == 1.0
    assert comparison["rankingStability"] is None

# NEW
    assert assessment["composite_fit_score"] == 87.5
    assert "overall_recommendation" not in assessment
    assert "meets_min_qualifications" not in assessment
    assert assessment["requirement_scores"] == {"Python": 100.0}
    assert assessment["seniority_years_fit_score"] == 75.0
    assert assessment["education_fit_score"] is None
    assert "strengths" not in assessment
    comparison = data["comparison"]["jd-1"]
    assert comparison["meanFitScore"] == 87.5
    assert "meetsMinRate" not in comparison
    assert "hireRate" not in comparison
    assert comparison["rankingStability"] is None
```

Edit 3 — `test_export_console_web_data_includes_repeat_samples_when_test_retest_exists`'s assertions (the fixture's `test_retest` list keeps its `overall_recommendation`/`meets_min_qualifications` per repeat unchanged — it simulates an existing on-disk `test_retest.json` from before sub-project 4/5, and the point of this edit is proving `_repeat_samples_by_row` now ignores those two keys):

```python
# OLD
    assert samples[0] == {"compositeFitScore": 86.0, "overallRecommendation": "hire", "meetsMinQualifications": True}
    assert samples[2]["overallRecommendation"] == "maybe"

# NEW
    assert samples[0] == {"compositeFitScore": 86.0}
    assert samples[2] == {"compositeFitScore": 87.0}
```

`test_export_console_web_data_includes_evaluation_summary_when_report_exists` and `test_export_console_web_data_repeat_samples_empty_when_no_test_retest` need no changes — neither references `overall_recommendation`/`meets_min_qualifications`/`hireRate`/`meetsMinRate` anywhere in their own bodies.

- [ ] **Step 2: Run the test file and confirm it currently fails**

Run: `uv run pytest tests/output/test_console_export.py -v`

Expected: `test_export_console_web_data_maps_jev_scores` fails with `KeyError: 'overall_recommendation'` (raised inside `_comparison_from_assessments`, since the fixture no longer has that key but the current `console_export.py` still reads it unconditionally). `test_export_console_web_data_includes_repeat_samples_when_test_retest_exists` fails on the changed assertion (current code still includes `overallRecommendation`/`meetsMinQualifications` in each sample, so the dict doesn't equal `{"compositeFitScore": 86.0}`).

- [ ] **Step 3: Update `src/candidate_ranking/output/console_export.py`**

Edit 1 — `_null_comparison`:

```python
# OLD
def _null_comparison(jd_id: str) -> dict:
    return {"jdId": jd_id, "meanFitScore": None, "meetsMinRate": None, "hireRate": None, "rankingStability": None}

# NEW
def _null_comparison(jd_id: str) -> dict:
    return {"jdId": jd_id, "meanFitScore": None, "rankingStability": None}
```

Edit 2 — `_comparison_from_assessments`:

```python
# OLD
def _comparison_from_assessments(jd_id: str, jd_assessments: dict[str, dict], ranking_stability: float | None) -> dict:
    if not jd_assessments:
        return _null_comparison(jd_id)
    fit_scores = [_composite_fit_score(a) for a in jd_assessments.values()]
    meets_min_flags = [a["meets_min_qualifications"] for a in jd_assessments.values()]
    recommendations = [a["overall_recommendation"] for a in jd_assessments.values()]
    return {
        "jdId": jd_id,
        "meanFitScore": statistics.mean(fit_scores),
        "meetsMinRate": sum(meets_min_flags) / len(meets_min_flags),
        "hireRate": recommendations.count("hire") / len(recommendations),
        "rankingStability": ranking_stability,
    }

# NEW
def _comparison_from_assessments(jd_id: str, jd_assessments: dict[str, dict], ranking_stability: float | None) -> dict:
    if not jd_assessments:
        return _null_comparison(jd_id)
    fit_scores = [_composite_fit_score(a) for a in jd_assessments.values()]
    return {
        "jdId": jd_id,
        "meanFitScore": statistics.mean(fit_scores),
        "rankingStability": ranking_stability,
    }
```

Edit 3 — `_repeat_samples_by_row`:

```python
# OLD
def _repeat_samples_by_row(records: list[dict] | None) -> dict[str, list[dict]]:
    if records is None:
        return {}
    result: dict[str, list[dict]] = {}
    for record in records:
        row_id = f"{record['candidate_id']}::{record['jd_id']}"
        result[row_id] = [
            {
                "compositeFitScore": repeat["composite_fit_score"],
                "overallRecommendation": repeat["overall_recommendation"],
                "meetsMinQualifications": repeat["meets_min_qualifications"],
            }
            for repeat in record["repeats"]
        ]
    return result

# NEW
def _repeat_samples_by_row(records: list[dict] | None) -> dict[str, list[dict]]:
    if records is None:
        return {}
    result: dict[str, list[dict]] = {}
    for record in records:
        row_id = f"{record['candidate_id']}::{record['jd_id']}"
        result[row_id] = [
            {"compositeFitScore": repeat["composite_fit_score"]}
            for repeat in record["repeats"]
        ]
    return result
```

Edit 4 — main export loop's per-candidate assessment dict:

```python
# OLD
            assessments[row_id] = {
                "composite_fit_score": _composite_fit_score(assessment_entry),
                "overall_recommendation": assessment_entry["overall_recommendation"],
                "meets_min_qualifications": assessment_entry["meets_min_qualifications"],
                "requirement_scores": assessment_entry.get("requirement_scores", {}),
                "seniority_years_fit_score": assessment_entry.get("seniority_years_fit_score"),
                "education_fit_score": assessment_entry.get("education_fit_score"),
            }

# NEW
            assessments[row_id] = {
                "composite_fit_score": _composite_fit_score(assessment_entry),
                "requirement_scores": assessment_entry.get("requirement_scores", {}),
                "seniority_years_fit_score": assessment_entry.get("seniority_years_fit_score"),
                "education_fit_score": assessment_entry.get("education_fit_score"),
            }
```

- [ ] **Step 4: Run the test file and confirm it passes**

Run: `uv run pytest tests/output/test_console_export.py -v`

Expected: all tests `PASS`.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest`

Expected: all tests `PASS` (no `XFAIL` remaining anywhere — sub-project 1's two markers were removed in Task 1, and Task 2 touches no other xfailed test).

- [ ] **Step 6: Commit**

```bash
git add src/candidate_ranking/output/console_export.py tests/output/test_console_export.py
git commit -m "$(cat <<'EOF'
fix: stop console_export.py reading removed Assessment fields

console_export.py's exported real-data.json drops hireRate,
meetsMinRate, overallRecommendation, and meetsMinQualifications
entirely -- these depended on Assessment fields removed in
sub-project 1/5. Sub-project 3/5 (frontend) picks up the new,
smaller data.ts shape from here.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** formatter.py markdown/JSON changes and caption wording → Task 1 Step 4; all four console_export.py call sites (`_null_comparison`, `_comparison_from_assessments`, `_repeat_samples_by_row`, main export loop) → Task 2 Step 3; frontend interface handoff documented in Task 2's "Produces" and this plan's Global Constraints; out-of-scope items (`console-web/`, `run_jev_evaluation_study.py`, `_evaluation_summary`) explicitly called out in Global Constraints so neither task's implementer touches them.
- **Placeholder scan:** none — every step has literal code, exact commands, and expected output.
- **Type/name consistency:** `_assessment(candidate_id, score)`'s new two-parameter signature in Task 1's test rewrite is used consistently across both of that file's tests. Task 2's fixture and assertion edits use the same key names (`overall_recommendation`, `meets_min_qualifications`, `hireRate`, `meetsMinRate`, `overallRecommendation`, `meetsMinQualifications`) as the source-code edits in the same task, so a reviewer can check them against each other directly.
