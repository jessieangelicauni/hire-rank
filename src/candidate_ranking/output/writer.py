from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict

from candidate_ranking.models import Assessment, JobProfile


class AssessmentResultRecord(TypedDict):
    jp_id: str
    candidate_id: str
    status: Literal["ok", "failed"]
    assessment: Assessment | None
    error: str | None


def format_jd_ranking(
    jp: JobProfile,
    assessments: dict[str, Assessment],
    utilities: dict[str, float],
) -> tuple[str, dict]:
    ranked = sorted(utilities, key=utilities.__getitem__, reverse=True)

    lines = [
        f"# Ranking: {jp.title} ({jp.id})",
        "",
        "> Ranked by Plackett-Luce utility via pairwise comparisons.",
        "",
    ]
    rows = []
    for rank, candidate_id in enumerate(ranked, start=1):
        a = assessments.get(candidate_id)
        utility = utilities[candidate_id]
        flag = " â€” DISQUALIFIED (failed a must-have)" if a and not a.qualified else ""
        lines.append(f"{rank}. **{candidate_id}** (utility={utility:.4f}){flag}")
        rows.append({
            "rank": rank,
            "candidate_id": candidate_id,
            "utility": utility,
            "qualified": a.qualified if a else None,
            "total_relevant_years": a.total_relevant_years if a else None,
        })

    return "\n".join(lines) + "\n", {"job_profile_id": jp.id, "rankings": rows}


def write_jd_ranking(
    run_dir: Path,
    jp: JobProfile,
    assessments: dict[str, Assessment],
    utilities: dict[str, float],
) -> None:
    markdown_text, json_payload = format_jd_ranking(jp, assessments, utilities)
    jp_dir = run_dir / jp.id
    jp_dir.mkdir(parents=True, exist_ok=True)
    (jp_dir / "ranking.md").write_text(markdown_text, encoding="utf-8")
    (jp_dir / "ranking.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")


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
        {"jp_id": r["jp_id"], "candidate_id": r["candidate_id"], "error": r["error"]}
        for r in assessment_results
        if r["status"] == "failed"
    ]
    (run_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")

    assessments_by_jp: dict[str, dict[str, dict]] = {}
    for r in assessment_results:
        if r["status"] != "ok":
            continue
        assessments_by_jp.setdefault(r["jp_id"], {})[r["candidate_id"]] = r["assessment"].model_dump()

    for jp_id, candidate_assessments in assessments_by_jp.items():
        jp_dir = run_dir / jp_id
        jp_dir.mkdir(parents=True, exist_ok=True)
        (jp_dir / "assessments.json").write_text(json.dumps(candidate_assessments, indent=2), encoding="utf-8")

    succeeded = sum(1 for r in assessment_results if r["status"] == "ok")
    failed = sum(1 for r in assessment_results if r["status"] == "failed")

    per_jp: dict[str, dict[str, int]] = {}
    for r in assessment_results:
        bucket = per_jp.setdefault(r["jp_id"], {"succeeded": 0, "failed": 0})
        bucket["succeeded" if r["status"] == "ok" else "failed"] += 1

    summary = {
        "succeeded": succeeded,
        "failed": failed,
        "per_jp": per_jp,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return run_dir
