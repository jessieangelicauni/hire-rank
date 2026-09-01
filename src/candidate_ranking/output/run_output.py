from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict

from candidate_ranking.models import Assessment


class AssessmentResultRecord(TypedDict):
    jd_id: str
    candidate_id: str
    status: Literal["ok", "failed"]
    assessment: Assessment | None
    error: str | None


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

    summary = {
        "succeeded": succeeded,
        "failed": failed,
        "per_jd": per_jd,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return run_dir
