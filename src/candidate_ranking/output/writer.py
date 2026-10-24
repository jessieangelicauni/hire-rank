from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict

from candidate_ranking.models import Assessment, JobProfile
from candidate_ranking.scoring.criterion_score import criterion_score, criterion_score_breakdown


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
        f"# Ranking: {jp.id}",
        "",
        "> Ranked primarily by a transparent, per-criterion evidence score; "
        "Jev's pairwise judgment only breaks ties among candidates the "
        "transparent score alone can't separate.",
        "",
    ]
    rows = []
    for rank, candidate_id in enumerate(ranked, start=1):
        utility = utilities[candidate_id]
        assessment = assessments.get(candidate_id)
        score = criterion_score(assessment) if assessment is not None else None
        score_display = f"score={score:.3f}, " if score is not None else ""
        lines.append(f"{rank}. **{candidate_id}** ({score_display}utility={utility:.4f})")
        rows.append({
            "rank": rank,
            "candidate_id": candidate_id,
            "utility": utility,
            "criterion_score": score,
            "criterion_breakdown": criterion_score_breakdown(assessment) if assessment is not None else [],
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
    assessment_results: list[AssessmentResultRecord],
) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    assessments_by_jp: dict[str, dict[str, dict]] = {}
    for r in assessment_results:
        if r["status"] != "ok":
            continue
        assessments_by_jp.setdefault(r["jp_id"], {})[r["candidate_id"]] = r["assessment"].model_dump()

    for jp_id, candidate_assessments in assessments_by_jp.items():
        jp_dir = run_dir / jp_id
        jp_dir.mkdir(parents=True, exist_ok=True)
        (jp_dir / "assessments.json").write_text(json.dumps(candidate_assessments, indent=2), encoding="utf-8")

    return run_dir
