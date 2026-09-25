from __future__ import annotations

import json
from pathlib import Path

from candidate_ranking.models import Assessment, JobDescription


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


def write_jd_ranking(run_dir: Path, jd: JobDescription, assessments: dict[str, Assessment]) -> None:
    markdown_text, json_payload = format_jd_ranking(jd, assessments)
    jd_dir = run_dir / jd.id
    jd_dir.mkdir(parents=True, exist_ok=True)
    (jd_dir / "ranking.md").write_text(markdown_text, encoding="utf-8")
    (jd_dir / "ranking.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
