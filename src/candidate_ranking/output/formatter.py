from __future__ import annotations
import json
from pathlib import Path
from candidate_ranking.models import Assessment, JobDescription


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
