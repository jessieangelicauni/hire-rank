from __future__ import annotations

import json
import math
from pathlib import Path

from candidate_ranking.evaluation.evaluation import K_FRACTIONS
from candidate_ranking.models import Assessment, JobDescription, TournamentResult


def _utility_std(tournament_result: TournamentResult, candidate_id: str) -> float | None:
    variance = tournament_result.final_utility_variance.get(candidate_id)
    return math.sqrt(variance) if variance is not None and variance >= 0 else None


def _is_borderline(rank: int, ranked: list[str], tournament_result: TournamentResult) -> bool:
    candidate_id = ranked[rank - 1]
    utility = tournament_result.final_utilities[candidate_id]
    std = _utility_std(tournament_result, candidate_id)
    if std is None:
        return False

    n = len(ranked)
    for fraction in K_FRACTIONS.values():
        k = max(1, round(fraction * n))
        if k >= n:
            continue
        boundary_utility = (
            tournament_result.final_utilities[ranked[k - 1]] + tournament_result.final_utilities[ranked[k]]
        ) / 2
        if abs(utility - boundary_utility) < std:
            return True
    return False


def format_jd_ranking(
    jd: JobDescription, tournament_result: TournamentResult, assessments: dict[str, Assessment]
) -> tuple[str, dict]:
    times_ranked: dict[str, int] = dict.fromkeys(tournament_result.candidate_ids, 0)
    for record in tournament_result.iteration_history:
        for candidate_id in record.subset_candidate_ids:
            if candidate_id in times_ranked:
                times_ranked[candidate_id] += 1

    ranked = sorted(tournament_result.candidate_ids, key=lambda cid: tournament_result.final_utilities[cid], reverse=True)
    compared = [cid for cid in ranked if times_ranked[cid] > 0]
    uncompared = [cid for cid in ranked if times_ranked[cid] == 0]

    def summary_for(candidate_id: str) -> str:
        assessment = assessments[candidate_id]
        return f"{len(assessment.strengths)} strengths, {len(assessment.weaknesses)} weaknesses"

    lines = [
        f"# Ranking: {jd.title} ({jd.id})",
        "",
        "> Strengths and weaknesses state only facts drawn from the CV.",
        "> Ranks marked (borderline) are not clearly separated from a neighboring rank once utility",
        "> uncertainty is accounted for -- treat as a tie needing a closer look, not a confident order.",
        "",
    ]
    rows = []
    for rank, candidate_id in enumerate(compared, start=1):
        utility = tournament_result.final_utilities[candidate_id]
        utility_std = _utility_std(tournament_result, candidate_id)
        borderline = _is_borderline(rank, compared, tournament_result)
        assessment = assessments[candidate_id]
        borderline_tag = " (borderline)" if borderline else ""
        lines.append(f"{rank}. **{candidate_id}** (utility={utility:.4f}){borderline_tag} — {summary_for(candidate_id)}")
        rows.append(
            {
                "rank": rank,
                "candidate_id": candidate_id,
                "utility": utility,
                "utility_std": utility_std,
                "borderline": borderline,
                "strengths": assessment.strengths,
                "weaknesses": assessment.weaknesses,
                "additional_skills": assessment.additional_skills,
                "times_ranked": times_ranked[candidate_id],
            }
        )

    if uncompared:
        lines.append("")
        lines.append("## Not compared (0 tournament iterations)")
        lines.append("")
        for candidate_id in uncompared:
            utility = tournament_result.final_utilities[candidate_id]
            assessment = assessments[candidate_id]
            lines.append(f"- **{candidate_id}** (utility={utility:.4f}) — {summary_for(candidate_id)}")
            rows.append(
                {
                    "rank": None,
                    "candidate_id": candidate_id,
                    "utility": utility,
                    "utility_std": _utility_std(tournament_result, candidate_id),
                    "borderline": False,
                    "strengths": assessment.strengths,
                    "weaknesses": assessment.weaknesses,
                    "times_ranked": 0,
                }
            )

    markdown_text = "\n".join(lines) + "\n"
    json_payload = {
        "job_description_id": jd.id,
        "repeat_index": tournament_result.repeat_index,
        "rankings": rows,
    }
    return markdown_text, json_payload


def write_jd_ranking(
    run_dir: Path, jd: JobDescription, tournament_result: TournamentResult, assessments: dict[str, Assessment]
) -> None:
    markdown_text, json_payload = format_jd_ranking(jd, tournament_result, assessments)
    jd_dir = run_dir / jd.id
    jd_dir.mkdir(parents=True, exist_ok=True)
    (jd_dir / "ranking.md").write_text(markdown_text, encoding="utf-8")
    (jd_dir / "ranking.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
