from __future__ import annotations

from candidate_ranking.models import Assessment, JobDescription


def _extract_jd_summary(jd: JobDescription) -> str:
    words = jd.raw_text.split()
    return f"{jd.title}\n\n" + " ".join(words[:200])


def _render_candidate_evidence(candidate_id: str, assessments: dict[str, Assessment]) -> str:
    """Single-candidate evidence block — used for pairwise comparisons."""
    a = assessments[candidate_id]
    must_text = "\n".join(
        f"  [{ev.verdict}] {ev.requirement} ({ev.evidence_strength}, {ev.years_demonstrated}yr): {ev.evidence}"
        for ev in a.must_have_evaluations
    ) or "  (none)"
    met_nice = [ev for ev in a.nice_to_have_evaluations if ev.verdict in ("MET", "PARTIALLY_MET")]
    nice_text = ", ".join(f"{ev.requirement} [{ev.verdict}]" for ev in met_nice) or "(none)"
    traj = a.career_trajectory
    traj_text = (
        f"{traj.direction}, avg {traj.average_tenure_months}mo/role"
        + (", multinational exp." if traj.has_multinational_experience else "")
    ) if traj else "unknown"
    return (
        f"Must-have evaluations:\n{must_text}\n"
        f"Nice-to-have met: {nice_text}\n"
        f"Trajectory: {traj_text} | Relevant years: {a.total_relevant_years}"
    )


def _render_evidence_for_ranking(candidate_ids: list[str], assessments: dict[str, Assessment]) -> str:
    """Numbered multi-candidate block — used for listwise ranking."""
    blocks = []
    for index, candidate_id in enumerate(candidate_ids, start=1):
        blocks.append(f"{index}:\n" + "\n".join(
            "  " + line for line in _render_candidate_evidence(candidate_id, assessments).splitlines()
        ))
    return "\n\n".join(blocks)
