from __future__ import annotations

from candidate_ranking.models import Assessment, JobProfile


def _extract_jd_summary(jp: JobProfile) -> str:
    words = jp.raw_text.split()
    return f"{jd.title}\n\n" + " ".join(words[:200])


def _render_candidate_evidence(candidate_id: str, assessments: dict[str, Assessment]) -> str:
    """Single-candidate evidence block — used for pairwise comparisons."""
    a = assessments[candidate_id]
    must_text = "\n".join(
        f"  [{ev.evidence_strength}] {ev.requirement}: {ev.evidence}"
        for ev in a.must_have_evaluations
    ) or "  (none)"
    met_nice = [ev for ev in a.nice_to_have_evaluations if ev.evidence_strength != "none"]
    nice_text = ", ".join(f"{ev.requirement} [{ev.evidence_strength}]" for ev in met_nice) or "(none)"
    other_text = "\n".join(
        f"  [{ev.match_level}] {ev.requirement} ({ev.source})"
        for ev in a.adaptive_requirement_evaluations
    ) or "  (none)"
    return (
        f"Must-have evaluations:\n{must_text}\n"
        f"Nice-to-have met: {nice_text}\n"
        f"Other requirements:\n{other_text}\n"
        f"Relevant years: {a.total_relevant_years}"
    )


def _render_evidence_for_ranking(candidate_ids: list[str], assessments: dict[str, Assessment]) -> str:
    """Numbered multi-candidate block — used for pairwise ranking."""
    blocks = []
    for index, candidate_id in enumerate(candidate_ids, start=1):
        blocks.append(f"{index}:\n" + "\n".join(
            "  " + line for line in _render_candidate_evidence(candidate_id, assessments).splitlines()
        ))
    return "\n\n".join(blocks)
