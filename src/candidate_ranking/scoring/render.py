from __future__ import annotations

from candidate_ranking.models import Assessment, JobProfile


def _extract_jd_summary(jp: JobProfile) -> str:
    words = jp.raw_text.split()
    return f"{jp.id}\n\n" + " ".join(words[:200])


def _render_candidate_evidence(candidate_id: str, assessments: dict[str, Assessment]) -> str:
    a = assessments[candidate_id]
    ordered_evaluations = sorted(a.evaluations, key=lambda ev: ev.requirement.lower())
    evidence_text = "\n".join(
        f"  [{ev.evidence_strength}] {ev.requirement}: {ev.evidence}"
        for ev in ordered_evaluations
    ) or "  (none)"
    return f"Evaluations:\n{evidence_text}"


def _render_evidence_for_ranking(candidate_ids: list[str], assessments: dict[str, Assessment]) -> str:
    blocks = []
    for index, candidate_id in enumerate(candidate_ids, start=1):
        blocks.append(f"{index}:\n" + "\n".join(
            "  " + line for line in _render_candidate_evidence(candidate_id, assessments).splitlines()
        ))
    return "\n\n".join(blocks)
