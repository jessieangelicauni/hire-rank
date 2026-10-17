from __future__ import annotations

from candidate_ranking.models import AdaptiveRequirementEvaluation, Assessment, RequirementEvaluation
from candidate_ranking.scoring.render import _render_candidate_evidence


def _assessment_with_adaptive(evals: list[AdaptiveRequirementEvaluation]) -> Assessment:
    return Assessment(
        job_profile_id="jd-1", candidate_id="cand-1", generated_by_model="jev/jev-latest",
        must_have_evaluations=[
            RequirementEvaluation(requirement="Python", evidence="FastAPI", evidence_strength="strong"),
        ],
        adaptive_requirement_evaluations=evals,
        qualified=True,
    )


def test_render_candidate_evidence_includes_other_requirements_block():
    evals = [
        AdaptiveRequirementEvaluation(
            source="education", requirement="Bachelor's in CS", evidence="BSc Computer Science, 2020",
            match_level="meets", confidence=0.9,
        ),
    ]
    text = _render_candidate_evidence("cand-1", {"cand-1": _assessment_with_adaptive(evals)})
    assert "Other requirements:" in text
    assert "[meets] Bachelor's in CS (education)" in text


def test_render_candidate_evidence_shows_none_when_no_adaptive_requirements():
    text = _render_candidate_evidence("cand-1", {"cand-1": _assessment_with_adaptive([])})
    assert "Other requirements:\n  (none)" in text


def test_render_candidate_evidence_no_longer_mentions_trajectory():
    text = _render_candidate_evidence("cand-1", {"cand-1": _assessment_with_adaptive([])})
    assert "Trajectory" not in text
