from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
from candidate_ranking.config import RunConfig
from candidate_ranking.graphs.pipeline import (
    PipelineState,
    assessments_by_jd,
    build_pipeline_graph,
)
from candidate_ranking.models import Assessment, Candidate, JobProfile, RequirementEvaluation
from candidate_ranking.scoring.job_profile import JobProfileGenerationError


def _qualified_assessment(jd_id: str, candidate_id: str) -> Assessment:
    return Assessment(
        job_profile_id=jd_id, candidate_id=candidate_id,
        generated_by_model="qwen2.5:14b", qualified=True,
        must_have_evaluations=[
            RequirementEvaluation(requirement="Python", verdict="MET", evidence="FastAPI",
                                  evidence_strength="strong", recency="current", years_demonstrated=3)
        ],
    )


def _ok_result(jd_id: str, candidate_id: str) -> dict:
    return {
        "jp_id": jd_id, "candidate_id": candidate_id, "status": "ok",
        "assessment": _qualified_assessment(jd_id, candidate_id), "error": None,
    }


def _failed_result(jd_id: str, candidate_id: str) -> dict:
    return {"jd_id": jd_id, "candidate_id": candidate_id, "status": "failed",
            "assessment": None, "error": "boom"}


def test_pipeline_state_has_job_profiles_key():
    assert "job_profiles" in PipelineState.__annotations__
    assert "job_postings" in PipelineState.__annotations__


def test_pipeline_state_has_no_jev_fields():
    for banned in ("jev_client", "jev_results"):
        assert banned not in PipelineState.__annotations__


def test_assessments_by_jd_groups_ok_and_skips_failed():
    results = [_ok_result("jd-1", "a"), _ok_result("jd-1", "b"), _failed_result("jd-1", "c")]
    grouped = assessments_by_jd(results)
    assert set(grouped["jd-1"]) == {"a", "b"}
    assert grouped["jd-1"]["a"].qualified is True


def test_build_pipeline_graph_has_profile_node_and_no_jev_nodes(tmp_path: Path):
    cfg = RunConfig.full(tmp_path)
    graph = build_pipeline_graph(
        cfg=cfg,
        job_profile_chain=MagicMock(),
        jev_client=MagicMock(),

        skill_index=np.zeros((1, 10)),
        skill_row_map=[("jd-1", "cand-1")],
        skill_embedder=MagicMock(return_value=np.zeros((1, 10))),
        run_id="test-run",
    )
    node_names = set(graph.nodes)
    assert "extract_job_profile_for_jd" in node_names
    assert "generate_assessment_for_pair" in node_names
    assert "rank_and_format_jd_node" in node_names
    assert "extract_skills_for_jd" not in node_names
    assert "extract_requirements_for_jd" not in node_names
    assert not any("jev" in n.lower() for n in node_names)


def test_rank_and_format_writes_empty_ranking_for_jd_with_no_assessments(tmp_path: Path):
    cfg = RunConfig.full(tmp_path)
    jp_ok = JobProfile(id="jd-ok", title="Engineer", raw_text="Python", source_path="a.pdf")
    jp_fail = JobProfile(id="jd-fail", title="Other", raw_text="...", source_path="b.pdf")
    candidate = Candidate(
        id="cand-1", source_path="cv.pdf", raw_text="...", num_pages=1, char_count=10, parse_status="ok",
    )

    def fake_load_or_generate_job_profile(jp, chain, model_name, cache_path):
        if jp.id == "jd-fail":
            raise JobProfileGenerationError("boom")
        return JobProfile(id=jp.id, title=jp.title, raw_text=jp.raw_text, source_path=jp.source_path, generated_by_model=model_name, must_have_skills=["Python"])

    def fake_shortlist_candidates(jp, index, row_map, embedder, **kwargs):
        return ["cand-1"] if jp.id == "jd-ok" else []

    def fake_load_or_generate_assessment(jp, candidate_, jev_client, cache_dir):
        return Assessment(
            job_profile_id=jp.id, candidate_id=candidate_.id,
            generated_by_model="jev/jev-latest",
            must_have_evaluations=[
                RequirementEvaluation(requirement="Python", verdict="MET", evidence="x",
                                      evidence_strength="strong", recency="current", years_demonstrated=1)
            ],
        )

    with patch(
        "candidate_ranking.graphs.pipeline.load_or_generate_job_profile",
        side_effect=fake_load_or_generate_job_profile,
    ), patch(
        "candidate_ranking.graphs.pipeline.shortlist_candidates", side_effect=fake_shortlist_candidates
    ), patch(
        "candidate_ranking.graphs.pipeline.load_or_generate_assessment",
        side_effect=fake_load_or_generate_assessment,
    ), patch(
        "candidate_ranking.graphs.pipeline.rank_active_listwise", return_value={"cand-1": 1.0}
    ), patch(
        "candidate_ranking.graphs.pipeline.write_jd_ranking"
    ) as mock_write:
        graph = build_pipeline_graph(
            cfg=cfg,
            job_profile_chain=MagicMock(),
            jev_client=MagicMock(),
    
            skill_index=np.zeros((1, 10)),
            skill_row_map=[("jd-ok", "cand-1")],
            skill_embedder=MagicMock(return_value=np.zeros((1, 10))),
            run_id="test-run",
        )
        compiled = graph.compile()
        compiled.invoke({
            "job_postings": [jp_ok, jp_fail], "candidates": [candidate],
            "job_profiles": {}, "shortlists": {}, "assessment_results": [],
        })

    written_jp_ids = {call.args[1].id for call in mock_write.call_args_list}
    assert "jd-fail" in written_jp_ids
    assert "jd-ok" in written_jp_ids
