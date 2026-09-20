from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from candidate_ranking.config import RunConfig
from candidate_ranking.graphs.pipeline import PipelineState, assessments_by_jd, build_pipeline_graph, rank_and_format_jd
from candidate_ranking.models import Assessment, JobDescription


def _ok_result(jd_id: str, candidate_id: str, score: float) -> dict:
    return {
        "jd_id": jd_id,
        "candidate_id": candidate_id,
        "status": "ok",
        "assessment": Assessment(
            job_description_id=jd_id, candidate_id=candidate_id, generated_by_model="typesafe/jev",
            overall_fit_score=score, overall_recommendation="hire", meets_min_qualifications=True,
        ),
        "error": None,
    }


def _failed_result(jd_id: str, candidate_id: str) -> dict:
    return {"jd_id": jd_id, "candidate_id": candidate_id, "status": "failed", "assessment": None, "error": "boom"}


def test_pipeline_state_has_no_tournament_results_key():
    assert "tournament_results" not in PipelineState.__annotations__
    assert "assessment_results" in PipelineState.__annotations__


def test_assessments_by_jd_groups_ok_results_and_skips_failed():
    results = [_ok_result("jd-1", "cand-a", 80.0), _ok_result("jd-1", "cand-b", 60.0), _failed_result("jd-1", "cand-c")]

    grouped = assessments_by_jd(results)

    assert set(grouped["jd-1"]) == {"cand-a", "cand-b"}
    assert grouped["jd-1"]["cand-a"].overall_fit_score == 80.0


def test_rank_and_format_jd_writes_ranking_files(tmp_path: Path):
    jd = JobDescription(id="jd-1", title="Backend Engineer", raw_text="...", source_path="jd.pdf")
    assessments = {
        "cand-a": Assessment(
            job_description_id="jd-1", candidate_id="cand-a", generated_by_model="typesafe/jev",
            overall_fit_score=80.0, overall_recommendation="hire", meets_min_qualifications=True,
        )
    }

    rank_and_format_jd(tmp_path, "run-1", jd, assessments)

    ranking = json.loads((tmp_path / "run-1" / "jd-1" / "ranking.json").read_text(encoding="utf-8"))
    assert ranking["rankings"][0]["candidate_id"] == "cand-a"


def test_build_pipeline_graph_has_no_tournament_nodes(tmp_path: Path):
    cfg = RunConfig.full(tmp_path)
    graph = build_pipeline_graph(
        cfg,
        jd_skills_chain=None,
        jev_client=None,
        skill_index=np.zeros(0),
        skill_row_map=[],
        skill_embedder=lambda texts: np.zeros((len(texts), 0)),
        run_id="run-1",
    )

    node_names = set(graph.nodes)
    assert "rank_and_format_jd_node" in node_names
    assert not any("tournament" in name for name in node_names)


def test_build_pipeline_graph_accepts_min_must_have_matches(tmp_path: Path):
    cfg = RunConfig.full(tmp_path)
    graph = build_pipeline_graph(
        cfg,
        jd_skills_chain=None,
        jev_client=None,
        skill_index=np.zeros(0),
        skill_row_map=[],
        skill_embedder=lambda texts: np.zeros((len(texts), 0)),
        run_id="run-1",
        min_must_have_matches=3,
    )

    assert "build_shortlist_for_jd" in set(graph.nodes)
