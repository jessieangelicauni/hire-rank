from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
from candidate_ranking.config import RunConfig
from candidate_ranking.graphs.pipeline import (
    PipelineState,
    assessments_by_jd,
    build_pipeline_graph,
)
from candidate_ranking.models import Assessment, JDRequirements, JobDescription, RequirementEvaluation


def _qualified_assessment(jd_id: str, candidate_id: str) -> Assessment:
    return Assessment(
        job_description_id=jd_id, candidate_id=candidate_id,
        generated_by_model="qwen2.5:14b", qualified=True,
        must_have_evaluations=[
            RequirementEvaluation(requirement="Python", verdict="MET", evidence="FastAPI",
                                  evidence_strength="strong", recency="current", years_demonstrated=3)
        ],
    )


def _ok_result(jd_id: str, candidate_id: str) -> dict:
    return {
        "jd_id": jd_id, "candidate_id": candidate_id, "status": "ok",
        "assessment": _qualified_assessment(jd_id, candidate_id), "error": None,
    }


def _failed_result(jd_id: str, candidate_id: str) -> dict:
    return {"jd_id": jd_id, "candidate_id": candidate_id, "status": "failed",
            "assessment": None, "error": "boom"}


def test_pipeline_state_has_jd_requirements_key():
    assert "jd_requirements" in PipelineState.__annotations__


def test_pipeline_state_has_no_jev_fields():
    for banned in ("jev_client", "jev_results"):
        assert banned not in PipelineState.__annotations__


def test_assessments_by_jd_groups_ok_and_skips_failed():
    results = [_ok_result("jd-1", "a"), _ok_result("jd-1", "b"), _failed_result("jd-1", "c")]
    grouped = assessments_by_jd(results)
    assert set(grouped["jd-1"]) == {"a", "b"}
    assert grouped["jd-1"]["a"].qualified is True


def test_build_pipeline_graph_has_requirements_and_no_jev_nodes(tmp_path: Path):
    cfg = RunConfig.full(tmp_path)
    graph = build_pipeline_graph(
        cfg=cfg,
        jd_skills_chain=MagicMock(),
        jd_requirements_chain=MagicMock(),
        qualification_chain=MagicMock(),
        build_sliding_window_chain=MagicMock(),
        skill_index=np.zeros((1, 10)),
        skill_row_map=[("jd-1", "cand-1")],
        skill_embedder=MagicMock(return_value=np.zeros((1, 10))),
        run_id="test-run",
    )
    node_names = set(graph.nodes)
    assert "extract_requirements_for_jd" in node_names
    assert "generate_assessment_for_pair" in node_names
    assert "rank_and_format_jd_node" in node_names
    assert not any("jev" in n.lower() for n in node_names)
