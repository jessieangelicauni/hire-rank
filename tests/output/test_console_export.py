from __future__ import annotations

import json
from pathlib import Path

import pytest
from pypdf import PdfWriter

from candidate_ranking.config import RunConfig
from candidate_ranking.output.console_export import export_console_web_data


@pytest.fixture
def run_with_jev_ranking(tmp_path: Path, monkeypatch) -> tuple[RunConfig, str]:
    project_root = tmp_path / "project"
    jp_dir = project_root / "job-profile"
    cv_dir = project_root / "cv"
    runs_dir = project_root / "runs"
    jp_dir.mkdir(parents=True)
    cv_dir.mkdir(parents=True)

    (jp_dir / "jd-1.txt").write_text("Backend Engineer\n\nNeeds Python.", encoding="utf-8")
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(cv_dir / "cand-a.pdf", "wb") as f:
        writer.write(f)

    run_id = "run-1"
    run_dir = runs_dir / run_id
    (run_dir / "jd-1").mkdir(parents=True)

    manifest = {"ollama_model": "qwen2.5:14b", "jd_ids": ["jd-1"], "candidate_ids": ["cand-a"]}
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    ranking = {
        "job_profile_id": "jd-1",
        "rankings": [
            {
                "rank": 1, "candidate_id": "cand-a", "utility": 87.5,
            }
        ],
    }
    (run_dir / "jd-1" / "ranking.json").write_text(json.dumps(ranking), encoding="utf-8")

    assessments = {
        "cand-a": {
            "job_profile_id": "jd-1",
            "candidate_id": "cand-a",
            "generated_by_model": "typesafe/jev",
            "qualified": True,
            "total_relevant_years": 3,
            "must_have_evaluations": [
                {"requirement": "Python", "verdict": "MET"},
                {"requirement": "Django", "verdict": "MET"},
            ],
            "nice_to_have_evaluations": [
                {"requirement": "Docker", "verdict": "MET"},
            ],
            "career_trajectory": {"has_multinational_experience": False},
        }
    }
    (run_dir / "jd-1" / "assessments.json").write_text(json.dumps(assessments), encoding="utf-8")

    cfg = RunConfig.full(project_root)

    monkeypatch.setattr(
        "candidate_ranking.output.console_export.load_or_extract_candidate_name",
        lambda candidate, chain, model_name, cache_path: None,
    )
    monkeypatch.setattr(
        "candidate_ranking.output.console_export.build_name_extraction_chain", lambda llm: None
    )
    monkeypatch.setattr("candidate_ranking.output.console_export.ChatOllama", lambda **kwargs: None)

    return cfg, run_id


def test_export_console_web_data_maps_jev_scores(run_with_jev_ranking, tmp_path):
    cfg, run_id = run_with_jev_ranking
    output_path = tmp_path / "real-data.json"

    result_path = export_console_web_data(cfg, run_id, output_path=output_path)

    assert result_path == output_path
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["candidates"][0]["utility"] == 87.5
    assessment = data["assessments"]["cand-a::jd-1"]
    assert assessment["qualified"] is True
    assert assessment["total_relevant_years"] == 3
    assert assessment["must_have_met"] == 2
    assert assessment["nice_to_have_met"] == 1
    assert assessment["has_multinational_experience"] is False
    assert "composite_fit_score" not in assessment
    assert "requirement_scores" not in assessment
    assert "seniority_years_fit_score" not in assessment
    assert "education_fit_score" not in assessment
    assert "strengths" not in assessment
    comparison = data["comparison"]["jd-1"]
    assert comparison["qualifiedCount"] == 1
    assert comparison["totalAssessed"] == 1
    assert "meanFitScore" not in comparison
    assert "meetsMinRate" not in comparison
    assert "hireRate" not in comparison
    assert "rankingStability" not in comparison
    assert "evaluationSummary" not in data


