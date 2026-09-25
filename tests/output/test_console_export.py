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
    jd_dir = project_root / "job-description"
    cv_dir = project_root / "cv"
    runs_dir = project_root / "runs"
    jd_dir.mkdir(parents=True)
    cv_dir.mkdir(parents=True)

    (jd_dir / "jd-1.txt").write_text("Backend Engineer\n\nNeeds Python.", encoding="utf-8")
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
        "job_description_id": "jd-1",
        "rankings": [
            {
                "rank": 1, "candidate_id": "cand-a", "composite_fit_score": 87.5,
                "requirement_scores": {"Python": 100.0},
            }
        ],
    }
    (run_dir / "jd-1" / "ranking.json").write_text(json.dumps(ranking), encoding="utf-8")

    assessments = {
        "cand-a": {
            "job_description_id": "jd-1", "candidate_id": "cand-a", "generated_by_model": "typesafe/jev",
            "composite_fit_score": 87.5,
            "requirement_scores": {"Python": 100.0}, "confidence": {"requirement::Python": 0.9},
            "seniority_years_fit_score": 75.0, "education_fit_score": None,
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
    assert assessment["composite_fit_score"] == 87.5
    assert "overall_recommendation" not in assessment
    assert "meets_min_qualifications" not in assessment
    assert assessment["requirement_scores"] == {"Python": 100.0}
    assert assessment["seniority_years_fit_score"] == 75.0
    assert assessment["education_fit_score"] is None
    assert "strengths" not in assessment
    comparison = data["comparison"]["jd-1"]
    assert comparison["meanFitScore"] == 87.5
    assert "meetsMinRate" not in comparison
    assert "hireRate" not in comparison
    assert comparison["rankingStability"] is None
    assert data["evaluationSummary"] is None


def test_export_console_web_data_includes_evaluation_summary_when_report_exists(run_with_jev_ranking, tmp_path):
    cfg, run_id = run_with_jev_ranking
    run_dir = cfg.runs_dir / run_id
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True)
    report = {
        "run_id": run_id,
        "test_retest": {
            "n_pairs": 338, "n_repeats_per_pair": 3,
            "composite_fit_score_stdev": {"mean": 0.8, "median": 0.66},
        },
        "ranking_convergence": {
            "per_job_profile": {"jd-1": {"n_candidates": 22, "n_repeats": 3, "mean_kendall_tau": 0.93}},
            "mean_kendall_tau_across_all_profiles": 0.906,
        },
        "internal_coherence": {
            "n_assessments": 338,
            "requirement_vs_composite_score_correlation": {"spearman_rho": 0.667, "p_value": 6.7e-45, "n": 338},
        },
    }
    (eval_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    output_path = tmp_path / "real-data.json"

    export_console_web_data(cfg, run_id, output_path=output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["comparison"]["jd-1"]["rankingStability"] == 0.93
    summary = data["evaluationSummary"]
    assert "recommendationAgreementRate" not in summary
    assert summary["meanRankingConvergence"] == 0.906
    assert summary["compositeScoreStdev"] == 0.8
    assert summary["coherenceSpearmanRho"] == 0.667


def test_export_console_web_data_includes_repeat_samples_when_test_retest_exists(run_with_jev_ranking, tmp_path):
    cfg, run_id = run_with_jev_ranking
    run_dir = cfg.runs_dir / run_id
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True)
    test_retest = [
        {
            "jd_id": "jd-1",
            "candidate_id": "cand-a",
            "repeats": [
                {
                    "composite_fit_score": 86.0, "overall_recommendation": "hire",
                    "meets_min_qualifications": True, "requirement_scores": {"Python": 95.0},
                    "latency_seconds": 1.1,
                },
                {
                    "composite_fit_score": 88.5, "overall_recommendation": "hire",
                    "meets_min_qualifications": True, "requirement_scores": {"Python": 100.0},
                    "latency_seconds": 1.0,
                },
                {
                    "composite_fit_score": 87.0, "overall_recommendation": "maybe",
                    "meets_min_qualifications": True, "requirement_scores": {"Python": 98.0},
                    "latency_seconds": 1.3,
                },
            ],
        }
    ]
    (eval_dir / "test_retest.json").write_text(json.dumps(test_retest), encoding="utf-8")
    output_path = tmp_path / "real-data.json"

    export_console_web_data(cfg, run_id, output_path=output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    samples = data["repeatSamples"]["cand-a::jd-1"]
    assert len(samples) == 3
    assert samples[0] == {"compositeFitScore": 86.0}
    assert samples[2] == {"compositeFitScore": 87.0}


def test_export_console_web_data_repeat_samples_empty_when_no_test_retest(run_with_jev_ranking, tmp_path):
    cfg, run_id = run_with_jev_ranking
    output_path = tmp_path / "real-data.json"

    export_console_web_data(cfg, run_id, output_path=output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["repeatSamples"] == {}
