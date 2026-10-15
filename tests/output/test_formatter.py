from __future__ import annotations
import json
from pathlib import Path
from candidate_ranking.models import Assessment, JobDescription, RequirementEvaluation
from candidate_ranking.output.writer import format_jd_ranking, write_jd_ranking


def _jd() -> JobDescription:
    return JobDescription(id="jd-1", title="Backend Engineer", raw_text="...", source_path="jd.pdf")


def _assessment(candidate_id: str) -> Assessment:
    return Assessment(
        job_description_id="jd-1", candidate_id=candidate_id,
        generated_by_model="qwen2.5:14b", qualified=True,
        must_have_evaluations=[
            RequirementEvaluation(requirement="Python", verdict="MET", evidence="FastAPI",
                                  evidence_strength="strong", recency="current", years_demonstrated=3)
        ],
    )


def test_format_jd_ranking_orders_by_utility():
    assessments = {
        "cand-a": _assessment("cand-a"),
        "cand-b": _assessment("cand-b"),
        "cand-c": _assessment("cand-c"),
    }
    utilities = {"cand-a": 0.5, "cand-b": 2.1, "cand-c": 1.0}

    _, json_payload = format_jd_ranking(_jd(), assessments, utilities)

    rows = json_payload["rankings"]
    assert [r["candidate_id"] for r in rows] == ["cand-b", "cand-c", "cand-a"]
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert rows[0]["utility"] == 2.1
    assert "composite_fit_score" not in rows[0]


def test_format_jd_ranking_includes_qualified_and_years():
    assessments = {"cand-a": _assessment("cand-a")}
    utilities = {"cand-a": 1.0}

    _, json_payload = format_jd_ranking(_jd(), assessments, utilities)

    row = json_payload["rankings"][0]
    assert row["qualified"] is True
    assert row["total_relevant_years"] == 0


def test_write_jd_ranking_writes_markdown_and_json(tmp_path: Path):
    assessments = {"cand-a": _assessment("cand-a")}
    utilities = {"cand-a": 1.5}

    write_jd_ranking(tmp_path, _jd(), assessments, utilities)

    assert (tmp_path / "jd-1" / "ranking.md").exists()
    json_payload = json.loads((tmp_path / "jd-1" / "ranking.json").read_text(encoding="utf-8"))
    assert json_payload["rankings"][0]["candidate_id"] == "cand-a"
    assert json_payload["rankings"][0]["utility"] == 1.5
