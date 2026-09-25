from __future__ import annotations

import json
from pathlib import Path

from candidate_ranking.models import Assessment, JobDescription
from candidate_ranking.output.formatter import format_jd_ranking, write_jd_ranking


def _jd() -> JobDescription:
    return JobDescription(id="jd-1", title="Backend Engineer", raw_text="...", source_path="jd.pdf")


def _assessment(candidate_id: str, score: float) -> Assessment:
    return Assessment(
        job_description_id="jd-1",
        candidate_id=candidate_id,
        generated_by_model="typesafe/jev",
        requirement_scores={"Python": score},
        confidence={"requirement::Python": 0.9},
    )


def test_format_jd_ranking_sorts_descending_by_score():
    assessments = {
        "cand-a": _assessment("cand-a", 40.0),
        "cand-b": _assessment("cand-b", 90.0),
        "cand-c": _assessment("cand-c", 65.0),
    }

    markdown_text, json_payload = format_jd_ranking(_jd(), assessments)

    rows = json_payload["rankings"]
    assert [row["candidate_id"] for row in rows] == ["cand-b", "cand-c", "cand-a"]
    assert [row["rank"] for row in rows] == [1, 2, 3]
    assert rows[0]["composite_fit_score"] == 90.0
    assert rows[0]["requirement_scores"] == {"Python": 90.0}
    assert "overall_recommendation" not in rows[0]
    assert "meets_min_qualifications" not in rows[0]
    assert json_payload["job_description_id"] == "jd-1"
    assert "cand-b" in markdown_text
    assert markdown_text.index("cand-b") < markdown_text.index("cand-c") < markdown_text.index("cand-a")


def test_write_jd_ranking_writes_markdown_and_json(tmp_path: Path):
    assessments = {"cand-a": _assessment("cand-a", 55.0)}

    write_jd_ranking(tmp_path, _jd(), assessments)

    assert (tmp_path / "jd-1" / "ranking.md").exists()
    json_payload = json.loads((tmp_path / "jd-1" / "ranking.json").read_text(encoding="utf-8"))
    assert json_payload["rankings"][0]["candidate_id"] == "cand-a"
