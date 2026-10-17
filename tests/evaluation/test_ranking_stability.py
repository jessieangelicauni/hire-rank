from __future__ import annotations
import json
import pytest
from unittest.mock import MagicMock, patch
from candidate_ranking.models import Assessment, JobProfile, RequirementEvaluation
from candidate_ranking.evaluation.ranking_stability import (
    StabilityReport,
    kendall_tau,
    rank_biased_overlap,
    evaluate_jd_stability,
    run_stability_report,
)

ACTIVE_LISTWISE_MODULE = "candidate_ranking.scoring.active_pairwise_ranking"


def test_kendall_tau_identical_rankings_returns_one():
    utils = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert kendall_tau(utils, utils) == pytest.approx(1.0)


def test_kendall_tau_reversed_rankings_returns_minus_one():
    a = {"a": 3.0, "b": 2.0, "c": 1.0}
    b = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert kendall_tau(a, b) == pytest.approx(-1.0)


def test_kendall_tau_partial_agreement():
    a = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}
    b = {"a": 4.0, "b": 3.0, "c": 1.0, "d": 2.0}
    tau = kendall_tau(a, b)
    assert -1.0 <= tau <= 1.0
    assert tau > 0.0


def test_stability_report_fields():
    report = StabilityReport(
        per_jd={"jd-1": 0.95, "jd-2": 0.88},
        mean_tau=0.915,
        min_tau=0.88,
        min_tau_jd="jd-2",
        rbo_per_jd={"jd-1": 0.97, "jd-2": 0.80},
        mean_rbo=0.885,
        min_rbo=0.80,
        min_rbo_jd="jd-2",
    )
    assert report.mean_tau == 0.915
    assert report.min_tau_jd == "jd-2"
    assert report.mean_rbo == 0.885
    assert report.min_rbo_jd == "jd-2"


def test_rank_biased_overlap_identical_rankings_returns_one():
    utils = {"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.0}
    assert rank_biased_overlap(utils, utils) == pytest.approx(1.0)


def test_rank_biased_overlap_fully_reversed_is_low():
    a = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}
    b = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    rbo = rank_biased_overlap(a, b)
    assert 0.0 <= rbo < 0.5


def test_rank_biased_overlap_weighs_top_agreement_more_than_kendall_tau():
    # Runs agree perfectly on who's #1 and #2, but the bottom two swap.
    a = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}
    b = {"a": 4.0, "b": 3.0, "c": 1.0, "d": 2.0}
    rbo = rank_biased_overlap(a, b, p=0.9)
    tau = kendall_tau(a, b)
    # Top-of-list agreement is total, so RBO should reward that more than tau does.
    assert rbo > tau


def _jp(jp_id: str = "jd-1") -> JobProfile:
    return JobProfile(id=jp_id, title="Backend Engineer",
                          raw_text="Python required.", source_path="jd.pdf")


def _assessment(candidate_id: str, jd_id: str = "jd-1", years: int = 3) -> Assessment:
    return Assessment(
        job_profile_id=jd_id, candidate_id=candidate_id,
        generated_by_model="qwen2.5:14b",
        must_have_evaluations=[
            RequirementEvaluation(
                requirement="Python", evidence="FastAPI",
                evidence_strength="strong",
            )
        ],
        total_relevant_years=years,
    )


def test_evaluate_jd_stability_returns_float_between_minus_one_and_one():
    jp = _jp()
    assessments = {f"cand-{i}": _assessment(f"cand-{i}", years=i + 1) for i in range(6)}

    with patch(f"{ACTIVE_LISTWISE_MODULE}._rank_subset", side_effect=lambda jp, subset_ids, asmt, llm: list(subset_ids)):
        tau, rbo = evaluate_jd_stability(jp, assessments, MagicMock(), subset_size=3)

    assert isinstance(tau, float)
    assert -1.0 <= tau <= 1.0
    assert isinstance(rbo, float)
    assert 0.0 <= rbo <= 1.0


def test_run_stability_report_aggregates_correctly():
    jp1 = _jp("jd-1")
    jp2 = _jp("jd-2")
    assessments1 = {f"cand-{i}": _assessment(f"cand-{i}", "jd-1", years=i + 1) for i in range(4)}
    assessments2 = {f"cand-{i}": _assessment(f"cand-{i}", "jd-2", years=i + 1) for i in range(4)}
    jp_assessments = {"jd-1": (jp1, assessments1), "jd-2": (jp2, assessments2)}

    with patch(f"{ACTIVE_LISTWISE_MODULE}._rank_subset", side_effect=lambda jp, subset_ids, asmt, llm: list(subset_ids)):
        report = run_stability_report(jp_assessments, MagicMock(), subset_size=3)

    assert set(report.per_jd.keys()) == {"jd-1", "jd-2"}
    assert report.min_tau_jd in {"jd-1", "jd-2"}
    assert report.mean_tau == pytest.approx(sum(report.per_jd.values()) / 2)


def test_run_stability_report_raises_when_fewer_than_two_candidates():
    jp = _jp()
    jp_assessments = {"jd-1": (jp, {"cand-a": _assessment("cand-a")})}
    with pytest.raises(ValueError, match="No JDs"):
        run_stability_report(jp_assessments, MagicMock())


from pathlib import Path
from candidate_ranking.evaluation.ranking_stability import load_cached_assessments


def _write_assessment_cache(jd_dir: Path, candidate_id: str, requirement: str) -> None:
    a = _assessment(candidate_id)
    a.must_have_evaluations[0].requirement = requirement
    cache_entry = json.dumps({"cache_key": "irrelevant", "assessment": a.model_dump()})
    (jd_dir / f"{candidate_id}.json").write_text(cache_entry, encoding="utf-8")


def test_load_cached_assessments_reads_by_jd(tmp_path: Path):
    assessments_dir = tmp_path / "assessments"
    jd_dir = assessments_dir / "jd-1"
    jd_dir.mkdir(parents=True)
    _write_assessment_cache(jd_dir, "cand-a", "Python")
    result = load_cached_assessments(tmp_path)
    assert "jd-1" in result
    assert "cand-a" in result["jd-1"]
    assert result["jd-1"]["cand-a"].candidate_id == "cand-a"


def test_load_cached_assessments_skips_entries_scored_against_a_stale_profile(tmp_path: Path):
    assessments_dir = tmp_path / "assessments"
    jd_dir = assessments_dir / "jd-1"
    jd_dir.mkdir(parents=True)
    _write_assessment_cache(jd_dir, "cand-fresh", "Python")
    _write_assessment_cache(jd_dir, "cand-stale", "Wireshark")  # scored against an old must-have list

    current_profile = JobProfile(
        id="jd-1", title="Job", raw_text="...", source_path="j.txt",
        generated_by_model="qwen2.5:14b", must_have_skills=["Python"]
    )
    result = load_cached_assessments(tmp_path, job_profiles={"jd-1": current_profile})

    assert set(result["jd-1"]) == {"cand-fresh"}
