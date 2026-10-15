from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from typing import Callable
from candidate_ranking.models import Assessment, JobDescription, RequirementEvaluation
from candidate_ranking.scoring.sliding_window_ranking import _CandidateEvaluation
from candidate_ranking.evaluation.ranking_stability import (
    StabilityReport,
    kendall_tau,
    evaluate_jd_stability,
    run_stability_report,
)


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
    )
    assert report.mean_tau == 0.915
    assert report.min_tau_jd == "jd-2"


def _jd(jd_id: str = "jd-1") -> JobDescription:
    return JobDescription(id=jd_id, title="Backend Engineer",
                          raw_text="Python required.", source_path="jd.pdf")


def _assessment(candidate_id: str, jd_id: str = "jd-1", years: int = 3) -> Assessment:
    return Assessment(
        job_description_id=jd_id, candidate_id=candidate_id,
        generated_by_model="qwen2.5:14b", qualified=True,
        must_have_evaluations=[
            RequirementEvaluation(
                requirement="Python", verdict="MET", evidence="FastAPI",
                evidence_strength="strong", recency="current", years_demonstrated=years,
            )
        ],
        total_relevant_years=years,
    )


def _identity_chain_factory() -> Callable:
    def _build(model):
        chain = MagicMock()
        def _invoke(payload):
            blocks = [b for b in payload["candidates_text"].split("\n\n") if b.strip()]
            size = len(blocks)
            evals = [_CandidateEvaluation(id=i, key_strengths="s", key_gaps="g")
                     for i in range(1, size + 1)]
            return model(candidate_evaluations=evals, ranking=list(range(1, size + 1)))
        chain.invoke.side_effect = _invoke
        return chain
    return _build


def test_evaluate_jd_stability_returns_float_between_minus_one_and_one():
    jd = _jd()
    assessments = {f"cand-{i}": _assessment(f"cand-{i}", years=i + 1) for i in range(6)}
    tau = evaluate_jd_stability(jd, assessments, _identity_chain_factory(), window_size=3, step=2)
    assert isinstance(tau, float)
    assert -1.0 <= tau <= 1.0


def test_run_stability_report_aggregates_correctly():
    jd1 = _jd("jd-1")
    jd2 = _jd("jd-2")
    assessments1 = {f"cand-{i}": _assessment(f"cand-{i}", "jd-1", years=i + 1) for i in range(4)}
    assessments2 = {f"cand-{i}": _assessment(f"cand-{i}", "jd-2", years=i + 1) for i in range(4)}
    jd_assessments = {"jd-1": (jd1, assessments1), "jd-2": (jd2, assessments2)}
    report = run_stability_report(jd_assessments, _identity_chain_factory(), window_size=3, step=2)
    assert set(report.per_jd.keys()) == {"jd-1", "jd-2"}
    assert report.min_tau_jd in {"jd-1", "jd-2"}
    assert report.mean_tau == pytest.approx(sum(report.per_jd.values()) / 2)


def test_run_stability_report_raises_when_no_qualified_candidates():
    jd = _jd()
    jd_assessments = {"jd-1": (jd, {"cand-a": _assessment("cand-a")})}
    with pytest.raises(ValueError, match="No JDs"):
        run_stability_report(jd_assessments, _identity_chain_factory())
