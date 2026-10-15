from __future__ import annotations
import pytest
from candidate_ranking.evaluation.ranking_stability import (
    StabilityReport,
    kendall_tau,
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
