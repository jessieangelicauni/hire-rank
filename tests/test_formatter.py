
from candidate_ranking.models import Assessment, JobDescription, TournamentIterationRecord, TournamentResult
from candidate_ranking.output.formatter import format_jd_ranking

JD = JobDescription(id="backend-engineer", title="Backend Engineer", raw_text="Build APIs.", source_path="/jd/be.txt")


def _assessment(candidate_id: str) -> Assessment:
    return Assessment(
        job_description_id="backend-engineer", candidate_id=candidate_id, generated_by_model="fake-model",
        strengths=["Knows Python."], weaknesses=[],
    )


def _result(utilities: dict[str, float], variances: dict[str, float]) -> TournamentResult:
    candidate_ids = list(utilities)
    return TournamentResult(
        job_description_id="backend-engineer",
        repeat_index=0,
        candidate_ids=candidate_ids,
        final_utilities=utilities,
        final_utility_variance=variances,
        iteration_history=[
            TournamentIterationRecord(iteration=0, subset_candidate_ids=candidate_ids, ranking=candidate_ids, delta_u=0.1)
        ],
        status="ok",
    )


def test_candidate_near_a_cutoff_boundary_is_flagged_borderline():
    utilities = {f"cv-{i}": float(19 - i) for i in range(20)}
    variances = dict.fromkeys(utilities, 0.36)
    result = _result(utilities, variances)
    assessments = {cid: _assessment(cid) for cid in result.candidate_ids}

    _, payload = format_jd_ranking(JD, result, assessments)
    rows_by_id = {row["candidate_id"]: row for row in payload["rankings"]}

    assert rows_by_id["cv-2"]["borderline"] is True


def test_candidate_far_from_every_cutoff_boundary_is_not_flagged():
    utilities = {f"cv-{i}": float(19 - i) for i in range(20)}
    variances = dict.fromkeys(utilities, 0.36)
    result = _result(utilities, variances)
    assessments = {cid: _assessment(cid) for cid in result.candidate_ids}

    _, payload = format_jd_ranking(JD, result, assessments)
    rows_by_id = {row["candidate_id"]: row for row in payload["rankings"]}

    assert rows_by_id["cv-0"]["borderline"] is False
    assert rows_by_id["cv-10"]["borderline"] is False


def test_a_densely_packed_pool_does_not_flag_every_candidate_as_borderline():
    n = 55
    utilities = {f"cv-{i}": 2.2 - i * (4.34 / (n - 1)) for i in range(n)}
    variances = dict.fromkeys(utilities, 0.55**2)
    result = _result(utilities, variances)
    assessments = {cid: _assessment(cid) for cid in result.candidate_ids}

    _, payload = format_jd_ranking(JD, result, assessments)
    flagged = [row["candidate_id"] for row in payload["rankings"] if row["borderline"]]

    assert len(flagged) > 0
    assert len(flagged) < n
    rows_by_id = {row["candidate_id"]: row for row in payload["rankings"]}
    assert rows_by_id["cv-40"]["borderline"] is False


def test_missing_variance_data_is_never_flagged_borderline():
    result = _result(utilities={"cv-1": 0.50, "cv-2": 0.48}, variances={})
    assessments = {cid: _assessment(cid) for cid in result.candidate_ids}

    _, payload = format_jd_ranking(JD, result, assessments)

    assert all(row["borderline"] is False for row in payload["rankings"])
    assert all(row["utility_std"] is None for row in payload["rankings"])


def test_utility_std_is_the_square_root_of_the_variance():
    result = _result(utilities={"cv-1": 0.50, "cv-2": 0.10}, variances={"cv-1": 0.09, "cv-2": 0.0})
    assessments = {cid: _assessment(cid) for cid in result.candidate_ids}

    _, payload = format_jd_ranking(JD, result, assessments)
    rows_by_id = {row["candidate_id"]: row for row in payload["rankings"]}

    assert rows_by_id["cv-1"]["utility_std"] == 0.3
    assert rows_by_id["cv-2"]["utility_std"] == 0.0


def test_borderline_tag_appears_in_markdown_output():
    utilities = {f"cv-{i}": float(19 - i) for i in range(20)}
    variances = dict.fromkeys(utilities, 0.36)
    result = _result(utilities, variances)
    assessments = {cid: _assessment(cid) for cid in result.candidate_ids}

    markdown_text, _ = format_jd_ranking(JD, result, assessments)

    assert "(borderline)" in markdown_text
