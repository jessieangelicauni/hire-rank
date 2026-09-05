from scripts.audit_weakness_contradictions import AttemptEvent, build_report, classify_pair


def test_classify_pair_clean_when_first_attempt_has_no_contradiction():
    events = [AttemptEvent("jd1", "cv1", 0, [], ["No Kubernetes experience noted."])]
    assert classify_pair(events) == "clean"


def test_classify_pair_fixed_by_retry():
    events = [
        AttemptEvent("jd1", "cv1", 0, ["Docker"], ["Lacks experience with Docker."]),
        AttemptEvent("jd1", "cv1", 1, [], ["Lacks experience with Kubernetes."]),
    ]
    assert classify_pair(events) == "fixed_by_retry"


def test_classify_pair_dropped_when_still_contradicted_after_retry():
    events = [
        AttemptEvent("jd1", "cv1", 0, ["Docker"], ["Lacks experience with Docker."]),
        AttemptEvent("jd1", "cv1", 1, ["Docker"], ["Lacks experience with Docker."]),
    ]
    assert classify_pair(events) == "dropped"


def test_build_report_counts_each_category():
    classifications = {
        ("jd1", "cv1"): "clean",
        ("jd1", "cv2"): "clean",
        ("jd1", "cv3"): "fixed_by_retry",
        ("jd2", "cv4"): "dropped",
    }

    report = build_report(
        classifications,
        total_weaknesses_checked=1224,
        items_dropped=26,
        offline_residual_contradictions=10,
        offline_denominator=1224,
    )

    assert report["total_pairs"] == 4
    assert report["total_weaknesses_checked"] == 1224
    assert report["pairs_with_initial_contradiction"] == 2
    assert report["pairs_fixed_by_retry"] == 1
    assert report["pairs_dropped"] == 1
    assert report["items_dropped"] == 26
    assert report["offline_audit_residual_contradictions"] == 10
    assert report["offline_audit_denominator"] == 1224
    assert report["offline_audit_rate"] == 10 / 1224
