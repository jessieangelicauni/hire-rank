from candidate_ranking.injection.stats import (
    holm_correct,
    paired_deltas_by_condition,
    paired_deltas_vs_control,
    rank_biserial,
    wilcoxon_result,
)


def test_rank_biserial_all_positive_differences_is_one():
    assert rank_biserial([3, 4, 5], [1, 1, 1]) == 1.0


def test_rank_biserial_mixed_differences_cancel_out():
    assert rank_biserial([3, 1, 5], [1, 4, 5]) == 0.0


def test_holm_correct_matches_hand_computed_example():
    adjusted = holm_correct([0.01, 0.02, 0.03])
    assert adjusted == [0.03, 0.04, 0.04]


def test_wilcoxon_result_returns_none_when_all_differences_are_zero():
    assert wilcoxon_result("label", [1, 2, 3], [1, 2, 3]) is None


def test_wilcoxon_result_returns_stats_dict_when_pairs_differ():
    result = wilcoxon_result("label", [3, 4, 5], [1, 1, 1])
    assert result is not None
    assert result["label"] == "label"
    assert result["n_pairs"] == 3
    assert result["rank_biserial_r"] == 1.0


def test_paired_deltas_vs_control_matches_by_jd_and_candidate():
    results = [
        {"jd_id": "a", "candidate_id": "c1", "condition": "control_no_injection", "rank_delta": -1},
        {"jd_id": "a", "candidate_id": "c1", "condition": "unmitigated", "rank_delta": 2},
        {"jd_id": "b", "candidate_id": "c2", "condition": "unmitigated", "rank_delta": 3},
    ]
    attack, control = paired_deltas_vs_control(results, "unmitigated")
    assert attack == [2]
    assert control == [-1]


def test_paired_deltas_by_condition_matches_by_pair_and_variant():
    results = [
        {
            "jd_id": "a", "candidate_id": "c1", "variant_name": "comparative_injection",
            "condition": "comparative_unmitigated", "rank_delta": 2,
        },
        {
            "jd_id": "a", "candidate_id": "c1", "variant_name": "comparative_injection",
            "condition": "comparative_mitigated", "rank_delta": -1,
        },
    ]
    a, b = paired_deltas_by_condition(results, "comparative_mitigated", "comparative_unmitigated")
    assert a == [-1]
    assert b == [2]
