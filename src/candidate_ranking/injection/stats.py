from __future__ import annotations

from scipy.stats import wilcoxon


def rank_biserial(first_deltas: list[float], second_deltas: list[float]) -> float:
    n_first_higher = sum(1 for first, second in zip(first_deltas, second_deltas) if first > second)
    n_second_higher = sum(1 for first, second in zip(first_deltas, second_deltas) if first < second)
    n = len(first_deltas)
    return (n_first_higher - n_second_higher) / n if n else 0.0


def holm_correct(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction, adjusted p-values returned in
    the same order as the input."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted_sorted = []
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (m - rank) * p_values[i])
        adjusted_sorted.append(min(running_max, 1.0))
    result = [0.0] * m
    for rank, i in enumerate(order):
        result[i] = adjusted_sorted[rank]
    return result


def wilcoxon_result(label: str, first_deltas: list[float], second_deltas: list[float]) -> dict | None:
    n_pairs = len(first_deltas)
    if n_pairs >= 1 and any(first != second for first, second in zip(first_deltas, second_deltas)):
        stat, p_value = wilcoxon(first_deltas, second_deltas)
        r = rank_biserial(first_deltas, second_deltas)
        return {
            "label": label, "n_pairs": n_pairs, "statistic": float(stat),
            "p_value": float(p_value), "rank_biserial_r": r,
        }
    return None


def _deltas_by_pair_key(results: list[dict], condition: str) -> dict[tuple, float]:
    return {(r["jd_id"], r["candidate_id"]): r["rank_delta"] for r in results if r["condition"] == condition}


def paired_deltas_vs_control(
    results: list[dict], attack_condition: str, control_condition: str = "control_no_injection",
) -> tuple[list[float], list[float]]:
    control_by_pair = _deltas_by_pair_key(results, control_condition)
    attack_deltas, control_deltas = [], []
    for r in results:
        if r["condition"] != attack_condition:
            continue
        key = (r["jd_id"], r["candidate_id"])
        if key in control_by_pair:
            attack_deltas.append(r["rank_delta"])
            control_deltas.append(control_by_pair[key])
    return attack_deltas, control_deltas


def paired_deltas_by_condition(
    results: list[dict], condition_a: str, condition_b: str,
) -> tuple[list[float], list[float]]:
    deltas_by_condition_at_key: dict[tuple, dict[str, float]] = {}
    for r in results:
        if r["condition"] in (condition_a, condition_b):
            key = (r["jd_id"], r["candidate_id"], r["variant_name"])
            deltas_by_condition_at_key.setdefault(key, {})[r["condition"]] = r["rank_delta"]
    deltas_a, deltas_b = [], []
    for conditions_seen in deltas_by_condition_at_key.values():
        if condition_a in conditions_seen and condition_b in conditions_seen:
            deltas_a.append(conditions_seen[condition_a])
            deltas_b.append(conditions_seen[condition_b])
    return deltas_a, deltas_b
