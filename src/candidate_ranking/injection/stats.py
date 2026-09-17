from __future__ import annotations

from scipy.stats import wilcoxon


def rank_biserial(a: list[float], b: list[float]) -> float:
    n_pos = sum(1 for x, y in zip(a, b) if x > y)
    n_neg = sum(1 for x, y in zip(a, b) if x < y)
    n = len(a)
    return (n_pos - n_neg) / n if n else 0.0


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


def wilcoxon_result(label: str, a: list[float], b: list[float]) -> dict | None:
    n_pairs = len(a)
    if n_pairs >= 1 and any(x != y for x, y in zip(a, b)):
        stat, p_value = wilcoxon(a, b)
        r = rank_biserial(a, b)
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
    attack, control = [], []
    for r in results:
        if r["condition"] != attack_condition:
            continue
        key = (r["jd_id"], r["candidate_id"])
        if key in control_by_pair:
            attack.append(r["rank_delta"])
            control.append(control_by_pair[key])
    return attack, control


def paired_deltas_by_condition(
    results: list[dict], condition_a: str, condition_b: str,
) -> tuple[list[float], list[float]]:
    by_key: dict[tuple, dict[str, float]] = {}
    for r in results:
        if r["condition"] in (condition_a, condition_b):
            key = (r["jd_id"], r["candidate_id"], r["variant_name"])
            by_key.setdefault(key, {})[r["condition"]] = r["rank_delta"]
    a, b = [], []
    for pair in by_key.values():
        if condition_a in pair and condition_b in pair:
            a.append(pair[condition_a])
            b.append(pair[condition_b])
    return a, b
