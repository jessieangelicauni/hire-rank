from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scipy.stats import wilcoxon

from candidate_ranking.config import RunConfig, apply_env_overrides

CONDITIONS = ("control_no_injection", "unmitigated", "mitigated", "mitigated_no_filter")
LABELS = {
    "control_no_injection": "control",
    "unmitigated": "unmitigated",
    "mitigated": "mitigated",
    "mitigated_no_filter": "mitigated_no_filter",
}

_FILTERED_CONDITIONS = ("mitigated",)


def summarize_by_condition(results: list[dict]) -> dict:
    summary = {}
    for condition in CONDITIONS:
        rows = [r for r in results if r["condition"] == condition]
        n = len(rows)
        n_survived = sum(1 for r in rows if r["marker_survived"])
        deltas = [r["rank_delta"] for r in rows]
        entry = {
            "n": n,
            "attack_success_rate": n_survived / n if n else 0.0,
            "mean_rank_delta": sum(deltas) / n if n else 0.0,
            "n_worsened": sum(1 for d in deltas if d < 0),
        }
        if condition in _FILTERED_CONDITIONS:
            dropped = [r["lines_dropped"] for r in rows if "lines_dropped" in r]
            entry["mean_lines_dropped"] = sum(dropped) / len(dropped) if dropped else 0.0
            entry["frac_with_lines_dropped"] = (
                sum(1 for d in dropped if d > 0) / len(dropped) if dropped else 0.0
            )
        summary[LABELS[condition]] = entry
    return summary


def _deltas_by_pair_key(results: list[dict], condition: str) -> dict[tuple, float]:
    return {(r["jd_id"], r["candidate_id"]): r["rank_delta"] for r in results if r["condition"] == condition}


def paired_deltas_vs_control(results: list[dict], attack_condition: str) -> tuple[list[float], list[float]]:
    control_by_pair = _deltas_by_pair_key(results, "control_no_injection")
    attack, control = [], []
    for r in results:
        if r["condition"] != attack_condition:
            continue
        key = (r["jd_id"], r["candidate_id"])
        if key in control_by_pair:
            attack.append(r["rank_delta"])
            control.append(control_by_pair[key])
    return attack, control


def paired_deltas_by_variant(results: list[dict], condition_a: str, condition_b: str) -> tuple[list[float], list[float]]:
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


def rank_biserial(a: list[float], b: list[float]) -> float:
    n_pos = sum(1 for x, y in zip(a, b) if x > y)
    n_neg = sum(1 for x, y in zip(a, b) if x < y)
    n = len(a)
    return (n_pos - n_neg) / n if n else 0.0


def _wilcoxon_result(label: str, a: list[float], b: list[float]) -> dict | None:
    n_pairs = len(a)
    if n_pairs >= 1 and any(x != y for x, y in zip(a, b)):
        stat, p_value = wilcoxon(a, b)
        r = rank_biserial(a, b)
        return {"label": label, "n_pairs": n_pairs, "statistic": float(stat), "p_value": float(p_value), "rank_biserial_r": r}
    print(f"Wilcoxon test skipped ({label}): fewer than 1 paired sample or all differences are zero.")
    return None


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


def main(run_id: str) -> None:
    cfg = apply_env_overrides(RunConfig.full(PROJECT_ROOT))
    run_dir = cfg.runs_dir / run_id
    results = json.loads((run_dir / "injection_study" / "results.json").read_text(encoding="utf-8"))

    summary = summarize_by_condition(results)
    print(json.dumps(summary, indent=2))

    # Fixed, pre-specified family of six comparisons (matches paper Table III),
    # so Holm-Bonferroni is always corrected across the complete set actually
    # performed, not a family chosen after seeing results.
    comparisons: list[tuple[str, list[float], list[float]]] = []
    for condition in ("unmitigated", "mitigated", "mitigated_no_filter"):
        a, control = paired_deltas_vs_control(results, condition)
        comparisons.append((f"{LABELS[condition]}_vs_control", a, control))

    a, b = paired_deltas_by_variant(results, "mitigated", "unmitigated")
    comparisons.append(("mitigated_vs_unmitigated", a, b))

    a, b = paired_deltas_by_variant(results, "mitigated_no_filter", "unmitigated")
    comparisons.append(("mitigated_no_filter_vs_unmitigated", a, b))

    a, b = paired_deltas_by_variant(results, "mitigated", "mitigated_no_filter")
    comparisons.append(("mitigated_vs_mitigated_no_filter", a, b))

    raw_results = [_wilcoxon_result(label, a, b) for label, a, b in comparisons]
    p_values = [r["p_value"] for r in raw_results if r is not None]
    p_holm_values = iter(holm_correct(p_values))

    wilcoxon_report: dict = {}
    for r in raw_results:
        if r is None:
            continue
        r["p_holm"] = next(p_holm_values)
        wilcoxon_report[r["label"]] = r
        print(
            f"Wilcoxon signed-rank ({r['label']}): n_pairs={r['n_pairs']}, "
            f"statistic={r['statistic']:.3f}, p={r['p_value']:.5f}, p_holm={r['p_holm']:.5f}, "
            f"r={r['rank_biserial_r']:+.3f}"
        )

    report: dict = {"summary": summary, "wilcoxon": wilcoxon_report}

    out_path = run_dir / "injection_study" / "significance_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    main(args.run_id)
