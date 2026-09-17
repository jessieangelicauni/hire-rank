from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.injection.measurement import holm_correct, paired_deltas_by_condition, paired_deltas_vs_control, wilcoxon_result


def main(run_id: str) -> None:
    cfg = apply_env_overrides(RunConfig.full(PROJECT_ROOT))
    run_dir = cfg.runs_dir / run_id
    results = json.loads((run_dir / "injection_study" / "extended_results.json").read_text(encoding="utf-8"))

    comparisons: list[tuple[str, list[float], list[float]]] = []
    for condition in ("comparative_unmitigated", "comparative_mitigated", "adaptive_unmitigated", "adaptive_mitigated", "defense_a_classifier", "defense_b_self_reminder"):
        a, control = paired_deltas_vs_control(results, condition)
        comparisons.append((f"{condition}_vs_control", a, control))

    a, b = paired_deltas_by_condition(results, "comparative_mitigated", "comparative_unmitigated")
    comparisons.append(("comparative_mitigated_vs_comparative_unmitigated", a, b))

    a, b = paired_deltas_by_condition(results, "adaptive_mitigated", "adaptive_unmitigated")
    comparisons.append(("adaptive_mitigated_vs_adaptive_unmitigated", a, b))

    prior_results = json.loads((run_dir / "injection_study" / "results.json").read_text(encoding="utf-8"))
    original_unmitigated_by_pair = {
        (r["jd_id"], r["candidate_id"]): r["rank_delta"] for r in prior_results if r["condition"] == "unmitigated"
    }
    original_mitigated_by_pair = {
        (r["jd_id"], r["candidate_id"]): r["rank_delta"] for r in prior_results if r["condition"] == "mitigated"
    }
    for defense_condition in ("defense_a_classifier", "defense_b_self_reminder"):
        defense_rows = [r for r in results if r["condition"] == defense_condition]
        vs_unmitigated_a, vs_unmitigated_b = [], []
        vs_mitigated_a, vs_mitigated_b = [], []
        for r in defense_rows:
            key = (r["jd_id"], r["candidate_id"])
            if key in original_unmitigated_by_pair:
                vs_unmitigated_a.append(r["rank_delta"])
                vs_unmitigated_b.append(original_unmitigated_by_pair[key])
            if key in original_mitigated_by_pair:
                vs_mitigated_a.append(r["rank_delta"])
                vs_mitigated_b.append(original_mitigated_by_pair[key])
        comparisons.append((f"{defense_condition}_vs_original_unmitigated", vs_unmitigated_a, vs_unmitigated_b))
        comparisons.append((f"{defense_condition}_vs_original_mitigated", vs_mitigated_a, vs_mitigated_b))

    raw_results = [wilcoxon_result(label, a, b) for label, a, b in comparisons]
    p_values = [r["p_value"] for r in raw_results if r is not None]
    p_holm_values = iter(holm_correct(p_values))

    report: dict = {}
    for r in raw_results:
        if r is None:
            continue
        r["p_holm"] = next(p_holm_values)
        report[r["label"]] = r
        print(
            f"{r['label']}: n_pairs={r['n_pairs']}, statistic={r['statistic']:.3f}, "
            f"p={r['p_value']:.5f}, p_holm={r['p_holm']:.5f}, r={r['rank_biserial_r']:+.3f}"
        )

    out_path = run_dir / "injection_study" / "extended_significance_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "20260911-154235")
