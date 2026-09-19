from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scipy.stats import kendalltau, mannwhitneyu, spearmanr, wilcoxon

from candidate_ranking.config import RunConfig, apply_env_overrides

_VENDOR_PUBLISHED_LATENCY_CLAIM = "70ms-500ms end-to-end (TypeSafe, self-reported, West Coast laptop)"
_VENDOR_PUBLISHED_COMPARISON_CLAIM = "existing frontier LLMs: 3 to 329 seconds end-to-end (TypeSafe, self-reported)"
_VENDOR_PUBLISHED_SPEEDUP_CLAIM = "40x-200x faster for comparable System One task intelligence (TypeSafe, self-reported)"


def _rank_biserial(first: list[float], second: list[float]) -> float:
    """Directional effect size for a paired comparison: (pairs where `first` wins minus
    pairs where `second` wins) / n, ties excluded from neither count nor sign.

    This is a sign-based proportion difference, not the matched-pairs rank-biserial
    correlation derived from the Wilcoxon signed-rank statistic (which would weight each
    pair by the magnitude-rank of its difference, via (W+ - W-)/(W+ + W-)). It ignores how
    large each difference is, only its direction -- a deliberate simplification, reported
    here (and in the paper, as "rank-biserial r") as an easily-interpreted companion to the
    Wilcoxon p-value, not a drop-in replacement for the textbook statistic of that name.
    """
    n_first_higher = sum(1 for a, b in zip(first, second) if a > b)
    n_second_higher = sum(1 for a, b in zip(first, second) if a < b)
    n = len(first)
    return (n_first_higher - n_second_higher) / n if n else 0.0


def _wilcoxon_result(label: str, first: list[float], second: list[float]) -> dict | None:
    if len(first) >= 1 and any(a != b for a, b in zip(first, second)):
        stat, p_value = wilcoxon(first, second)
        return {
            "label": label,
            "n_pairs": len(first),
            "statistic": float(stat),
            "p_value": float(p_value),
            "rank_biserial_r": _rank_biserial(first, second),
            "mean_first": statistics.mean(first),
            "mean_second": statistics.mean(second),
        }
    return None


def analyze_test_retest(records: list[dict]) -> dict:
    score_stdevs: list[float] = []
    requirement_score_stdevs: list[float] = []
    recommendation_full_agreement = 0
    latencies: list[float] = []

    for record in records:
        repeats = record["repeats"]
        scores = [r["overall_fit_score"] for r in repeats]
        if len(scores) >= 2:
            score_stdevs.append(statistics.stdev(scores))

        recommendations = {r["overall_recommendation"] for r in repeats}
        if len(recommendations) == 1:
            recommendation_full_agreement += 1

        requirement_keys = set(repeats[0]["requirement_scores"])
        for key in requirement_keys:
            values = [r["requirement_scores"].get(key) for r in repeats if key in r["requirement_scores"]]
            if len(values) >= 2:
                requirement_score_stdevs.append(statistics.stdev(values))

        latencies.extend(r["latency_seconds"] for r in repeats)

    n = len(records)
    return {
        "n_pairs": n,
        "n_repeats_per_pair": len(records[0]["repeats"]) if records else 0,
        "overall_fit_score_stdev": {
            "mean": statistics.mean(score_stdevs) if score_stdevs else None,
            "median": statistics.median(score_stdevs) if score_stdevs else None,
        },
        "requirement_score_stdev": {
            "mean": statistics.mean(requirement_score_stdevs) if requirement_score_stdevs else None,
            "median": statistics.median(requirement_score_stdevs) if requirement_score_stdevs else None,
            "n_requirement_observations": len(requirement_score_stdevs),
        },
        "recommendation_full_agreement_rate": recommendation_full_agreement / n if n else None,
        "latency_seconds": {
            "n": len(latencies),
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
    }


def analyze_ranking_convergence(records: list[dict]) -> dict:
    by_jd: dict[str, list[dict]] = {}
    for record in records:
        by_jd.setdefault(record["jd_id"], []).append(record)

    per_job_profile: dict[str, dict] = {}
    all_mean_taus: list[float] = []

    for jd_id, jd_records in by_jd.items():
        candidate_ids = [r["candidate_id"] for r in jd_records]
        n_repeats = len(jd_records[0]["repeats"]) if jd_records else 0
        if len(candidate_ids) < 2 or n_repeats < 2:
            per_job_profile[jd_id] = {"n_candidates": len(candidate_ids), "n_repeats": n_repeats, "mean_kendall_tau": None}
            continue

        rankings = []
        for i in range(n_repeats):
            scores = {r["candidate_id"]: r["repeats"][i]["overall_fit_score"] for r in jd_records}
            rankings.append(sorted(candidate_ids, key=lambda cid: scores[cid], reverse=True))

        pair_taus = []
        for i in range(len(rankings)):
            for j in range(i + 1, len(rankings)):
                position_i = {cid: idx for idx, cid in enumerate(rankings[i])}
                position_j = {cid: idx for idx, cid in enumerate(rankings[j])}
                tau, _ = kendalltau(
                    [position_i[cid] for cid in candidate_ids],
                    [position_j[cid] for cid in candidate_ids],
                )
                if tau == tau:  # exclude NaN (e.g. all-tied scores)
                    pair_taus.append(float(tau))

        mean_tau = statistics.mean(pair_taus) if pair_taus else None
        per_job_profile[jd_id] = {"n_candidates": len(candidate_ids), "n_repeats": n_repeats, "mean_kendall_tau": mean_tau}
        if mean_tau is not None:
            all_mean_taus.append(mean_tau)

    return {
        "per_job_profile": per_job_profile,
        "mean_kendall_tau_across_all_profiles": statistics.mean(all_mean_taus) if all_mean_taus else None,
        "min_kendall_tau_profile": min(per_job_profile.items(), key=lambda kv: (kv[1]["mean_kendall_tau"] is None, kv[1]["mean_kendall_tau"] or 0))[0] if per_job_profile else None,
    }


def analyze_score_decomposition_diagnostic(records: list[dict], ranking_convergence: dict) -> dict:
    """Correlate each job profile's shortlisted-pool signal-to-noise ratio against its
    ranking-convergence Kendall-tau, to test whether ranking instability traces to
    genuine score-noise proximity between similarly qualified applicants rather than
    to model unreliability (paper2_jev.tex, Section III-E)."""
    by_jd: dict[str, list[dict]] = {}
    for record in records:
        by_jd.setdefault(record["jd_id"], []).append(record)

    per_job_profile: dict[str, dict] = {}
    snrs: list[float] = []
    taus: list[float] = []

    for jd_id, jd_records in by_jd.items():
        candidate_means: list[float] = []
        candidate_stdevs: list[float] = []
        for r in jd_records:
            scores = [rep["overall_fit_score"] for rep in r["repeats"]]
            if len(scores) >= 2:
                candidate_means.append(statistics.mean(scores))
                candidate_stdevs.append(statistics.stdev(scores))

        tau = ranking_convergence["per_job_profile"].get(jd_id, {}).get("mean_kendall_tau")

        if len(candidate_means) < 2 or not candidate_stdevs:
            per_job_profile[jd_id] = {
                "n_candidates": len(candidate_means), "signal": None, "noise": None,
                "snr": None, "mean_kendall_tau": tau,
            }
            continue

        signal = statistics.stdev(candidate_means)
        noise = statistics.mean(candidate_stdevs)
        snr = signal / noise if noise > 0 else None
        per_job_profile[jd_id] = {
            "n_candidates": len(candidate_means), "signal": signal, "noise": noise,
            "snr": snr, "mean_kendall_tau": tau,
        }
        if snr is not None and tau is not None:
            snrs.append(snr)
            taus.append(tau)

    correlation = None
    if len(snrs) >= 3:
        rho, p_value = spearmanr(snrs, taus)
        correlation = {"spearman_rho": float(rho), "p_value": float(p_value), "n": len(snrs)}

    return {"per_job_profile": per_job_profile, "snr_vs_kendall_tau_correlation": correlation}


def analyze_internal_coherence(run_dir: Path, jd_ids: list[str]) -> dict:
    assessments: list[dict] = []
    for jd_id in jd_ids:
        path = run_dir / jd_id / "assessments.json"
        if not path.exists():
            continue
        assessments.extend(json.loads(path.read_text(encoding="utf-8")).values())

    mean_requirement_scores = []
    overall_scores_for_correlation = []
    for a in assessments:
        if a["requirement_scores"]:
            mean_requirement_scores.append(statistics.mean(a["requirement_scores"].values()))
            overall_scores_for_correlation.append(a["overall_fit_score"])

    correlation = None
    if len(mean_requirement_scores) >= 3:
        rho, p_value = spearmanr(mean_requirement_scores, overall_scores_for_correlation)
        correlation = {"spearman_rho": float(rho), "p_value": float(p_value), "n": len(mean_requirement_scores)}

    hire_scores = [a["overall_fit_score"] for a in assessments if a["overall_recommendation"] == "hire"]
    no_scores = [a["overall_fit_score"] for a in assessments if a["overall_recommendation"] == "no"]
    recommendation_vs_score = None
    if hire_scores and no_scores:
        stat, p_value = mannwhitneyu(hire_scores, no_scores, alternative="greater")
        recommendation_vs_score = {
            "hire_mean": statistics.mean(hire_scores),
            "hire_n": len(hire_scores),
            "no_mean": statistics.mean(no_scores),
            "no_n": len(no_scores),
            "mannwhitneyu_statistic": float(stat),
            "p_value_hire_greater_than_no": float(p_value),
        }

    meets_min_scores = [a["overall_fit_score"] for a in assessments if a["meets_min_qualifications"]]
    fails_min_scores = [a["overall_fit_score"] for a in assessments if not a["meets_min_qualifications"]]
    qualifications_vs_score = None
    if meets_min_scores and fails_min_scores:
        stat, p_value = mannwhitneyu(meets_min_scores, fails_min_scores, alternative="greater")
        qualifications_vs_score = {
            "meets_min_mean": statistics.mean(meets_min_scores),
            "meets_min_n": len(meets_min_scores),
            "fails_min_mean": statistics.mean(fails_min_scores),
            "fails_min_n": len(fails_min_scores),
            "mannwhitneyu_statistic": float(stat),
            "p_value_meets_greater_than_fails": float(p_value),
        }

    return {
        "n_assessments": len(assessments),
        "requirement_vs_overall_score_correlation": correlation,
        "recommendation_vs_score": recommendation_vs_score,
        "min_qualifications_vs_score": qualifications_vs_score,
    }


_OVERALL_FIT_KEY = "overall_fit_score"
_UNAFFECTED_CONTROL_KEYS = ("overall_recommendation", "meets_min_qualifications")


def _paired_bucket(concrete: list[float], vague: list[float], label: str) -> dict:
    return {
        "wilcoxon": _wilcoxon_result(label, concrete, vague),
        "concrete_pct_below_0.5": 100 * sum(1 for v in concrete if v < 0.5) / len(concrete) if concrete else None,
        "vague_pct_below_0.5": 100 * sum(1 for v in vague if v < 0.5) / len(vague) if vague else None,
        "n_observations": len(concrete),
    }


def analyze_ablation(records: list[dict]) -> dict:
    concrete_overall: list[float] = []
    vague_overall: list[float] = []
    concrete_control: list[float] = []
    vague_control: list[float] = []
    concrete_requirement: list[float] = []
    vague_requirement: list[float] = []
    vague_latencies: list[float] = []

    for record in records:
        concrete_conf = record["concrete_confidence"]
        vague_conf = record["vague_confidence"]

        if _OVERALL_FIT_KEY in concrete_conf and _OVERALL_FIT_KEY in vague_conf:
            concrete_overall.append(concrete_conf[_OVERALL_FIT_KEY])
            vague_overall.append(vague_conf[_OVERALL_FIT_KEY])

        for key in _UNAFFECTED_CONTROL_KEYS:
            if key in concrete_conf and key in vague_conf:
                concrete_control.append(concrete_conf[key])
                vague_control.append(vague_conf[key])

        shared_requirement_keys = {
            k for k in concrete_conf if k.startswith("requirement::")
        } & {k for k in vague_conf if k.startswith("requirement::")}
        for key in shared_requirement_keys:
            concrete_requirement.append(concrete_conf[key])
            vague_requirement.append(vague_conf[key])

        vague_latencies.append(record["vague_latency_seconds"])

    return {
        "n_pairs": len(records),
        "overall_fit_score": _paired_bucket(concrete_overall, vague_overall, "overall_fit_score_confidence"),
        "unaffected_control": _paired_bucket(concrete_control, vague_control, "unaffected_control_confidence"),
        "requirement_questions": _paired_bucket(concrete_requirement, vague_requirement, "requirement_confidence"),
        "vague_latency_seconds_mean": statistics.mean(vague_latencies) if vague_latencies else None,
    }


def analyze_seniority_education_ablation(records: list[dict]) -> dict:
    concrete_by_key: dict[str, list[float]] = {"seniority": [], "education": []}
    vague_by_key: dict[str, list[float]] = {"seniority": [], "education": []}
    vague_latencies: list[float] = []

    for record in records:
        concrete_conf = record["concrete_confidence"]
        vague_conf = record["vague_confidence"]
        for key in ("seniority", "education"):
            if key in concrete_conf and key in vague_conf:
                concrete_by_key[key].append(concrete_conf[key])
                vague_by_key[key].append(vague_conf[key])
        vague_latencies.append(record["vague_latency_seconds"])

    buckets = {
        key: _paired_bucket(concrete_by_key[key], vague_by_key[key], f"{key}_confidence")
        for key in ("seniority", "education")
    }
    concrete_pooled = concrete_by_key["seniority"] + concrete_by_key["education"]
    vague_pooled = vague_by_key["seniority"] + vague_by_key["education"]

    return {
        "n_pairs": len(records),
        "seniority": buckets["seniority"],
        "education": buckets["education"],
        "pooled": _paired_bucket(concrete_pooled, vague_pooled, "seniority_education_pooled_confidence"),
        "vague_latency_seconds_mean": statistics.mean(vague_latencies) if vague_latencies else None,
    }


def render_markdown(report: dict) -> str:
    tr = report.get("test_retest")
    conv = report.get("ranking_convergence")
    coh = report.get("internal_coherence")
    abl = report.get("ablation")

    lines = [f"# Jev Evaluation Study -- Run {report['run_id']}"]

    if tr:
        lat = tr["latency_seconds"]
        lines += [
            "",
            "## Test-retest reliability",
            f"- {tr['n_pairs']} pairs x {tr['n_repeats_per_pair']} repeats",
            f"- overall_fit_score stdev across repeats: mean={tr['overall_fit_score_stdev']['mean']:.2f}, "
            f"median={tr['overall_fit_score_stdev']['median']:.2f}",
            f"- per-requirement score stdev: mean={tr['requirement_score_stdev']['mean']:.2f} "
            f"(n={tr['requirement_score_stdev']['n_requirement_observations']} requirement observations)",
            f"- recommendation full agreement rate: {tr['recommendation_full_agreement_rate']:.1%}",
        ]
    if conv:
        lines += [
            "",
            "## Ranking convergence (Paper-1-equivalent: candidate order stability across repeats)",
            f"- mean Kendall-tau across all {len(conv['per_job_profile'])} job profiles: "
            f"{conv['mean_kendall_tau_across_all_profiles']:.3f}" if conv['mean_kendall_tau_across_all_profiles'] is not None else "- insufficient data",
        ]
        for jd_id, jd_conv in sorted(conv["per_job_profile"].items(), key=lambda kv: (kv[1]["mean_kendall_tau"] is None, kv[1]["mean_kendall_tau"] or 0)):
            tau_str = f"{jd_conv['mean_kendall_tau']:.3f}" if jd_conv["mean_kendall_tau"] is not None else "N/A"
            lines.append(f"  - {jd_id} (n={jd_conv['n_candidates']}): tau={tau_str}")
    if coh:
        lines += [
            "",
            "## Internal coherence",
            f"- n={coh['n_assessments']} assessments",
        ]
        if coh["requirement_vs_overall_score_correlation"]:
            c = coh["requirement_vs_overall_score_correlation"]
            lines.append(f"- mean(requirement_scores) vs overall_fit_score: Spearman rho={c['spearman_rho']:.3f} (p={c['p_value']:.4g}, n={c['n']})")
        if coh["recommendation_vs_score"]:
            r = coh["recommendation_vs_score"]
            lines.append(
                f"- hire (n={r['hire_n']}, mean={r['hire_mean']:.1f}) vs no (n={r['no_n']}, mean={r['no_mean']:.1f}): "
                f"Mann-Whitney U p={r['p_value_hire_greater_than_no']:.4g}"
            )
        if coh["min_qualifications_vs_score"]:
            q = coh["min_qualifications_vs_score"]
            lines.append(
                f"- meets_min_qualifications=True (n={q['meets_min_n']}, mean={q['meets_min_mean']:.1f}) vs "
                f"False (n={q['fails_min_n']}, mean={q['fails_min_mean']:.1f}): "
                f"Mann-Whitney U p={q['p_value_meets_greater_than_fails']:.4g}"
            )
    if abl:
        lines += ["", "## Criteria-design ablation (concrete vs. vague Score criteria)", f"- n={abl['n_pairs']} sampled pairs"]
        if abl["overall_fit_score"]["wilcoxon"]:
            w = abl["overall_fit_score"]["wilcoxon"]
            lines.append(
                f"- overall_fit_score confidence (the one Score-type fixed question, directly affected by the criteria change): "
                f"concrete mean={w['mean_first']:.3f} vs vague mean={w['mean_second']:.3f} "
                f"-- Wilcoxon p={w['p_value']:.4g}, r={w['rank_biserial_r']:.3f} "
                f"(<0.5: concrete {abl['overall_fit_score']['concrete_pct_below_0.5']:.0f}% vs vague {abl['overall_fit_score']['vague_pct_below_0.5']:.0f}%)"
            )
        if abl["unaffected_control"]["wilcoxon"]:
            w = abl["unaffected_control"]["wilcoxon"]
            lines.append(
                f"- unaffected_control confidence (overall_recommendation + meets_min_qualifications, criteria "
                f"NOT changed by the ablation -- expected null effect): "
                f"concrete mean={w['mean_first']:.3f} vs vague mean={w['mean_second']:.3f} "
                f"-- Wilcoxon p={w['p_value']:.4g}, r={w['rank_biserial_r']:.3f}"
            )
        if abl["requirement_questions"]["wilcoxon"]:
            w = abl["requirement_questions"]["wilcoxon"]
            lines.append(
                f"- Requirement questions confidence (n_obs={abl['requirement_questions']['n_observations']}): "
                f"concrete mean={w['mean_first']:.3f} vs vague mean={w['mean_second']:.3f} "
                f"-- Wilcoxon p={w['p_value']:.4g}, r={w['rank_biserial_r']:.3f} "
                f"(<0.5: concrete {abl['requirement_questions']['concrete_pct_below_0.5']:.0f}% vs vague {abl['requirement_questions']['vague_pct_below_0.5']:.0f}%)"
            )
    snd = report.get("score_decomposition_diagnostic")
    if snd:
        lines += [
            "",
            "## Structured score decomposition diagnostic (signal-to-noise ratio vs. ranking convergence)",
        ]
        for jd_id, jd_snd in sorted(
            snd["per_job_profile"].items(),
            key=lambda kv: (kv[1]["snr"] is None, kv[1]["snr"] or 0),
        ):
            if jd_snd["snr"] is not None:
                lines.append(
                    f"  - {jd_id} (n={jd_snd['n_candidates']}): signal={jd_snd['signal']:.2f}, "
                    f"noise={jd_snd['noise']:.2f}, snr={jd_snd['snr']:.2f}, "
                    f"tau={jd_snd['mean_kendall_tau']:.3f}" if jd_snd["mean_kendall_tau"] is not None
                    else f"  - {jd_id} (n={jd_snd['n_candidates']}): snr={jd_snd['snr']:.2f}, tau=N/A"
                )
            else:
                lines.append(f"  - {jd_id} (n={jd_snd['n_candidates']}): insufficient data")
        if snd["snr_vs_kendall_tau_correlation"]:
            c = snd["snr_vs_kendall_tau_correlation"]
            lines.append(f"- SNR vs. Kendall-tau: Spearman rho={c['spearman_rho']:.3f} (p={c['p_value']:.4g}, n={c['n']})")
        else:
            lines.append("- SNR vs. Kendall-tau: insufficient data for correlation")

    sen_edu_abl = report.get("seniority_education_ablation")
    if sen_edu_abl:
        lines += ["", "## Seniority/education criteria ablation (concrete vs. bare-label Score)", f"- n={sen_edu_abl['n_pairs']} eligible pairs"]
        for label, key in (("seniority", "seniority"), ("education", "education"), ("pooled", "pooled")):
            bucket = sen_edu_abl[key]
            if bucket["wilcoxon"]:
                w = bucket["wilcoxon"]
                lines.append(
                    f"- {label} confidence (n_obs={bucket['n_observations']}): "
                    f"concrete mean={w['mean_first']:.3f} vs bare-label mean={w['mean_second']:.3f} "
                    f"-- Wilcoxon p={w['p_value']:.4g}, r={w['rank_biserial_r']:.3f} "
                    f"(<0.5: concrete {bucket['concrete_pct_below_0.5']:.0f}% vs bare-label {bucket['vague_pct_below_0.5']:.0f}%)"
                )
            else:
                lines.append(f"- {label} confidence: insufficient paired data (n_obs={bucket['n_observations']})")

    if tr:
        lat = tr["latency_seconds"]
        lines += [
            "",
            "## Efficiency",
            f"- Measured Jev latency (this study, n={lat['n']} calls): mean={lat['mean']*1000:.0f}ms, "
            f"median={lat['median']*1000:.0f}ms"
            + (f", p95={lat['p95']*1000:.0f}ms" if lat["p95"] is not None else ""),
            f"- Vendor-published (not independently verified against a reconstructed baseline in this study):",
            f"  - {_VENDOR_PUBLISHED_LATENCY_CLAIM}",
            f"  - vs. {_VENDOR_PUBLISHED_COMPARISON_CLAIM}",
            f"  - claimed speedup: {_VENDOR_PUBLISHED_SPEEDUP_CLAIM}",
        ]
    lines.append("")
    return "\n".join(lines)


def main(run_id: str) -> None:
    cfg = apply_env_overrides(RunConfig.full(PROJECT_ROOT))
    run_dir = cfg.runs_dir / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    jd_ids = manifest["jd_ids"]

    eval_dir = run_dir / "evaluation"
    report: dict = {"run_id": run_id}

    test_retest_path = eval_dir / "test_retest.json"
    if test_retest_path.exists():
        test_retest_records = json.loads(test_retest_path.read_text(encoding="utf-8"))
        report["test_retest"] = analyze_test_retest(test_retest_records)
        report["ranking_convergence"] = analyze_ranking_convergence(test_retest_records)
        report["internal_coherence"] = analyze_internal_coherence(run_dir, jd_ids)
        report["score_decomposition_diagnostic"] = analyze_score_decomposition_diagnostic(
            test_retest_records, report["ranking_convergence"]
        )

    ablation_path = eval_dir / "ablation.json"
    if ablation_path.exists():
        ablation_records = json.loads(ablation_path.read_text(encoding="utf-8"))
        report["ablation"] = analyze_ablation(ablation_records)

    seniority_education_ablation_path = eval_dir / "seniority_education_ablation.json"
    if seniority_education_ablation_path.exists():
        seniority_education_records = json.loads(seniority_education_ablation_path.read_text(encoding="utf-8"))
        report["seniority_education_ablation"] = analyze_seniority_education_ablation(seniority_education_records)

    (eval_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = render_markdown(report)
    (eval_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\nWrote {eval_dir / 'report.json'} and {eval_dir / 'report.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    main(args.run_id)
