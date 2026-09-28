from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from candidate_ranking.config import RunConfig, apply_env_overrides


def _informative_requirements(assessments: dict[str, dict], min_pool_median: float) -> dict[str, float]:
    """Requirements where the pool's own median score is high enough that scoring near-zero on them
    is a real outlier, not just a hard/rare skill nobody in this pool has -- computed per job profile,
    not against a fixed global list, so it adapts to whatever that role's requirements are."""
    by_requirement: dict[str, list[float]] = {}
    for assessment in assessments.values():
        for req, score in assessment["requirement_scores"].items():
            by_requirement.setdefault(req, []).append(score)
    medians = {req: statistics.median(scores) for req, scores in by_requirement.items()}
    return {req: m for req, m in medians.items() if m >= min_pool_median}


def _flag_candidate(
    requirement_scores: dict[str, float], informative: dict[str, float], near_zero_threshold: float, flag_fraction: float
) -> tuple[bool, float, list[str]]:
    relevant = {req: requirement_scores[req] for req in informative if req in requirement_scores}
    if not relevant:
        return False, 0.0, []
    near_zero_reqs = [req for req, score in relevant.items() if score <= near_zero_threshold]
    frac_near_zero = len(near_zero_reqs) / len(relevant)
    return frac_near_zero >= flag_fraction, frac_near_zero, near_zero_reqs


def audit_job_profile(
    jd_id: str,
    assessments: dict[str, dict],
    min_pool_median: float,
    near_zero_threshold: float,
    flag_fraction: float,
    profile_flag_fraction: float,
) -> dict:
    informative = _informative_requirements(assessments, min_pool_median)

    flagged_candidates = []
    for candidate_id, assessment in assessments.items():
        is_flagged, frac_near_zero, near_zero_reqs = _flag_candidate(
            assessment["requirement_scores"], informative, near_zero_threshold, flag_fraction
        )
        if is_flagged:
            flagged_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "composite_fit_score": assessment["composite_fit_score"],
                    "frac_near_zero_informative": frac_near_zero,
                    "n_informative_requirements": len(informative),
                    "near_zero_requirements": near_zero_reqs,
                }
            )
    flagged_candidates.sort(key=lambda c: c["composite_fit_score"])

    n_total = len(assessments)
    n_flagged = len(flagged_candidates)
    flagged_pool_fraction = n_flagged / n_total if n_total else 0.0

    return {
        "jd_id": jd_id,
        "n_candidates": n_total,
        "n_informative_requirements": len(informative),
        "informative_requirements": informative,
        "n_flagged": n_flagged,
        "flagged_pool_fraction": flagged_pool_fraction,
        "profile_flagged": bool(informative) and flagged_pool_fraction >= profile_flag_fraction,
        "flagged_candidates": flagged_candidates,
    }


def render_markdown(report: dict) -> str:
    lines = [f"# Shortlisting Quality Audit -- Run {report['run_id']}", ""]
    lines.append(
        f"For each job profile, a requirement counts as \"informative\" only if the shortlisted pool's own "
        f"median score on it is at least {report['min_pool_median']:.0f} -- i.e.\\ most peers in this same "
        f"pool actually have it, so scoring near-zero on it is a real outlier rather than a hard/rare skill "
        f"nobody has. A candidate is flagged when at least {report['flag_fraction']:.0%} of their informative "
        f"requirements are at or below {report['near_zero_threshold']:.0f} (0-100 scale). A job profile is "
        f"flagged when at least {report['profile_flag_fraction']:.0%} of its shortlisted pool is flagged."
    )
    lines.append("")
    for jd in sorted(report["per_job_profile"], key=lambda j: -j["flagged_pool_fraction"]):
        marker = " -- POSSIBLE SHORTLISTING DEFECT" if jd["profile_flagged"] else ""
        lines.append(f"## {jd['jd_id']}{marker}")
        lines.append(
            f"- {jd['n_informative_requirements']} informative requirement(s) out of this profile's full list; "
            f"{jd['n_flagged']}/{jd['n_candidates']} shortlisted candidates flagged ({jd['flagged_pool_fraction']:.0%})"
        )
        for c in jd["flagged_candidates"][:10]:
            reqs = ", ".join(c["near_zero_requirements"][:6])
            more = f" (+{len(c['near_zero_requirements']) - 6} more)" if len(c["near_zero_requirements"]) > 6 else ""
            lines.append(
                f"  - {c['candidate_id']}: composite_fit_score={c['composite_fit_score']:.1f}, near-zero on {reqs}{more}"
            )
        if len(jd["flagged_candidates"]) > 10:
            lines.append(f"  - ... and {len(jd['flagged_candidates']) - 10} more")
        lines.append("")
    return "\n".join(lines)


def main(
    run_id: str, min_pool_median: float, near_zero_threshold: float, flag_fraction: float, profile_flag_fraction: float
) -> None:
    cfg = apply_env_overrides(RunConfig.full(PROJECT_ROOT))
    run_dir = cfg.runs_dir / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    jd_ids = manifest["jd_ids"]

    per_job_profile = []
    for jd_id in jd_ids:
        path = run_dir / jd_id / "assessments.json"
        if not path.exists():
            continue
        assessments = json.loads(path.read_text(encoding="utf-8"))
        per_job_profile.append(
            audit_job_profile(jd_id, assessments, min_pool_median, near_zero_threshold, flag_fraction, profile_flag_fraction)
        )

    report = {
        "run_id": run_id,
        "min_pool_median": min_pool_median,
        "near_zero_threshold": near_zero_threshold,
        "flag_fraction": flag_fraction,
        "profile_flag_fraction": profile_flag_fraction,
        "per_job_profile": per_job_profile,
    }

    out_dir = run_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "shortlisting_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = render_markdown(report)
    (out_dir / "shortlisting_audit.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\nWrote {out_dir / 'shortlisting_audit.json'} and {out_dir / 'shortlisting_audit.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--min-pool-median", type=float, default=30.0,
        help="A requirement is only 'informative' if the shortlisted pool's own median score on it is at "
        "least this (0-100 scale) -- excludes skills that are just hard/rare for everyone (default: 30).",
    )
    parser.add_argument(
        "--near-zero-threshold", type=float, default=15.0,
        help="A requirement score at or below this (0-100 scale) counts as near-zero (default: 15).",
    )
    parser.add_argument(
        "--flag-fraction", type=float, default=0.7,
        help="Flag a candidate if at least this fraction of their informative requirements are near-zero (default: 0.7).",
    )
    parser.add_argument(
        "--profile-flag-fraction", type=float, default=0.2,
        help="Flag a whole job profile if at least this fraction of its shortlisted pool is flagged (default: 0.2).",
    )
    args = parser.parse_args()
    main(args.run_id, args.min_pool_median, args.near_zero_threshold, args.flag_fraction, args.profile_flag_fraction)
