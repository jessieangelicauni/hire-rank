from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
from scipy.stats import kendalltau

from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.scoring.jev_client import JevClient

from run_jev_evaluation_study import collect_test_retest, load_corpus, load_run_pairs

load_dotenv()


def _trial_scores_by_jd(records: list[dict]) -> dict[str, dict[str, float]]:
    by_jd: dict[str, dict[str, float]] = {}
    for record in records:
        scores = [rep["composite_fit_score"] for rep in record["repeats"]]
        by_jd.setdefault(record["jd_id"], {})[record["candidate_id"]] = statistics.mean(scores)
    return by_jd


def compute_averaged_ranking_convergence(trials: list[dict[str, dict[str, float]]]) -> dict:
    jd_ids = set(trials[0])
    for t in trials[1:]:
        jd_ids &= set(t)

    per_job_profile: dict[str, dict] = {}
    all_mean_taus: list[float] = []

    for jd_id in jd_ids:
        candidate_ids = sorted(trials[0][jd_id])
        if len(candidate_ids) < 2:
            per_job_profile[jd_id] = {"n_candidates": len(candidate_ids), "mean_kendall_tau": None}
            continue

        rankings = [
            sorted(candidate_ids, key=lambda cid: t[jd_id][cid], reverse=True) for t in trials
        ]

        pair_taus = []
        for i in range(len(rankings)):
            for j in range(i + 1, len(rankings)):
                position_i = {cid: idx for idx, cid in enumerate(rankings[i])}
                position_j = {cid: idx for idx, cid in enumerate(rankings[j])}
                tau, _ = kendalltau(
                    [position_i[cid] for cid in candidate_ids],
                    [position_j[cid] for cid in candidate_ids],
                )
                if tau == tau:  # exclude NaN
                    pair_taus.append(float(tau))

        mean_tau = statistics.mean(pair_taus) if pair_taus else None
        per_job_profile[jd_id] = {"n_candidates": len(candidate_ids), "mean_kendall_tau": mean_tau}
        if mean_tau is not None:
            all_mean_taus.append(mean_tau)

    return {
        "per_job_profile": per_job_profile,
        "mean_kendall_tau_across_all_profiles": statistics.mean(all_mean_taus) if all_mean_taus else None,
    }


def main(run_id: str, max_workers: int) -> None:
    cfg = apply_env_overrides(RunConfig.full(PROJECT_ROOT))
    run_dir = cfg.runs_dir / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    jd_ids = manifest["jd_ids"]

    eval_dir = run_dir / "evaluation"
    trial_a_records = json.loads((eval_dir / "test_retest.json").read_text(encoding="utf-8"))
    trial_a = _trial_scores_by_jd(trial_a_records)
    print(f"Trial A: reused {len(trial_a_records)} pairs from existing test_retest.json")

    if not cfg.jev_api_key:
        raise RuntimeError("Set CANDIDATE_RANKING_JEV_API_KEY (e.g. in .env) before running.")
    jev_client = JevClient(api_token=cfg.jev_api_key)

    pairs = load_run_pairs(run_dir, jd_ids)
    jds_by_id, candidates_by_id, jd_skills_by_jd = load_corpus(cfg, jd_ids)

    trials = [trial_a]
    for trial_name in ("B", "C"):
        print(f"Collecting trial {trial_name}: {len(pairs)} pairs x 3 fresh repeats...")
        records = collect_test_retest(
            jev_client, jds_by_id, candidates_by_id, jd_skills_by_jd, pairs, repeats=3, max_workers=max_workers
        )
        (eval_dir / f"test_retest_trial_{trial_name}.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        trials.append(_trial_scores_by_jd(records))

    result = compute_averaged_ranking_convergence(trials)
    (eval_dir / "multi_call_averaging_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print()
    print(f"Averaged-score (n_calls=3) ranking convergence, mean Kendall-tau across "
          f"{len(result['per_job_profile'])} job profiles: {result['mean_kendall_tau_across_all_profiles']:.3f}")
    for jd_id, jd_result in sorted(result["per_job_profile"].items(), key=lambda kv: (kv[1]["mean_kendall_tau"] is None, kv[1]["mean_kendall_tau"] or 0)):
        tau_str = f"{jd_result['mean_kendall_tau']:.3f}" if jd_result["mean_kendall_tau"] is not None else "N/A"
        print(f"  {jd_id} (n={jd_result['n_candidates']}): tau={tau_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    main(args.run_id, args.max_workers)
