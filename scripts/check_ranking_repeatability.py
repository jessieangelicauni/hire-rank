"""Full-ranking repeat-to-repeat stability check (ground-truth-free).

Runs the active pairwise ranking tournament for a small candidate subset of
each job profile `--repeats` times, independently (fresh Jev calls each
time -- there is no cache for pairwise ranking decisions), and reports the
mean Kendall's tau between every pair of the resulting rankings. This
measures whether the SAME inputs produce the SAME final order across
independent runs of the whole ranking process -- a reliability metric, not
a validity one: a high tau says the ranking is reproducible, not that it is
correct. Cost is self-limiting: a sampled subset with no genuine tie
cluster costs zero Jev calls, since the active tournament only runs inside
a tie cluster.

Usage:
    .venv/bin/python scripts/check_ranking_repeatability.py [--sample-size 8] [--repeats 3] [--max-jps 3]

Requires a prior pipeline run so `runs/_cache/assessments/<jp_id>/*.json` exists.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
from scipy.stats import kendalltau

from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.models import Assessment, JobProfile
from candidate_ranking.scoring.active_pairwise_ranking import RankingDiagnostics, rank_active_pairwise
from candidate_ranking.scoring.jev_client import JevClient


def _load_assessments_for_jp(cache_dir: Path, jp_id: str) -> dict[str, Assessment]:
    out: dict[str, Assessment] = {}
    for path in (cache_dir / "assessments" / jp_id).glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[path.stem] = Assessment.model_validate(data["assessment"])
    return out


def _load_cached_job_profile(cache_path: Path, jp_id: str) -> JobProfile:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return JobProfile.model_validate(data[jp_id]["job_profile"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=8, help="candidates per job profile")
    parser.add_argument("--repeats", type=int, default=3, help="independent full ranking runs per job profile")
    parser.add_argument("--ensemble-repeats", type=int, default=1, help="Jev calls per pairwise comparison")
    parser.add_argument("--max-jps", type=int, default=None, help="limit how many job profiles to test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jp-id", default=None, help="limit to a single job profile id")
    parser.add_argument("--json-out", default=None, help="write structured result to this JSON path")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    cfg = apply_env_overrides(RunConfig.full(PROJECT_ROOT))
    if not cfg.jev_api_key:
        print("CANDIDATE_RANKING_JEV_API_KEY is required.", file=sys.stderr)
        sys.exit(1)

    assessments_dir = cfg.cache_dir / "assessments"
    if not assessments_dir.exists():
        print("No cached assessments found — run the pipeline first.", file=sys.stderr)
        sys.exit(1)

    jev_client = JevClient(api_token=cfg.jev_api_key)
    rng = random.Random(args.seed)

    jp_dirs = sorted(p for p in assessments_dir.iterdir() if p.is_dir())
    if args.jp_id is not None:
        jp_dirs = [p for p in jp_dirs if p.name == args.jp_id]
    if args.max_jps is not None:
        jp_dirs = jp_dirs[: args.max_jps]

    all_taus: list[float] = []
    total_candidates = 0
    total_candidates_in_clusters = 0
    total_tie_clusters = 0
    total_jev_calls = 0

    for jp_dir in jp_dirs:
        jp_id = jp_dir.name
        full_assessments = _load_assessments_for_jp(cfg.cache_dir, jp_id)
        candidate_ids = list(full_assessments)
        if len(candidate_ids) < 3:
            continue

        sample_ids = rng.sample(candidate_ids, min(args.sample_size, len(candidate_ids)))
        assessments = {cid: full_assessments[cid] for cid in sample_ids}
        jp = _load_cached_job_profile(cfg.cache_dir / "job_profile.json", jp_id)

        rankings: list[list[str]] = []
        candidates_in_clusters = 0
        tie_cluster_count = 0
        jev_calls = 0
        for _ in range(args.repeats):
            diagnostics = RankingDiagnostics()
            utilities = rank_active_pairwise(
                jp, assessments, jev_client=jev_client, subset_size=2,
                ensemble_repeats=args.ensemble_repeats, diagnostics=diagnostics,
            )
            rankings.append(sorted(utilities, key=utilities.__getitem__, reverse=True))
            candidates_in_clusters = max(candidates_in_clusters, sum(diagnostics.cluster_sizes))
            tie_cluster_count = max(tie_cluster_count, diagnostics.tie_clusters)
            jev_calls += diagnostics.position_a_wins + diagnostics.position_b_wins

        rank_of = [{cid: i for i, cid in enumerate(r)} for r in rankings]
        taus = []
        for i in range(len(rankings)):
            for j in range(i + 1, len(rankings)):
                a = [rank_of[i][cid] for cid in sample_ids]
                b = [rank_of[j][cid] for cid in sample_ids]
                tau, _ = kendalltau(a, b)
                taus.append(tau)

        total_candidates += len(sample_ids)
        total_candidates_in_clusters += candidates_in_clusters
        total_tie_clusters += tie_cluster_count
        total_jev_calls += jev_calls

        if taus:
            mean_tau = sum(taus) / len(taus)
            all_taus.extend(taus)
            print(
                f"{jp_id:35s} n={len(sample_ids)}  mean tau={mean_tau:.3f}  (repeat-pairs={len(taus)})  "
                f"live-tested={candidates_in_clusters}/{len(sample_ids)} in {tie_cluster_count} cluster(s), "
                f"{jev_calls} Jev call(s)"
            )

    print()
    if not all_taus:
        print("No job profile had enough cached candidates to compare.")
        if args.json_out:
            Path(args.json_out).write_text(json.dumps({
                "check": "ranking_repeatability", "jp_id": args.jp_id, "n_repeat_pairs": 0, "mean_kendall_tau": None,
                "n_candidates": total_candidates, "n_candidates_in_tie_clusters": total_candidates_in_clusters,
                "n_tie_clusters": total_tie_clusters, "n_jev_calls": total_jev_calls,
            }), encoding="utf-8")
        return
    mean_tau = sum(all_taus) / len(all_taus)
    print(f"Overall mean Kendall's tau across {len(all_taus)} repeat-pair(s): {mean_tau:.3f}")
    print(
        f"Candidates actually re-tested live via Jev (inside a tie cluster): "
        f"{total_candidates_in_clusters}/{total_candidates}, across {total_tie_clusters} cluster(s), "
        f"{total_jev_calls} Jev call(s) total."
    )
    if total_candidates_in_clusters == 0:
        print(
            "WARNING: zero candidates landed in a tie cluster -- this tau is trivially perfect "
            "because no live comparison was ever made, not because the ranking was proven stable."
        )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "check": "ranking_repeatability", "jp_id": args.jp_id,
            "n_repeat_pairs": len(all_taus), "mean_kendall_tau": mean_tau,
            "n_candidates": total_candidates, "n_candidates_in_tie_clusters": total_candidates_in_clusters,
            "n_tie_clusters": total_tie_clusters, "n_jev_calls": total_jev_calls,
        }), encoding="utf-8")


if __name__ == "__main__":
    main()
