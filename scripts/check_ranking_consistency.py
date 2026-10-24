"""Pairwise-ranking consistency and cosmetic-invariance spot-check.

Samples a handful of already-assessed candidate pairs per job profile and
runs two fairness checks, fresh each time (no cache):

1. Consistency (test-retest): ask Jev to pick the stronger fit `--repeats`
   times with the evidence unchanged. If the winner flips across repeats,
   that pair's ranking signal is noise, not a real preference.
2. Cosmetic invariance: same question, but with each candidate's evidence
   lines reordered (content identical, presentation order shuffled) and
   re-asked `--repeats` times. If the majority winner flips versus the
   original order, the ranking is sensitive to formatting/order luck rather
   than substance -- a fairness bug, not just a reliability one, since two
   candidates with identical qualifications could rank differently purely
   because of how their evidence happened to be listed.

Usage:
    .venv/bin/python scripts/check_ranking_consistency.py [--sample-pairs 8] [--repeats 3]

Requires a prior pipeline run so `runs/_cache/assessments/<jp_id>/*.json` exists.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv

from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.models import Assessment, JobProfile
from candidate_ranking.scoring.active_pairwise_ranking import _rank_pair_jev
from candidate_ranking.scoring.jev_client import JevClient


def _shuffle_evaluations(assessment: Assessment, rng: random.Random) -> Assessment:
    """Same evidence, different presentation order -- a pure formatting
    change with zero information change."""
    shuffled = list(assessment.evaluations)
    rng.shuffle(shuffled)
    return assessment.model_copy(update={"evaluations": shuffled})


def _majority_winner(winners: list[str]) -> str | None:
    if not winners:
        return None
    return Counter(winners).most_common(1)[0][0]


def _load_assessments_for_jp(cache_dir: Path, jp_id: str) -> dict[str, Assessment]:
    out: dict[str, Assessment] = {}
    for path in (cache_dir / "assessments" / jp_id).glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[path.stem] = Assessment.model_validate(data["assessment"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-pairs", type=int, default=8, help="pairs to check per job profile")
    parser.add_argument("--repeats", type=int, default=3, help="independent Jev calls per pair")
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

    total_pairs = 0
    flipped_pairs = 0
    cosmetic_total = 0
    cosmetic_flipped = 0

    for jp_dir in sorted(assessments_dir.iterdir()):
        if not jp_dir.is_dir():
            continue
        jp_id = jp_dir.name
        if args.jp_id is not None and jp_id != args.jp_id:
            continue
        assessments = _load_assessments_for_jp(cfg.cache_dir, jp_id)
        candidate_ids = list(assessments)
        if len(candidate_ids) < 2:
            continue

        all_pairs = [
            (a, b) for i, a in enumerate(candidate_ids) for b in candidate_ids[i + 1:]
        ]
        rng.shuffle(all_pairs)
        sampled = all_pairs[: args.sample_pairs]
        jp = JobProfile(id=jp_id, raw_text="")

        for a_id, b_id in sampled:
            winners = []
            for _ in range(args.repeats):
                result = _rank_pair_jev(jp, [a_id, b_id], assessments, jev_client)
                if result is not None:
                    winners.append(result[0])
            if len(winners) < 2:
                continue
            total_pairs += 1
            if len(set(winners)) > 1:
                flipped_pairs += 1
                print(f"  INCONSISTENT  {jp_id}: {a_id} vs {b_id} -> winners={winners}")

            original_winner = _majority_winner(winners)

            cosmetic_winners = []
            for _ in range(args.repeats):
                shuffled_assessments = dict(assessments)
                shuffled_assessments[a_id] = _shuffle_evaluations(assessments[a_id], rng)
                shuffled_assessments[b_id] = _shuffle_evaluations(assessments[b_id], rng)
                result = _rank_pair_jev(jp, [a_id, b_id], shuffled_assessments, jev_client)
                if result is not None:
                    cosmetic_winners.append(result[0])
            if len(cosmetic_winners) < 2:
                continue
            cosmetic_total += 1
            cosmetic_winner = _majority_winner(cosmetic_winners)
            if cosmetic_winner != original_winner:
                cosmetic_flipped += 1
                print(
                    f"  INCONSISTENT-COSMETIC  {jp_id}: {a_id} vs {b_id} -> "
                    f"original={original_winner} ({winners}), "
                    f"reordered={cosmetic_winner} ({cosmetic_winners})"
                )

    print()
    result = {"check": "ranking_consistency", "jp_id": args.jp_id}
    if total_pairs == 0:
        print("No pairs could be evaluated.")
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(result), encoding="utf-8")
        return

    rate = flipped_pairs / total_pairs
    print(f"Checked {total_pairs} pair(s) across all job profiles, {args.repeats}x each.")
    print(f"Inconsistent (winner changed across repeats): {flipped_pairs} ({rate:.1%})")
    result.update({"total_pairs": total_pairs, "flipped_pairs": flipped_pairs, "inconsistent_rate": rate})

    if cosmetic_total == 0:
        print("No pairs could be evaluated for cosmetic invariance.")
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(result), encoding="utf-8")
        return

    cosmetic_rate = cosmetic_flipped / cosmetic_total
    print()
    print(f"Checked {cosmetic_total} pair(s) for cosmetic invariance (evidence order shuffled).")
    print(f"Inconsistent (winner changed after reordering, by majority vote): "
          f"{cosmetic_flipped} ({cosmetic_rate:.1%})")
    result.update({
        "cosmetic_total": cosmetic_total, "cosmetic_flipped": cosmetic_flipped,
        "cosmetic_inconsistent_rate": cosmetic_rate,
    })
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
