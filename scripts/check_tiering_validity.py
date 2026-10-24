"""Tiering-validity check (ground-truth-free): does Jev's holistic pairwise
judgment agree with the transparent criterion_score on pairs CLEARLY
separated by score -- i.e. pairs the tiered ranking design would never send
to Jev at all, since they're outside any tie cluster?

This is a concurrent-validity check: it does not ask whether either signal
is "correct", only whether the two independently-computed signals agree
where the design assumes they would. High agreement supports skipping
Jev's pairwise call for clearly-separated candidates; frequent disagreement
would mean that skip is discarding real signal.

Usage:
    .venv/bin/python scripts/check_tiering_validity.py [--sample-pairs 8] [--gap-multiplier 2.0]

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

from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.models import Assessment, JobProfile
from candidate_ranking.scoring.active_pairwise_ranking import JEV_PROBABILITY_NOISE_FLOOR, _rank_pair_jev
from candidate_ranking.scoring.criterion_score import criterion_score
from candidate_ranking.scoring.jev_client import JevClient


def _load_cached_job_profiles(cache_path: Path) -> dict[str, JobProfile]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return {jp_id: JobProfile.model_validate(entry["job_profile"]) for jp_id, entry in data.items()}


def _load_assessments_for_jp(cache_dir: Path, jp_id: str) -> dict[str, Assessment]:
    out: dict[str, Assessment] = {}
    for path in (cache_dir / "assessments" / jp_id).glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[path.stem] = Assessment.model_validate(data["assessment"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-pairs", type=int, default=8, help="clearly-separated pairs per job profile")
    parser.add_argument("--gap-multiplier", type=float, default=2.0,
                         help="require a score gap of at least this many times tie_epsilon")
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

    job_profiles = _load_cached_job_profiles(cfg.cache_dir / "job_profile.json")
    jev_client = JevClient(api_token=cfg.jev_api_key)
    rng = random.Random(args.seed)

    total = 0
    agreed = 0

    for jp_dir in sorted(assessments_dir.iterdir()):
        if not jp_dir.is_dir():
            continue
        jp_id = jp_dir.name
        if args.jp_id is not None and jp_id != args.jp_id:
            continue
        jp = job_profiles.get(jp_id)
        if jp is None:
            continue
        assessments = _load_assessments_for_jp(cfg.cache_dir, jp_id)
        if len(assessments) < 2:
            continue

        scores = {cid: criterion_score(a) for cid, a in assessments.items()}
        tie_epsilon = max(1.0 / max(len(jp.evaluation_criteria), 1), JEV_PROBABILITY_NOISE_FLOOR)
        gap_threshold = args.gap_multiplier * tie_epsilon

        ordered = sorted(scores, key=scores.__getitem__, reverse=True)
        clear_pairs = [
            (ordered[i], ordered[j])
            for i in range(len(ordered))
            for j in range(i + 1, len(ordered))
            if scores[ordered[i]] - scores[ordered[j]] >= gap_threshold
        ]
        if not clear_pairs:
            continue
        rng.shuffle(clear_pairs)
        sampled = clear_pairs[: args.sample_pairs]

        jp_agreed = 0
        jp_total = 0
        for higher_id, lower_id in sampled:
            result = _rank_pair_jev(jp, [higher_id, lower_id], assessments, jev_client)
            if result is None:
                continue
            jp_total += 1
            total += 1
            winner_id = result[0]
            if winner_id == higher_id:
                agreed += 1
                jp_agreed += 1
            else:
                print(f"  DISAGREE  {jp_id}: {higher_id} (score={scores[higher_id]:.3f}) vs "
                      f"{lower_id} (score={scores[lower_id]:.3f}) -> Jev picked {winner_id}")
        if jp_total:
            print(f"{jp_id:35s} clear pairs tested={jp_total}  agreed={jp_agreed}")

    print()
    if total == 0:
        print("No clearly-separated pairs could be tested.")
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(
                {"check": "tiering_validity", "jp_id": args.jp_id, "total": 0, "agreed": 0, "agreement_rate": None}
            ), encoding="utf-8")
        return
    print(f"Overall: {agreed}/{total} clearly-separated pairs agreed with criterion_score "
          f"({agreed / total:.1%}).")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "check": "tiering_validity", "jp_id": args.jp_id,
            "total": total, "agreed": agreed, "agreement_rate": agreed / total,
        }), encoding="utf-8")


if __name__ == "__main__":
    main()
