"""Confidence vs. self-consistency calibration check (ground-truth-free).

For a sample of already-assessed (job profile, candidate, criterion)
triples, repeats Jev's per-criterion evidence-strength question `--repeats`
times (fresh calls, same single criterion each time) and measures the
majority-agreement rate: the fraction of repeats matching the most common
answer. Buckets triples by the ORIGINAL assessment's reported confidence
and reports mean agreement per bucket -- if confidence is a meaningful
signal, higher-confidence buckets should show higher agreement, without
ever needing to know which answer was actually "correct". This measures
reliability (is the confidence signal internally coherent), not validity
(is Jev actually right) -- the latter needs ground truth this check does
not use.

Usage:
    .venv/bin/python scripts/check_confidence_calibration.py [--sample-size 24] [--repeats 3]

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
from candidate_ranking.ingestion.cv import load_candidates
from candidate_ranking.models import Assessment, JobProfile
from candidate_ranking.scoring.jev_client import JevClient, JevClientError, JevQuestion

_BUCKET_ORDER = {"<0.5": 0, "0.5-0.7": 1, "0.7-0.9": 2, "0.9-1.0": 3}


def _load_cached_job_profiles(cache_path: Path) -> dict[str, JobProfile]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return {jp_id: JobProfile.model_validate(entry["job_profile"]) for jp_id, entry in data.items()}


def _load_assessments_for_jp(cache_dir: Path, jp_id: str) -> dict[str, Assessment]:
    out: dict[str, Assessment] = {}
    for path in (cache_dir / "assessments" / jp_id).glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[path.stem] = Assessment.model_validate(data["assessment"])
    return out


def _confidence_bucket(confidence: float) -> str:
    if confidence >= 0.9:
        return "0.9-1.0"
    if confidence >= 0.7:
        return "0.7-0.9"
    if confidence >= 0.5:
        return "0.5-0.7"
    return "<0.5"


def _majority_agreement(labels: list[str]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    return counts.most_common(1)[0][1] / len(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=24, help="(jp, candidate, criterion) triples to test")
    parser.add_argument("--repeats", type=int, default=3, help="independent Jev calls per triple")
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
    candidates_by_id = {c.id: c for c in load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")}
    jev_client = JevClient(api_token=cfg.jev_api_key)
    rng = random.Random(args.seed)

    pool: list[tuple[str, str, int, float]] = []
    for jp_dir in sorted(assessments_dir.iterdir()):
        if not jp_dir.is_dir():
            continue
        jp_id = jp_dir.name
        if args.jp_id is not None and jp_id != args.jp_id:
            continue
        jp = job_profiles.get(jp_id)
        if jp is None:
            continue
        for cid, assessment in _load_assessments_for_jp(cfg.cache_dir, jp_id).items():
            if cid not in candidates_by_id:
                continue
            for i, ev in enumerate(assessment.evaluations):
                if ev.probabilities and i < len(jp.evaluation_criteria):
                    pool.append((jp_id, cid, i, max(ev.probabilities.values())))

    if not pool:
        print("No evaluations with recorded probabilities found.", file=sys.stderr)
        sys.exit(1)

    rng.shuffle(pool)
    sampled = pool[: args.sample_size]

    buckets: dict[str, list[float]] = {}
    for jp_id, cid, crit_idx, original_confidence in sampled:
        jp = job_profiles[jp_id]
        candidate = candidates_by_id[cid]
        criterion = jp.evaluation_criteria[crit_idx]
        question = JevQuestion(
            key="evidence_strength", kind="choice", instructions=criterion.instructions,
            criteria={"strong": criterion.strong, "partial": criterion.partial, "none": criterion.none},
        )
        labels: list[str] = []
        for _ in range(args.repeats):
            try:
                answers = jev_client.evaluate(candidate.raw_text, [question])
                labels.append(answers[0].value)
            except JevClientError as exc:
                print(f"  call failed for {jp_id}/{cid}/crit{crit_idx}: {exc}", file=sys.stderr)
        if len(labels) < 2:
            continue
        agreement = _majority_agreement(labels)
        bucket = _confidence_bucket(original_confidence)
        buckets.setdefault(bucket, []).append(agreement)
        print(f"  {jp_id}/{cid} crit={crit_idx} orig_conf={original_confidence:.2f} "
              f"labels={labels} agreement={agreement:.2f}")

    print()
    print("Mean majority-agreement rate by original-confidence bucket:")
    if not buckets:
        print("  (no triples could be tested)")
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(
                {"check": "confidence_calibration", "jp_id": args.jp_id, "buckets": {}}
            ), encoding="utf-8")
        return
    bucket_results = {}
    for bucket in sorted(buckets, key=lambda b: _BUCKET_ORDER[b]):
        vals = buckets[bucket]
        mean = sum(vals) / len(vals)
        bucket_results[bucket] = {"mean": mean, "n": len(vals)}
        print(f"  {bucket}: mean={mean:.3f}  n={len(vals)}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"check": "confidence_calibration", "jp_id": args.jp_id, "buckets": bucket_results}
        ), encoding="utf-8")


if __name__ == "__main__":
    main()
