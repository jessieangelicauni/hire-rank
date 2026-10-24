"""Run the shortlist gate over the full corpus without touching Jev.

Loads all cached job-profile rubrics (`runs/_cache/job_profile.json`, local
Ollama output, no Jev involved) and all cached candidates
(`runs/_cache/cv.json`), then checks `passes_shortlist_gate` for every
(job profile, candidate) pair. Pure local computation (sentence-transformers
embeddings + regex) -- zero Jev API calls, safe to run against the full
301-candidate x 10-job-profile corpus to see real pass/fail numbers before
deciding whether to spend any Jev budget at all.

Usage:
    .venv/bin/python scripts/run_shortlist_only.py [--minimum 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.ingestion.cv import load_candidates
from candidate_ranking.models import JobProfile
from candidate_ranking.scoring.assessment import filter_assessable_candidates
from candidate_ranking.scoring.shortlist import matching_skill_count, passes_shortlist_gate


def _load_cached_job_profiles(cache_path: Path) -> list[JobProfile]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return [JobProfile.model_validate(entry["job_profile"]) for entry in data.values()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum", type=int, default=5, help="minimum matched skills to pass the gate")
    args = parser.parse_args()

    cfg = apply_env_overrides(RunConfig.full(PROJECT_ROOT))

    jps = _load_cached_job_profiles(cfg.cache_dir / "job_profile.json")
    candidates = filter_assessable_candidates(load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json"))

    print(f"{len(jps)} job profile(s), {len(candidates)} assessable candidate(s), minimum={args.minimum}")
    print()

    total_passed = 0
    total_checked = 0
    for jp in sorted(jps, key=lambda j: j.id):
        counts = [matching_skill_count(c, jp) for c in candidates]
        passed = sum(1 for c, n in zip(candidates, counts) if passes_shortlist_gate(c, jp, args.minimum))
        total_passed += passed
        total_checked += len(candidates)
        avg_match = sum(counts) / len(counts) if counts else 0.0
        print(
            f"{jp.id:35s} skills={len(jp.technical_skills):2d}  "
            f"passed={passed:3d}/{len(candidates)}  avg_matched_skills={avg_match:.1f}"
        )

    print()
    print(f"Total: {total_passed}/{total_checked} (jp, candidate) pairs would reach Jev "
          f"({total_passed / total_checked:.1%}) -- the rest are gated out before any Jev call.")


if __name__ == "__main__":
    main()
