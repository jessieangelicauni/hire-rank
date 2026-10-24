"""Run all 4 ground-truth-free reliability/validity checks for one job
profile CONCURRENTLY (each as its own subprocess, not sequentially), and
merge their results into a single JSON report.

Fixes the earlier workflow of running check_confidence_calibration.py,
check_ranking_consistency.py, check_ranking_repeatability.py, and
check_tiering_validity.py one after another -- they are fully independent
(separate HTTP calls to Jev, no shared state), so there is no reason to pay
their wall-clock cost serially.

Usage:
    .venv/bin/python scripts/run_reliability_suite.py --jp-id data-analyst [--full]

    --full sizes every check to cover its ENTIRE eligible population for
    that job profile (all C(n,2) pairs for consistency, all clear pairs for
    tiering-validity, all recorded triples for calibration, all candidates
    for repeatability) instead of the scripts' small default samples. Only
    sensible for small job profiles -- it does not time-cap itself, so a
    large role (e.g. 80+ candidates) can take a long time under --full even
    though the dollar cost stays low (Jev is priced per token, not per
    call). Omit --full to use each script's own sampling flags, which you
    can still override individually (--consistency-pairs, etc).

Writes runs/<run-dir>/reliability/<jp-id>.json (default run-dir: the most
recently modified directory under runs/ that isn't a cache/reliability dir).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from candidate_ranking.config import RunConfig, apply_env_overrides  # noqa: E402
from candidate_ranking.models import Assessment, JobProfile  # noqa: E402
from candidate_ranking.scoring.active_pairwise_ranking import (  # noqa: E402
    JEV_PROBABILITY_NOISE_FLOOR,
    _tie_clusters,
)
from candidate_ranking.scoring.criterion_score import criterion_score  # noqa: E402

PYTHON = sys.executable
_CHECKS = {
    "tiering_validity": "check_tiering_validity.py",
    "confidence_calibration": "check_confidence_calibration.py",
    "ranking_consistency": "check_ranking_consistency.py",
    "ranking_repeatability": "check_ranking_repeatability.py",
}


def _load_assessments_for_jp(cache_dir: Path, jp_id: str) -> dict[str, Assessment]:
    out: dict[str, Assessment] = {}
    for path in (cache_dir / "assessments" / jp_id).glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[path.stem] = Assessment.model_validate(data["assessment"])
    return out


def _full_sizes(cfg: RunConfig, jp_id: str) -> dict[str, int]:
    """Compute --full coverage sizes locally (no Jev calls needed): all of
    this is deterministic from already-cached assessment data."""
    jp_cache = json.loads((cfg.cache_dir / "job_profile.json").read_text(encoding="utf-8"))
    jp = JobProfile.model_validate(jp_cache[jp_id]["job_profile"])
    assessments = _load_assessments_for_jp(cfg.cache_dir, jp_id)
    n = len(assessments)
    n_crit = max(len(jp.evaluation_criteria), 1)
    tie_epsilon = max(1.0 / n_crit, JEV_PROBABILITY_NOISE_FLOOR)

    scores = {cid: criterion_score(a) for cid, a in assessments.items()}
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)
    gap_threshold = 2.0 * tie_epsilon
    clear_pairs = sum(
        1
        for i in range(len(ordered))
        for j in range(i + 1, len(ordered))
        if scores[ordered[i]] - scores[ordered[j]] >= gap_threshold
    )

    total_triples = sum(
        1 for a in assessments.values() for i, ev in enumerate(a.evaluations) if ev.probabilities and i < n_crit
    )

    return {
        "consistency_pairs": n * (n - 1) // 2,
        "tiering_pairs": clear_pairs,
        "calibration_size": total_triples,
        "repeatability_size": n,
    }


def _default_run_dir(cfg: RunConfig) -> Path:
    candidates = [
        p for p in cfg.runs_dir.iterdir() if p.is_dir() and p.name not in ("_cache",)
    ]
    if not candidates:
        raise SystemExit(f"No run directories found under {cfg.runs_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jp-id", required=True, help="job profile id to evaluate")
    parser.add_argument("--full", action="store_true", help="size every check to its full eligible population")
    parser.add_argument("--consistency-pairs", type=int, default=8)
    parser.add_argument("--tiering-pairs", type=int, default=8)
    parser.add_argument("--calibration-size", type=int, default=24)
    parser.add_argument("--repeatability-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ensemble-repeats", type=int, default=1, help="Jev calls per pairwise comparison")
    parser.add_argument("--run-dir", default=None, help="defaults to the most recently modified runs/<id> dir")
    args = parser.parse_args()

    cfg = apply_env_overrides(RunConfig.full(PROJECT_ROOT))

    if args.full:
        sizes = _full_sizes(cfg, args.jp_id)
        consistency_pairs = sizes["consistency_pairs"]
        tiering_pairs = sizes["tiering_pairs"]
        calibration_size = sizes["calibration_size"]
        repeatability_size = sizes["repeatability_size"]
        print(f"--full: {sizes}")
    else:
        consistency_pairs = args.consistency_pairs
        tiering_pairs = args.tiering_pairs
        calibration_size = args.calibration_size
        repeatability_size = args.repeatability_size

    scripts_dir = Path(__file__).resolve().parent
    commands: dict[str, list[str]] = {
        "tiering_validity": [
            PYTHON, str(scripts_dir / "check_tiering_validity.py"),
            "--jp-id", args.jp_id, "--sample-pairs", str(tiering_pairs),
        ],
        "confidence_calibration": [
            PYTHON, str(scripts_dir / "check_confidence_calibration.py"),
            "--jp-id", args.jp_id, "--sample-size", str(calibration_size), "--repeats", str(args.repeats),
        ],
        "ranking_consistency": [
            PYTHON, str(scripts_dir / "check_ranking_consistency.py"),
            "--jp-id", args.jp_id, "--sample-pairs", str(consistency_pairs), "--repeats", str(args.repeats),
        ],
        "ranking_repeatability": [
            PYTHON, str(scripts_dir / "check_ranking_repeatability.py"),
            "--jp-id", args.jp_id, "--sample-size", str(repeatability_size), "--repeats", str(args.repeats),
            "--ensemble-repeats", str(args.ensemble_repeats),
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        json_paths = {name: Path(tmpdir) / f"{name}.json" for name in commands}
        for name, cmd in commands.items():
            cmd.extend(["--json-out", str(json_paths[name])])

        def _run(name: str) -> tuple[str, subprocess.CompletedProcess]:
            return name, subprocess.run(commands[name], cwd=PROJECT_ROOT, capture_output=True, text=True)

        print(f"Running 4 reliability checks concurrently for {args.jp_id}...")
        results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            for name, proc in executor.map(_run, commands):
                if proc.returncode != 0:
                    print(f"  {name}: FAILED (exit {proc.returncode})\n{proc.stderr}", file=sys.stderr)
                    continue
                if json_paths[name].exists():
                    results[name] = json.loads(json_paths[name].read_text(encoding="utf-8"))
                    print(f"  {name}: done")
                else:
                    print(f"  {name}: no JSON output produced", file=sys.stderr)

    run_dir = Path(args.run_dir) if args.run_dir else _default_run_dir(cfg)
    out_dir = run_dir / "reliability"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.jp_id}.json"
    out_path.write_text(json.dumps({"jp_id": args.jp_id, "checks": results}, indent=2), encoding="utf-8")
    print(f"\nWrote combined reliability report to {out_path}")


if __name__ == "__main__":
    main()
