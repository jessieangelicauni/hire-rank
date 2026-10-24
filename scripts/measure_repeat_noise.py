"""Repeat-call probability-noise measurement for one (job, candidate,
criterion) triple -- the exact measurement behind Table "Five repeated Jev
calls" in docs/main.tex. Earlier runs of this measurement were done ad hoc
and reported to 2 decimal places, which left it undetectable whether a
reported 0.00 was a true zero or just a small value rounded down. This
script prints and saves FULL, unrounded precision specifically so that is
visible, and is re-runnable against any (job, candidate, criterion) triple
instead of a one-off, undocumented measurement.

The original table used n=1 (one candidate, one hand-picked "ambiguous"
criterion) -- thin evidence for a noise floor applied across the whole
pipeline. --all-candidates extends this to every cached, assessed candidate
for a job profile: for each, it locally (free, no Jev call) picks that
candidate's OWN most-ambiguous criterion -- the one whose originally
recorded top-label probability was closest to a flat 1/3 split, i.e. least
confident -- then repeat-tests that single (candidate, criterion) pair fresh
against the live API, same design as the original single-example
measurement, just across the whole job profile instead of one anecdote.

Usage:
    # Single, hand-picked triple (original table's design):
    .venv/bin/python scripts/measure_repeat_noise.py --jp-id data-scientist-engineer \
        --criterion-label "Python or Java Programming" --repeats 5

    # Every candidate in one job profile, each on their own most-ambiguous
    # criterion, aggregated:
    .venv/bin/python scripts/measure_repeat_noise.py --jp-id data-analyst \
        --all-candidates --repeats 5 --json-out /tmp/noise_data_analyst.json

Requires a prior pipeline run so `runs/_cache/assessments/<jp_id>/*.json` and
`runs/_cache/job_profile.json` exist (for the criterion's instructions/rubric).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv

from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.ingestion.cv import load_candidates
from candidate_ranking.models import JobProfile
from candidate_ranking.scoring.jev_client import JevClient, JevClientError, JevQuestion


def _load_cached_job_profile(cache_path: Path, jp_id: str) -> JobProfile:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return JobProfile.model_validate(data[jp_id]["job_profile"])


def _pick_candidate_id(cache_dir: Path, jp_id: str, criterion_label: str) -> str | None:
    """First cached assessment for this JP containing a matching criterion label."""
    jp_dir = cache_dir / "assessments" / jp_id
    if not jp_dir.exists():
        return None
    target = criterion_label.strip().lower()
    for path in sorted(jp_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for ev in data["assessment"]["evaluations"]:
            if ev["requirement"].strip().lower() == target:
                return path.stem
    return None


def _most_ambiguous_criterion_per_candidate(cache_dir: Path, jp_id: str) -> dict[str, str]:
    """Local, zero-cost: for every cached, assessed candidate in this job
    profile, the requirement label whose recorded top-label probability was
    lowest (closest to a flat 3-way split -- the least-confident, most
    "ambiguous" evaluation Jev gave that candidate), mirroring how the
    original single-example table picked its one "ambiguous case"."""
    jp_dir = cache_dir / "assessments" / jp_id
    picks: dict[str, str] = {}
    for path in sorted(jp_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        best_label, best_top_prob = None, 2.0
        for ev in data["assessment"]["evaluations"]:
            probs = ev.get("probabilities")
            if not probs:
                continue
            top_prob = max(probs.values())
            if top_prob < best_top_prob:
                best_top_prob, best_label = top_prob, ev["requirement"]
        if best_label is not None:
            picks[path.stem] = best_label
    return picks


def _repeat_test(
    jev_client: JevClient, candidate, criterion, repeats: int,
) -> list[dict]:
    question = JevQuestion(
        key="evidence_strength", kind="choice", instructions=criterion.instructions,
        criteria={"strong": criterion.strong, "partial": criterion.partial, "none": criterion.none},
    )
    calls: list[dict] = []
    for i in range(1, repeats + 1):
        try:
            answers = jev_client.evaluate(candidate.raw_text, [question])
        except JevClientError as exc:
            print(f"    call {i} failed: {exc}", file=sys.stderr)
            continue
        answer = answers[0]
        probs = answer.probabilities or {}
        calls.append({"call": i, "label": answer.value, "confidence": answer.confidence, "probabilities": probs})
    return calls


def _print_calls(label: str, calls: list[dict]) -> None:
    for c in calls:
        probs = c["probabilities"]
        strong, partial, none_ = probs.get("strong", 0.0), probs.get("partial", 0.0), probs.get("none", 0.0)
        print(
            f"    call {c['call']}: label={c['label']:7s} "
            f"P(strong)={strong!r:>10} P(partial)={partial!r:>10} P(none)={none_!r:>10}"
        )
    if len(calls) >= 2:
        for level in ("strong", "partial", "none"):
            values = [c["probabilities"].get(level, 0.0) for c in calls]
            spread = max(values) - min(values)
            print(f"    P({level}) spread: {spread!r} (min={min(values)!r}, max={max(values)!r})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jp-id", required=True, help="job profile id")
    parser.add_argument("--criterion-label", default=None, help="exact criterion label to repeat-test")
    parser.add_argument("--cv-id", default=None, help="candidate id; auto-picked from cache if omitted")
    parser.add_argument("--repeats", type=int, default=5, help="independent fresh Jev calls per triple")
    parser.add_argument(
        "--all-candidates", action="store_true",
        help="repeat-test every cached candidate's own most-ambiguous criterion, instead of one hand-picked triple",
    )
    parser.add_argument("--json-out", default=None, help="write full raw results to this JSON path")
    args = parser.parse_args()

    if not args.all_candidates and not args.criterion_label:
        print("--criterion-label is required unless --all-candidates is set.", file=sys.stderr)
        sys.exit(1)

    load_dotenv(PROJECT_ROOT / ".env")
    cfg = apply_env_overrides(RunConfig.full(PROJECT_ROOT))
    if not cfg.jev_api_key:
        print("CANDIDATE_RANKING_JEV_API_KEY is required.", file=sys.stderr)
        sys.exit(1)

    jp = _load_cached_job_profile(cfg.cache_dir / "job_profile.json", args.jp_id)
    criteria_by_label = {c.label.strip().lower(): c for c in jp.evaluation_criteria}
    candidates_by_id = {c.id: c for c in load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")}
    jev_client = JevClient(api_token=cfg.jev_api_key)

    if args.all_candidates:
        picks = _most_ambiguous_criterion_per_candidate(cfg.cache_dir, args.jp_id)
        if not picks:
            print(f"No cached, assessed candidates found for {args.jp_id!r}.", file=sys.stderr)
            sys.exit(1)
        print(f"Repeat-testing {len(picks)} candidate(s) in {args.jp_id!r}, each on their own most-ambiguous "
              f"criterion, {args.repeats}x each, fresh Jev calls, full precision:\n")

        all_results: list[dict] = []
        all_spreads = {"strong": [], "partial": [], "none": []}
        for cv_id, requirement in picks.items():
            candidate = candidates_by_id.get(cv_id)
            criterion = criteria_by_label.get(requirement.strip().lower())
            if candidate is None or criterion is None:
                continue
            print(f"  {cv_id} / {requirement!r}:")
            calls = _repeat_test(jev_client, candidate, criterion, args.repeats)
            _print_calls(requirement, calls)
            all_results.append({"cv_id": cv_id, "criterion_label": requirement, "calls": calls})
            if len(calls) >= 2:
                for level in ("strong", "partial", "none"):
                    values = [c["probabilities"].get(level, 0.0) for c in calls]
                    all_spreads[level].append(max(values) - min(values))
            print()

        print("Aggregate spread across all candidates' most-ambiguous criterion:")
        for level, spreads in all_spreads.items():
            if spreads:
                print(
                    f"  P({level}): mean={sum(spreads)/len(spreads):.4f}  "
                    f"max={max(spreads):.4f}  n={len(spreads)}"
                )

        if args.json_out:
            Path(args.json_out).write_text(json.dumps({
                "jp_id": args.jp_id, "mode": "all_candidates", "results": all_results,
            }, indent=2), encoding="utf-8")
            print(f"\nWrote full-precision results to {args.json_out}")
        return

    criterion = criteria_by_label.get(args.criterion_label.strip().lower())
    if criterion is None:
        print(f"No criterion labeled {args.criterion_label!r} in job profile {args.jp_id!r}.", file=sys.stderr)
        sys.exit(1)

    cv_id = args.cv_id or _pick_candidate_id(cfg.cache_dir, args.jp_id, args.criterion_label)
    if cv_id is None:
        print(f"No cached, assessed candidate found for criterion {args.criterion_label!r}.", file=sys.stderr)
        sys.exit(1)

    candidate = candidates_by_id.get(cv_id)
    if candidate is None:
        print(f"Candidate {cv_id!r} not found in CV cache.", file=sys.stderr)
        sys.exit(1)

    print(f"Repeating {args.jp_id}/{cv_id}/{criterion.label!r} {args.repeats}x, fresh Jev calls, full precision:\n")
    calls = _repeat_test(jev_client, candidate, criterion, args.repeats)
    _print_calls(criterion.label, calls)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "jp_id": args.jp_id, "cv_id": cv_id, "criterion_label": criterion.label, "calls": calls,
        }, indent=2), encoding="utf-8")
        print(f"\nWrote full-precision results to {args.json_out}")


if __name__ == "__main__":
    main()
