# src/candidate_ranking/cli.py
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.graphs.pipeline import build_pipeline_graph
from candidate_ranking.ingestion.cv import build_skill_extraction_chain, enrich_candidates_with_skills, load_candidates
from candidate_ranking.ingestion.jp import load_job_profiles
from candidate_ranking.logging_setup import configure_logging
from candidate_ranking.models import Assessment, Candidate, JobProfile
from candidate_ranking.output.console_export import export_console_web_data, seed_console_web_roles
from candidate_ranking.output.writer import write_run_output
from candidate_ranking.scoring.job_profile import build_job_profile_chain
from candidate_ranking.scoring.assessment import (
    filter_assessable_candidates,
)
from candidate_ranking.scoring.jev_client import JevClient
from candidate_ranking.evaluation.ranking_stability import (
    load_cached_assessments,
    load_job_profiles,
)
from candidate_ranking.evaluation.qualification_reliability import (
    repeat_qualification_gate,
    build_reliability_report,
)
from candidate_ranking.scoring.skills import build_candidate_skill_index, build_skill_embedder
from candidate_ranking.utils import OllamaLatencyCallback

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ASSESSMENT_SCOPE_VERSION = "jev-qualification-gate-v1"


def load_corpus(cfg: RunConfig) -> tuple[list[JobProfile], list[Candidate]]:
    jps = load_job_profiles(cfg.jp_dir)
    candidates = load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")
    if cfg.max_jps is not None:
        jps = jps[: cfg.max_jps]
    if cfg.max_candidates is not None:
        candidates = candidates[: cfg.max_candidates]
    return jps, candidates


_CHECKPOINT_ALLOWED_MSGPACK_MODULES = [
    ("candidate_ranking.models", name)
    for name in ("JobProfile", "Candidate", "Assessment")
]


@contextmanager
def sqlite_checkpointer(db_path: Path) -> Iterator[SqliteSaver]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    serde = JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_ALLOWED_MSGPACK_MODULES)
    with closing(sqlite3.connect(str(db_path), check_same_thread=False)) as conn:
        yield SqliteSaver(conn, serde=serde)


def run(
    run_id: str | None = None,
    skill_match_threshold: float = 0.8,
    min_skill_matches: int = 3,
) -> Path:
    cfg = RunConfig.full(PROJECT_ROOT)
    cfg = apply_env_overrides(cfg)

    run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    print(f"Run ID: {run_id}")
    configure_logging(cfg.runs_dir / run_id / "warnings.json")

    jps, candidates = load_corpus(cfg)
    try:
        seed_console_web_roles(cfg)
    except Exception as exc:
        print(f"WARNING: console-web role seeding failed: {exc}", file=sys.stderr)
    candidates = filter_assessable_candidates(candidates)

    llm = ChatOllama(
        model=cfg.ollama_model, base_url=cfg.ollama_base_url, temperature=0, num_ctx=cfg.ollama_num_ctx,
        callbacks=[OllamaLatencyCallback()],
    )

    if not cfg.jev_api_key:
        raise RuntimeError(
            "CANDIDATE_RANKING_JEV_API_KEY is required — set it in .env or the environment"
        )
    job_profile_chain = build_job_profile_chain(llm)
    jev_client = JevClient(api_token=cfg.jev_api_key)
    skill_extraction_chain = build_skill_extraction_chain(llm)
    candidates = enrich_candidates_with_skills(
        candidates, skill_extraction_chain, cfg.ollama_model,
        cfg.cache_dir / "cv_skills.json", max_workers=cfg.ollama_num_parallel,
    )
    skill_embedder = build_skill_embedder(cfg.skill_embedding_model)
    skill_index, skill_row_map = build_candidate_skill_index(candidates, skill_embedder)

    graph = build_pipeline_graph(
        cfg=cfg,
        job_profile_chain=job_profile_chain,
        jev_client=jev_client,
        skill_index=skill_index,
        skill_row_map=skill_row_map,
        skill_embedder=skill_embedder,
        run_id=run_id,
        skill_match_threshold=skill_match_threshold,
        min_skill_matches=min_skill_matches,
    )

    db_path = cfg.cache_dir / "checkpoints.db"
    initial_state = {
        "jps": jps,
        "candidates": candidates,
        "job_profiles": {},
        "shortlists": {},
        "assessment_results": [],
    }

    with sqlite_checkpointer(db_path) as saver:
        compiled = graph.compile(checkpointer=saver)
        thread_config = {"configurable": {"thread_id": run_id}, "max_concurrency": cfg.ollama_num_parallel}
        resuming = saver.get_tuple(thread_config) is not None
        final_state = compiled.invoke(None if resuming else initial_state, config=thread_config)

    for jp in jps:
        if len(final_state["shortlists"].get(jp.id, [])) == 0:
            print(
                f"WARNING: JP {jp.id}'s shortlist is empty — no candidates assessed.",
                file=sys.stderr,
            )

    manifest_path = cfg.runs_dir / run_id / "manifest.json"
    if resuming and manifest_path.exists():
        try:
            prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            skill_match_threshold = prior_manifest.get("skill_match_threshold", skill_match_threshold)
            min_skill_matches = prior_manifest.get("min_skill_matches", min_skill_matches)
        except (json.JSONDecodeError, OSError):
            pass

    manifest = {
        "run_id": run_id,
        "preset": cfg.preset,
        "ollama_model": cfg.ollama_model,
        "assessment_scope": ASSESSMENT_SCOPE_VERSION,
        "skill_embedding_model": cfg.skill_embedding_model,
        "skill_match_threshold": skill_match_threshold,
        "min_skill_matches": min_skill_matches,
        "shortlist_sizes": {jp_id: len(cids) for jp_id, cids in final_state["shortlists"].items()},
        "jp_ids": [jp.id for jp in jps],
        "candidate_ids": [c.id for c in candidates],
    }
    run_dir = write_run_output(cfg.runs_dir, run_id, manifest, final_state["assessment_results"])

    total_shortlisted = sum(len(v) for v in final_state["shortlists"].values())
    if total_shortlisted == 0:
        raise RuntimeError(
            f"Run produced zero shortlisted candidates across {len(jps)} JDs — no assessments generated."
        )

    try:
        exported_path = export_console_web_data(cfg, run_id)
        if exported_path:
            print(f"Exported console-web data to {exported_path}")
    except Exception as exc:
        print(f"WARNING: console-web export failed: {exc}", file=sys.stderr)

    return run_dir


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="candidate_ranking.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--run-id", default=None)
    run_parser.add_argument("--skill-match-threshold", type=float, default=0.8)
    run_parser.add_argument("--min-skill-matches", type=int, default=3)

    subparsers.add_parser("evaluate-audit")

    reliability_parser = subparsers.add_parser("evaluate-reliability")
    reliability_parser.add_argument("--sample-per-jp", type=int, default=2)
    reliability_parser.add_argument("--repeats", type=int, default=3)

    args = parser.parse_args()

    if args.command == "run":
        run_dir = run(
            run_id=args.run_id,
            skill_match_threshold=args.skill_match_threshold,
            min_skill_matches=args.min_skill_matches,
        )
        print(f"Run complete: {run_dir / 'summary.json'}")

    elif args.command == "evaluate-reliability":
        import json as _json
        import random

        cfg = RunConfig.full(PROJECT_ROOT)
        cfg = apply_env_overrides(cfg)

        if not cfg.jev_api_key:
            raise RuntimeError("CANDIDATE_RANKING_JEV_API_KEY is required for reliability evaluation")
        jev_client = JevClient(api_token=cfg.jev_api_key)

        jps = load_job_profiles(cfg.jp_dir)
        jp_map = {jp.id: jp for jp in jps}
        job_profiles = load_job_profiles(cfg.cache_dir)
        candidates = load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")
        candidate_map = {c.id: c for c in candidates}

        cached = load_cached_assessments(cfg.cache_dir, job_profiles)
        if not cached:
            print("No cached assessments found. Run the pipeline first.", file=sys.stderr)
            sys.exit(1)

        rng = random.Random(42)
        sampled_pairs: list[tuple[str, str]] = []
        for jp_id, assessments in cached.items():
            if jp_id not in jp_map:
                continue
            candidate_ids = [cid for cid in assessments if cid in candidate_map]
            rng.shuffle(candidate_ids)
            for cid in candidate_ids[: args.sample_per_jd]:
                sampled_pairs.append((jp_id, cid))

        if not sampled_pairs:
            print("No (JD, candidate) pairs available to sample.", file=sys.stderr)
            sys.exit(1)

        print(
            f"Repeating the qualification gate {args.repeats}x (no cache) for "
            f"{len(sampled_pairs)} sampled pairs — this makes real LLM calls...\n"
        )
        repeated_assessments: dict[str, list[Assessment]] = {}
        for jp_id, candidate_id in sampled_pairs:
            jp = jp_map[jp_id]
            profile = job_profiles.get(jp_id)
            candidate = candidate_map[candidate_id]
            pair_id = f"{jp_id}/{candidate_id}"
            try:
                repeated_assessments[pair_id] = repeat_qualification_gate(
                    jp, candidate, jev_client, repeats=args.repeats
                )
            except Exception as exc:
                print(f"WARNING: skipping {pair_id}: {exc}", file=sys.stderr)

        report = build_reliability_report(repeated_assessments)

        sep = "─" * 70
        print(f"\nQualification Gate Reliability Report (no ground truth — self-consistency only)")
        print(sep)
        print(f"Pairs sampled     : {report.n_pairs}")
        print(f"Repeats per pair  : {report.n_repeats}")
        print(f"qualified agreement rate : {report.qualified_agreement_rate:.3f}")
        print(f"qualified Cohen's kappa  : {report.qualified_cohens_kappa:.3f}")
        print(f"per-requirement verdict agreement : {report.verdict_agreement_rate:.3f}")
        print(sep)
        print("Interpretation (kappa): > 0.80 = near-perfect, 0.60–0.80 = substantial,")
        print("  0.40–0.60 = moderate, < 0.40 = low self-consistency — the verdict is")
        print("  largely noise, independent of whether it happens to be correct.\n")

        report_path = cfg.cache_dir / "reliability_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            _json.dumps({
                "n_pairs": report.n_pairs, "n_repeats": report.n_repeats,
                "qualified_agreement_rate": report.qualified_agreement_rate,
                "qualified_cohens_kappa": report.qualified_cohens_kappa,
                "verdict_agreement_rate": report.verdict_agreement_rate,
                "per_pair_qualified": report.per_pair_qualified,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"Saved to {report_path}")

    elif args.command == "evaluate-audit":
        import json as _json
        from candidate_ranking.evaluation.assessment_audit import audit_assessments
        cfg = RunConfig.full(PROJECT_ROOT)
        cfg = apply_env_overrides(cfg)

        cached = load_cached_assessments(cfg.cache_dir)
        if not cached:
            print("No cached assessments found. Run the pipeline first.", file=sys.stderr)
            sys.exit(1)

        report = audit_assessments(cached)
        sep = "─" * 70
        print(f"\nAssessment Audit Report (zero LLM calls)")
        print(sep)
        print(f"JPs assessed          : {report.n_jps}")
        print(f"Candidates assessed   : {report.n_assessments}")
        print(f"\n── Evidence Coverage (BM25) ──")
        print(f"Must-have evaluations : {report.must_have_evals_total}")
        print(f"With non-empty evidence: {report.must_have_evals_with_evidence}")
        print(f"Coverage rate         : {report.evidence_coverage_rate:.1%}")
        print(f"\n── Jev Assessment Distribution ──")
        for strength in ("strong", "adequate", "weak", "none"):
            frac = report.evidence_strength_distribution.get(strength, 0.0)
            print(f"  {strength:<10}: {frac:.1%}")
        print(f"\n── Internal Consistency ──")
        print(f"Coherence violations  : {len(report.coherence_violations)} ({report.coherence_violation_rate:.1%})")
        print(f"Qualified flag errors : {report.qualified_flag_errors}")
        if report.coherence_violations:
            for v in report.coherence_violations[:5]:
                print(f"  [{v.jp_id}/{v.candidate_id}] {v.requirement}: {v.issue}")
            if len(report.coherence_violations) > 5:
                print(f"  ... and {len(report.coherence_violations) - 5} more")
        print(f"\n── Qualification Funnel ──")
        print(f"Mean qualification rate: {report.mean_qualification_rate:.1%}")
        if report.zero_qualified_jps:
            print(f"WARNING — 0% qualified : {', '.join(report.zero_qualified_jps)}")
        if report.fully_qualified_jps:
            print(f"WARNING — 100% qualified: {', '.join(report.fully_qualified_jps)}")
        for jp_id, rate in sorted(report.qualification_rate_per_jp.items()):
            print(f"  {jp_id:<40}: {rate:.1%}")
        print(f"\n── Top Disqualifying Requirements ──")
        if report.most_common_disqualifiers:
            for req, count in report.most_common_disqualifiers[:10]:
                print(f"  {count:4d}x  {req}")
        else:
            print("  (none — all candidates qualified on all must-haves)")
        print(sep)

        report_path = cfg.cache_dir / "audit_report.json"
        report_path.write_text(
            _json.dumps({
                "n_jps": report.n_jps,
                "n_assessments": report.n_assessments,
                "evidence_coverage_rate": report.evidence_coverage_rate,
                "must_have_evals_total": report.must_have_evals_total,
                "must_have_evals_with_evidence": report.must_have_evals_with_evidence,
                "evidence_strength_distribution": report.evidence_strength_distribution,
                "coherence_violation_rate": report.coherence_violation_rate,
                "coherence_violations": [
                    {"jp_id": v.jp_id, "candidate_id": v.candidate_id,
                     "requirement": v.requirement, "issue": v.issue}
                    for v in report.coherence_violations
                ],
                "qualified_flag_errors": report.qualified_flag_errors,
                "mean_qualification_rate": report.mean_qualification_rate,
                "qualification_rate_per_jp": report.qualification_rate_per_jp,
                "fully_qualified_jps": report.fully_qualified_jps,
                "zero_qualified_jps": report.zero_qualified_jps,
                "most_common_disqualifiers": report.most_common_disqualifiers,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"Saved to {report_path}")


if __name__ == "__main__":
    main()
