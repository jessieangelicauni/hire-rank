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
from candidate_ranking.ingestion.jd import load_job_descriptions
from candidate_ranking.logging_setup import configure_logging
from candidate_ranking.models import Assessment, Candidate, JobDescription
from candidate_ranking.output.console_export import export_console_web_data, seed_console_web_roles
from candidate_ranking.output.writer import write_run_output
from candidate_ranking.scoring.job_profile import build_job_profile_chain
from candidate_ranking.scoring.llm_assessment import (
    build_qualification_chain,
    filter_assessable_candidates,
)
from candidate_ranking.evaluation.ranking_stability import (
    load_cached_assessments,
    load_job_profiles,
    run_stability_report,
)
from candidate_ranking.evaluation.qualification_reliability import (
    repeat_qualification_gate,
    build_reliability_report,
)
from candidate_ranking.scoring.skills import build_candidate_skill_index, build_skill_embedder

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ASSESSMENT_SCOPE_VERSION = "llm-qualification-gate-v1"


def load_corpus(cfg: RunConfig) -> tuple[list[JobDescription], list[Candidate]]:
    jds = load_job_descriptions(cfg.jd_dir)
    candidates = load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")
    if cfg.max_jds is not None:
        jds = jds[: cfg.max_jds]
    if cfg.max_candidates is not None:
        candidates = candidates[: cfg.max_candidates]
    return jds, candidates


_CHECKPOINT_ALLOWED_MSGPACK_MODULES = [
    ("candidate_ranking.models", name)
    for name in ("JobDescription", "Candidate", "JobProfile", "Assessment")
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

    jds, candidates = load_corpus(cfg)
    try:
        seed_console_web_roles(cfg)
    except Exception as exc:
        print(f"WARNING: console-web role seeding failed: {exc}", file=sys.stderr)
    candidates = filter_assessable_candidates(candidates)

    llm = ChatOllama(
        model=cfg.ollama_model, base_url=cfg.ollama_base_url, temperature=0, num_ctx=cfg.ollama_num_ctx
    )
    llm_ranking = ChatOllama(
        model=cfg.ollama_model, base_url=cfg.ollama_base_url,
        temperature=cfg.ranking_temperature, num_ctx=cfg.ollama_num_ctx,
    )

    job_profile_chain = build_job_profile_chain(llm)
    qualification_chain = build_qualification_chain(llm)
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
        qualification_chain=qualification_chain,
        ranking_llm=llm_ranking,
        skill_index=skill_index,
        skill_row_map=skill_row_map,
        skill_embedder=skill_embedder,
        run_id=run_id,
        skill_match_threshold=skill_match_threshold,
        min_skill_matches=min_skill_matches,
    )

    db_path = cfg.cache_dir / "checkpoints.db"
    initial_state = {
        "jds": jds,
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

    for jd in jds:
        if len(final_state["shortlists"].get(jd.id, [])) == 0:
            print(
                f"WARNING: JD {jd.id}'s shortlist is empty — no candidates assessed.",
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
        "ranking_temperature": cfg.ranking_temperature,
        "shortlist_sizes": {jd_id: len(cids) for jd_id, cids in final_state["shortlists"].items()},
        "jd_ids": [jd.id for jd in jds],
        "candidate_ids": [c.id for c in candidates],
    }
    run_dir = write_run_output(cfg.runs_dir, run_id, manifest, final_state["assessment_results"])

    total_shortlisted = sum(len(v) for v in final_state["shortlists"].values())
    if total_shortlisted == 0:
        raise RuntimeError(
            f"Run produced zero shortlisted candidates across {len(jds)} JDs — no assessments generated."
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

    subparsers.add_parser("evaluate-stability")

    reliability_parser = subparsers.add_parser("evaluate-reliability")
    reliability_parser.add_argument("--sample-per-jd", type=int, default=2)
    reliability_parser.add_argument("--repeats", type=int, default=3)

    args = parser.parse_args()

    if args.command == "run":
        run_dir = run(
            run_id=args.run_id,
            skill_match_threshold=args.skill_match_threshold,
            min_skill_matches=args.min_skill_matches,
        )
        print(f"Run complete: {run_dir / 'summary.json'}")

    elif args.command == "evaluate-stability":
        import json as _json
        cfg = RunConfig.full(PROJECT_ROOT)
        cfg = apply_env_overrides(cfg)

        llm_ranking = ChatOllama(
            model=cfg.ollama_model, base_url=cfg.ollama_base_url,
            temperature=cfg.ranking_temperature, num_ctx=cfg.ollama_num_ctx,
        )

        jds = load_job_descriptions(cfg.jd_dir)
        jd_map = {jd.id: jd for jd in jds}
        cached = load_cached_assessments(cfg.cache_dir)

        jd_assessments = {
            jd_id: (jd_map[jd_id], assessments)
            for jd_id, assessments in cached.items()
            if jd_id in jd_map
        }
        if not jd_assessments:
            print("No cached assessments found. Run the pipeline first.", file=sys.stderr)
            sys.exit(1)

        report = run_stability_report(
            jd_assessments, llm_ranking,
            cfg.ranking_subset_size, cfg.ranking_max_iterations,
            cfg.ranking_min_delta_u, cfg.ranking_pl_prior_variance,
        )

        sep = "─" * 70
        print(f"\nRanking Stability Report")
        print(f"subset_size={cfg.ranking_subset_size}  max_iterations={cfg.ranking_max_iterations}")
        print(sep)
        print(f"{'JD':<35} {'Candidates':>10}  {'τ':>6}  {'RBO':>6}")
        for jd_id, tau in sorted(report.per_jd.items()):
            n = len(cached[jd_id])
            print(f"{jd_id:<35} {n:>10}  {tau:>6.3f}  {report.rbo_per_jd[jd_id]:>6.3f}")
        print(sep)
        print(f"Mean τ   : {report.mean_tau:.3f}      Mean RBO : {report.mean_rbo:.3f}")
        print(f"Min  τ   : {report.min_tau:.3f}  ({report.min_tau_jd})")
        print(f"Min  RBO : {report.min_rbo:.3f}  ({report.min_rbo_jd})")
        print(sep)
        print("τ (Kendall-tau): agreement across ALL rank pairs, equally weighted.")
        print("RBO (rank-biased overlap): agreement weighted toward the TOP of the list —")
        print("  what matters for shortlisting. τ ≥ 0.90 / RBO ≥ 0.90 = stable,")
        print("  0.75–0.90 = moderate, < 0.75 = unstable.\n")

        report_path = cfg.cache_dir / "stability_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            _json.dumps({
                "subset_size": cfg.ranking_subset_size, "max_iterations": cfg.ranking_max_iterations,
                "per_jd": report.per_jd, "mean_tau": report.mean_tau,
                "min_tau": report.min_tau, "min_tau_jd": report.min_tau_jd,
                "rbo_per_jd": report.rbo_per_jd, "mean_rbo": report.mean_rbo,
                "min_rbo": report.min_rbo, "min_rbo_jd": report.min_rbo_jd,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"Saved to {report_path}")

    elif args.command == "evaluate-reliability":
        import json as _json
        import random

        cfg = RunConfig.full(PROJECT_ROOT)
        cfg = apply_env_overrides(cfg)

        llm = ChatOllama(
            model=cfg.ollama_model, base_url=cfg.ollama_base_url, temperature=0, num_ctx=cfg.ollama_num_ctx
        )
        qualification_chain = build_qualification_chain(llm)

        jds = load_job_descriptions(cfg.jd_dir)
        jd_map = {jd.id: jd for jd in jds}
        job_profiles = load_job_profiles(cfg.cache_dir)
        candidates = load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")
        candidate_map = {c.id: c for c in candidates}

        cached = load_cached_assessments(cfg.cache_dir, job_profiles)
        if not cached:
            print("No cached assessments found. Run the pipeline first.", file=sys.stderr)
            sys.exit(1)

        rng = random.Random(42)
        sampled_pairs: list[tuple[str, str]] = []
        for jd_id, assessments in cached.items():
            if jd_id not in jd_map:
                continue
            candidate_ids = [cid for cid in assessments if cid in candidate_map]
            rng.shuffle(candidate_ids)
            for cid in candidate_ids[: args.sample_per_jd]:
                sampled_pairs.append((jd_id, cid))

        if not sampled_pairs:
            print("No (JD, candidate) pairs available to sample.", file=sys.stderr)
            sys.exit(1)

        print(
            f"Repeating the qualification gate {args.repeats}x (no cache) for "
            f"{len(sampled_pairs)} sampled pairs — this makes real LLM calls...\n"
        )
        repeated_assessments: dict[str, list[Assessment]] = {}
        for jd_id, candidate_id in sampled_pairs:
            jd = jd_map[jd_id]
            profile = job_profiles.get(jd_id)
            candidate = candidate_map[candidate_id]
            pair_id = f"{jd_id}/{candidate_id}"
            try:
                repeated_assessments[pair_id] = repeat_qualification_gate(
                    jd, candidate, profile, qualification_chain, cfg.ollama_model, repeats=args.repeats
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


if __name__ == "__main__":
    main()
