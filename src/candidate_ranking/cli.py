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
from candidate_ranking.models import Assessment, Candidate, JDRequirements, JobDescription
from candidate_ranking.output.console_export import export_console_web_data, seed_console_web_roles
from candidate_ranking.output.run_output import write_run_output
from candidate_ranking.scoring.jd_requirements import build_jd_requirements_chain
from candidate_ranking.scoring.jd_skills import build_jd_skills_chain
from candidate_ranking.scoring.llm_assessment import (
    build_qualification_chain,
    filter_assessable_candidates,
)
from candidate_ranking.scoring.sliding_window_ranking import build_sliding_window_chain_factory
from candidate_ranking.evaluation.ranking_stability import (
    load_cached_assessments,
    run_stability_report,
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
    for name in ("JobDescription", "Candidate", "JDSkills", "Assessment", "JDRequirements")
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
    min_skill_matches: int = 5,
    min_must_have_matches: int = 2,
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

    jd_skills_chain = build_jd_skills_chain(llm)
    jd_requirements_chain = build_jd_requirements_chain(llm)
    qualification_chain = build_qualification_chain(llm)
    build_sliding_window_chain = build_sliding_window_chain_factory(llm_ranking)

    skill_extraction_chain = build_skill_extraction_chain(llm)
    candidates = enrich_candidates_with_skills(
        candidates, skill_extraction_chain, cfg.ollama_model,
        cfg.cache_dir / "cv_skills.json", max_workers=cfg.ollama_num_parallel,
    )
    skill_embedder = build_skill_embedder(cfg.skill_embedding_model)
    skill_index, skill_row_map = build_candidate_skill_index(candidates, skill_embedder)

    graph = build_pipeline_graph(
        cfg=cfg,
        jd_skills_chain=jd_skills_chain,
        jd_requirements_chain=jd_requirements_chain,
        qualification_chain=qualification_chain,
        build_sliding_window_chain=build_sliding_window_chain,
        skill_index=skill_index,
        skill_row_map=skill_row_map,
        skill_embedder=skill_embedder,
        run_id=run_id,
        skill_match_threshold=skill_match_threshold,
        min_skill_matches=min_skill_matches,
        min_must_have_matches=min_must_have_matches,
    )

    db_path = cfg.cache_dir / "checkpoints.db"
    initial_state = {
        "jds": jds,
        "candidates": candidates,
        "jd_skills": {},
        "jd_requirements": {},
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
            min_must_have_matches = prior_manifest.get("min_must_have_matches", min_must_have_matches)
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
        "min_must_have_matches": min_must_have_matches,
        "ranking_temperature": cfg.ranking_temperature,
        "ranking_pool_threshold": cfg.ranking_pool_threshold,
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
    run_parser.add_argument("--min-skill-matches", type=int, default=5)
    run_parser.add_argument("--min-must-have-matches", type=int, default=2)

    stability_parser = subparsers.add_parser("evaluate-stability")
    stability_parser.add_argument("--window-size", type=int, default=None)
    stability_parser.add_argument("--step", type=int, default=None)
    stability_parser.add_argument("--seeds", type=int, nargs=2, default=[42, 137])

    args = parser.parse_args()

    if args.command == "run":
        run_dir = run(
            run_id=args.run_id,
            skill_match_threshold=args.skill_match_threshold,
            min_skill_matches=args.min_skill_matches,
            min_must_have_matches=args.min_must_have_matches,
        )
        print(f"Run complete: {run_dir / 'summary.json'}")

    elif args.command == "evaluate-stability":
        import json as _json
        import sys as _sys
        cfg = RunConfig.full(PROJECT_ROOT)
        cfg = apply_env_overrides(cfg)

        window_size = args.window_size or cfg.sliding_window_size
        step = args.step or cfg.sliding_window_step
        seeds = tuple(args.seeds)

        llm_ranking = ChatOllama(
            model=cfg.ollama_model, base_url=cfg.ollama_base_url,
            temperature=cfg.ranking_temperature, num_ctx=cfg.ollama_num_ctx,
        )
        build_chain = build_sliding_window_chain_factory(llm_ranking)

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

        report = run_stability_report(jd_assessments, build_chain, window_size, step, seeds)

        sep = "─" * 56
        print(f"\nRanking Stability Report")
        print(f"window_size={window_size}  step={step}  seeds={list(seeds)}")
        print(sep)
        print(f"{'JD':<35} {'Candidates':>10}  {'τ':>6}")
        for jd_id, tau in sorted(report.per_jd.items()):
            n = len([a for a in cached[jd_id].values() if a.qualified])
            print(f"{jd_id:<35} {n:>10}  {tau:>6.3f}")
        print(sep)
        print(f"Mean τ : {report.mean_tau:.3f}")
        print(f"Min  τ : {report.min_tau:.3f}  ({report.min_tau_jd})")
        print(sep)
        print("Interpretation: τ ≥ 0.90 = stable, 0.75–0.90 = moderate, < 0.75 = unstable\n")

        report_path = cfg.cache_dir / "stability_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            _json.dumps({
                "window_size": window_size, "step": step, "seeds": list(seeds),
                "per_jd": report.per_jd, "mean_tau": report.mean_tau,
                "min_tau": report.min_tau, "min_tau_jd": report.min_tau_jd,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"Saved to {report_path}")


if __name__ == "__main__":
    main()
