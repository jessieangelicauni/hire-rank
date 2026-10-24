from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from contextlib import closing, contextmanager
import sqlite3

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.graphs.pipeline import build_pipeline_graph
from candidate_ranking.ingestion.cv import load_candidates
from candidate_ranking.ingestion.jp import load_job_profiles
from candidate_ranking.logging_setup import configure_logging
from candidate_ranking.models import Candidate, JobProfile
from candidate_ranking.output.console_export import export_console_web_data, seed_console_web_roles
from candidate_ranking.output.writer import write_run_output
from candidate_ranking.scoring.job_profile import build_rubric_chain
from candidate_ranking.scoring.assessment import filter_assessable_candidates
from candidate_ranking.scoring.cv_profile import build_cv_profile_chain, load_or_generate_cv_profile
from candidate_ranking.scoring.jev_client import JevClient
from candidate_ranking.utils import OllamaLatencyCallback

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_corpus(cfg: RunConfig) -> tuple[list[JobProfile], list[Candidate]]:
    jps = load_job_profiles(cfg.jp_dir)
    candidates = load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")
    if cfg.max_jps is not None:
        jps = jps[: cfg.max_jps]
    if cfg.max_candidates is not None:
        candidates = candidates[: cfg.max_candidates]
    return jps, candidates


def attach_cv_profiles(
    candidates: list[Candidate], chain, model_name: str, cache_path: Path, max_workers: int,
) -> list[Candidate]:
    def _attach(candidate: Candidate) -> Candidate:
        profile = load_or_generate_cv_profile(candidate, chain, model_name, cache_path)
        return candidate.model_copy(update={"profile": profile})

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_attach, candidates))


_CHECKPOINT_ALLOWED_MSGPACK_MODULES = [
    ("candidate_ranking.models", name)
    for name in ("JobProfile", "Candidate", "Assessment", "CandidateProfile", "SkillEntry")
]


@contextmanager
def sqlite_checkpointer(db_path: Path) -> Iterator[SqliteSaver]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    serde = JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_ALLOWED_MSGPACK_MODULES)
    with closing(sqlite3.connect(str(db_path), check_same_thread=False)) as conn:
        yield SqliteSaver(conn, serde=serde)


def run(run_id: str | None = None) -> Path:
    cfg = RunConfig.full(PROJECT_ROOT)
    cfg = apply_env_overrides(cfg)

    run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    print(f"Run ID: {run_id}")
    configure_logging()

    jps, candidates = load_corpus(cfg)
    try:
        seed_console_web_roles(cfg)
    except Exception as exc:
        print(f"WARNING: console-web role seeding failed: {exc}", file=sys.stderr)
    candidates = filter_assessable_candidates(candidates)

    llm = ChatOllama(
        model=cfg.ollama_model, base_url=cfg.ollama_base_url, temperature=0, num_ctx=cfg.ollama_num_ctx,
        reasoning=False, callbacks=[OllamaLatencyCallback()],
    )

    if not cfg.jev_api_key:
        raise RuntimeError(
            "CANDIDATE_RANKING_JEV_API_KEY is required — set it in .env or the environment"
        )
    rubric_chain = build_rubric_chain(llm)
    cv_profile_chain = build_cv_profile_chain(llm)
    jev_client = JevClient(api_token=cfg.jev_api_key)

    candidates = attach_cv_profiles(
        candidates, cv_profile_chain, cfg.ollama_model, cfg.cache_dir / "cv_profile.json", cfg.ollama_num_parallel,
    )

    graph = build_pipeline_graph(
        cfg=cfg,
        rubric_chain=rubric_chain,
        jev_client=jev_client,
        run_id=run_id,
    )

    db_path = cfg.cache_dir / "checkpoints.db"
    initial_state = {
        "job_postings": jps,
        "candidates": candidates,
        "job_profiles": {},
        "assessment_results": [],
    }

    with sqlite_checkpointer(db_path) as saver:
        compiled = graph.compile(checkpointer=saver)
        thread_config = {"configurable": {"thread_id": run_id}, "max_concurrency": cfg.ollama_num_parallel}
        resuming = saver.get_tuple(thread_config) is not None
        final_state = compiled.invoke(None if resuming else initial_state, config=thread_config)

    run_dir = write_run_output(cfg.runs_dir, run_id, final_state["assessment_results"])

    total_assessed = sum(1 for r in final_state["assessment_results"] if r["status"] == "ok")
    if total_assessed == 0:
        raise RuntimeError(
            f"Run produced zero assessments across {len(jps)} JDs and {len(candidates)} candidates."
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

    args = parser.parse_args()

    if args.command == "run":
        run_dir = run(run_id=args.run_id)
        print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
