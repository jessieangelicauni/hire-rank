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

from candidate_ranking.scoring.assessment import (
    ASSESSMENT_SCOPE_VERSION,
    build_assessment_chain,
    filter_assessable_candidates,
)
from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.output.console_export import export_console_web_data, seed_console_web_roles
from candidate_ranking.scoring.jd_skills import build_jd_skills_chain
from candidate_ranking.graphs.pipeline import build_pipeline_graph
from candidate_ranking.ingestion.cv import build_skill_extraction_chain, enrich_candidates_with_skills, load_candidates
from candidate_ranking.ingestion.jd import load_job_descriptions
from candidate_ranking.logging_setup import configure_logging
from candidate_ranking.models import Candidate, JobDescription
from candidate_ranking.output.run_output import write_run_output
from candidate_ranking.scoring.skills import build_candidate_skill_index, build_skill_embedder
from candidate_ranking.ranking.tournament import (
    RANKING_PROMPT_VERSION,
    build_listwise_ranking_chain,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    for name in (
        "JobDescription",
        "Candidate",
        "JDSkills",
        "Assessment",
        "TournamentIterationRecord",
        "TournamentResult",
        "ConvergencePoint",
        "JDEvaluation",
    )
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

    llm = ChatOllama(model=cfg.ollama_model, base_url=cfg.ollama_base_url, temperature=0, num_ctx=cfg.ollama_num_ctx)
    jd_skills_chain = build_jd_skills_chain(llm)
    assessment_chain = build_assessment_chain(llm)
    build_ranking_chain = build_listwise_ranking_chain(llm)
    skill_extraction_chain = build_skill_extraction_chain(llm)

    candidates = enrich_candidates_with_skills(
        candidates, skill_extraction_chain, cfg.ollama_model, cfg.cache_dir / "cv_skills.json",
        max_workers=cfg.ollama_num_parallel,
    )
    skill_embedder = build_skill_embedder(cfg.skill_embedding_model)
    skill_index, skill_row_map = build_candidate_skill_index(candidates, skill_embedder)

    graph = build_pipeline_graph(
        cfg,
        jd_skills_chain,
        assessment_chain,
        skill_index,
        skill_row_map,
        skill_embedder,
        build_ranking_chain,
        run_id,
        skill_match_threshold=skill_match_threshold,
        min_skill_matches=min_skill_matches,
    )

    db_path = cfg.cache_dir / "checkpoints.db"
    initial_state = {
        "jds": jds,
        "candidates": candidates,
        "jd_skills": {},
        "shortlists": {},
        "assessment_results": [],
        "tournament_results": [],
    }

    with sqlite_checkpointer(db_path) as saver:
        compiled = graph.compile(checkpointer=saver)
        thread_config = {"configurable": {"thread_id": run_id}, "max_concurrency": cfg.ollama_num_parallel}
        resuming = saver.get_tuple(thread_config) is not None
        final_state = compiled.invoke(None if resuming else initial_state, config=thread_config)

    for jd in jds:
        if len(final_state["shortlists"].get(jd.id, [])) == 0:
            print(
                f"WARNING: JD {jd.id}'s skill-match shortlist is empty (skill_match_threshold="
                f"{skill_match_threshold}, min_skill_matches={min_skill_matches}); no candidates will be "
                "assessed for this JD.",
                file=sys.stderr,
            )

    manifest_path = cfg.runs_dir / run_id / "manifest.json"
    if resuming and manifest_path.exists():
        try:
            prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            skill_match_threshold = prior_manifest.get("skill_match_threshold", skill_match_threshold)
            min_skill_matches = prior_manifest.get("min_skill_matches", min_skill_matches)
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"WARNING: could not read prior manifest at {manifest_path} to preserve original "
                f"skill-shortlist thresholds on resume; using the values passed to this invocation "
                f"instead: {exc}",
                file=sys.stderr,
            )

    manifest = {
        "run_id": run_id,
        "preset": cfg.preset,
        "ollama_model": cfg.ollama_model,
        "ranking_prompt_version": RANKING_PROMPT_VERSION,
        "assessment_scope": ASSESSMENT_SCOPE_VERSION,
        "skill_embedding_model": cfg.skill_embedding_model,
        "skill_match_threshold": skill_match_threshold,
        "min_skill_matches": min_skill_matches,
        "shortlist_sizes": {jd_id: len(cids) for jd_id, cids in final_state["shortlists"].items()},
        "jd_ids": [jd.id for jd in jds],
        "candidate_ids": [c.id for c in candidates],
    }
    run_dir = write_run_output(
        cfg.runs_dir, run_id, manifest, final_state["assessment_results"]
    )

    total_shortlisted = sum(len(v) for v in final_state["shortlists"].values())
    if total_shortlisted == 0:
        raise RuntimeError(
            f"Run produced zero shortlisted candidates across all {len(jds)} JDs at "
            f"skill_match_threshold={skill_match_threshold}, min_skill_matches={min_skill_matches} -- no "
            "assessments were generated. Check extracted skill vocabularies or loosen the thresholds. "
            f"Run output (manifest, empty summary) was still written to {run_dir}."
        )

    try:
        exported_path = export_console_web_data(cfg, run_id)
        if exported_path is not None:
            print(f"Exported console-web data to {exported_path}")
        else:
            print("INFO: console-web export skipped (see warnings above for the reason, if any)")
    except Exception as exc:
        print(f"WARNING: console-web export failed: {exc}", file=sys.stderr)

    return run_dir


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="candidate_ranking.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--run-id",
        default=None,
        help="Reuse a prior run's id to resume it from its checkpoint. Defaults to a new timestamped id.",
    )
    run_parser.add_argument(
        "--skill-match-threshold",
        type=float,
        default=0.8,
        help="Cosine similarity at or above which a candidate's skill counts as matching a JD's technical skill (default: 0.8).",
    )
    run_parser.add_argument(
        "--min-skill-matches",
        type=int,
        default=5,
        help=(
            "Number of a JD's technical skills a candidate must match to be shortlisted for assessment "
            "(default: 5; automatically reduced to the JD's total technical-skill count if that's smaller)."
        ),
    )
    args = parser.parse_args()

    if args.command == "run":
        run_dir = run(
            run_id=args.run_id,
            skill_match_threshold=args.skill_match_threshold,
            min_skill_matches=args.min_skill_matches,
        )
        print(f"Run complete: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
