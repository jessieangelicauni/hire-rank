"""Runs Ragas' Faithfulness metric directly over one already-completed run's
strengths, as an offline content-quality monitor -- not part of the live
pipeline. Weaknesses are checked separately (see --skip-weakness-check
below), not through Ragas: Faithfulness can only verify a claim positively
entailed by the context, and a weakness is almost always an absence claim
("lacks X") that CV text can never positively entail. Read-only: does not
touch the production pipeline or the run being evaluated other than writing
its own report file. Requires the `eval` dependency group:
`uv sync --group eval`.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv

from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.evaluation.ragas_eval import (
    LOW_SCORE_CUTOFF,
    run_ragas_faithfulness_eval,
    run_weakness_contradiction_check,
)

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Run id under runs_dir to evaluate, e.g. 20260826-010039")
    parser.add_argument(
        "--output", default=None,
        help="Report output path. Defaults to runs/<run-id>/ragas_faithfulness_report.json",
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help=(
            "Evaluate a stratified random sample of this many rows (an equal share per JD) instead of the "
            "full population -- a full corpus can be thousands of rows at 2 LLM calls each, which on a "
            "local model easily takes many hours. Omit to evaluate every row."
        ),
    )
    parser.add_argument("--sample-seed", type=int, default=42, help="RNG seed for --sample-size (default: 42)")
    parser.add_argument(
        "--skip-weakness-check", action="store_true",
        help=(
            "Skip the weakness-contradiction check (flags a weakness claiming a candidate lacks a JD skill "
            "they actually have listed) -- a check Ragas Faithfulness structurally can't perform on absence "
            "claims. Runs by default; it's cheap since it reuses each JD's/candidate's already-cached "
            "extracted skills rather than calling an LLM judge."
        ),
    )
    args = parser.parse_args()

    cfg = apply_env_overrides(RunConfig.full(PROJECT_ROOT))
    output_path = Path(args.output) if args.output else cfg.runs_dir / args.run_id / "ragas_faithfulness_report.json"

    report = run_ragas_faithfulness_eval(
        cfg, args.run_id, output_path, sample_size=args.sample_size, sample_seed=args.sample_seed
    )
    print(f"Wrote {output_path}")
    print()
    if report.population_size is not None and report.population_size != report.n_rows:
        print(f"Rows evaluated:            {report.n_rows} (sampled from {report.population_size})")
    else:
        print(f"Rows evaluated:            {report.n_rows}")
    print(f"Ragas failures (excluded): {report.n_ragas_failures}")
    mean_score = report.mean_score
    print(
        f"Mean faithfulness score (strengths only): {mean_score:.4f}"
        if mean_score is not None
        else "Mean faithfulness score (strengths only): n/a"
    )
    print(f"Low-score items (< {LOW_SCORE_CUTOFF}): {len(report.low_score_items)}")

    if not args.skip_weakness_check:
        from langchain_ollama import ChatOllama

        from candidate_ranking.ingestion.cv import build_skill_extraction_chain
        from candidate_ranking.scoring.jd_skills import build_jd_skills_chain
        from candidate_ranking.scoring.skills import build_skill_embedder

        llm = ChatOllama(model=cfg.ollama_model, base_url=cfg.ollama_base_url, temperature=0, num_ctx=cfg.ollama_num_ctx)
        contradictions = run_weakness_contradiction_check(
            cfg, args.run_id,
            jd_skills_chain=build_jd_skills_chain(llm),
            skill_extraction_chain=build_skill_extraction_chain(llm),
            embedder=build_skill_embedder(cfg.skill_embedding_model),
        )
        report = dataclasses.replace(report, weakness_contradictions=contradictions)
        output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print()
        print(f"Weakness contradictions (candidate has a skill their weakness claims they lack): {len(contradictions)}")


if __name__ == "__main__":
    main()
