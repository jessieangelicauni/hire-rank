"""Run the automated, accuracy-guarded criteria-design optimizer for the
per-requirement Score question, and write the winning criteria plus the full
round history to runs/<run-id>/evaluation/criteria_optimization.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import anthropic
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.evaluation.criteria_optimization import (
    load_assessments_by_jd,
    load_requirement_triples,
    run_optimization,
    triple_key,
)
from candidate_ranking.evaluation.criteria_proposer import build_criteria_proposer_chain, propose_criteria
from candidate_ranking.evaluation.proxy_labeler import ProxyLabelClient
from candidate_ranking.scoring.assessment import _REQUIREMENT_FIT_CRITERIA
from candidate_ranking.scoring.jev_client import JevClient

load_dotenv()

# Re-declared here rather than importing from run_jev_evaluation_study.py: scripts/ is a
# collection of standalone CLIs, not a shared package, so cross-script imports would need a
# second sys.path hack for no real benefit over these ~10 lines.
from candidate_ranking.ingestion.cv import load_candidates
from candidate_ranking.ingestion.jd import load_job_descriptions
from candidate_ranking.models import Candidate, JobDescription


def load_corpus(cfg: RunConfig, jd_ids: list[str]) -> tuple[dict[str, JobDescription], dict[str, Candidate]]:
    jds_by_id = {jd.id: jd for jd in load_job_descriptions(cfg.jd_dir) if jd.id in jd_ids}
    candidates_by_id = {c.id: c for c in load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")}
    cv_skills_cache = json.loads((cfg.cache_dir / "cv_skills.json").read_text(encoding="utf-8"))
    for cid, candidate in candidates_by_id.items():
        if cid in cv_skills_cache:
            candidates_by_id[cid] = candidate.model_copy(update={"skills": cv_skills_cache[cid]["skills"]})
    return jds_by_id, candidates_by_id


def main(
    run_id: str, max_rounds: int, hard_case_count: int, patience: int,
    train_fraction: float, validation_fraction: float, seed: int, dry_run: bool,
) -> None:
    cfg = apply_env_overrides(RunConfig.full(PROJECT_ROOT))
    run_dir = cfg.runs_dir / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    jd_ids = manifest["jd_ids"]

    assessments_by_jd = load_assessments_by_jd(run_dir, jd_ids)
    all_triples = load_requirement_triples(assessments_by_jd)
    print(f"Loaded {len(all_triples)} requirement triple(s) from run {run_id}")

    rng = random.Random(seed)
    shuffled = all_triples[:]
    rng.shuffle(shuffled)

    if train_fraction + validation_fraction > 1.0:
        raise ValueError(
            f"train_fraction ({train_fraction}) + validation_fraction ({validation_fraction}) "
            "must not exceed 1.0"
        )

    n_train = int(len(shuffled) * train_fraction)
    n_val = int(len(shuffled) * validation_fraction)
    train_triples = shuffled[:n_train]
    validation_triples = shuffled[n_train:n_train + n_val]
    test_triples = shuffled[n_train + n_val:]
    print(f"Split: {len(train_triples)} train, {len(validation_triples)} validation, {len(test_triples)} held-out test")

    if dry_run:
        print(f"Dry run: would run up to {max_rounds} optimization round(s) with hard_case_count={hard_case_count}.")
        return

    if not cfg.jev_api_key:
        raise RuntimeError("Set CANDIDATE_RANKING_JEV_API_KEY before running.")
    if not cfg.anthropic_api_key:
        raise RuntimeError("Set CANDIDATE_RANKING_ANTHROPIC_API_KEY before running.")

    jev_client = JevClient(api_token=cfg.jev_api_key)
    anthropic_client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    proxy_labeler = ProxyLabelClient(
        anthropic_client, cache_path=cfg.cache_dir / "proxy_labels.json", model=cfg.proxy_label_model
    )
    llm = ChatOllama(model=cfg.ollama_model, base_url=cfg.ollama_base_url, temperature=0, num_ctx=cfg.ollama_num_ctx)
    proposer_chain = build_criteria_proposer_chain(llm)

    jds_by_id, candidates_by_id = load_corpus(cfg, jd_ids)

    proxy_labels: dict[str, int] = {}
    for triple in train_triples + validation_triples + test_triples:
        jd = jds_by_id[triple.job_description_id]
        candidate = candidates_by_id[triple.candidate_id]
        proxy_labels[triple_key(triple)] = proxy_labeler.label(triple, jd, candidate)
    print(f"Collected {len(proxy_labels)} proxy label(s)")

    def bound_propose_fn(current_criteria: list[str], hard_case_summaries: list[str]) -> list[str]:
        return propose_criteria(current_criteria, hard_case_summaries, proposer_chain)

    best_criteria, history = run_optimization(
        seed_criteria=_REQUIREMENT_FIT_CRITERIA,
        train_triples=train_triples,
        validation_triples=validation_triples,
        jds_by_id=jds_by_id,
        candidates_by_id=candidates_by_id,
        jev_client=jev_client,
        proxy_labels=proxy_labels,
        propose_fn=bound_propose_fn,
        max_rounds=max_rounds,
        hard_case_count=hard_case_count,
        patience=patience,
    )

    out_dir = run_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "criteria_optimization.json"
    out_path.write_text(
        json.dumps(
            {
                "best_criteria": best_criteria,
                "history": [round_.model_dump() for round_ in history],
                "test_triples": [t.model_dump() for t in test_triples],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote optimization result to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--hard-case-count", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(
        args.run_id, args.max_rounds, args.hard_case_count, args.patience,
        args.train_fraction, args.validation_fraction, args.seed, args.dry_run,
    )
