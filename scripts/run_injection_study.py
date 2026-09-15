from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.injection.attack_corpus import (
    ATTACK_CATEGORY,
    build_injected_candidate,
    marker_survived,
    stratified_sample_pairs,
)
from candidate_ranking.injection.mitigation import build_hardened_assessment_chain
from candidate_ranking.injection.semantic_filter import filter_suspicious_lines
from candidate_ranking.injection.semantic_filter_reference import REFERENCE_SUSPICIOUS_PHRASES
from candidate_ranking.injection.rank_shift import compute_rank_shift
from candidate_ranking.ingestion.cv import load_candidates
from candidate_ranking.ingestion.jd import load_job_descriptions
from candidate_ranking.models import Assessment
from candidate_ranking.ranking.tournament import build_listwise_ranking_chain
from candidate_ranking.scoring.assessment import build_assessment_chain, generate_assessment
from candidate_ranking.scoring.skills import build_skill_embedder

load_dotenv()


def _load_jd_assessments(run_dir: Path, jd_id: str) -> dict[str, Assessment]:
    raw = json.loads((run_dir / jd_id / "assessments.json").read_text(encoding="utf-8"))
    return {cv_id: Assessment.model_validate(entry) for cv_id, entry in raw.items()}


def _load_jd_rank_by_candidate_id(run_dir: Path, jd_id: str) -> dict[str, int]:
    ranking = json.loads((run_dir / jd_id / "ranking.json").read_text(encoding="utf-8"))
    return {row["candidate_id"]: row["rank"] for row in ranking["rankings"] if row["rank"] is not None}


def _load_combined_repeat_history(run_dir: Path, jd_id: str) -> list[dict]:
    combined: list[dict] = []
    for state_path in sorted((run_dir / jd_id / "tournament").glob("repeat_*/state.json")):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        combined.extend(state["history"])
    return combined


def main(run_id: str, per_profile: int, seed: int, dry_run: bool) -> None:
    cfg = apply_env_overrides(RunConfig.full(PROJECT_ROOT))
    run_dir = cfg.runs_dir / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    jd_ids = manifest["jd_ids"]

    jds_by_id = {jd.id: jd for jd in load_job_descriptions(cfg.jd_dir) if jd.id in jd_ids}
    candidates_by_id = {c.id: c for c in load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")}
    cv_skills_cache = json.loads((cfg.cache_dir / "cv_skills.json").read_text(encoding="utf-8"))
    for cid, candidate in candidates_by_id.items():
        if cid in cv_skills_cache:
            candidates_by_id[cid] = candidate.model_copy(update={"skills": cv_skills_cache[cid]["skills"]})

    assessments_by_jd = {jd_id: _load_jd_assessments(run_dir, jd_id) for jd_id in jd_ids}
    rank_by_jd = {jd_id: _load_jd_rank_by_candidate_id(run_dir, jd_id) for jd_id in jd_ids}
    history_by_jd = {jd_id: _load_combined_repeat_history(run_dir, jd_id) for jd_id in jd_ids}

    candidate_ids_by_jd = {
        jd_id: [cid for cid in assessments_by_jd[jd_id] if cid in rank_by_jd[jd_id]] for jd_id in jd_ids
    }
    pairs = stratified_sample_pairs(candidate_ids_by_jd, per_profile=per_profile, seed=seed)
    print(f"Sampled {len(pairs)} pairs across {len(jd_ids)} job profiles.")

    if dry_run:
        for jd_id, cv_id in pairs:
            assert cv_id in candidates_by_id, f"missing candidate {cv_id}"
            assert cv_id in assessments_by_jd[jd_id], f"missing assessment {jd_id}/{cv_id}"
            assert cv_id in rank_by_jd[jd_id], f"missing rank {jd_id}/{cv_id}"
            assert history_by_jd[jd_id], f"empty tournament history for {jd_id}"
        total_combinations = len(pairs) * (1 + 3)
        print(
            f"Dry run OK: would run {total_combinations} (pair, condition) combinations "
            f"({len(pairs)} control_no_injection regenerations + {len(pairs) * 3} "
            "attack-condition combinations)."
        )
        return

    llm = ChatOllama(model=cfg.ollama_model, base_url=cfg.ollama_base_url, temperature=0)
    unmitigated_chain = build_assessment_chain(llm)
    mitigated_chain = build_hardened_assessment_chain(llm)
    build_ranking_chain = build_listwise_ranking_chain(llm)
    embedder = build_skill_embedder(cfg.skill_embedding_model)
    reference_embeddings = embedder(REFERENCE_SUSPICIOUS_PHRASES)

    results: list[dict] = []
    for jd_id, cv_id in pairs:
        jd = jds_by_id[jd_id]
        pool_assessments = assessments_by_jd[jd_id]
        clean_rank = rank_by_jd[jd_id][cv_id]
        item_ids = sorted(pool_assessments)
        history = history_by_jd[jd_id]

        control_assessment = generate_assessment(jd, candidates_by_id[cv_id], unmitigated_chain, cfg.ollama_model)
        control_shift = compute_rank_shift(
            jd=jd, pool_assessments=pool_assessments, candidate_id=cv_id, clean_rank=clean_rank,
            item_ids=item_ids, combined_history=history, modified_assessment=control_assessment,
            build_ranking_chain=build_ranking_chain, pl_prior_variance=cfg.pl_prior_variance,
        )
        results.append({
            "jd_id": jd_id, "candidate_id": cv_id, "variant_name": "none", "condition": "control_no_injection",
            "marker_survived": False, **control_shift,
        })
        print(f"{jd_id}/{cv_id} none/control_no_injection: {control_shift}")

        injected_candidate, marker = build_injected_candidate(candidates_by_id[cv_id], jd_id, seed)
        filtered_text, dropped_count = filter_suspicious_lines(injected_candidate.raw_text, embedder, reference_embeddings)
        filtered_candidate = injected_candidate.model_copy(update={"raw_text": filtered_text})
        for condition, chain, candidate_for_condition in (
            ("unmitigated", unmitigated_chain, injected_candidate),
            ("mitigated", mitigated_chain, filtered_candidate),
            ("mitigated_no_filter", mitigated_chain, injected_candidate),
        ):
            modified_assessment = generate_assessment(jd, candidate_for_condition, chain, cfg.ollama_model)
            shift = compute_rank_shift(
                jd=jd, pool_assessments=pool_assessments, candidate_id=cv_id, clean_rank=clean_rank,
                item_ids=item_ids, combined_history=history, modified_assessment=modified_assessment,
                build_ranking_chain=build_ranking_chain, pl_prior_variance=cfg.pl_prior_variance,
            )
            record = {
                "jd_id": jd_id, "candidate_id": cv_id, "variant_name": ATTACK_CATEGORY, "condition": condition,
                "marker_survived": marker_survived(modified_assessment, marker), **shift,
            }
            if condition == "mitigated":
                record["lines_dropped"] = dropped_count
            results.append(record)
            print(f"{jd_id}/{cv_id} {ATTACK_CATEGORY}/{condition}: {shift}")

    out_path = run_dir / "injection_study" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} record(s) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--per-profile", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Validate sampling/data availability without calling the LLM.")
    args = parser.parse_args()
    main(args.run_id, args.per_profile, args.seed, args.dry_run)
