from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.injection.attacks import (
    INSTRUCTION_INJECTION_PARAPHRASES,
    build_injected_candidate,
    marker_survived,
    optimize_evasive_attack,
    select_pair_subsample,
    stratified_sample_pairs,
)
from candidate_ranking.injection.defenses import (
    REFERENCE_SUSPICIOUS_PHRASES,
    build_classifier_filter,
    build_hardened_assessment_chain,
    build_self_reminder_assessment_chain,
    filter_suspicious_lines,
    filter_suspicious_lines_classifier,
)
from candidate_ranking.injection.measurement import compute_rank_shift
from candidate_ranking.ingestion.cv import load_candidates
from candidate_ranking.ingestion.jd import load_job_descriptions
from candidate_ranking.models import Assessment, Candidate
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


def _control_by_pair(prior_results: list[dict]) -> dict[tuple[str, str], dict]:
    return {
        (r["jd_id"], r["candidate_id"]): r
        for r in prior_results
        if r["condition"] == "control_no_injection"
    }


_COMPARATIVE_CONDITIONS = ("comparative_unmitigated", "comparative_mitigated")
_NON_COMPARATIVE_CONDITIONS = (
    "adaptive_unmitigated", "adaptive_mitigated",
    "defense_a_classifier", "defense_b_self_reminder",
)


def _existing_conditions_by_pair(existing_results: list[dict]) -> dict[tuple[str, str], set[str]]:
    conditions_by_pair: dict[tuple[str, str], set[str]] = {}
    for r in existing_results:
        key = (r["jd_id"], r["candidate_id"])
        conditions_by_pair.setdefault(key, set()).add(r["condition"])
    return conditions_by_pair


def main(run_id: str, prior_run_id: str, per_profile: int, seed: int, dry_run: bool, skip_comparative: bool) -> None:
    target_conditions = _NON_COMPARATIVE_CONDITIONS if skip_comparative else (_COMPARATIVE_CONDITIONS + _NON_COMPARATIVE_CONDITIONS)
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
    original_pairs = stratified_sample_pairs(candidate_ids_by_jd, per_profile=10, seed=seed)
    all_pairs = select_pair_subsample(original_pairs, per_profile=per_profile)
    print(f"Selected {len(all_pairs)} pairs (subset of the original {len(original_pairs)}-pair sample).")

    prior_results_path = cfg.runs_dir / prior_run_id / "injection_study" / "results.json"
    prior_results = json.loads(prior_results_path.read_text(encoding="utf-8"))
    control_by_pair = _control_by_pair(prior_results)
    for jd_id, cv_id in all_pairs:
        assert (jd_id, cv_id) in control_by_pair, f"missing reused control for {(jd_id, cv_id)}"

    out_path = run_dir / "injection_study" / "extended_results.json"
    results: list[dict] = []
    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))
    existing = _existing_conditions_by_pair(results)
    pairs = [p for p in all_pairs if not set(target_conditions) <= existing.get(p, set())]
    missing_combinations = sum(len(set(target_conditions) - existing.get(p, set())) for p in pairs)
    print(
        f"{len(all_pairs) - len(pairs)} pairs already fully complete for the target conditions in "
        f"{out_path.name}; {len(pairs)} pair(s) have at least one missing condition."
    )
    if skip_comparative:
        print("Skipping Attack A (comparative) conditions for this run.")

    if dry_run:
        print(f"Dry run OK: would run {missing_combinations} new (pair, condition) combinations.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    llm = ChatOllama(model=cfg.ollama_model, base_url=cfg.ollama_base_url, temperature=0)
    unmitigated_chain = build_assessment_chain(llm)
    hardened_chain = build_hardened_assessment_chain(llm)
    self_reminder_chain = build_self_reminder_assessment_chain(llm)
    build_ranking_chain = build_listwise_ranking_chain(llm)
    embedder = build_skill_embedder(cfg.skill_embedding_model)
    reference_embeddings = embedder(REFERENCE_SUSPICIOUS_PHRASES)
    classify = build_classifier_filter()
    rng = random.Random(seed)

    for jd_id, cv_id in pairs:
        have = existing.setdefault((jd_id, cv_id), set())
        if "control_no_injection" not in have:
            results.append(control_by_pair[(jd_id, cv_id)])
            have.add("control_no_injection")

        jd = jds_by_id[jd_id]
        pool_assessments = assessments_by_jd[jd_id]
        clean_rank = rank_by_jd[jd_id][cv_id]
        item_ids = sorted(pool_assessments)
        history = history_by_jd[jd_id]

        def _measure(condition, variant_name, candidate_for_condition, chain, marker):
            if condition in have:
                print(f"{jd_id}/{cv_id} {variant_name}/{condition}: already present, skipping")
                return
            modified_assessment = generate_assessment(jd, candidate_for_condition, chain, cfg.ollama_model)
            shift = compute_rank_shift(
                jd=jd, pool_assessments=pool_assessments, candidate_id=cv_id, clean_rank=clean_rank,
                item_ids=item_ids, combined_history=history, modified_assessment=modified_assessment,
                build_ranking_chain=build_ranking_chain, pl_prior_variance=cfg.pl_prior_variance,
            )
            record = {
                "jd_id": jd_id, "candidate_id": cv_id, "variant_name": variant_name, "condition": condition,
                "marker_survived": marker_survived(modified_assessment, marker) if marker else False,
                **shift,
            }
            results.append(record)
            have.add(condition)
            print(f"{jd_id}/{cv_id} {variant_name}/{condition}: {shift}")

        if not skip_comparative and {"comparative_unmitigated", "comparative_mitigated"} - have:
            comparative_candidate, comparative_marker = build_injected_candidate(
                candidates_by_id[cv_id], jd_id, seed, category="comparative_injection",
            )
            comparative_filtered_text, _ = filter_suspicious_lines(
                comparative_candidate.raw_text, embedder, reference_embeddings,
            )
            comparative_filtered_candidate = comparative_candidate.model_copy(
                update={"raw_text": comparative_filtered_text}
            )
            _measure(
                "comparative_unmitigated", "comparative_injection", comparative_candidate,
                unmitigated_chain, comparative_marker,
            )
            _measure(
                "comparative_mitigated", "comparative_injection", comparative_filtered_candidate,
                hardened_chain, comparative_marker,
            )

        if {"adaptive_unmitigated", "adaptive_mitigated"} - have:
            seed_paraphrase = rng.choice(INSTRUCTION_INJECTION_PARAPHRASES)
            evasive_text = optimize_evasive_attack(
                seed_text=seed_paraphrase["attack_text"],
                embedder=embedder,
                reference_embeddings=reference_embeddings,
                counter_fitted_vectors_path=str(cfg.cache_dir / "counter-fitted-vectors.txt"),
            )
            adaptive_candidate: Candidate = candidates_by_id[cv_id].model_copy(
                update={"raw_text": candidates_by_id[cv_id].raw_text + "\n\n" + evasive_text}
            )
            adaptive_filtered_text, _ = filter_suspicious_lines(
                adaptive_candidate.raw_text, embedder, reference_embeddings,
            )
            adaptive_filtered_candidate = adaptive_candidate.model_copy(update={"raw_text": adaptive_filtered_text})
            _measure("adaptive_unmitigated", "adaptive_evasion", adaptive_candidate, unmitigated_chain, None)
            _measure("adaptive_mitigated", "adaptive_evasion", adaptive_filtered_candidate, hardened_chain, None)

        if {"defense_a_classifier", "defense_b_self_reminder"} - have:
            original_candidate, original_marker = build_injected_candidate(candidates_by_id[cv_id], jd_id, seed)
            classifier_filtered_text, _ = filter_suspicious_lines_classifier(original_candidate.raw_text, classify)
            classifier_filtered_candidate = original_candidate.model_copy(update={"raw_text": classifier_filtered_text})
            _measure(
                "defense_a_classifier", "instruction_injection", classifier_filtered_candidate,
                unmitigated_chain, original_marker,
            )
            _measure(
                "defense_b_self_reminder", "instruction_injection", original_candidate,
                self_reminder_chain, original_marker,
            )

        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} record(s) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prior-run-id", required=True)
    parser.add_argument("--per-profile", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-comparative", action="store_true", help="Skip Attack A (comparative) conditions.")
    args = parser.parse_args()
    main(args.run_id, args.prior_run_id, args.per_profile, args.seed, args.dry_run, args.skip_comparative)
