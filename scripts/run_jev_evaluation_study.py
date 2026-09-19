from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv

from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.ingestion.cv import load_candidates
from candidate_ranking.ingestion.jd import load_job_descriptions
from candidate_ranking.models import Candidate, JDSkills, JobDescription
from candidate_ranking.scoring.assessment import JEV_MODEL_NAME, _answers_to_assessment, _build_questions, _build_state
from candidate_ranking.scoring.jev_client import JevAnswer, JevClient, JevQuestion

load_dotenv()

_VAGUE_SCORE_CRITERIA = ["0", "25", "50", "75", "100"]


def _vague_build_questions(jd_technical_skills: list[str] | None) -> list[JevQuestion]:
    questions = [
        JevQuestion(
            key="overall_fit_score", kind="score",
            instructions="How well does this candidate's CV fit the job description overall?",
            criteria=_VAGUE_SCORE_CRITERIA,
        ),
        JevQuestion(
            key="overall_recommendation", kind="choice",
            instructions="What is the hiring recommendation for this candidate against this job description?",
            criteria={
                "hire": "Candidate clearly meets or exceeds the role's requirements",
                "maybe": "Candidate partially meets the role's requirements",
                "no": "Candidate does not meet the role's requirements",
            },
        ),
        JevQuestion(
            key="meets_min_qualifications", kind="noul",
            instructions="Does the candidate meet the job description's minimum qualifications?",
            criteria={
                "true": "Meets every minimum qualification stated in the job description",
                "false": "Fails at least one minimum qualification stated in the job description",
            },
        ),
    ]
    for requirement in jd_technical_skills or []:
        questions.append(
            JevQuestion(
                key=f"requirement::{requirement}", kind="score",
                instructions=f"How well does the candidate's CV support the requirement '{requirement}'?",
                criteria=_VAGUE_SCORE_CRITERIA,
            )
        )
    return questions


def _vague_seniority_education_questions(jd_skills: JDSkills | None) -> list[JevQuestion]:
    questions = []
    if jd_skills and jd_skills.seniority_requirement:
        questions.append(
            JevQuestion(
                key="seniority", kind="score",
                instructions=(
                    "How well does the candidate meet the job description's stated seniority/experience "
                    f"requirement: '{jd_skills.seniority_requirement}'?"
                ),
                criteria=_VAGUE_SCORE_CRITERIA,
            )
        )
    if jd_skills and jd_skills.education_requirement:
        questions.append(
            JevQuestion(
                key="education", kind="score",
                instructions=(
                    "How well does the candidate meet the job description's stated education "
                    f"requirement: '{jd_skills.education_requirement}'?"
                ),
                criteria=_VAGUE_SCORE_CRITERIA,
            )
        )
    return questions


def load_run_pairs(run_dir: Path, jd_ids: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for jd_id in jd_ids:
        path = run_dir / jd_id / "assessments.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        pairs.extend((jd_id, candidate_id) for candidate_id in data)
    return pairs


def load_corpus(
    cfg: RunConfig, jd_ids: list[str]
) -> tuple[dict[str, JobDescription], dict[str, Candidate], dict[str, JDSkills]]:
    jds_by_id = {jd.id: jd for jd in load_job_descriptions(cfg.jd_dir) if jd.id in jd_ids}
    candidates_by_id = {c.id: c for c in load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")}
    cv_skills_cache = json.loads((cfg.cache_dir / "cv_skills.json").read_text(encoding="utf-8"))
    for cid, candidate in candidates_by_id.items():
        if cid in cv_skills_cache:
            candidates_by_id[cid] = candidate.model_copy(update={"skills": cv_skills_cache[cid]["skills"]})
    jd_skills_cache = json.loads((cfg.cache_dir / "jd_skills.json").read_text(encoding="utf-8"))
    jd_skills_by_jd = {
        jd_id: JDSkills.model_validate(entry["jd_skills"]) for jd_id, entry in jd_skills_cache.items() if jd_id in jd_ids
    }
    return jds_by_id, candidates_by_id, jd_skills_by_jd


def _timed_evaluate(jev_client: JevClient, state: str, questions: list[JevQuestion]) -> tuple[list[JevAnswer], float]:
    start = time.perf_counter()
    answers = jev_client.evaluate(state, questions)
    elapsed = time.perf_counter() - start
    return answers, elapsed


def collect_test_retest(
    jev_client: JevClient,
    jds_by_id: dict[str, JobDescription],
    candidates_by_id: dict[str, Candidate],
    jd_skills_by_jd: dict[str, JDSkills],
    pairs: list[tuple[str, str]],
    repeats: int,
    max_workers: int,
) -> list[dict]:
    def run_one(pair: tuple[str, str]) -> dict:
        jd_id, candidate_id = pair
        jd = jds_by_id[jd_id]
        candidate = candidates_by_id[candidate_id]
        questions = _build_questions(jd_skills_by_jd.get(jd_id))
        state = _build_state(jd, candidate)

        repeats_out = []
        for _ in range(repeats):
            answers, elapsed = _timed_evaluate(jev_client, state, questions)
            assessment = _answers_to_assessment(jd, candidate, JEV_MODEL_NAME, answers)
            repeats_out.append(
                {
                    "overall_fit_score": assessment.overall_fit_score,
                    "overall_recommendation": assessment.overall_recommendation,
                    "meets_min_qualifications": assessment.meets_min_qualifications,
                    "requirement_scores": assessment.requirement_scores,
                    "latency_seconds": elapsed,
                }
            )
        return {"jd_id": jd_id, "candidate_id": candidate_id, "repeats": repeats_out}

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, result in enumerate(executor.map(run_one, pairs), start=1):
            results.append(result)
            if i % 25 == 0 or i == len(pairs):
                print(f"test-retest: {i}/{len(pairs)} pairs done")
    return results


def collect_ablation(
    jev_client: JevClient,
    jds_by_id: dict[str, JobDescription],
    candidates_by_id: dict[str, Candidate],
    jd_skills_by_jd: dict[str, JDSkills],
    run_dir: Path,
    sample_pairs: list[tuple[str, str]],
) -> list[dict]:
    def run_one(pair: tuple[str, str]) -> dict:
        jd_id, candidate_id = pair
        jd = jds_by_id[jd_id]
        candidate = candidates_by_id[candidate_id]

        cached = json.loads((run_dir / jd_id / "assessments.json").read_text(encoding="utf-8"))[candidate_id]
        concrete_confidence = cached["confidence"]

        jd_skills = jd_skills_by_jd.get(jd_id)
        vague_questions = _vague_build_questions(jd_skills.technical_skills if jd_skills else None)
        state = _build_state(jd, candidate)
        answers, elapsed = _timed_evaluate(jev_client, state, vague_questions)
        vague_assessment = _answers_to_assessment(jd, candidate, JEV_MODEL_NAME, answers)

        return {
            "jd_id": jd_id,
            "candidate_id": candidate_id,
            "concrete_confidence": concrete_confidence,
            "vague_confidence": vague_assessment.confidence,
            "vague_latency_seconds": elapsed,
        }

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i, result in enumerate(executor.map(run_one, sample_pairs), start=1):
            results.append(result)
            if i % 10 == 0 or i == len(sample_pairs):
                print(f"ablation: {i}/{len(sample_pairs)} pairs done")
    return results


def collect_seniority_education_ablation(
    jev_client: JevClient,
    jds_by_id: dict[str, JobDescription],
    candidates_by_id: dict[str, Candidate],
    jd_skills_by_jd: dict[str, JDSkills],
    run_dir: Path,
    pairs: list[tuple[str, str]],
) -> list[dict]:
    eligible_pairs = [
        pair
        for pair in pairs
        if _vague_seniority_education_questions(jd_skills_by_jd.get(pair[0]))
    ]

    def run_one(pair: tuple[str, str]) -> dict:
        jd_id, candidate_id = pair
        jd = jds_by_id[jd_id]
        candidate = candidates_by_id[candidate_id]
        jd_skills = jd_skills_by_jd.get(jd_id)

        cached = json.loads((run_dir / jd_id / "assessments.json").read_text(encoding="utf-8"))[candidate_id]
        vague_questions = _vague_seniority_education_questions(jd_skills)
        concrete_confidence = {
            q.key: cached["confidence"][q.key] for q in vague_questions if q.key in cached["confidence"]
        }

        state = _build_state(jd, candidate)
        answers, elapsed = _timed_evaluate(jev_client, state, vague_questions)
        vague_confidence = {a.key: a.confidence for a in answers}

        return {
            "jd_id": jd_id,
            "candidate_id": candidate_id,
            "concrete_confidence": concrete_confidence,
            "vague_confidence": vague_confidence,
            "vague_latency_seconds": elapsed,
        }

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i, result in enumerate(executor.map(run_one, eligible_pairs), start=1):
            results.append(result)
            if i % 25 == 0 or i == len(eligible_pairs):
                print(f"seniority-education-ablation: {i}/{len(eligible_pairs)} pairs done")
    return results


def main(run_id: str, repeats: int, ablation_sample_size: int, seed: int, max_workers: int, dry_run: bool, seniority_education_only: bool) -> None:
    cfg = apply_env_overrides(RunConfig.full(PROJECT_ROOT))
    run_dir = cfg.runs_dir / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    jd_ids = manifest["jd_ids"]

    pairs = load_run_pairs(run_dir, jd_ids)
    ablation_size = min(ablation_sample_size, len(pairs))
    print(f"Loaded {len(pairs)} successfully-assessed (jd, candidate) pair(s) from run {run_id}")

    jds_by_id, candidates_by_id, jd_skills_by_jd = load_corpus(cfg, jd_ids)
    eligible_count = sum(1 for jd_id, _ in pairs if _vague_seniority_education_questions(jd_skills_by_jd.get(jd_id)))

    if dry_run:
        if seniority_education_only:
            print(
                f"Dry run: would collect seniority/education ablation for {eligible_count} pair(s) whose job "
                f"description has a seniority or education requirement (out of {len(pairs)} total pairs)."
            )
        else:
            print(
                f"Dry run: would collect test-retest for {len(pairs)} pair(s) x {repeats} repeat(s) "
                f"({len(pairs) * repeats} call(s)), plus ablation for {ablation_size} pair(s) "
                f"({ablation_size} call(s))."
            )
        return

    if not cfg.jev_api_key:
        raise RuntimeError("Set CANDIDATE_RANKING_JEV_API_KEY (e.g. in .env) before running.")
    jev_client = JevClient(api_token=cfg.jev_api_key)

    out_dir = run_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    if seniority_education_only:
        print("Collecting seniority/education criteria-design ablation data (concrete vs. bare-label Score)...")
        sen_edu_results = collect_seniority_education_ablation(
            jev_client, jds_by_id, candidates_by_id, jd_skills_by_jd, run_dir, pairs
        )
        (out_dir / "seniority_education_ablation.json").write_text(json.dumps(sen_edu_results, indent=2), encoding="utf-8")
        print(f"Wrote {len(sen_edu_results)} seniority/education ablation record(s) to {out_dir / 'seniority_education_ablation.json'}")
        return

    print(f"Collecting test-retest reliability data ({len(pairs)} pairs x {repeats} repeats)...")
    test_retest_results = collect_test_retest(
        jev_client, jds_by_id, candidates_by_id, jd_skills_by_jd, pairs, repeats, max_workers
    )
    (out_dir / "test_retest.json").write_text(json.dumps(test_retest_results, indent=2), encoding="utf-8")
    print(f"Wrote {len(test_retest_results)} test-retest record(s) to {out_dir / 'test_retest.json'}")

    rng = random.Random(seed)
    ablation_pairs = rng.sample(pairs, ablation_size)
    print(f"Collecting criteria-design ablation data ({len(ablation_pairs)} pairs)...")
    ablation_results = collect_ablation(
        jev_client, jds_by_id, candidates_by_id, jd_skills_by_jd, run_dir, ablation_pairs
    )
    (out_dir / "ablation.json").write_text(json.dumps(ablation_results, indent=2), encoding="utf-8")
    print(f"Wrote {len(ablation_results)} ablation record(s) to {out_dir / 'ablation.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ablation-sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Validate pair counts without calling Jev.")
    parser.add_argument(
        "--seniority-education-only", action="store_true",
        help="Skip test-retest and technical-requirement ablation; only collect the seniority/education "
        "concrete-vs-bare-label Score criteria ablation, over all eligible pairs.",
    )
    args = parser.parse_args()
    main(
        args.run_id, args.repeats, args.ablation_sample_size, args.seed, args.max_workers, args.dry_run,
        args.seniority_education_only,
    )
