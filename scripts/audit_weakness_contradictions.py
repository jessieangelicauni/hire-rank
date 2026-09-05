"""Re-verifies the weakness self-correction retry loop's real numbers
against the live model, for run 20260831-010721.

Regenerates assessments for all 348 shortlisted (job, candidate) pairs by
calling generate_assessment directly (bypassing the production assessment
cache), using an on_attempt hook to observe what the original run's logging
never captured: how many pairs had a weakness contradiction on the first
attempt, how many were fixed by the single retry, and how many still
contradicted afterward (and were dropped).

Because every ChatOllama call in this codebase uses temperature=0, this is
expected to be a deterministic replay of what already happened during the
original run -- it should reproduce the 26-pair drop count already visible
in runs/20260831-010721/warnings.json, while also surfacing the
previously-unlogged first-attempt and fixed-by-retry counts.

Does not touch runs/_cache/assessments/ or any existing run output; writes
only a new runs/20260831-010721/weakness_retry_audit_report.json.

Run with: uv run python scripts/audit_weakness_contradictions.py
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AttemptEvent:
    jd_id: str
    candidate_id: str
    attempt: int
    contradicted_skills: list[str]
    weaknesses: list[str]


def classify_pair(events: list[AttemptEvent]) -> str:
    first = events[0]
    if not first.contradicted_skills:
        return "clean"
    if len(events) == 1:
        return "dropped"
    return "dropped" if events[1].contradicted_skills else "fixed_by_retry"


def build_report(
    classifications: dict[tuple[str, str], str],
    total_weaknesses_checked: int,
    items_dropped: int,
    offline_residual_contradictions: int,
    offline_denominator: int,
) -> dict:
    total_pairs = len(classifications)
    pairs_with_initial_contradiction = sum(1 for c in classifications.values() if c != "clean")
    pairs_fixed_by_retry = sum(1 for c in classifications.values() if c == "fixed_by_retry")
    pairs_dropped = sum(1 for c in classifications.values() if c == "dropped")
    return {
        "total_pairs": total_pairs,
        "total_weaknesses_checked": total_weaknesses_checked,
        "pairs_with_initial_contradiction": pairs_with_initial_contradiction,
        "pairs_fixed_by_retry": pairs_fixed_by_retry,
        "pairs_dropped": pairs_dropped,
        "items_dropped": items_dropped,
        "offline_audit_residual_contradictions": offline_residual_contradictions,
        "offline_audit_denominator": offline_denominator,
        "offline_audit_rate": offline_residual_contradictions / offline_denominator,
    }


def _process_pair(
    jd,
    candidate,
    jd_skills,
    assessment_chain,
    skill_extraction_chain,
    skill_embedder,
    cfg,
) -> tuple[tuple[str, str], list[AttemptEvent], object]:
    """Runs one (jd, candidate) pair end-to-end and returns its own results.

    Executed inside a worker thread. Everything mutated here (the local
    `events` list, the `enriched` candidate) is local to this call, so
    nothing here is shared with any other in-flight worker -- the caller is
    responsible for folding the returned tuple into shared state back on the
    main thread.
    """
    from candidate_ranking.ingestion.cv import enrich_candidates_with_skills
    from candidate_ranking.scoring.assessment import generate_assessment

    pair_key = (jd.id, candidate.id)
    enriched = enrich_candidates_with_skills(
        [candidate], skill_extraction_chain, cfg.ollama_model, cfg.cache_dir / "cv_skills.json",
    )[0]

    events: list[AttemptEvent] = []

    def on_attempt(attempt: int, contradicted: list[str], weaknesses: list[str]) -> None:
        events.append(AttemptEvent(pair_key[0], pair_key[1], attempt, contradicted, weaknesses))

    final = generate_assessment(
        jd, enriched, assessment_chain, cfg.ollama_model,
        jd_skills=jd_skills, skill_embedder=skill_embedder, on_attempt=on_attempt,
    )
    return pair_key, events, final


def _run(run_id: str = "20260831-010721") -> dict:
    from langchain_ollama import ChatOllama

    from candidate_ranking.config import RunConfig, apply_env_overrides
    from candidate_ranking.evaluation.evaluation import load_assessment_results
    from candidate_ranking.ingestion.cv import build_skill_extraction_chain, load_candidates
    from candidate_ranking.ingestion.jd import load_job_descriptions
    from candidate_ranking.scoring.assessment import build_assessment_chain
    from candidate_ranking.scoring.jd_skills import build_jd_skills_chain, load_or_generate_jd_skills
    from candidate_ranking.scoring.skills import build_skill_embedder

    project_root = Path(__file__).resolve().parents[1]
    cfg = apply_env_overrides(RunConfig.full(project_root))
    run_dir = cfg.runs_dir / run_id

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    jd_ids: list[str] = manifest["jd_ids"]

    jds_by_id = {jd.id: jd for jd in load_job_descriptions(cfg.jd_dir)}
    all_candidates_by_id = {c.id: c for c in load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")}

    llm = ChatOllama(model=cfg.ollama_model, base_url=cfg.ollama_base_url, temperature=0, num_ctx=cfg.ollama_num_ctx)
    jd_skills_chain = build_jd_skills_chain(llm)
    assessment_chain = build_assessment_chain(llm)
    skill_extraction_chain = build_skill_extraction_chain(llm)
    skill_embedder = build_skill_embedder(cfg.skill_embedding_model)

    # Loading each JD's skills is cheap (one cached call per JD, not per pair)
    # and stays sequential on the main thread; only the per-(jd, candidate)
    # assessment work below is fanned out.
    pending: list[tuple] = []
    for jd_id in jd_ids:
        jd = jds_by_id[jd_id]
        jd_skills = load_or_generate_jd_skills(jd, jd_skills_chain, cfg.ollama_model, cfg.cache_dir / "jd_skills.json")
        existing_assessments = load_assessment_results(run_dir, jd_id)

        for existing in existing_assessments:
            candidate = all_candidates_by_id[existing.candidate_id]
            pending.append((jd, candidate, jd_skills))

    events_by_pair: dict[tuple[str, str], list[AttemptEvent]] = {}
    total_weaknesses_checked = 0
    items_dropped = 0

    # max_workers matches cfg.ollama_num_parallel (4) so this replay hits
    # Ollama at the same concurrency as the original run, per
    # RunConfig.ollama_num_parallel / cli.py's max_concurrency. Each worker
    # (_process_pair) only touches its own local `events` list and returns
    # a self-contained (pair_key, events, final) tuple; all shared state
    # (events_by_pair, total_weaknesses_checked, items_dropped) is mutated
    # only here on the main thread as results come back, so no lock is
    # needed. A future.result() exception (e.g. AssessmentGenerationError)
    # propagates out of this loop and aborts the run -- no report is written.
    with ThreadPoolExecutor(max_workers=cfg.ollama_num_parallel) as executor:
        futures = [
            executor.submit(
                _process_pair, jd, candidate, jd_skills, assessment_chain, skill_extraction_chain, skill_embedder, cfg
            )
            for jd, candidate, jd_skills in pending
        ]
        for future in as_completed(futures):
            pair_key, events, final = future.result()
            events_by_pair[pair_key] = events
            total_weaknesses_checked += len(final.weaknesses)
            last_attempt_weaknesses = events[-1].weaknesses
            items_dropped += max(0, len(last_attempt_weaknesses) - len(final.weaknesses))

    classifications = {pair: classify_pair(events) for pair, events in events_by_pair.items()}

    ragas_report = json.loads((run_dir / "ragas_faithfulness_report.json").read_text(encoding="utf-8"))
    offline_residual = len(ragas_report["weakness_contradictions"])

    report = build_report(
        classifications,
        total_weaknesses_checked=total_weaknesses_checked,
        items_dropped=items_dropped,
        offline_residual_contradictions=offline_residual,
        offline_denominator=total_weaknesses_checked,
    )

    out_path = run_dir / "weakness_retry_audit_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    _run()
