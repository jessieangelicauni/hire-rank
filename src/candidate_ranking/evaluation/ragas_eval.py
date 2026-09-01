from __future__ import annotations

import json
import logging
import math
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from candidate_ranking.config import RunConfig
from candidate_ranking.evaluation.evaluation import load_assessment_results
from candidate_ranking.ingestion.cv import enrich_candidates_with_skills, load_candidates
from candidate_ranking.ingestion.jd import load_job_descriptions
from candidate_ranking.models import Candidate, JDSkills, JobDescription
from candidate_ranking.scoring.jd_skills import load_or_generate_jd_skills
from candidate_ranking.scoring.skills import find_negated_skill_mentions

if TYPE_CHECKING:
    import pandas
    from langchain_core.runnables import Runnable
    from ragas import EvaluationDataset
    from ragas.llms import LangchainLLMWrapper

logger = logging.getLogger(__name__)

LOW_SCORE_CUTOFF = 0.5

NEUTRAL_USER_INPUT = "What skills, experience, and qualifications does the candidate have, based on their resume?"


@dataclass(frozen=True)
class FaithfulnessRow:
    jd_id: str
    candidate_id: str
    item_type: str
    response: str
    retrieved_contexts: list[str]
    user_input: str


def build_rows(cfg: RunConfig, run_id: str) -> list[FaithfulnessRow]:
    run_dir = cfg.runs_dir / run_id
    jds_by_id: dict[str, JobDescription] = {jd.id: jd for jd in load_job_descriptions(cfg.jd_dir)}
    candidates_by_id: dict[str, Candidate] = {
        c.id: c for c in load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")
    }
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    rows: list[FaithfulnessRow] = []
    for jd_id in manifest.get("jd_ids", []):
        jd = jds_by_id.get(jd_id)
        if jd is None:
            logger.warning("Skipping %s: JD no longer present in the corpus", jd_id)
            continue

        for assessment in load_assessment_results(run_dir, jd_id):
            candidate = candidates_by_id.get(assessment.candidate_id)
            if candidate is None:
                logger.warning(
                    "Skipping %s/%s: candidate no longer present in the corpus",
                    jd_id, assessment.candidate_id,
                )
                continue

            for strength in assessment.strengths:
                rows.append(
                    FaithfulnessRow(
                        jd_id=jd_id, candidate_id=candidate.id, item_type="strength",
                        response=strength, retrieved_contexts=[candidate.raw_text],
                        user_input=NEUTRAL_USER_INPUT,
                    )
                )
            for weakness in assessment.weaknesses:
                rows.append(
                    FaithfulnessRow(
                        jd_id=jd_id, candidate_id=candidate.id, item_type="weakness",
                        response=weakness, retrieved_contexts=[candidate.raw_text],
                        user_input=NEUTRAL_USER_INPUT,
                    )
                )
    return rows


@dataclass(frozen=True)
class FaithfulnessReport:
    run_id: str
    n_rows: int
    n_ragas_failures: int
    mean_score: float | None
    per_jd: dict[str, float]
    low_score_items: list[dict]
    population_size: int | None = None
    weakness_contradictions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "n_rows": self.n_rows,
            "n_ragas_failures": self.n_ragas_failures,
            "mean_score": self.mean_score,
            "per_jd": self.per_jd,
            "low_score_items": self.low_score_items,
            "population_size": self.population_size,
            "weakness_contradictions": self.weakness_contradictions,
        }


def stratified_sample(rows: list[FaithfulnessRow], sample_size: int, seed: int) -> list[FaithfulnessRow]:
    if sample_size >= len(rows):
        return list(rows)

    rng = random.Random(seed)
    groups = sorted(_group_by_jd(rows))
    base_share, remainder = divmod(sample_size, len(groups))

    sampled: list[FaithfulnessRow] = []
    for index, (_jd_id, jd_rows) in enumerate(groups):
        target = base_share + (1 if index < remainder else 0)
        n = min(target, len(jd_rows))
        sampled.extend(rng.sample(jd_rows, n))
    return sampled


def merge_and_report(
    run_id: str, rows: list[FaithfulnessRow], ragas_scores: list[float], population_size: int | None = None
) -> FaithfulnessReport:
    if len(rows) != len(ragas_scores):
        raise ValueError(f"rows ({len(rows)}) and ragas_scores ({len(ragas_scores)}) must be the same length")

    if not rows:
        return FaithfulnessReport(
            run_id=run_id, n_rows=0, n_ragas_failures=0, mean_score=None, per_jd={}, low_score_items=[],
            population_size=population_size,
        )

    finite_rows: list[FaithfulnessRow] = []
    finite_scores: list[float] = []
    for row, score in zip(rows, ragas_scores):
        if math.isfinite(score):
            finite_rows.append(row)
            finite_scores.append(score)
    n_ragas_failures = len(rows) - len(finite_rows)

    if not finite_rows:
        return FaithfulnessReport(
            run_id=run_id, n_rows=len(rows), n_ragas_failures=n_ragas_failures, mean_score=None,
            per_jd={}, low_score_items=[], population_size=population_size,
        )

    mean_score = sum(finite_scores) / len(finite_scores)

    scores_by_jd: dict[str, list[float]] = {}
    for row, score in zip(finite_rows, finite_scores):
        scores_by_jd.setdefault(row.jd_id, []).append(score)
    per_jd = {jd_id: sum(scores) / len(scores) for jd_id, scores in scores_by_jd.items()}

    low_score_items = [
        {
            "jd_id": row.jd_id,
            "candidate_id": row.candidate_id,
            "item_type": row.item_type,
            "item_text": row.response,
            "ragas_score": score,
        }
        for row, score in zip(finite_rows, finite_scores)
        if score < LOW_SCORE_CUTOFF
    ]

    return FaithfulnessReport(
        run_id=run_id, n_rows=len(rows), n_ragas_failures=n_ragas_failures,
        mean_score=mean_score, per_jd=per_jd, low_score_items=low_score_items, population_size=population_size,
    )


def find_weakness_contradictions(
    rows: list[FaithfulnessRow],
    jd_skills_by_id: dict[str, JDSkills],
    candidates_by_id: dict[str, Candidate],
    embedder: Callable[[list[str]], np.ndarray],
    threshold: float = 0.8,
) -> list[dict]:
    contradictions: list[dict] = []
    for row in rows:
        if row.item_type != "weakness":
            continue
        jd_skills = jd_skills_by_id.get(row.jd_id)
        candidate = candidates_by_id.get(row.candidate_id)
        if jd_skills is None or candidate is None or not jd_skills.technical_skills or not candidate.skills:
            continue

        mentioned_skills = find_negated_skill_mentions(row.response, jd_skills.technical_skills)
        if not mentioned_skills:
            continue

        mentioned_vectors = embedder(mentioned_skills)
        candidate_skill_vectors = embedder(candidate.skills)
        similarities = mentioned_vectors @ candidate_skill_vectors.T
        for i, skill in enumerate(mentioned_skills):
            best_index = int(np.argmax(similarities[i]))
            best_similarity = float(similarities[i][best_index])
            if best_similarity >= threshold:
                contradictions.append(
                    {
                        "jd_id": row.jd_id,
                        "candidate_id": row.candidate_id,
                        "weakness_text": row.response,
                        "contradicted_skill": skill,
                        "matched_candidate_skill": candidate.skills[best_index],
                        "similarity": best_similarity,
                    }
                )
    return contradictions


def run_weakness_contradiction_check(
    cfg: RunConfig,
    run_id: str,
    jd_skills_chain: "Runnable",
    skill_extraction_chain: "Runnable",
    embedder: Callable[[list[str]], np.ndarray],
) -> list[dict]:
    rows = build_rows(cfg, run_id)
    weakness_rows = [row for row in rows if row.item_type == "weakness"]
    if not weakness_rows:
        return []

    jds_by_id = {jd.id: jd for jd in load_job_descriptions(cfg.jd_dir)}
    relevant_jd_ids = {row.jd_id for row in weakness_rows}
    jd_skills_by_id = {
        jd_id: load_or_generate_jd_skills(
            jds_by_id[jd_id], jd_skills_chain, cfg.ollama_model, cfg.cache_dir / "jd_skills.json"
        )
        for jd_id in relevant_jd_ids
        if jd_id in jds_by_id
    }

    all_candidates_by_id = {c.id: c for c in load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")}
    relevant_candidate_ids = {row.candidate_id for row in weakness_rows}
    relevant_candidates = [c for c in all_candidates_by_id.values() if c.id in relevant_candidate_ids]
    enriched_candidates = enrich_candidates_with_skills(
        relevant_candidates, skill_extraction_chain, cfg.ollama_model, cfg.cache_dir / "cv_skills.json"
    )
    candidates_by_id = {c.id: c for c in enriched_candidates}

    return find_weakness_contradictions(weakness_rows, jd_skills_by_id, candidates_by_id, embedder)


def _default_evaluate_fn(
    dataset: "EvaluationDataset", llm: "LangchainLLMWrapper", max_workers: int = 16
) -> "pandas.DataFrame":
    from ragas import evaluate as ragas_evaluate
    from ragas.run_config import RunConfig as RagasRunConfig

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import Faithfulness

    result = ragas_evaluate(
        dataset=dataset,
        metrics=[Faithfulness()],
        llm=llm,
        run_config=RagasRunConfig(max_workers=max_workers, timeout=300),
    )
    return result.to_pandas()


def _group_by_jd(rows: list[FaithfulnessRow]) -> list[tuple[str, list[FaithfulnessRow]]]:
    groups: dict[str, list[FaithfulnessRow]] = {}
    for row in rows:
        groups.setdefault(row.jd_id, []).append(row)
    return list(groups.items())


def run_ragas_faithfulness_eval(
    cfg: RunConfig,
    run_id: str,
    output_path: Path,
    evaluate_fn: "Callable[[EvaluationDataset, LangchainLLMWrapper], pandas.DataFrame] | None" = None,
    sample_size: int | None = None,
    sample_seed: int = 42,
) -> FaithfulnessReport:
    all_rows = [row for row in build_rows(cfg, run_id) if row.item_type == "strength"]
    if not all_rows:
        raise SystemExit(
            f"No Ragas-evaluable data for run {run_id!r} -- either it has no per-JD assessments/ "
            "subdirectories, every assessment had zero strengths, or every candidate is missing "
            "from the corpus. See log output above for details."
        )

    population_size = len(all_rows)
    rows = all_rows if sample_size is None else stratified_sample(all_rows, sample_size, sample_seed)
    if sample_size is not None:
        logger.info(
            "Sampling %d/%d row(s) (seed=%d), an equal share per JD, instead of evaluating the full population",
            len(rows), population_size, sample_seed,
        )

    from langchain_ollama import ChatOllama
    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.llms import LangchainLLMWrapper

    judge_model = cfg.faithfulness_model or cfg.ollama_model

    def _build_llm() -> "LangchainLLMWrapper":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return LangchainLLMWrapper(
                ChatOllama(model=judge_model, base_url=cfg.ollama_base_url, temperature=0, num_ctx=cfg.ollama_num_ctx)
            )

    if evaluate_fn is None:
        def evaluate_fn(dataset, llm):
            return _default_evaluate_fn(dataset, llm, max_workers=cfg.ollama_num_parallel)

    processed_rows: list[FaithfulnessRow] = []
    ragas_scores: list[float] = []
    for jd_id, jd_rows in _group_by_jd(rows):
        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(user_input=row.user_input, response=row.response, retrieved_contexts=row.retrieved_contexts)
                for row in jd_rows
            ]
        )
        llm = _build_llm()
        result_df = evaluate_fn(dataset, llm)
        if result_df["response"].tolist() != [row.response for row in jd_rows]:
            raise RuntimeError(
                "Ragas returned results in an unexpected order (response text doesn't match row order) "
                "-- refusing to merge scores positionally, since that would silently misattribute them."
            )
        processed_rows.extend(jd_rows)
        ragas_scores.extend(result_df["faithfulness"].tolist())

        report = merge_and_report(run_id, processed_rows, ragas_scores, population_size=population_size)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        logger.info(
            "Checkpoint after jd_id=%r: wrote %d/%d sampled row(s) (of %d in the full population) to %s",
            jd_id, len(processed_rows), len(rows), population_size, output_path,
        )

    return report
