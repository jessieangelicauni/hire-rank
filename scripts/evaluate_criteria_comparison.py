"""Compare the automatically-optimized per-requirement criteria against the hand-written
baseline on a held-out test split, reporting mean confidence, accuracy against
Claude proxy labels, and a paired Wilcoxon signed-rank test.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import anthropic
from scipy.stats import wilcoxon

from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.evaluation.criteria_optimization import (
    CriteriaEvalRecord,
    RequirementTriple,
    evaluate_criteria,
    load_assessments_by_jd,
    triple_key,
)
from candidate_ranking.evaluation.proxy_labeler import ProxyLabelClient
from candidate_ranking.ingestion.cv import load_candidates
from candidate_ranking.ingestion.jd import load_job_descriptions
from candidate_ranking.models import Candidate, JobDescription
from candidate_ranking.scoring.assessment import _REQUIREMENT_FIT_CRITERIA
from candidate_ranking.scoring.jev_client import JevClient


def _accuracy(records: list[CriteriaEvalRecord], proxy_labels: dict[str, int]) -> float:
    agreements = [abs(r.score - proxy_labels[triple_key(r.triple)]) <= 1 for r in records]
    return statistics.mean(agreements) if agreements else 0.0


def _rank_biserial(first: list[float], second: list[float]) -> float:
    diffs = [b - a for a, b in zip(first, second)]
    positive = sum(1 for d in diffs if d > 0)
    negative = sum(1 for d in diffs if d < 0)
    total = positive + negative
    return (positive - negative) / total if total else 0.0


def load_corpus(cfg: RunConfig, jd_ids: list[str]) -> tuple[dict[str, JobDescription], dict[str, Candidate]]:
    jds_by_id = {jd.id: jd for jd in load_job_descriptions(cfg.jd_dir) if jd.id in jd_ids}
    candidates_by_id = {c.id: c for c in load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")}
    cv_skills_cache = json.loads((cfg.cache_dir / "cv_skills.json").read_text(encoding="utf-8"))
    for cid, candidate in candidates_by_id.items():
        if cid in cv_skills_cache:
            candidates_by_id[cid] = candidate.model_copy(update={"skills": cv_skills_cache[cid]["skills"]})
    return jds_by_id, candidates_by_id


def compare_criteria(
    baseline_records: list[CriteriaEvalRecord],
    optimized_records: list[CriteriaEvalRecord],
    proxy_labels: dict[str, int],
) -> dict:
    baseline_confidences = [r.confidence for r in baseline_records]
    optimized_confidences = [r.confidence for r in optimized_records]

    wilcoxon_result = None
    # Check if there are any non-zero differences
    diffs = [b - a for a, b in zip(baseline_confidences, optimized_confidences)]
    if any(d != 0 for d in diffs):
        try:
            stat, p_value = wilcoxon(baseline_confidences, optimized_confidences)
            wilcoxon_result = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "rank_biserial_r": _rank_biserial(baseline_confidences, optimized_confidences),
            }
        except ValueError:
            pass  # scipy.wilcoxon raised an error

    return {
        "baseline": {
            "mean_confidence": statistics.mean(baseline_confidences) if baseline_confidences else 0.0,
            "accuracy": _accuracy(baseline_records, proxy_labels),
        },
        "optimized": {
            "mean_confidence": statistics.mean(optimized_confidences) if optimized_confidences else 0.0,
            "accuracy": _accuracy(optimized_records, proxy_labels),
        },
        "wilcoxon": wilcoxon_result,
    }


def main(run_id: str) -> None:
    cfg = apply_env_overrides(RunConfig.full(PROJECT_ROOT))
    run_dir = cfg.runs_dir / run_id
    opt_path = run_dir / "evaluation" / "criteria_optimization.json"
    opt_result = json.loads(opt_path.read_text(encoding="utf-8"))
    test_triples = [RequirementTriple.model_validate(t) for t in opt_result["test_triples"]]
    optimized_criteria = opt_result["best_criteria"]

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assessments_by_jd = load_assessments_by_jd(run_dir, manifest["jd_ids"])

    if not cfg.jev_api_key:
        raise RuntimeError("Set CANDIDATE_RANKING_JEV_API_KEY before running.")
    jev_client = JevClient(api_token=cfg.jev_api_key)

    if not cfg.anthropic_api_key:
        raise RuntimeError("Set CANDIDATE_RANKING_ANTHROPIC_API_KEY before running.")
    anthropic_client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    proxy_labeler = ProxyLabelClient(
        anthropic_client, cache_path=cfg.cache_dir / "proxy_labels.json", model=cfg.proxy_label_model
    )

    jds_by_id, candidates_by_id = load_corpus(cfg, manifest["jd_ids"])
    proxy_labels = {
        triple_key(t): proxy_labeler.label(t, jds_by_id[t.job_description_id], candidates_by_id[t.candidate_id])
        for t in test_triples
    }

    baseline_records = evaluate_criteria(
        _REQUIREMENT_FIT_CRITERIA, test_triples, jds_by_id, candidates_by_id, jev_client
    )
    optimized_records = evaluate_criteria(
        optimized_criteria, test_triples, jds_by_id, candidates_by_id, jev_client
    )

    report = compare_criteria(baseline_records, optimized_records, proxy_labels)
    out_path = run_dir / "evaluation" / "criteria_comparison.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote comparison to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    main(args.run_id)
