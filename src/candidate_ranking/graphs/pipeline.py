# src/candidate_ranking/graphs/pipeline.py
from __future__ import annotations

import logging
import operator
from typing import TYPE_CHECKING, Annotated, Callable, Literal, TypedDict

import numpy as np
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph
from langgraph.types import Send

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from candidate_ranking.scoring.jev_client import JevClient

from candidate_ranking.config import RunConfig
from candidate_ranking.models import Assessment, Candidate, JobDescription, JobProfile
from candidate_ranking.output.writer import write_jd_ranking
from candidate_ranking.scoring.job_profile import (
    JobProfileGenerationError,
    load_or_generate_job_profile,
)
from candidate_ranking.scoring.llm_assessment import (
    AssessmentGenerationError,
    load_or_generate_assessment,
)
from candidate_ranking.scoring.active_listwise_ranking import rank_active_listwise
from candidate_ranking.scoring.skills import shortlist_candidates

logger = logging.getLogger(__name__)


class AssessmentResult(TypedDict):
    jd_id: str
    candidate_id: str
    status: Literal["ok", "failed"]
    assessment: Assessment | None
    error: str | None


class PipelineState(TypedDict):
    jds: list[JobDescription]
    candidates: list[Candidate]
    job_profiles: Annotated[dict[str, JobProfile], operator.or_]
    shortlists: Annotated[dict[str, list[str]], operator.or_]
    assessment_results: Annotated[list[AssessmentResult], operator.add]


def assessments_by_jd(assessment_results: list[AssessmentResult]) -> dict[str, dict[str, Assessment]]:
    grouped: dict[str, dict[str, Assessment]] = {}
    for result in assessment_results:
        if result["status"] != "ok":
            continue
        grouped.setdefault(result["jd_id"], {})[result["candidate_id"]] = result["assessment"]
    return grouped


def build_pipeline_graph(
    cfg: RunConfig,
    job_profile_chain: Runnable,
    jev_client: "JevClient",
    free_text_chain: Runnable,
    ranking_llm: "BaseChatModel",
    skill_index: np.ndarray,
    skill_row_map: list[tuple[str, str]],
    skill_embedder: Callable[[list[str]], np.ndarray],
    run_id: str,
    skill_match_threshold: float = 0.8,
    min_skill_matches: int = 3,
) -> StateGraph:
    job_profile_cache_path = cfg.cache_dir / "job_profile.json"

    def _noop(state: PipelineState) -> dict:
        return {}

    def fanout_job_profile_extraction(state: PipelineState) -> list[Send]:
        return [Send("extract_job_profile_for_jd", {"jd": jd}) for jd in state["jds"]]

    def extract_job_profile_for_jd(payload: dict) -> dict:
        jd = payload["jd"]
        try:
            profile = load_or_generate_job_profile(jd, job_profile_chain, cfg.ollama_model, job_profile_cache_path)
        except JobProfileGenerationError as exc:
            logger.warning("Excluding %s from this run: %s", jd.id, exc)
            return {}
        return {"job_profiles": {jd.id: profile}}

    def fanout_shortlist(state: PipelineState) -> list[Send]:
        return [
            Send("build_shortlist_for_jd", {"jd": jd, "profile": state["job_profiles"][jd.id]})
            for jd in state["jds"]
            if jd.id in state["job_profiles"]
        ]

    def build_shortlist_for_jd(payload: dict) -> dict:
        jd, profile = payload["jd"], payload["profile"]
        candidate_ids = shortlist_candidates(
            jd, profile, skill_index, skill_row_map, skill_embedder,
            threshold=skill_match_threshold, min_matches=min_skill_matches,
        )
        return {"shortlists": {jd.id: candidate_ids}}

    def fanout_assessments(state: PipelineState) -> list[Send]:
        candidates_by_id = {c.id: c for c in state["candidates"]}
        return [
            Send(
                "generate_assessment_for_pair",
                {
                    "jd": jd,
                    "candidate": candidates_by_id[cid],
                    "job_profile": state["job_profiles"].get(jd.id),
                },
            )
            for jd in state["jds"]
            for cid in state["shortlists"].get(jd.id, [])
        ]

    def generate_assessment_for_pair(payload: dict) -> dict:
        jd, candidate = payload["jd"], payload["candidate"]
        profile = payload.get("job_profile")
        try:
            assessment = load_or_generate_assessment(
                jd, candidate, profile, jev_client, free_text_chain, cfg.ollama_model, cfg.cache_dir
            )
            result: AssessmentResult = {
                "jd_id": jd.id, "candidate_id": candidate.id, "status": "ok",
                "assessment": assessment, "error": None,
            }
        except AssessmentGenerationError as exc:
            result = {
                "jd_id": jd.id, "candidate_id": candidate.id, "status": "failed",
                "assessment": None, "error": str(exc),
            }
        return {"assessment_results": [result]}

    def fanout_rank_and_format(state: PipelineState) -> list[Send]:
        grouped = assessments_by_jd(state["assessment_results"])
        return [
            Send("rank_and_format_jd_node", {"jd": jd, "assessments": grouped.get(jd.id, {})})
            for jd in state["jds"]
        ]

    def rank_and_format_jd_node(payload: dict) -> dict:
        jd: JobDescription = payload["jd"]
        all_assessments: dict[str, Assessment] = payload["assessments"]

        if not all_assessments:
            logger.warning("No assessed candidates for JD %s — skipping ranking", jd.id)
            write_jd_ranking(cfg.runs_dir / run_id, jd, {}, {})
            return {}

        utilities = rank_active_listwise(
            jd,
            all_assessments,
            ranking_llm,
            subset_size=cfg.ranking_subset_size,
            max_iterations=cfg.ranking_max_iterations,
            min_delta_u=cfg.ranking_min_delta_u,
            pl_prior_variance=cfg.ranking_pl_prior_variance,
        )

        write_jd_ranking(cfg.runs_dir / run_id, jd, all_assessments, utilities)
        return {}

    graph = StateGraph(PipelineState)
    graph.add_node("start", _noop)
    graph.add_node("extract_job_profile_for_jd", extract_job_profile_for_jd)
    graph.add_node("profile_barrier", _noop)
    graph.add_node("build_shortlist_for_jd", build_shortlist_for_jd)
    graph.add_node("shortlist_barrier", _noop)
    graph.add_node("generate_assessment_for_pair", generate_assessment_for_pair)
    graph.add_node("assessments_barrier", _noop)
    graph.add_node("rank_and_format_jd_node", rank_and_format_jd_node)

    graph.set_entry_point("start")
    graph.add_conditional_edges("start", fanout_job_profile_extraction, ["extract_job_profile_for_jd"])
    graph.add_edge("extract_job_profile_for_jd", "profile_barrier")
    graph.add_conditional_edges("profile_barrier", fanout_shortlist, ["build_shortlist_for_jd"])
    graph.add_edge("build_shortlist_for_jd", "shortlist_barrier")
    graph.add_conditional_edges("shortlist_barrier", fanout_assessments, ["generate_assessment_for_pair"])
    graph.add_edge("generate_assessment_for_pair", "assessments_barrier")
    graph.add_conditional_edges("assessments_barrier", fanout_rank_and_format, ["rank_and_format_jd_node"])
    graph.add_edge("rank_and_format_jd_node", END)

    return graph
