from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from pathlib import Path
from pydantic import ValidationError

from candidate_ranking.models import (
    Assessment, Candidate, EvaluationCriterion, JobProfile, RequirementEvaluation,
)
from candidate_ranking.scoring.jev_client import JEV_MODEL_ID, JevClientError, JevQuestion
from candidate_ranking.scoring.shortlist import (
    DEFAULT_SIMILARITY_THRESHOLD, _model, _split_sentences, _word_pattern,
)
logger = logging.getLogger(__name__)

_cache_lock = threading.Lock()

_SKILL_EVIDENCE_CRITERIA: dict[str, str] = {
    "strong": (
        "Skill demonstrated in work experience with a specific system, project, or measurable outcome. "
        "E.g. 'Built REST APIs using Django at PT X for 3 years' -> STRONG for Django."
    ),
    "partial": (
        "Skill mentioned, but not clearly demonstrated: either used in an experience context without "
        "a named project/system, or listed only in a Skills/Summary/Certifications section. "
        "E.g. 'Used Python for data processing tasks' -> PARTIAL."
    ),
    "none": (
        "No mention of the skill anywhere in the CV. "
        "Check ALL sections (experience, projects, education, skills, summary, certifications, volunteer) "
        "before answering NONE. "
        "Remember implied skills: 'Django' implies 'Python'; 'Spring Boot' implies 'Java'; "
        "'React' implies 'JavaScript'; 'Kubernetes' implies 'Docker'."
    ),
}

_GENERIC_SKILL_INSTRUCTION_TEMPLATE = (
    "Role context: {jp_role}. Assess the evidence strength for the skill '{skill}' in this CV, "
    "against the STRONG/PARTIAL/NONE definitions given below. Apply them strictly and "
    "consistently regardless of role or industry."
)


def fallback_evaluation_criterion(jp_id: str, label: str) -> EvaluationCriterion:
    """Generic, role-agnostic evidence rubric for one evaluation item — used when
    Call 2 (scoring/job_profile.py) fails or returns no criteria at all."""
    return EvaluationCriterion(
        label=label,
        instructions=_GENERIC_SKILL_INSTRUCTION_TEMPLATE.format(jp_role=jp_id, skill=label),
        strong=_SKILL_EVIDENCE_CRITERIA["strong"],
        partial=_SKILL_EVIDENCE_CRITERIA["partial"],
        none=_SKILL_EVIDENCE_CRITERIA["none"],
    )


def _build_jev_questions(jp: JobProfile) -> list[JevQuestion]:
    return [
        JevQuestion(
            key=f"criterion_{i}_evidence_strength",
            kind="choice",
            instructions=criterion.instructions,
            criteria={
                "strong": criterion.strong, "partial": criterion.partial, "none": criterion.none,
            },
        )
        for i, criterion in enumerate(jp.evaluation_criteria)
    ]


_MAX_EVIDENCE_SENTENCES = 3


def _find_evidence_sentence(candidate: Candidate, requirement: str) -> str | None:
    """Evidence straight from the candidate's own CV text, never from an LLM's
    attribution of it: split into sentences (`shortlist.py`'s own splitter, so
    the same unit of evidence used to decide shortlist admission is used here),
    then

    1. a per-word, not whole-phrase, literal match (so a requirement like
       "Kubernetes (Containers)" or "AWS Services (EC2, EBS, S3, Lambda)" is
       still grounded by whichever single word in it the CV actually uses) --
       every sentence containing any requirement word is genuine evidence, so
       all of them are kept (joined, most relevant first per BM25) rather than
       arbitrarily picking just one;
    2. if no requirement word appears anywhere, cosine similarity between the
       requirement and every sentence (same embedding model and threshold as
       the shortlist gate) -- catches a requirement phrased as a broad
       category ("Cloud-native technologies and frameworks") with no word in
       common with any single sentence.

    None if neither finds a sentence clearing the bar -- no guessed snippet."""
    sentences = _split_sentences(candidate.raw_text)
    if not sentences:
        return None

    query_tokens = requirement.lower().split()
    patterns = [_word_pattern(t) for t in query_tokens]
    matches = [s for s in sentences if any(p.search(s) for p in patterns)]
    if matches:
        if len(matches) > 1:
            from rank_bm25 import BM25Okapi
            corpus = [s.lower().split() for s in matches]
            scores = BM25Okapi(corpus).get_scores(query_tokens)
            matches = [s for s, _ in sorted(zip(matches, scores), key=lambda pair: -pair[1])]
        return " ".join(matches[:_MAX_EVIDENCE_SENTENCES])

    vectors = _model().encode(sentences, normalize_embeddings=True)
    req_vector = _model().encode([requirement], normalize_embeddings=True)
    similarities = (req_vector @ vectors.T)[0]
    best_idx = int(similarities.argmax())
    if similarities[best_idx] >= DEFAULT_SIMILARITY_THRESHOLD:
        return sentences[best_idx]
    return None


class AssessmentGenerationError(Exception):
    pass


def _answers_to_assessment(
    answers: list,
    jp: JobProfile,
    candidate: Candidate,
) -> Assessment:
    by_key = {a.key: a for a in answers}

    evaluations = []
    for i, criterion in enumerate(jp.evaluation_criteria):
        answer = by_key.get(f"criterion_{i}_evidence_strength")
        if answer is None:
            logger.warning(
                "Dropped Jev answer for criterion_%d (%r) on %s/%s; skipping evaluation.",
                i, criterion.label, jp.id, candidate.id,
            )
            continue
        # No evidence text for "none" -- would otherwise look like
        # supporting evidence for a criterion Jev found none for.
        evidence = _find_evidence_sentence(candidate, criterion.label) if answer.value != "none" else None
        evaluations.append(RequirementEvaluation(
            requirement=criterion.label,
            evidence=evidence or "",
            evidence_strength=answer.value,
            probabilities=answer.probabilities,
        ))

    return Assessment(
        job_profile_id=jp.id,
        candidate_id=candidate.id,
        evaluations=evaluations,
    )


def _assessment_cache_key(
    jp: JobProfile,
    candidate: Candidate,
    questions: list,
) -> str:
    questions_repr = json.dumps(
        [q.model_dump() for q in questions], sort_keys=True
    )
    digest_input = (
        f"{jp.id}||{jp.raw_text}||{candidate.id}||{candidate.raw_text}"
        f"||{questions_repr}||{JEV_MODEL_ID}"
    )
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


def generate_assessment(
    jp: JobProfile,
    candidate: Candidate,
    jev_client: "JevClient",
) -> Assessment:
    questions = _build_jev_questions(jp)

    try:
        answers = jev_client.evaluate(candidate.raw_text, questions)
    except JevClientError as exc:
        raise AssessmentGenerationError(
            f"Jev call failed for {jp.id}/{candidate.id}: {exc}"
        ) from exc

    return _answers_to_assessment(answers, jp, candidate)


def _assessment_cache_path(cache_dir: Path, jp_id: str, candidate_id: str) -> Path:
    return cache_dir / "assessments" / jp_id / f"{candidate_id}.json"


def load_or_generate_assessment(
    jp: JobProfile,
    candidate: Candidate,
    jev_client: "JevClient",
    cache_dir: Path,
) -> Assessment:
    questions = _build_jev_questions(jp)
    path = _assessment_cache_path(cache_dir, jp.id, candidate.id)
    key = _assessment_cache_key(jp, candidate, questions)

    with _cache_lock:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("cache_key") == key:
                    return Assessment.model_validate(data["assessment"])
            except (json.JSONDecodeError, KeyError, ValidationError, OSError) as exc:
                logger.warning("Discarding unreadable assessment cache %s: %s", path, exc)

    assessment = generate_assessment(jp, candidate, jev_client)

    with _cache_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(
            json.dumps(
                {"cache_key": key, "assessment": assessment.model_dump()}, indent=2
            ),
            encoding="utf-8",
        )
        tmp.replace(path)
    return assessment


def filter_assessable_candidates(candidates: list[Candidate]) -> list[Candidate]:
    assessable = [c for c in candidates if c.parse_status == "ok"]
    excluded = len(candidates) - len(assessable)
    if excluded > 0:
        logger.warning("Excluded %d candidate(s) with failed CV parsing", excluded)
    return assessable
