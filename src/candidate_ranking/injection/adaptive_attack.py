from __future__ import annotations

import random
import re
from typing import Callable

import numpy as np

DEFAULT_SYNONYM_BANK: dict[str, list[str]] = {
    "disregard": ["ignore", "overlook", "dismiss"],
    "requirement": ["criterion", "qualification", "condition"],
    "criteria": ["benchmarks", "standards", "guidelines"],
    "evaluation": ["assessment", "review", "appraisal"],
    "instruction": ["directive", "guidance", "note"],
    "outstanding": ["excellent", "exceptional", "remarkable"],
    "ideal": ["perfect", "optimal", "flawless"],
    "accordingly": ["appropriately", "correspondingly", "suitably"],
}

_TRAILING_PUNCT_RE = re.compile(r"[.,;:!?)\"']+$")


def _split_trailing_punct(word: str) -> tuple[str, str]:
    match = _TRAILING_PUNCT_RE.search(word)
    if match is None:
        return word, ""
    return word[: match.start()], word[match.start():]


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _max_reference_similarity(vector: np.ndarray, reference_embeddings: np.ndarray) -> float:
    norms = np.linalg.norm(reference_embeddings, axis=1) * np.linalg.norm(vector)
    norms = np.where(norms == 0.0, 1e-9, norms)
    return float((reference_embeddings @ vector / norms).max())


def optimize_evasive_attack(
    seed_text: str,
    embedder: Callable[[list[str]], np.ndarray],
    reference_embeddings: np.ndarray,
    rng: random.Random,
    synonym_bank: dict[str, list[str]],
    max_iterations: int = 200,
    semantic_floor: float = 0.75,
) -> str:
    """Deterministic (given rng) hill-climb: repeatedly substitutes one word
    for a curated synonym, keeping the substitution only if it both (a)
    lowers the candidate's max cosine similarity to reference_embeddings and
    (b) keeps its similarity to the original seed_text at or above
    semantic_floor. No LLM calls -- only the embedder already used by the
    semantic filter."""
    words = seed_text.split(" ")
    seed_vector = np.array(embedder([seed_text])[0])

    original_cores: dict[int, tuple[str, str]] = {}
    for i, word in enumerate(words):
        core, trailing = _split_trailing_punct(word)
        if core.lower() in synonym_bank:
            original_cores[i] = (core, trailing)

    positions = list(original_cores)
    if not positions:
        return seed_text
    rng.shuffle(positions)

    current_words = list(words)
    current_score = _max_reference_similarity(seed_vector, reference_embeddings)

    for iteration in range(max_iterations):
        position = positions[iteration % len(positions)]
        core, trailing = original_cores[position]
        candidates = synonym_bank[core.lower()]
        candidate = candidates[(iteration // len(positions)) % len(candidates)]
        trial_words = list(current_words)
        trial_words[position] = _match_case(core, candidate) + trailing
        trial_text = " ".join(trial_words)

        trial_vector = np.array(embedder([trial_text])[0])
        if _cosine_similarity(trial_vector, seed_vector) < semantic_floor:
            continue

        trial_score = _max_reference_similarity(trial_vector, reference_embeddings)
        if trial_score < current_score:
            current_words = trial_words
            current_score = trial_score

    return " ".join(current_words)
