from __future__ import annotations

from typing import Callable

import numpy as np


def build_classifier_filter(
    model_name: str = "protectai/deberta-v3-base-prompt-injection-v2",
    injection_label: str = "INJECTION",
) -> Callable[[list[str]], np.ndarray]:
    """Loads the HF text-classification pipeline once; returns a callable
    with the same (lines -> scores) contract the semantic filter's embedder
    exposes, so it slots into the same filtering shape."""
    from transformers import pipeline

    classify = pipeline("text-classification", model=model_name, top_k=None)

    def score(lines: list[str]) -> np.ndarray:
        if not lines:
            return np.array([])
        raw_results = classify(lines)
        scores = []
        for result in raw_results:
            entries = result if isinstance(result, list) else [result]
            injection_entries = [e["score"] for e in entries if e["label"] == injection_label]
            scores.append(injection_entries[0] if injection_entries else 0.0)
        return np.array(scores, dtype="float32")

    return score


def filter_suspicious_lines_classifier(
    text: str,
    classify: Callable[[list[str]], np.ndarray],
    threshold: float = 0.5,
) -> tuple[str, int]:
    lines = text.split("\n")
    non_blank = [(i, line) for i, line in enumerate(lines) if line.strip()]
    if not non_blank:
        return text, 0
    indices, texts = zip(*non_blank)
    scores = classify(list(texts))
    drop_indices = {indices[j] for j, score in enumerate(scores) if score >= threshold}
    kept = [line for i, line in enumerate(lines) if i not in drop_indices]
    return "\n".join(kept), len(drop_indices)
