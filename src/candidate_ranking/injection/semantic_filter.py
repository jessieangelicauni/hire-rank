from __future__ import annotations

from typing import Callable

import numpy as np


def filter_suspicious_lines(
    text: str,
    embedder: Callable[[list[str]], np.ndarray],
    reference_embeddings: np.ndarray,
    threshold: float = 0.5587,
) -> tuple[str, int]:
    lines = text.split("\n")
    non_blank = [(i, line) for i, line in enumerate(lines) if line.strip()]
    if not non_blank:
        return text, 0
    indices, texts = zip(*non_blank)
    vectors = embedder(list(texts))
    similarities = vectors @ reference_embeddings.T
    max_similarities = similarities.max(axis=1)
    drop_indices = {indices[j] for j, sim in enumerate(max_similarities) if sim >= threshold}
    kept = [line for i, line in enumerate(lines) if i not in drop_indices]
    return "\n".join(kept), len(drop_indices)
