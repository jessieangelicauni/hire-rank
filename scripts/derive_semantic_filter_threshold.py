from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from candidate_ranking.injection.attack_corpus import INSTRUCTION_INJECTION_PARAPHRASES
from candidate_ranking.injection.semantic_filter_reference import REFERENCE_SUSPICIOUS_PHRASES
from candidate_ranking.scoring.skills import build_skill_embedder


def load_cv_lines(cache_path: Path) -> tuple[list[str], list[str]]:
    with open(cache_path, encoding="utf-8") as f:
        cache = json.load(f)
    lines: list[str] = []
    cv_ids: list[str] = []
    for cache_key, entry in cache.items():
        candidate = entry["candidate"]
        cv_id = candidate.get("id", cache_key)
        for line in candidate["raw_text"].split("\n"):
            if line.strip():
                lines.append(line)
                cv_ids.append(cv_id)
    return lines, cv_ids


def main(margin: float) -> None:
    cache_path = PROJECT_ROOT / "runs" / "_cache" / "cv.json"
    embedder = build_skill_embedder("sentence-transformers/all-MiniLM-L6-v2")
    reference_embeddings = embedder(REFERENCE_SUSPICIOUS_PHRASES)

    lines, cv_ids = load_cv_lines(cache_path)
    print(f"Loaded {len(lines)} non-blank lines from {len(set(cv_ids))} CVs in {cache_path}")

    vectors = embedder(lines)
    similarities = vectors @ reference_embeddings.T
    max_similarities = similarities.max(axis=1)

    ceiling_index = int(max_similarities.argmax())
    ceiling = float(max_similarities[ceiling_index])
    ceiling_line = lines[ceiling_index]
    ceiling_cv_id = cv_ids[ceiling_index]

    print(f"\nceiling (max similarity across all clean lines): {ceiling:.4f}")
    print(f"  line: {ceiling_line!r}")
    print(f"  from CV: {ceiling_cv_id}")

    threshold = round(ceiling + margin, 4)
    n_false_positives = int((max_similarities >= threshold).sum())
    fp_rate = n_false_positives / len(lines) if lines else 0.0
    print(f"\nthreshold (ceiling + margin={margin}): {threshold:.4f}")
    print(f"false positives at threshold: {n_false_positives}/{len(lines)} lines ({fp_rate:.4%})")

    eval_vectors = embedder([p["attack_text"] for p in INSTRUCTION_INJECTION_PARAPHRASES])
    eval_sims = (eval_vectors @ reference_embeddings.T).max(axis=1)
    caught = int((eval_sims >= threshold).sum())
    print(f"\nconfirmatory-only: eval-paraphrase catch rate at threshold={threshold:.4f}")
    for paraphrase, sim in zip(INSTRUCTION_INJECTION_PARAPHRASES, eval_sims):
        status = "CAUGHT" if sim >= threshold else "missed"
        print(f"  {sim:.4f} [{status}] {paraphrase['attack_text'][:60]!r}")
    print(f"caught: {caught}/{len(INSTRUCTION_INJECTION_PARAPHRASES)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margin", type=float, default=0.04)
    args = parser.parse_args()
    main(args.margin)
