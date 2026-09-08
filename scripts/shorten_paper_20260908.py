"""Shortens docs/candidate-ranking-paper-new.docx's main body (Title through
end of Section VI. Conclusion) from ~6.85 pages to 5 pages or fewer, per
docs/superpowers/specs/2026-09-08-paper-5-page-shortening-design.md.

Cuts: Figures 5 and 6 (each verified redundant with existing data -- Fig. 5
duplicates Fig. 4's endpoints, Fig. 6 duplicates Table II's rows), a
condensed Literature Review (all 19 citations kept), and light
prose-tightening in the Introduction. No number, claim, or citation is
dropped -- only redundant charts and verbose wording.

docs/ is gitignored, so SHORTENING_BACKUP_PATH is the only surviving copy of
the pre-shortening (but post-accuracy-corrections) paper. Distinct from the
older docs/candidate-ranking-paper-new-backup.docx, which must not be
touched by this script.

Run with: uv run python scripts/shorten_paper_20260908.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import docx
from docx.oxml.ns import qn

from scripts.apply_paper_corrections import find_paragraph, replace_paragraph_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_PATH = PROJECT_ROOT / "docs" / "candidate-ranking-paper-new.docx"
SHORTENING_BACKUP_PATH = PAPER_PATH.with_name("candidate-ranking-paper-new-preshortening-backup.docx")
FIGURES_ARCHIVE_DIR = PROJECT_ROOT / "docs" / "figures" / "_archive"


def delete_paragraph(paragraph) -> None:
    element = paragraph._p
    element.getparent().remove(element)


def paragraph_element_immediately_before(document, paragraph):
    body_children = list(document.element.body)
    index = body_children.index(paragraph._p)
    return body_children[index - 1]


def remove_figure_and_fold_text(
    document, caption_marker: str, sentence_marker: str, new_sentence: str, archive_path: Path
) -> None:
    """Deletes a figure's image + caption paragraph, keeping every number the
    figure carried by rewriting the paragraph that used to narrate it
    (`sentence_marker`) into a self-contained sentence (`new_sentence`) that
    no longer references the now-deleted figure. Archives the original image
    bytes to `archive_path` first (only if not already archived, so a second
    run doesn't overwrite the archive with nothing -- the image is gone from
    the document after the first run)."""
    caption_paragraph = find_paragraph(document, caption_marker)
    image_element = paragraph_element_immediately_before(document, caption_paragraph)
    if not image_element.findall(".//" + qn("w:drawing")):
        raise ValueError(
            f"paragraph immediately before the {caption_marker!r} caption has no image -- "
            "document structure may have changed, refusing to delete"
        )

    blip = image_element.findall(".//" + qn("a:blip"))[0]
    rel_id = blip.get(qn("r:embed"))
    image_part = document.part.related_parts[rel_id]
    if not archive_path.exists():
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(image_part.blob)

    sentence_paragraph = find_paragraph(document, sentence_marker)
    replace_paragraph_text(sentence_paragraph, new_sentence)

    delete_paragraph(caption_paragraph)
    image_element.getparent().remove(image_element)


FIGURE_5_NEW_SENTENCE = (
    "Final Kendall's Tau per role, averaged across stability repeats, spans 0.930 "
    "(frontend-engineer, n=7) to 0.979 (data-engineer, n=77), with smaller shortlists "
    "scoring lower — consistent with Kendall's Tau being a noisier estimator over "
    "fewer candidates rather than evidence that smaller pools rank less reliably."
)

FIGURE_5_ARCHIVE_PATH = FIGURES_ARCHIVE_DIR / "table7_kendall_tau_per_role.removed.png"


def cut_figure_5(document) -> None:
    remove_figure_and_fold_text(
        document,
        caption_marker="Fig. 5. Final Kendall",
        sentence_marker="As Fig. 5 shows",
        new_sentence=FIGURE_5_NEW_SENTENCE,
        archive_path=FIGURE_5_ARCHIVE_PATH,
    )


FIGURE_6_NEW_SENTENCE = (
    "As Table II shows, per-role Faithfulness scores range from full-stack-engineer "
    "(lowest, 0.764) to devops-engineer (highest, 0.944), bracketing the run-wide mean "
    "of 0.880."
)

FIGURE_6_ARCHIVE_PATH = FIGURES_ARCHIVE_DIR / "table5_ragas_faithfulness.removed.png"


def cut_figure_6(document) -> None:
    remove_figure_and_fold_text(
        document,
        caption_marker="Fig. 6. Faithfulness scores",
        sentence_marker="As shown in Fig. 6",
        new_sentence=FIGURE_6_NEW_SENTENCE,
        archive_path=FIGURE_6_ARCHIVE_PATH,
    )




_LITERATURE_REVIEW_II_A_MARKER = "Most recent papers apply LLMs to extraction"
_LITERATURE_REVIEW_II_A_NEW = (
    "Most recent papers apply LLMs to extraction and pointwise scoring inside RAG or multi-agent pipelines, "
    "without a comparative ranking step. Tran and Tran [2] build a RAG resume agent (Faithfulness 0.882) but "
    "do not rank applicants. Lo et al. [3] use a four-agent CrewAI scorer; Chowdhury et al. [4] fine-tune "
    "open-LLM sub-dimension agents that collapse without data augmentation (R² as low as −8.15); "
    "Jahan et al. [5]’s zero-shot multi-agent consensus vote sees recall drop to 27.9% from requiring "
    "agent agreement; Walid et al. [7] tune an ensemble for field-extraction accuracy, not ranking; Gera et al. "
    "[8] report only a single-resume case study with no benchmark; and Chong et al. [17] pair an LLM "
    "relevance scorer with a fine-tuned-BERT chatbot (99.15% Q&A accuracy) but report no accuracy metric for "
    "the scoring/ranking component itself. None validates hallucination mitigation with a verified retry loop "
    "against the applicant’s own extracted skills, as this paper’s constrained, retry-on-contradiction "
    "generation does. Synthesizing 141 such papers, Dasaklis et al. [15] find hallucination and cross-session "
    "score instability to be recurring, unresolved risks across the field and call for dedicated "
    "hallucination-tracking metrics — precisely the mechanism this paper’s retry-on-contradiction loop "
    "provides."
)

_LITERATURE_REVIEW_II_B_MARKER = "Li et al. [6] deploy a production"
_LITERATURE_REVIEW_II_B_NEW = (
    "Li et al. [6] deploy a production talent-search ranker at Alibaba where the LLM only extracts hiring "
    "preferences for a Mixture-of-Experts network’s pointwise predicted rates, not a comparative judgment. "
    "Hoque et al. [9] combine a fine-tuned DistilRoBERTa classifier with Fuzzy TOPSIS under fixed, "
    "expert-elicited weights, reaching high agreement with human rankings (NDCG = 0.926) but on only 100 "
    "profiles from a single job family. Rosenberger et al. [13] embed resumes and ESCO job descriptions into "
    "a shared space and rank jobs by cosine similarity, validated on only five resumes and ten HR experts by "
    "the authors’ own admission. Xue et al. [14] fuse BERT-based semantic features with a graph neural "
    "network into a single pointwise fit score (94.6% accuracy on binary hire/no-hire classification); neither "
    "[13] nor [14] compares candidates against each other within a shared context the way a listwise judgment "
    "would. This paper instead ranks applicants through a position-robust listwise tournament rather than "
    "pointwise or fixed-weight scoring."
)

_LITERATURE_REVIEW_II_C_MARKER = "Yuksel et al. [1] are the closest prior work"
_LITERATURE_REVIEW_II_C_NEW = (
    "Yuksel et al. [1] are the closest prior work: an LLM ranks a small subset of applicants at once, "
    "aggregating orderings into global utilities via a Plackett-Luce model under an active-learning loop with "
    "a Monte Carlo knowledge-gradient acquisition strategy. It reports a real result against human judgment "
    "(87% of ratings within one rubric level; peak NDCG@25% of 0.5703), but does not name its LLM, disclose "
    "its dataset, report run-to-run variance, discuss identifier-drift failure, or self-correct hallucinated "
    "claims beyond a single internal prompt check — and motivates its listwise design by noting pairwise "
    "comparison “does not scale well to large applicant pools” without addressing how many applicants "
    "reach the tournament in the first place. Section III gives the full structured comparison."
)


def cut_literature_review(document) -> None:
    replace_paragraph_text(
        find_paragraph(document, _LITERATURE_REVIEW_II_A_MARKER), _LITERATURE_REVIEW_II_A_NEW
    )
    replace_paragraph_text(
        find_paragraph(document, _LITERATURE_REVIEW_II_B_MARKER), _LITERATURE_REVIEW_II_B_NEW
    )
    replace_paragraph_text(
        find_paragraph(document, _LITERATURE_REVIEW_II_C_MARKER), _LITERATURE_REVIEW_II_C_NEW
    )

def main() -> None:
    if not SHORTENING_BACKUP_PATH.exists():
        shutil.copy2(PAPER_PATH, SHORTENING_BACKUP_PATH)
        print(f"Backed up {PAPER_PATH} -> {SHORTENING_BACKUP_PATH}")
    else:
        print(f"Backup already exists at {SHORTENING_BACKUP_PATH}, not overwriting")

    document = docx.Document(PAPER_PATH)

    cut_figure_5(document)
    cut_figure_6(document)
    cut_literature_review(document)
    # cut_* calls are added here by later tasks, in this order.

    document.save(PAPER_PATH)
    print(f"Saved shortened paper to {PAPER_PATH}")


if __name__ == "__main__":
    main()
