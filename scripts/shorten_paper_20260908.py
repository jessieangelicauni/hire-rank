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


def main() -> None:
    if not SHORTENING_BACKUP_PATH.exists():
        shutil.copy2(PAPER_PATH, SHORTENING_BACKUP_PATH)
        print(f"Backed up {PAPER_PATH} -> {SHORTENING_BACKUP_PATH}")
    else:
        print(f"Backup already exists at {SHORTENING_BACKUP_PATH}, not overwriting")

    document = docx.Document(PAPER_PATH)

    cut_figure_5(document)
    cut_figure_6(document)
    # cut_* calls are added here by later tasks, in this order.

    document.save(PAPER_PATH)
    print(f"Saved shortened paper to {PAPER_PATH}")


if __name__ == "__main__":
    main()
