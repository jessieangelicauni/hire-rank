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


def main() -> None:
    if not SHORTENING_BACKUP_PATH.exists():
        shutil.copy2(PAPER_PATH, SHORTENING_BACKUP_PATH)
        print(f"Backed up {PAPER_PATH} -> {SHORTENING_BACKUP_PATH}")
    else:
        print(f"Backup already exists at {SHORTENING_BACKUP_PATH}, not overwriting")

    document = docx.Document(PAPER_PATH)

    # cut_* calls are added here by later tasks, in this order.

    document.save(PAPER_PATH)
    print(f"Saved shortened paper to {PAPER_PATH}")


if __name__ == "__main__":
    main()
