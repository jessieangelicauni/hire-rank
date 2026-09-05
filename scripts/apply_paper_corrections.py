"""Applies the paper-accuracy corrections from
docs/superpowers/specs/2026-09-05-paper-accuracy-corrections-design.md to
docs/candidate-ranking-paper.docx.

docs/ is not git-tracked, so this script backs up the original file before
overwriting it. Each fix_* function is independently unit-tested against a
small synthetic document in tests/test_apply_paper_corrections.py; main()
runs all of them in order against the real paper.

Run with: uv run python scripts/apply_paper_corrections.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import docx
from docx.shared import Inches

PAPER_PATH = Path(__file__).resolve().parents[1] / "docs" / "candidate-ranking-paper.docx"
BACKUP_PATH = PAPER_PATH.with_name(PAPER_PATH.stem + ".backup" + PAPER_PATH.suffix)


def find_paragraph(document, contains: str):
    for paragraph in document.paragraphs:
        if contains in paragraph.text:
            return paragraph
    raise ValueError(f"no paragraph found containing: {contains!r}")


def replace_paragraph_text(paragraph, new_text: str) -> None:
    if not paragraph.runs:
        paragraph.add_run(new_text)
        return
    paragraph.runs[0].text = new_text
    for run in paragraph.runs[1:]:
        run.text = ""


def insert_paragraph_before(anchor, text: str, bold: bool = False, style: str = "Body Text"):
    try:
        new_paragraph = anchor.insert_paragraph_before(text, style=style)
    except KeyError:
        new_paragraph = anchor.insert_paragraph_before(text)
    if bold and new_paragraph.runs:
        new_paragraph.runs[0].bold = True
    return new_paragraph


def insert_table_before(document, anchor, rows: int, cols: int, col_widths_in: list[float] | None = None):
    table = document.add_table(rows=rows, cols=cols)
    try:
        table.style = document.styles["Table Grid"]
    except KeyError:
        pass
    anchor._p.addprevious(table._tbl)
    if col_widths_in is not None:
        table.autofit = False
        for row in table.rows:
            for cell, width in zip(row.cells, col_widths_in):
                cell.width = Inches(width)
    return table


def main() -> None:
    shutil.copy2(PAPER_PATH, BACKUP_PATH)
    print(f"Backed up {PAPER_PATH} -> {BACKUP_PATH}")

    document = docx.Document(PAPER_PATH)

    # fix_* calls are added here by later tasks, in this order:
    # fix_abstract_and_contributions(document)
    # fix_methods_iii_c_d_e(document)
    # fix_experimental_setup_and_table_ii(document)
    # insert_results_headers_and_ranking_failure_table(document)
    # fix_conclusion_and_limitations(document)
    # insert_weakness_retry_audit_table(document)
    # apply_table_and_figure_layout_fixes(document)

    document.save(PAPER_PATH)
    print(f"Saved corrections to {PAPER_PATH}")


if __name__ == "__main__":
    main()
