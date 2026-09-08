"""Compiles docs/bare_jrnl.tex with pdflatex (twice, to settle floats and
cross-references) and reports which page "ACKNOWLEDGMENT" first appears on,
using it as the boundary for where the main body (Title through end of
Section VI. Conclusion) ends -- the paper's Acknowledgment section
immediately follows the Conclusion, with no other section between them.
Exits nonzero if that page is greater than 5, per
docs/superpowers/specs/2026-09-08-paper-tex-5-page-shortening-design.md's
target.

Run with: python3 scripts/measure_paper_tex_body_pages.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
TEX_PATH = DOCS_DIR / "bare_jrnl.tex"
BODY_PAGE_LIMIT = 5


def compile_to_pdf(tex_path: Path, work_dir: Path) -> Path:
    for source_name in ("bare_jrnl.tex", "IEEEtran.cls", "IEEEtran.bst"):
        shutil.copy(DOCS_DIR / source_name, work_dir / source_name)
    figures_link = work_dir / "figures"
    if not figures_link.exists():
        figures_link.symlink_to(DOCS_DIR / "figures")
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "bare_jrnl.tex"],
            cwd=work_dir,
            check=True,
            capture_output=True,
        )
    return work_dir / "bare_jrnl.pdf"


def find_body_end_page(pdf_path: Path) -> int:
    page_count = int(
        subprocess.run(
            ["pdfinfo", str(pdf_path)], check=True, capture_output=True, text=True
        ).stdout.split("Pages:")[1].split()[0]
    )
    for page in range(1, page_count + 1):
        text = subprocess.run(
            ["pdftotext", "-f", str(page), "-l", str(page), str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if "ACKNOWLEDGMENT" in text:
            return page
    raise ValueError(f"'ACKNOWLEDGMENT' not found in any of the {page_count} rendered pages")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = compile_to_pdf(TEX_PATH, Path(tmp_dir))
        body_end_page = find_body_end_page(pdf_path)

    print(f"Body (Title through Conclusion) ends on page {body_end_page} "
          f"(ACKNOWLEDGMENT starts there). Target: <= {BODY_PAGE_LIMIT}.")
    if body_end_page > BODY_PAGE_LIMIT:
        print(
            f"OVER TARGET by {body_end_page - BODY_PAGE_LIMIT} page(s). Do not silently "
            "cut a figure, a table row, or a reference to close the gap -- see "
            "docs/superpowers/specs/2026-09-08-paper-tex-5-page-shortening-design.md's "
            "Phase 3 / Open parameters."
        )
        sys.exit(1)
    print("Within target.")


if __name__ == "__main__":
    main()
