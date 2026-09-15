from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_PATH = PROJECT_ROOT / "docs" / "candidate-ranking-paper-new.docx"
BODY_PAGE_LIMIT = 5


def render_to_pdf(docx_path: Path, out_dir: Path) -> Path:
    subprocess.run(
        ["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(docx_path)],
        check=True,
        capture_output=True,
    )
    return out_dir / (docx_path.stem + ".pdf")


def find_body_end_page(pdf_path: Path) -> int:
    page_count = int(
        subprocess.run(
            ["pdfinfo", str(pdf_path)], check=True, capture_output=True, text=True
        ).stdout.split("Pages:")[1].split()[0]
    )
    for page in range(1, page_count + 1):
        text = subprocess.run(
            ["pdftotext", "-layout", "-f", str(page), "-l", str(page), str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if "Acknowledgments" in text:
            return page
    raise ValueError(f"'Acknowledgments' not found in any of the {page_count} rendered pages")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = render_to_pdf(PAPER_PATH, Path(tmp_dir))
        body_end_page = find_body_end_page(pdf_path)

    print(f"Body (Title through Conclusion) ends on page {body_end_page} "
          f"(Acknowledgments starts there). Target: <= {BODY_PAGE_LIMIT}.")
    if body_end_page > BODY_PAGE_LIMIT:
        print(
            f"OVER TARGET by {body_end_page - BODY_PAGE_LIMIT} page(s). See this plan's "
            "Task 7 Step 3 for next steps -- do not silently cut Figures 1-3, table rows, "
            "or references to close the gap."
        )
        sys.exit(1)
    print("Within target.")


if __name__ == "__main__":
    main()
