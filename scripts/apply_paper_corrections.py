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
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
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


ABSTRACT_TEXT = (
    "Abstract—Talent-acquisition systems increasingly pair large language models (LLMs) with statistical ranking "
    "to shortlist and order job applicants. The most closely related recent system combines an LLM-driven active "
    "listwise tournament with Plackett-Luce aggregation for human-resources applicant ranking, but it reports no "
    "mechanism against LLM identifier-drift failures during ranking, uses a fixed iteration count that leaves its "
    "convergence unquantified, and offers only prompt-level mitigation against hallucinated assessment claims. "
    "This paper's objective is to validate three pipeline-level mechanisms that close these gaps in an open, "
    "reproducible applicant-ranking system, implemented as a LangGraph-orchestrated pipeline with embedding-based "
    "skill shortlisting. This research's three contributions are: (i) resume-grounded strength/weakness "
    "assessments generated with an automated self-correction retry loop; (ii) ranking of shortlisted applicants "
    "with a position-robust, schema-constrained listwise tournament, adapting the Monte Carlo knowledge-gradient "
    "(MC-KG) subset-selection rule and Bayesian Plackett-Luce aggregation from prior work; and (iii) pool-size-aware "
    "adaptive iteration scaling. The results show that mean Faithfulness reached 0.896 and mean within-repeat "
    "ranking convergence (Kendall-τ) reached 0.953, both measured across all 10 job profiles. In conclusion, "
    "these results — measured with metrics the closest prior system does not report or disclose — offer "
    "transparent, empirical evidence that this research addresses those gaps."
)


def fix_abstract_and_contributions(document) -> None:
    paragraph = find_paragraph(document, "gives no cross-run stability")
    replace_paragraph_text(paragraph, ABSTRACT_TEXT)


FIGURE_2_CAPTION_TEXT = (
    "As shown, each generated weakness is checked in code against the applicant's own previously-extracted "
    "skill list for a self-contradiction (e.g., claiming an absent skill the applicant's skill list shows); on "
    "a detected contradiction, the assessment is regenerated once with corrective feedback, and any weakness "
    "still contradicting after the retry is dropped rather than kept. Strengths and additional skills are not "
    "live-checked by this mechanism."
)

_ADAPTIVE_ITERATION_FORMULA_ADDENDUM = (
    " Concretely, the resolved iteration budget is "
    "clamp(ceil(target_appearances_per_candidate × n_candidates / tournament_subset_size), "
    "iterations_min, iterations_max). With this paper's Table I values (target=8, subset_size=5, floor=30, "
    "ceiling=125): a 77-candidate shortlist (data-engineer) resolves to ceil(8×77/5)=124 iterations, while a "
    "7-candidate shortlist (frontend-engineer) resolves to the floor of 30 iterations — though that repeat "
    "stops earlier, at 21 iterations, once every one of its C(7,5)=21 distinct 5-candidate subsets has been "
    "sampled."
)

WEAKNESS_AUDIT_GROUND_TRUTH_TEXT = (
    "are instead audited with the job-profile-skill-bridging half of the same skill-contradiction heuristic "
    "used in the live retry loop: each weakness is scanned for a negated mention using the job profile's skill "
    "vocabulary — since a weakness typically echoes the job description's wording rather than the applicant's "
    "own — and that mention is then matched by embedding similarity against the applicant's own extracted "
    "skills, which is what actually decides whether the weakness is a genuine contradiction."
)


def fix_methods_iii_c_d_e(document) -> None:
    caption_paragraph = find_paragraph(document, "evidence retrieved from the resume")
    replace_paragraph_text(caption_paragraph, FIGURE_2_CAPTION_TEXT)

    iteration_paragraph = find_paragraph(document, "subject to a floor and a ceiling.")
    replace_paragraph_text(
        iteration_paragraph, iteration_paragraph.text + _ADAPTIVE_ITERATION_FORMULA_ADDENDUM
    )

    audit_paragraph = find_paragraph(
        document, "applied here against the job profile's required skills rather than the applicant's own skill list."
    )
    old_tail = (
        "are instead audited with the same skill-contradiction heuristic used in the live retry loop, applied "
        "here against the job profile's required skills rather than the applicant's own skill list."
    )
    new_text = audit_paragraph.text.replace(old_tail, WEAKNESS_AUDIT_GROUND_TRUTH_TEXT)
    replace_paragraph_text(audit_paragraph, new_text)


def fix_experimental_setup_and_table_ii(document) -> None:
    setup_paragraph = find_paragraph(document, "All results runs against")
    replace_paragraph_text(
        setup_paragraph,
        setup_paragraph.text.replace(
            "All results runs against", "All results were obtained by running the pipeline against"
        ),
    )

    table_ii_paragraph = find_paragraph(document, "full-coverage audit")
    old_clause = (
        "from the full-coverage audit against the latest assessments (run-id 20260831-010721), covering all "
        "10 job profiles (2 evaluation failures excluded from the 300-item sample)"
    )
    new_clause = (
        "from a 300-item stratified sample of the run's assessments (run-id 20260831-010721), covering all 10 "
        "job profiles, with 298 of the 300 sampled items scored (2 excluded as Ragas evaluation failures)"
    )
    replace_paragraph_text(table_ii_paragraph, table_ii_paragraph.text.replace(old_clause, new_clause))


FIGURE_2_PNG_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "figures" /
    "Hallucination-Aware Self-Correction Mechanism for Applicant Assessment.png"
)


def swap_figure_2_image(document) -> None:
    # Figure 2 is the second inline shape (index 1) at ~3.23in x 1.90in per the design doc.
    target_shape = document.inline_shapes[1]
    rel_id = target_shape._inline.graphic.graphicData.pic.blipFill.blip.embed
    image_part = document.part.related_parts[rel_id]

    backup_path = FIGURE_2_PNG_PATH.parent / "_archive" / "image2.original.png"
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path.write_bytes(image_part.blob)

    image_part._blob = FIGURE_2_PNG_PATH.read_bytes()


RANKING_FAILURE_EVIDENCE_INTRO = (
    "The position-robust tournament's reliability claim is backed by the following counts from the full run "
    "(run-id 20260831-010721), verified directly against repeats.json and warnings.json:"
)

_RANKING_FAILURE_ROWS = [
    ("Metric", "Value"),
    ("Total tournament ranking calls (rank_subset invocations)", "1713"),
    ("Initial invalid outputs (failed schema/permutation validation on attempt 1)", "0"),
    ("Successful retries", "0 (none needed)"),
    ("Unrecovered failures (ListwiseRankingError, both attempts invalid)", "0"),
]


def insert_results_headers_and_ranking_failure_table(document) -> None:
    fig4_paragraph = find_paragraph(document, "Fig. 4. Within-repeat convergence")
    insert_paragraph_before(fig4_paragraph, "A. Tournament Convergence", bold=True)

    table_ii_paragraph = find_paragraph(document, "Table II. Faithfulness scores")
    insert_paragraph_before(table_ii_paragraph, "B. Generation Faithfulness and Contradiction Audit", bold=True)

    comparison_paragraph = find_paragraph(document, "C. Comparison with the Closest Prior System")
    intro_paragraph = insert_paragraph_before(comparison_paragraph, RANKING_FAILURE_EVIDENCE_INTRO)
    table = insert_table_before(document, comparison_paragraph, rows=len(_RANKING_FAILURE_ROWS), cols=2,
                                 col_widths_in=[4.5, 1.5])
    for row_index, (metric, value) in enumerate(_RANKING_FAILURE_ROWS):
        table.cell(row_index, 0).text = metric
        table.cell(row_index, 1).text = value


NEW_CONCLUSION_TEXT = (
    "This paper presented three pipeline-level mechanisms for LLM-driven applicant ranking, validated against a "
    "real, locally-hosted 14-billion-parameter model rather than an undisclosed one. Self-correcting assessment "
    "generation guarantees zero known skill-contradictions in its own output by construction (Section V-B), "
    "with a 0.82% residual rate surfaced only by a separate offline post-hoc audit using the same heuristic; "
    "the position-robust tournament completes reliably, with zero unrecovered ranking failures out of 1,713 "
    "ranking calls (Section V-B); and adaptive iteration scaling settles convergence at Kendall-τ 0.91–0.98 "
    "across all ten job profiles, a reproducible number the closest prior system's rescaled curve never "
    "discloses. Mean Faithfulness reaches 0.896; the residual gap may partly reflect Faithfulness's own "
    "architectural sensitivity to phrasing and entailment strictness, but because the same model serves as both "
    "generator and judge, this evaluation cannot rule out that part of the gap reflects genuine generation "
    "quality — disentangling the two requires the independent judge model proposed as future work."
)

LIMITATIONS_TEXT = (
    "This study's corpus consists of synthetic resumes over synthetic job profiles authored by the paper's own "
    "authors, not real hiring data; its claims are correspondingly scoped to pipeline-level system reliability "
    "— self-correction, ranking robustness, and convergence — rather than real-world hiring validity, and no "
    "demographic or fairness analysis has been performed on the shortlisting, assessment, or ranking stages."
)

NEW_FUTURE_WORK_TEXT = (
    "Building on these results, future work will pursue three directions: 1) human-rater validation — "
    "comparing this pipeline against independent human-expert rankings using the same protocol as [1]; 2) an "
    "independent judge model — sourcing Faithfulness scoring from a separate model than the one used for "
    "generation, to rule out shared-model bias; and 3) demographic and fairness evaluation — checking whether "
    "shortlisting, assessment, or ranking behavior varies systematically across candidate demographic groups, "
    "once a suitable dataset with demographic labels is available. Taken together, these mechanisms move an "
    "already-strong architecture — the LLM listwise tournament with Plackett-Luce aggregation — toward one "
    "whose failure modes on a real, imperfect local model are documented, fixed, and measured, with the future "
    "work above set to close the remaining validation gaps."
)


def fix_conclusion_and_limitations(document) -> None:
    conclusion_paragraph = find_paragraph(document, "0.0% live skill-contradiction rate")
    replace_paragraph_text(conclusion_paragraph, NEW_CONCLUSION_TEXT)

    future_work_paragraph = find_paragraph(document, "future work will pursue two directions")
    insert_paragraph_before(future_work_paragraph, LIMITATIONS_TEXT)
    replace_paragraph_text(future_work_paragraph, NEW_FUTURE_WORK_TEXT)


def _set_row_cant_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:cantSplit"))


def _set_row_repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:tblHeader"))


def _find_table_by_header_cell_text(document, header_text: str):
    for table in document.tables:
        if table.rows[0].cells[0].text == header_text:
            return table
    raise ValueError(f"no table found with header cell text: {header_text!r}")


def _widen_to_full_page_width(anchor_element) -> None:
    """Wraps anchor_element (a table's <w:tbl> or a paragraph's <w:p>) in a
    continuous, single-column section so it can span the full page width
    instead of being squeezed into one column of the surrounding 2-column
    layout. Setting width alone without this wrapping section would just
    overflow/clip inside the narrow column."""
    start_section = OxmlElement("w:p")
    start_ppr = OxmlElement("w:pPr")
    start_sect_pr = OxmlElement("w:sectPr")
    start_cols = OxmlElement("w:cols")
    start_cols.set(qn("w:num"), "1")
    start_sect_pr.append(start_cols)
    start_ppr.append(start_sect_pr)
    start_section.append(start_ppr)
    anchor_element.addprevious(start_section)

    end_section = OxmlElement("w:p")
    end_ppr = OxmlElement("w:pPr")
    end_sect_pr = OxmlElement("w:sectPr")
    end_cols = OxmlElement("w:cols")
    end_cols.set(qn("w:num"), "2")
    end_sect_pr.append(end_cols)
    end_ppr.append(end_sect_pr)
    end_section.append(end_ppr)
    anchor_element.addnext(end_section)


def _paragraph_element_for_inline_shape(shape):
    """Returns the raw <w:p> element containing an inline image, by walking
    up from the shape's <wp:inline> through its <w:drawing> and <w:r>."""
    drawing = shape._inline.getparent()
    run = drawing.getparent()
    return run.getparent()


def apply_table_and_figure_layout_fixes(document) -> None:
    for table in document.tables:
        for row in table.rows:
            _set_row_cant_split(row)
        _set_row_repeat_header(table.rows[0])

    table_iii = _find_table_by_header_cell_text(document, "Dimension")
    table_iii.autofit = False
    full_width_cols_in = [1.6, 3.0, 3.0]
    for idx, width in enumerate(full_width_cols_in):
        table_iii.columns[idx].width = Inches(width)
    for row in table_iii.rows:
        for cell, width in zip(row.cells, full_width_cols_in):
            cell.width = Inches(width)
    _widen_to_full_page_width(table_iii._tbl)

    figure_1_shape = document.inline_shapes[0]
    aspect_ratio = figure_1_shape.height / figure_1_shape.width
    figure_1_shape.width = Inches(6.8)
    figure_1_shape.height = int(Inches(6.8) * aspect_ratio)
    _widen_to_full_page_width(_paragraph_element_for_inline_shape(figure_1_shape))


def main() -> None:
    shutil.copy2(PAPER_PATH, BACKUP_PATH)
    print(f"Backed up {PAPER_PATH} -> {BACKUP_PATH}")

    document = docx.Document(PAPER_PATH)

    # fix_* calls are added here by later tasks, in this order:
    fix_abstract_and_contributions(document)
    fix_methods_iii_c_d_e(document)
    swap_figure_2_image(document)
    fix_experimental_setup_and_table_ii(document)
    insert_results_headers_and_ranking_failure_table(document)
    fix_conclusion_and_limitations(document)
    # insert_weakness_retry_audit_table(document)
    apply_table_and_figure_layout_fixes(document)

    document.save(PAPER_PATH)
    print(f"Saved corrections to {PAPER_PATH}")


if __name__ == "__main__":
    main()
