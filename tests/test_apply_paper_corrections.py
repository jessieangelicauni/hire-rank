import docx

from scripts.apply_paper_corrections import (
    ABSTRACT_TEXT,
    find_paragraph,
    fix_abstract_and_contributions,
    insert_paragraph_before,
    insert_table_before,
    replace_paragraph_text,
)

_OLD_ABSTRACT = (
    "Abstract—Talent-acquisition systems increasingly pair large language models (LLMs) with statistical ranking "
    "to shortlist and order job applicants. The most closely related recent system combines an LLM-driven active "
    "listwise tournament with Plackett-Luce aggregation for human-resources applicant ranking, but it reports no "
    "mechanism against LLM identifier-drift failures during ranking, gives no cross-run stability, and offers "
    "only prompt-level mitigation against hallucinated assessment claims. This paper's objective is to validate "
    "three pipeline-level mechanisms that close these gaps in an open, reproducible applicant-ranking system. "
    "This research's methods are: 1) a LangGraph-orchestrated pipeline that shortlists applicants by "
    "embedding-based skill overlap; 2) resume-grounded strength/weakness assessments generated with an automated "
    "self-correction retry loop; and 3) ranking of shortlisted applicants with a position-robust, "
    "schema-constrained listwise tournament, adapting the Monte Carlo knowledge-gradient (MC-KG) "
    "subset-selection rule and Bayesian Plackett-Luce aggregation from prior work. The results show that mean "
    "Faithfulness reached 0.896 and mean within-repeat ranking convergence (Kendall-τ) reached 0.953, both "
    "measured across all 10 job profiles. In conclusion, these results — measured with metrics the closest prior "
    "system does not report or disclose — offer transparent, empirical evidence that this research addresses "
    "those gaps."
)


def test_fix_abstract_and_contributions_removes_cross_run_stability_claim():
    document = docx.Document()
    document.add_paragraph(_OLD_ABSTRACT)

    fix_abstract_and_contributions(document)

    paragraph = document.paragraphs[0]
    assert "cross-run stability" not in paragraph.text
    assert paragraph.text == ABSTRACT_TEXT


def test_fix_abstract_and_contributions_lists_the_three_introduction_contributions_in_order():
    document = docx.Document()
    document.add_paragraph(_OLD_ABSTRACT)

    fix_abstract_and_contributions(document)

    text = document.paragraphs[0].text
    self_correcting_idx = text.index("self-correcting" if "self-correcting" in text else "self-correction")
    position_robust_idx = text.index("position-robust")
    adaptive_idx = text.index("pool-size-aware adaptive iteration")
    assert self_correcting_idx < position_robust_idx < adaptive_idx


def test_find_paragraph_returns_matching_paragraph():
    document = docx.Document()
    document.add_paragraph("Some other text.")
    target = document.add_paragraph("This contains the marker phrase.")

    found = find_paragraph(document, "marker phrase")

    # python-docx constructs a brand-new Paragraph wrapper object on every
    # access of `.paragraphs` (see docx/blkcntnr.py: `[Paragraph(p, self) for
    # p in self._element.p_lst]`), so wrapper identity (`found is target`) is
    # never preserved even when both wrap the same underlying paragraph.
    # `._p` is the underlying lxml element, whose identity IS stable, and is
    # what actually matters: mutating through `found` mutates the same node
    # `target` wraps.
    assert found._p is target._p


def test_find_paragraph_raises_when_not_found():
    document = docx.Document()
    document.add_paragraph("Nothing relevant here.")

    try:
        find_paragraph(document, "missing marker")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_replace_paragraph_text_single_run():
    document = docx.Document()
    paragraph = document.add_paragraph("Old sentence to replace.")

    replace_paragraph_text(paragraph, "New sentence.")

    assert paragraph.text == "New sentence."


def test_replace_paragraph_text_preserves_bold_formatting():
    document = docx.Document()
    paragraph = document.add_paragraph()
    run = paragraph.add_run("Old bold sentence.")
    run.bold = True

    replace_paragraph_text(paragraph, "New bold sentence.")

    assert paragraph.text == "New bold sentence."
    assert paragraph.runs[0].bold is True


def test_replace_paragraph_text_multi_run_collapses_to_one():
    document = docx.Document()
    paragraph = document.add_paragraph()
    paragraph.add_run("Part one. ")
    paragraph.add_run("Part two.")

    replace_paragraph_text(paragraph, "Single replacement sentence.")

    assert paragraph.text == "Single replacement sentence."


def test_insert_paragraph_before_places_new_paragraph_immediately_before_anchor():
    document = docx.Document()
    document.add_paragraph("Before anchor.")
    anchor = document.add_paragraph("Anchor paragraph.")
    document.add_paragraph("After anchor.")

    inserted = insert_paragraph_before(anchor, "Inserted heading", bold=True)

    texts = [p.text for p in document.paragraphs]
    assert texts == ["Before anchor.", "Inserted heading", "Anchor paragraph.", "After anchor."]
    assert inserted.runs[0].bold is True


def test_insert_table_before_places_table_immediately_before_anchor():
    document = docx.Document()
    document.add_paragraph("Before anchor.")
    anchor = document.add_paragraph("Anchor paragraph.")

    table = insert_table_before(document, anchor, rows=2, cols=2, col_widths_in=[1.5, 1.5])
    table.cell(0, 0).text = "Header"

    body_children = list(document.element.body)
    tbl_index = next(i for i, el in enumerate(body_children) if el.tag.endswith("}tbl"))
    anchor_index = next(i for i, el in enumerate(body_children) if el is anchor._p)
    assert tbl_index == anchor_index - 1
    assert table.cell(0, 0).text == "Header"


def test_insert_table_before_with_col_widths_disables_autofit():
    document = docx.Document()
    anchor = document.add_paragraph("Anchor paragraph.")

    table = insert_table_before(document, anchor, rows=2, cols=2, col_widths_in=[1.5, 1.5])

    assert table.autofit is False
    tbl_layout = table._tbl.tblPr.find(
        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tblLayout"
    )
    assert tbl_layout is not None
    assert tbl_layout.get(
        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}type"
    ) == "fixed"


def test_insert_table_before_without_col_widths_keeps_default_autofit():
    document = docx.Document()
    anchor = document.add_paragraph("Anchor paragraph.")

    table = insert_table_before(document, anchor, rows=2, cols=2)

    assert table.autofit is True
