import docx

from scripts.shorten_paper_20260908 import delete_paragraph, paragraph_element_immediately_before


def test_delete_paragraph_removes_it_from_the_document():
    document = docx.Document()
    document.add_paragraph("Keep me.")
    target = document.add_paragraph("Delete me.")
    document.add_paragraph("Keep me too.")

    delete_paragraph(target)

    texts = [p.text for p in document.paragraphs]
    assert texts == ["Keep me.", "Keep me too."]


def test_paragraph_element_immediately_before_returns_the_preceding_element():
    document = docx.Document()
    before = document.add_paragraph("Before.")
    anchor = document.add_paragraph("Anchor.")

    element = paragraph_element_immediately_before(document, anchor)

    assert element is before._p
