import docx

from scripts.apply_paper_corrections import (
    ABSTRACT_TEXT,
    find_paragraph,
    fix_abstract_and_contributions,
    fix_conclusion_and_limitations,
    fix_experimental_setup_and_table_ii,
    fix_methods_iii_c_d_e,
    insert_paragraph_before,
    insert_results_headers_and_ranking_failure_table,
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


_OLD_FIG2_CAPTION = (
    "As shown, each generated claim is checked against evidence retrieved from the resume for unsupported, "
    "contradicted, or unverifiable content, with the assessment regenerated once on any detected issue before "
    "it is accepted."
)

_OLD_ADAPTIVE_ITERATION_PARAGRAPH = (
    "The pipeline's third contribution addresses the fixed-iteration-count gap identified in Section I: at a "
    "fixed iteration count, a large shortlist receives too few comparisons per applicant to rank reliably, while "
    "a small shortlist receives far more than it needs. Rather than using a fixed count for every job profile "
    "regardless of shortlist size, the iteration count is scaled so that every shortlisted applicant is expected "
    "to appear in roughly a fixed target number of sampled subsets on average, subject to a floor and a ceiling."
)

_OLD_SECTION_III_E = (
    "Each job profile's final ranking averages applicant utilities across its successful stability repeats, "
    "which reduces the noise a single repeat's point estimate would otherwise carry into the ranking. "
    "Within-repeat convergence — via the Kendall-τ rank correlation between the utility ordering at consecutive "
    "iterations — is computed directly from the tournament traces, with no LLM call required. Separately, an "
    "offline harness audits assessment quality after the fact: strengths are scored with the Faithfulness metric "
    "(an LLM judge estimates how well each strength is entailed by the resume text, given a neutral, "
    "job-agnostic question rather than one naming the job title, so the score reflects only resume support and "
    "not job-profile-specific framing), while weaknesses — which are almost always absence claims that a "
    "faithfulness-style entailment check cannot verify positively — are instead audited with the same "
    "skill-contradiction heuristic used in the live retry loop, applied here against the job profile's required "
    "skills rather than the applicant's own skill list."
)


def test_fix_methods_rewrites_figure_2_caption():
    document = docx.Document()
    document.add_paragraph(_OLD_FIG2_CAPTION)
    document.add_paragraph(_OLD_ADAPTIVE_ITERATION_PARAGRAPH)
    document.add_paragraph(_OLD_SECTION_III_E)

    fix_methods_iii_c_d_e(document)

    caption_text = document.paragraphs[0].text
    assert "evidence retrieved from the resume" not in caption_text
    assert "each generated weakness is checked" in caption_text
    assert "Strengths and additional skills are not live-checked" in caption_text


def test_fix_methods_adds_adaptive_iteration_formula():
    document = docx.Document()
    document.add_paragraph(_OLD_FIG2_CAPTION)
    document.add_paragraph(_OLD_ADAPTIVE_ITERATION_PARAGRAPH)
    document.add_paragraph(_OLD_SECTION_III_E)

    fix_methods_iii_c_d_e(document)

    formula_text = document.paragraphs[1].text
    assert "ceil(target_appearances_per_candidate" in formula_text
    assert "124 iterations" in formula_text
    assert "30 iterations" in formula_text


def test_fix_methods_corrects_weakness_audit_ground_truth():
    document = docx.Document()
    document.add_paragraph(_OLD_FIG2_CAPTION)
    document.add_paragraph(_OLD_ADAPTIVE_ITERATION_PARAGRAPH)
    document.add_paragraph(_OLD_SECTION_III_E)

    fix_methods_iii_c_d_e(document)

    audit_text = document.paragraphs[2].text
    assert "applied here against the job profile's required skills rather than the applicant's own skill list" not in audit_text
    assert "job-profile-skill-bridging half" in audit_text
    assert "which is what actually decides whether the weakness is a genuine contradiction" in audit_text


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


_OLD_EXPERIMENTAL_SETUP_START = (
    "All results runs against the Qwen2.5-14B-Instruct model, over the repository's corpus of resumes (500 "
    "sampled from the EraMatch CV Parsing Benchmark v3.0 synthetic resume dataset [14]) and 10 job profiles, "
    "authored synthetically by the authors. Shortlisting (Section III-B) narrowed this corpus to 348 (job, "
    "applicant) pairs across the 10 profiles."
)

_OLD_TABLE_II_CAPTION = (
    "Table II. Faithfulness scores for generated strengths, from the full-coverage audit against the latest "
    "assessments (run-id 20260831-010721), covering all 10 job profiles (2 evaluation failures excluded from "
    "the 300-item sample). The run-wide mean is 0.896, following successive prompt refinements described in "
    "Section III-C."
)


def test_fix_experimental_setup_grammar():
    document = docx.Document()
    document.add_paragraph(_OLD_EXPERIMENTAL_SETUP_START)
    document.add_paragraph(_OLD_TABLE_II_CAPTION)

    fix_experimental_setup_and_table_ii(document)

    assert "All results runs against" not in document.paragraphs[0].text
    assert "All results were obtained by running the pipeline against" in document.paragraphs[0].text


def test_fix_table_ii_caption_consistency():
    document = docx.Document()
    document.add_paragraph(_OLD_EXPERIMENTAL_SETUP_START)
    document.add_paragraph(_OLD_TABLE_II_CAPTION)

    fix_experimental_setup_and_table_ii(document)

    caption = document.paragraphs[1].text
    assert "full-coverage audit" not in caption
    assert "300-item stratified sample" in caption
    assert "298 of the 300 sampled items scored" in caption


def _results_section_skeleton():
    document = docx.Document()
    document.add_paragraph("V. Results and Discussion")
    document.add_paragraph("This section reports convergence, per-role stability, and generation faithfulness.")
    fig4 = document.add_paragraph("Fig. 4. Within-repeat convergence (Kendall-τ) across the 10 job profiles.")
    document.add_paragraph("Fig. 5. Final Kendall's Tau per job role after convergence.")
    table_ii_caption = document.add_paragraph(
        "Table II. Faithfulness scores for generated strengths, from a 300-item stratified sample."
    )
    fig6 = document.add_paragraph("Fig. 6. Faithfulness scores for generated strengths, by job profile.")
    comparison_header = document.add_paragraph("C. Comparison with the Closest Prior System")
    comparison_header.runs[0].bold = True
    return document, fig4, table_ii_caption, comparison_header


def test_insert_results_headers_adds_a_and_b_before_the_right_content():
    document, fig4, table_ii_caption, comparison_header = _results_section_skeleton()

    insert_results_headers_and_ranking_failure_table(document)

    texts = [p.text for p in document.paragraphs]
    assert "A. Tournament Convergence" in texts
    assert texts.index("A. Tournament Convergence") == texts.index(fig4.text) - 1
    assert "B. Generation Faithfulness and Contradiction Audit" in texts
    assert texts.index("B. Generation Faithfulness and Contradiction Audit") == texts.index(table_ii_caption.text) - 1


def test_insert_results_adds_ranking_failure_table_before_comparison_section():
    document, fig4, table_ii_caption, comparison_header = _results_section_skeleton()

    insert_results_headers_and_ranking_failure_table(document)

    body_children = list(document.element.body)
    tbl_indices = [i for i, el in enumerate(body_children) if el.tag.endswith("}tbl")]
    comparison_index = next(i for i, el in enumerate(body_children) if el is comparison_header._p)
    assert len(tbl_indices) == 1
    assert tbl_indices[0] < comparison_index

    table = document.tables[0]
    assert table.cell(0, 0).text == "Metric"
    assert table.cell(1, 1).text == "1713"
    assert table.cell(4, 1).text == "0"


_OLD_CONCLUSION = (
    "This paper presented three pipeline-level mechanisms for LLM-driven applicant ranking, validated against a "
    "real, locally-hosted 14-billion-parameter model rather than an undisclosed one. Self-correcting assessment "
    "generation achieves a 0.0% live skill-contradiction rate (0.6% under independent audit); the "
    "position-robust tournament completes reliably, with zero unrecovered ranking failures; and adaptive "
    "iteration scaling settles convergence at Kendall-τ 0.91–0.98 across all ten job profiles, a reproducible "
    "number the closest prior system's rescaled curve never discloses. Mean Faithfulness reaches 0.896, with "
    "the residual gap traced to Faithfulness's own architecture rather than generation quality."
)

_OLD_FUTURE_WORK = (
    "Building on these results, future work will pursue two directions: 1) human-rater validation — comparing "
    "this pipeline against independent human-expert rankings using the same protocol as [1]; and 2) an "
    "independent judge model — sourcing Faithfulness scoring from a separate model than the one used for "
    "generation, to rule out shared-model bias. Taken together, these mechanisms move an already-strong "
    "architecture — the LLM listwise tournament with Plackett-Luce aggregation — toward one whose failure modes "
    "on a real, imperfect local model are documented, fixed, and measured, with the future work above set to "
    "close the remaining validation gaps."
)


def test_fix_conclusion_renames_independent_audit_and_softens_faithfulness_claim():
    document = docx.Document()
    document.add_paragraph(_OLD_CONCLUSION)
    document.add_paragraph(_OLD_FUTURE_WORK)

    fix_conclusion_and_limitations(document)

    conclusion_text = document.paragraphs[0].text
    assert "under independent audit" not in conclusion_text
    assert "0.6%" not in conclusion_text
    assert "offline post-hoc audit" in conclusion_text
    assert "0.82%" in conclusion_text
    assert "traced to Faithfulness's own architecture rather than generation quality" not in conclusion_text
    assert "cannot rule out that part of the gap reflects genuine generation quality" in conclusion_text


def test_fix_conclusion_inserts_limitations_paragraph_and_extends_future_work():
    document = docx.Document()
    document.add_paragraph(_OLD_CONCLUSION)
    document.add_paragraph(_OLD_FUTURE_WORK)

    fix_conclusion_and_limitations(document)

    texts = [p.text for p in document.paragraphs]
    limitations = [t for t in texts if "synthetic resumes" in t]
    assert len(limitations) == 1
    assert "real-world hiring validity" in limitations[0]
    assert "no demographic or fairness analysis" in limitations[0]

    future_work_text = [t for t in texts if "future work will pursue" in t][0]
    assert "three directions" in future_work_text
    assert "demographic and fairness evaluation" in future_work_text
    limitations_index = texts.index(limitations[0])
    future_work_index = texts.index(future_work_text)
    assert limitations_index < future_work_index
