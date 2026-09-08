import docx
from docx.shared import Inches
from PIL import Image

from scripts.shorten_paper_20260908 import cut_figure_6, cut_literature_review, delete_paragraph, paragraph_element_immediately_before, remove_figure_and_fold_text


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


def _tiny_png_path(tmp_path, name="tiny.png"):
    path = tmp_path / name
    Image.new("RGB", (100, 40), color="white").save(path)
    return path


def test_remove_figure_and_fold_text_deletes_image_and_caption(tmp_path):
    document = docx.Document()
    document.add_paragraph("Preceding paragraph.")
    document.add_picture(str(_tiny_png_path(tmp_path)), width=Inches(3.5), height=Inches(2.0))
    document.add_paragraph("Fig. 5. Final Kendall's Tau per job role after convergence.")
    document.add_paragraph("As Fig. 5 shows, values range from 0.930 to 0.979.")
    document.add_paragraph("B. Generation Faithfulness and Contradiction Audit")

    archive_path = tmp_path / "_archive" / "removed.png"

    remove_figure_and_fold_text(
        document,
        caption_marker="Fig. 5. Final Kendall",
        sentence_marker="As Fig. 5 shows",
        new_sentence="Final Kendall's Tau per role ranges from 0.930 to 0.979.",
        archive_path=archive_path,
    )

    texts = [p.text for p in document.paragraphs]
    assert not any(t.startswith("Fig. 5.") for t in texts)
    assert "As Fig. 5 shows" not in "".join(texts)
    assert "Final Kendall's Tau per role ranges from 0.930 to 0.979." in texts
    assert len(document.inline_shapes) == 0
    assert archive_path.exists()


def test_remove_figure_and_fold_text_archives_the_original_image_bytes(tmp_path):
    document = docx.Document()
    png_path = _tiny_png_path(tmp_path)
    document.add_picture(str(png_path), width=Inches(3.5), height=Inches(2.0))
    document.add_paragraph("Fig. 5. Final Kendall's Tau per job role after convergence.")
    document.add_paragraph("As Fig. 5 shows, values range from 0.930 to 0.979.")

    archive_path = tmp_path / "_archive" / "removed.png"

    remove_figure_and_fold_text(
        document,
        caption_marker="Fig. 5. Final Kendall",
        sentence_marker="As Fig. 5 shows",
        new_sentence="Final Kendall's Tau per role ranges from 0.930 to 0.979.",
        archive_path=archive_path,
    )

    assert archive_path.read_bytes() == png_path.read_bytes()


def test_cut_figure_6_removes_image_and_rewrites_sentence_to_reference_table_ii(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.shorten_paper_20260908.FIGURE_6_ARCHIVE_PATH", tmp_path / "_archive" / "fig6.png"
    )
    document = docx.Document()
    document.add_picture(str(_tiny_png_path(tmp_path, "fig6.png")), width=Inches(3.5), height=Inches(2.0))
    document.add_paragraph("Fig. 6. Faithfulness scores for generated strengths, by job profile.")
    document.add_paragraph(
        "As shown in Fig. 6, per-role Faithfulness scores range from full-stack-engineer "
        "(lowest, 0.764) to devops-engineer (highest, 0.944), bracketing the run-wide mean "
        "of 0.880 (Table II)."
    )

    cut_figure_6(document)

    texts = [p.text for p in document.paragraphs]
    assert not any(t.startswith("Fig. 6.") for t in texts)
    joined = " ".join(texts)
    assert "As Table II shows" in joined
    assert "0.764" in joined and "0.944" in joined and "0.880" in joined
    assert len(document.inline_shapes) == 0

_OLD_II_A = (
    "Most recent papers apply LLMs to extraction and pointwise scoring inside RAG or multi-agent pipelines, "
    "without a comparative ranking step. Tran and Tran [2] build a RAG agent over a resume vector store "
    "(Faithfulness 0.882) but do not rank applicants; this pipeline’s own Faithfulness score (Table II) is "
    "measured on a more open-ended multi-claim task, not directly comparable. Lo et al. [3], Chowdhury et al. "
    "[4], Jahan et al. [5], Gera et al. [8], Walid et al. [7], and Chong et al. [17] similarly orchestrate "
    "multi-agent, extraction, or scoring pipelines — respectively a four-agent CrewAI scorer, fine-tuned "
    "open-LLM sub-dimension agents that collapse under fine-tuning without data augmentation (R² as low as "
    "−8.15), a zero-shot multi-agent consensus vote whose recall drops to 27.9% from requiring agent "
    "agreement, a single-resume case study with no benchmark, an ensemble tuned for field-extraction accuracy "
    "rather than ranking, and an LLM resume-relevance scorer paired with a fine-tuned-BERT recruitment chatbot "
    "(99.15% Q&A validation accuracy) that reports no accuracy metric for the scoring/ranking component itself. "
    "None validates hallucination mitigation with a verified retry loop against the applicant’s own extracted "
    "skills, as this paper’s constrained, retry-on-contradiction generation does. Synthesizing 141 such "
    "papers, Dasaklis et al. [15] find hallucination and cross-session score instability to be recurring, "
    "unresolved risks across the field and explicitly call for dedicated hallucination-tracking metrics as "
    "future work — precisely the mechanism this paper’s retry-on-contradiction loop provides."
)

_OLD_II_B = (
    "Li et al. [6] deploy a production talent-search ranker at Alibaba using an LLM only to extract hiring "
    "preferences, feeding a Mixture-of-Experts network whose ranking is a product of pointwise predicted rates, "
    "not a comparative judgment. Hoque et al. [9] combine a fine-tuned DistilRoBERTa classifier with Fuzzy "
    "TOPSIS under fixed, expert-elicited criterion weights, reaching high agreement with human rankings (NDCG "
    "= 0.926) but on only 100 profiles from a single job family, with weights fixed rather than adapted per "
    "query. Rosenberger et al. [13] similarly embed resumes and ESCO job descriptions into a shared vector "
    "space and rank jobs by cosine similarity, reporting a small-scale human-grounded evaluation (five resumes, "
    "ten HR experts) that the authors themselves flag as too limited to generalize, while Xue et al. [14] fuse "
    "BERT-based semantic features with a graph neural network over historical hiring records into a single "
    "pointwise fit score (94.6% accuracy on binary hire/no-hire classification); neither compares candidates "
    "against each other within a shared context the way a listwise judgment would. This paper instead ranks "
    "applicants through a position-robust listwise tournament rather than pointwise or fixed-weight scoring."
)

_OLD_II_C = (
    "Yuksel et al. [1] are the closest prior work: an LLM ranks a small subset of applicants at once, and "
    "these orderings are aggregated into global utilities with a Plackett-Luce model under an active-learning "
    "loop, including a Monte Carlo knowledge-gradient acquisition strategy. This design reports a real result "
    "against human judgment (87% of ratings within one rubric level; peak NDCG@25% of 0.5703), but does not "
    "name its LLM, disclose its dataset, report variance across runs, discuss identifier-drift failure, or "
    "self-correct hallucinated claims beyond a single internal prompt check — and motivates its listwise "
    "design by noting pairwise comparison “does not scale well to large applicant pools” without "
    "addressing how many applicants reach the tournament in the first place. These are exactly the gaps "
    "addressed in Section III that gives the full structured comparison."
)

_ALL_LIT_REVIEW_CITATIONS = [f"[{n}]" for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 17)]


def test_cut_literature_review_keeps_every_citation_and_shortens_each_paragraph():
    document = docx.Document()
    document.add_paragraph(_OLD_II_A)
    document.add_paragraph(_OLD_II_B)
    document.add_paragraph(_OLD_II_C)

    cut_literature_review(document)

    new_a, new_b, new_c = (p.text for p in document.paragraphs)
    combined = new_a + new_b + new_c
    for citation in _ALL_LIT_REVIEW_CITATIONS:
        assert citation in combined, f"citation {citation} was dropped"

    assert len(new_a) < len(_OLD_II_A) * 0.75
    assert len(new_b) < len(_OLD_II_B) * 0.75
    # II-C (Yuksel et al., the key comparator) gets the lightest cut.
    assert len(new_c) < len(_OLD_II_C)
    assert len(new_c) > len(_OLD_II_C) * 0.7