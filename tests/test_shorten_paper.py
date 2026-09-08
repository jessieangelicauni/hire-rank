import docx
from docx.shared import Inches
from PIL import Image

from scripts.shorten_paper_20260908 import cut_figure_5, cut_figure_6, cut_literature_review, cut_introduction_prose, delete_paragraph, paragraph_element_immediately_before, remove_figure_and_fold_text


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


def test_cut_figure_5_removes_image_and_preserves_all_four_folded_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.shorten_paper_20260908.FIGURE_5_ARCHIVE_PATH", tmp_path / "_archive" / "fig5.png"
    )
    document = docx.Document()
    document.add_picture(str(_tiny_png_path(tmp_path, "fig5.png")), width=Inches(3.5), height=Inches(2.0))
    document.add_paragraph("Fig. 5. Final Kendall's Tau per job role after convergence.")
    document.add_paragraph(
        "As Fig. 5 shows, final Kendall's Tau per role, averaged across stability repeats, "
        "spans 0.930 (frontend-engineer, n=7) to 0.979 (data-engineer, n=77), with smaller "
        "shortlists scoring lower — consistent with Kendall's Tau being a noisier estimator "
        "over fewer candidates rather than evidence that smaller pools rank less reliably."
    )

    cut_figure_5(document)

    texts = [p.text for p in document.paragraphs]
    assert not any(t.startswith("Fig. 5.") for t in texts)
    joined = " ".join(texts)
    assert "As Fig. 5 shows" not in joined
    for number in ("0.930", "n=7", "0.979", "n=77"):
        assert number in joined
    assert "noisier estimator" in joined
    assert len(document.inline_shapes) == 0


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

    assert len(new_a) < len(_OLD_II_A) * 0.85   # real ratio is ~0.808
    assert len(new_b) < len(_OLD_II_B) * 0.80   # tightened further in the contingency round; real ratio is ~0.753
    # II-C (Yuksel et al., the key comparator) gets the lightest cut.
    assert len(new_c) < len(_OLD_II_C)
    assert len(new_c) > len(_OLD_II_C) * 0.85   # was 0.7; real ratio is ~0.914, tighten the floor to match


_OLD_PARA_8 = (
    "Hiring teams routinely receive far more resumes than they can read carefully, so recruitment platforms "
    "have moved from plain keyword search toward machine-learning and, more recently, large-language-model "
    "(LLM) systems that read a resume the way a person would and explain what they find [3]. This shift favors "
    "LLMs because they can extract skills from unstructured text, write a short explanation of why an "
    "applicant fits a role, and even compare several applicants against each other in a single pass — none "
    "of which older keyword or embedding-similarity systems can do on their own. The closest prior work to "
    "this paper turns applicant ranking into a tournament: instead of scoring each applicant alone, the LLM "
    "looks at a small group of applicants side by side and orders them, and those orderings are combined "
    "statistically into one overall ranking using the Plackett-Luce model [1]."
)

_OLD_PARA_9 = (
    "This tournament idea is powerful, but the latest research leaves open several research gaps that stand "
    "out at the level of the LLM architecture and system pipeline. 1) No defense against LLM identifier or "
    "format drift during ranking — a failure mode common with smaller, locally-hosted models. 2) No "
    "self-correction for hallucinated claims — generated claims are not checked against the resume, nor is "
    "generation retried when a contradiction is found. 3) Fixed iteration counts — iterative methods [1] "
    "use a fixed value regardless of how many applicants are being ranked, which under-samples large pools and "
    "over-samples small ones."
)

_OLD_PARA_11 = (
    "The remainder of the paper reviews related work (Section II), describes the pipeline in implementation "
    "detail (Section III), details the experimental setup (Section IV), reports results and compares them "
    "against the closest prior system (Section V), and concludes with limitations and future work (Section VI)."
)


def test_cut_introduction_prose_keeps_the_three_numbered_gaps_and_the_plackett_luce_citation():
    document = docx.Document()
    document.add_paragraph(_OLD_PARA_8)
    document.add_paragraph(_OLD_PARA_9)
    document.add_paragraph(_OLD_PARA_11)

    cut_introduction_prose(document)

    new_8, new_9, new_11 = (p.text for p in document.paragraphs)
    assert "Plackett-Luce model [1]" in new_8
    assert len(new_8) < len(_OLD_PARA_8)
    for marker in ("1) No defense", "2) No self-correction", "3) Fixed iteration counts"):
        assert marker in new_9
    assert len(new_9) < len(_OLD_PARA_9)
    for section in ("Section II", "Section III", "Section IV", "Section V", "Section VI"):
        assert section in new_11
    assert len(new_11) < len(_OLD_PARA_11)
