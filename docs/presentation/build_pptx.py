"""
ICAIDES 2026 speaker deck.

Visual system: white background only, no solid fills anywhere except the
slide background itself; black outline shapes and black text on white;
Arial throughout. Emphasis uses bold + underline text, never an inverted
color block. Mirrors docs/paper2_jev.tex section by section.

Regenerate with: python3 docs/presentation/build_pptx.py
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

BLACK = RGBColor(0x00, 0x00, 0x00)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY = RGBColor(0x59, 0x59, 0x59)
FONT = "Arial"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]


def new_slide():
    slide = prs.slides.add_slide(BLANK)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    return slide


def add_textbox(slide, x, y, w, h, text, size=16, bold=False, align=PP_ALIGN.LEFT,
                 color=BLACK, italic=False, underline=False):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        for run in p.runs:
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.italic = italic
            run.font.underline = underline
            run.font.name = FONT
            run.font.color.rgb = color
    return box


def add_kicker(slide, text):
    add_textbox(slide, Inches(0.8), Inches(0.45), SLIDE_W - Inches(1.6), Inches(0.35),
                text.upper(), size=13, bold=True, color=GRAY)


def add_title(slide, text, top=Inches(0.85)):
    add_textbox(slide, Inches(0.8), top, SLIDE_W - Inches(1.6), Inches(1.1),
                text, size=30, bold=True, color=BLACK)


def add_footer(slide, text):
    add_textbox(slide, Inches(0.8), SLIDE_H - Inches(0.55), SLIDE_W - Inches(1.6), Inches(0.4),
                text, size=12, italic=True, color=GRAY)


def add_bullets(slide, x, y, w, h, items, size=18):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"•  {item}"
        for run in p.runs:
            run.font.size = Pt(size)
            run.font.name = FONT
            run.font.color.rgb = BLACK
        p.space_after = Pt(12)
    return box


def outline_box(slide, x, y, w, h, text=None, size=15, bold=False, align=PP_ALIGN.CENTER):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shape.fill.background()
    shape.line.color.rgb = BLACK
    shape.line.width = Pt(1.5)
    shape.shadow.inherit = False
    if text:
        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = Pt(10)
        tf.margin_right = Pt(10)
        for i, line in enumerate(text.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.alignment = align
            for run in p.runs:
                run.font.size = Pt(size)
                run.font.bold = bold
                run.font.name = FONT
                run.font.color.rgb = BLACK
    return shape


# ---------------------------------------------------------------------------
# Slide 1: Title
# ---------------------------------------------------------------------------
slide = new_slide()
add_textbox(slide, Inches(0.8), Inches(0.5), SLIDE_W - Inches(1.6), Inches(0.4),
            "ICAIDES 2026", size=15, bold=True, color=GRAY, align=PP_ALIGN.CENTER)
add_textbox(
    slide, Inches(1.2), Inches(1.8), SLIDE_W - Inches(2.4), Inches(2.2),
    "Reliable Structured-Decision Applicant Ranking with a Non-Autoregressive System One Model",
    size=30, bold=True, align=PP_ALIGN.CENTER,
)
add_textbox(
    slide, Inches(1.2), Inches(4.1), SLIDE_W - Inches(2.4), Inches(0.6),
    "Jessie Angelica · Hasanul Fahmi Zuhri · Rusdianto Roestam · Muhammad Arief",
    size=16, color=GRAY, align=PP_ALIGN.CENTER,
)
add_textbox(
    slide, Inches(1.5), Inches(5.0), SLIDE_W - Inches(3.0), Inches(1.0),
    "Can a model that only fills in structured answers rank applicants better than one that writes free text?",
    size=17, italic=True, align=PP_ALIGN.CENTER,
)

# ---------------------------------------------------------------------------
# Slide 2: Motivation & Gap
# ---------------------------------------------------------------------------
slide = new_slide()
add_kicker(slide, "Introduction")
add_title(slide, "Motivation & Gap")
add_bullets(
    slide, Inches(1.1), Inches(2.3), SLIDE_W - Inches(2.2), Inches(3.5),
    [
        "Free-text assessment can assert a strength or weakness the resume never supports — a hallucination — with no built-in confidence signal.",
        "In September 2026, TypeSafe released Jev, the first “System One Model.”",
        "No independent, peer-reviewed evaluation of it existed before this paper.",
    ],
    size=20,
)
add_footer(slide, "Section I — Introduction")

# ---------------------------------------------------------------------------
# Slide 3: Objective
# ---------------------------------------------------------------------------
slide = new_slide()
add_kicker(slide, "Introduction")
add_title(slide, "Objective")
labels = [
    "Criteria-Grounded\nQuestion Design",
    "Cost & Latency\nEfficiency",
    "Repeat-to-Repeat\nReliability",
]
box_w = Inches(3.7)
gap = Inches(0.4)
total_w = box_w * 3 + gap * 2
x0 = (SLIDE_W - total_w) / 2
top = Inches(2.3)
box_h = Inches(2.2)
for i, label in enumerate(labels):
    x = x0 + i * (box_w + gap)
    outline_box(slide, x, top, box_w, box_h, text=label, size=17, bold=True)
add_textbox(
    slide, Inches(1.0), top + box_h + Inches(0.5), SLIDE_W - Inches(2.0), Inches(0.8),
    "Three contributions for deploying Jev reliably in a real applicant-ranking pipeline.",
    size=16, align=PP_ALIGN.CENTER, color=GRAY, italic=True,
)
add_footer(slide, "Section I — Introduction")

def add_pipeline(slide, stages, top=Inches(1.5)):
    box_w = Inches(7.8)
    box_h = Inches(0.58)
    seg = Inches(0.36)
    x = (SLIDE_W - box_w) / 2
    y = top
    for i, (label, bold) in enumerate(stages):
        outline_box(slide, x, y, box_w, box_h, text=label, size=14, bold=bold)
        y += box_h
        if i < len(stages) - 1:
            add_textbox(slide, x, y, box_w, seg, "↓", size=16, align=PP_ALIGN.CENTER)
            y += seg


# ---------------------------------------------------------------------------
# Slide 4: Background - What Is a System One Model
# ---------------------------------------------------------------------------
slide = new_slide()
add_kicker(slide, "Background")
add_title(slide, "What Is a System One Model")
add_bullets(
    slide, Inches(1.1), Inches(2.3), SLIDE_W - Inches(2.2), Inches(3.5),
    [
        "Non-autoregressive: answers typed questions in one parallel pass — no generated tokens.",
        "Three typed primitives: Noul (yes/no + confidence), Choice (multiple-choice), Score (ordinal scale).",
        "This paper's questions all use Score.",
    ],
    size=20,
)
add_footer(slide, "Section II — Literature Review")

# ---------------------------------------------------------------------------
# Slide 5: Background - Why It Works
# ---------------------------------------------------------------------------
slide = new_slide()
add_kicker(slide, "Background")
add_title(slide, "Why It Works")
add_bullets(
    slide, Inches(1.1), Inches(2.3), SLIDE_W - Inches(2.2), Inches(3.5),
    [
        "Each Score level is judged alone — no view of neighboring levels, so a bare number carries no signal.",
        "A concrete situation gives the model resume evidence to check against.",
        "Trained via RLCD, a reward for calibration — not human preference (RLHF) or a verifiable outcome (RLVR).",
    ],
    size=20,
)
add_footer(slide, "Section II — Literature Review")

# ---------------------------------------------------------------------------
# Slide 6: Architecture
# ---------------------------------------------------------------------------
slide = new_slide()
add_kicker(slide, "Methodology")
add_title(slide, "Architecture")
add_pipeline(
    slide,
    [
        ("Extraction & Skill Classification", False),
        ("Shortlisting", False),
        ("Structured Assessment via Jev — parallel calls", True),
        ("Direct Sort", True),
        ("Score Breakdown Test (offline)", False),
    ],
    top=Inches(1.5),
)
add_textbox(
    slide, Inches(1.0), Inches(6.05), SLIDE_W - Inches(2.0), Inches(0.6),
    "Steps 1–2 are unchanged from the prior system. Contributions 1 and 2 are steps 3 and 4.",
    size=15, align=PP_ALIGN.CENTER, color=GRAY, italic=True,
)
add_footer(slide, "Section III — Methodology")


def add_compare(slide, left_title, left_items, right_title, right_items,
                 top=Inches(2.0), height=Inches(4.0)):
    box_w = Inches(5.3)
    left_x = Inches(0.8)
    right_x = SLIDE_W - Inches(0.8) - box_w

    outline_box(slide, left_x, top, box_w, height)
    add_textbox(slide, left_x + Inches(0.2), top + Inches(0.15), box_w - Inches(0.4), Inches(0.5),
                left_title, size=18, bold=True, align=PP_ALIGN.CENTER)
    add_bullets(slide, left_x + Inches(0.3), top + Inches(0.85), box_w - Inches(0.6),
                height - Inches(1.0), left_items, size=15)

    outline_box(slide, right_x, top, box_w, height)
    add_textbox(slide, right_x + Inches(0.2), top + Inches(0.15), box_w - Inches(0.4), Inches(0.5),
                right_title, size=18, bold=True, align=PP_ALIGN.CENTER)
    add_bullets(slide, right_x + Inches(0.3), top + Inches(0.85), box_w - Inches(0.6),
                height - Inches(1.0), right_items, size=15)

    arrow_x = left_x + box_w
    arrow_w = right_x - arrow_x
    add_textbox(slide, arrow_x, top + height / 2 - Inches(0.3), arrow_w, Inches(0.6),
                "→", size=28, align=PP_ALIGN.CENTER)


# ---------------------------------------------------------------------------
# Slide 7: Contribution 1 - Criteria-Grounded Question Design
# ---------------------------------------------------------------------------
slide = new_slide()
add_kicker(slide, "Methodology — Contribution 1")
add_title(slide, "Criteria-Grounded Question Design")
add_compare(
    slide,
    "Bare Ordinal Scale",
    ["0, 25, 50, 75, 100", "Each level judged alone", "No evidence to check against"],
    "Concrete Situation",
    ["“Skill only listed” vs. “used in a real role”",
     "Each level still judged alone", "Gives Jev resume evidence to match"],
)
add_footer(slide, "Result: per-requirement confidence 0.672 → 0.894 (Results, ahead)")

# ---------------------------------------------------------------------------
# Slide 8: Contribution 2 - Direct Sort Ranking
# ---------------------------------------------------------------------------
slide = new_slide()
add_kicker(slide, "Methodology — Contribution 2")
add_title(slide, "Direct Sort Ranking")
add_compare(
    slide,
    "Iterative Tournament",
    ["Bayesian Plackett-Luce", "Round-by-round, waits on each iteration", "No parallelism"],
    "Direct Sort",
    ["One score per candidate", "Sort directly — no tournament", "Fully parallel"],
)
add_footer(slide, "Result: well under half the model calls, ~4.5× lower latency (Results, ahead)")

# ---------------------------------------------------------------------------
# Slide 9: Experimental Setup
# ---------------------------------------------------------------------------
slide = new_slide()
add_kicker(slide, "Experimental Setup")
add_title(slide, "Experimental Setup")
stats = [
    ("273", "shortlisted (job, applicant) pairs"),
    ("10", "job profiles"),
    ("501", "resumes (500 synthetic + 1 real)"),
    ("3×", "independent repeats per pair"),
]
box_w = Inches(2.9)
gap = Inches(0.3)
total_w = box_w * 4 + gap * 3
x0 = (SLIDE_W - total_w) / 2
top = Inches(2.2)
box_h = Inches(2.0)
for i, (num, label) in enumerate(stats):
    x = x0 + i * (box_w + gap)
    outline_box(slide, x, top, box_w, box_h)
    add_textbox(slide, x, top + Inches(0.3), box_w, Inches(0.9), num, size=36, bold=True, align=PP_ALIGN.CENTER)
    add_textbox(slide, x + Inches(0.15), top + Inches(1.2), box_w - Inches(0.3), Inches(0.7),
                label, size=13, align=PP_ALIGN.CENTER, color=GRAY)
add_textbox(
    slide, Inches(1.0), top + box_h + Inches(0.5), SLIDE_W - Inches(2.0), Inches(0.8),
    "One variable changed between systems: the assessment step. 46.1% of extracted required skills were classified must-have.",
    size=15, align=PP_ALIGN.CENTER, color=GRAY, italic=True,
)
add_footer(slide, "Section IV — Experimental Setup")

out_path = Path(__file__).parent / "icaides2026_jev.pptx"
prs.save(out_path)
print("Saved", out_path, "with", len(prs.slides), "slides")
