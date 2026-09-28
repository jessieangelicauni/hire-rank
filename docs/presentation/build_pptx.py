from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

BLACK = RGBColor(0x1A, 0x1A, 0x1A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY = RGBColor(0x6B, 0x6B, 0x6B)
LIGHT_RULE = RGBColor(0xD9, 0xD9, 0xD9)
ACCENT = RGBColor(0x16, 0x3A, 0x5F)
FONT = "Calibri"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN = Inches(0.85)
CONTENT_W = SLIDE_W - 2 * MARGIN

TOTAL_SLIDES = 14
SHORT_TITLE = "Reliable Structured-Decision Applicant Ranking"

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]

_slide_counter = {"n": 0}


def new_slide():
    slide = prs.slides.add_slide(BLANK)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    _slide_counter["n"] += 1
    return slide


def add_textbox(slide, x, y, w, h, text, size=16, bold=False, align=PP_ALIGN.LEFT,
                 color=BLACK, italic=False, font=FONT, anchor=None, line_spacing=None):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    if anchor is not None:
        tf.vertical_anchor = anchor
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        if line_spacing:
            p.line_spacing = line_spacing
        for run in p.runs:
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.italic = italic
            run.font.name = font
            run.font.color.rgb = color
    return box


def add_rule(slide, x, y, w, color=LIGHT_RULE, weight=1.0):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x, y, x + w, y)
    line.line.color.rgb = color
    line.line.width = Pt(weight)
    return line


def add_vrule(slide, x, y, h, color=LIGHT_RULE, weight=1.0):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x, y, x, y + h)
    line.line.color.rgb = color
    line.line.width = Pt(weight)
    return line


def header(slide, section_label, title, title_size=27):
    spaced = "  ".join(list(section_label.upper()))
    add_textbox(slide, MARGIN, Inches(0.5), CONTENT_W, Inches(0.3),
                spaced, size=11, bold=True, color=ACCENT)
    add_rule(slide, MARGIN, Inches(0.82), CONTENT_W, color=ACCENT, weight=1.5)
    add_textbox(slide, MARGIN, Inches(0.98), CONTENT_W, Inches(0.85),
                title, size=title_size, bold=True, color=BLACK)


def footer(slide):
    y = SLIDE_H - Inches(0.5)
    add_rule(slide, MARGIN, y, CONTENT_W, color=LIGHT_RULE, weight=0.75)
    add_textbox(slide, MARGIN, y + Inches(0.06), Inches(8.0), Inches(0.35),
                SHORT_TITLE, size=10, color=GRAY)
    add_textbox(slide, SLIDE_W - MARGIN - Inches(1.5), y + Inches(0.06), Inches(1.5), Inches(0.35),
                f"{_slide_counter['n']:02d} / {TOTAL_SLIDES}", size=10, color=GRAY, align=PP_ALIGN.RIGHT)


def bullet_list(slide, x, y, w, items, size=16, gap=Pt(14), heading=None, heading_size=15):
    cy = y
    if heading:
        add_textbox(slide, x, cy, w, Inches(0.35), heading, size=heading_size, bold=True, color=ACCENT)
        cy += Inches(0.42)
    for item in items:
        mark = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, cy + Pt(7), Pt(7), Pt(7))
        mark.fill.solid()
        mark.fill.fore_color.rgb = ACCENT
        mark.line.fill.background()
        mark.shadow.inherit = False
        tb = add_textbox(slide, x + Inches(0.28), cy, w - Inches(0.28), Inches(0.55), item, size=size)
        n_lines = max(1, len(item) // 46 + 1)
        cy += Pt(size * 1.28) * n_lines + gap
    return cy


def two_column(slide, top, left_heading, left_items, right_heading, right_items, size=15.5):
    col_w = (CONTENT_W - Inches(0.6)) / 2
    left_x = MARGIN
    right_x = MARGIN + col_w + Inches(0.6)
    bullet_list(slide, left_x, top, col_w, left_items, size=size, heading=left_heading)
    add_vrule(slide, MARGIN + col_w + Inches(0.3), top, Inches(3.6), color=LIGHT_RULE, weight=1.0)
    bullet_list(slide, right_x, top, col_w, right_items, size=size, heading=right_heading)


def process_step(slide, x, y, w, number, title, desc, desc_size=12.5):
    d = Inches(0.5)
    circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, x, y, d, d)
    circle.fill.background()
    circle.line.color.rgb = ACCENT
    circle.line.width = Pt(1.5)
    circle.shadow.inherit = False
    tf = circle.text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = str(number)
    p.alignment = PP_ALIGN.CENTER
    for run in p.runs:
        run.font.size = Pt(15)
        run.font.bold = True
        run.font.name = FONT
        run.font.color.rgb = ACCENT
    add_textbox(slide, x + d + Inches(0.22), y - Inches(0.02), w - d - Inches(0.22), Inches(0.4),
                title, size=15.5, bold=True, color=BLACK)
    add_textbox(slide, x + d + Inches(0.22), y + Inches(0.38), w - d - Inches(0.22), Inches(0.7),
                desc, size=desc_size, color=GRAY)


def process_arrow(slide, x, y, size=13):
    add_textbox(slide, x, y, Inches(0.5), Inches(0.4), "→", size=size, color=ACCENT, align=PP_ALIGN.CENTER)


def metric_row(slide, top, metrics, num_size=40, height=Inches(1.7)):
    n = len(metrics)
    seg_w = CONTENT_W / n
    for i, (num, label) in enumerate(metrics):
        x = MARGIN + i * seg_w
        add_textbox(slide, x, top, seg_w - Inches(0.3), Inches(0.95), num,
                    size=num_size, bold=True, color=ACCENT, align=PP_ALIGN.LEFT)
        add_rule(slide, x, top + Inches(0.95), min(seg_w - Inches(0.5), Inches(1.6)),
                 color=LIGHT_RULE, weight=1.0)
        add_textbox(slide, x, top + Inches(1.05), seg_w - Inches(0.3), Inches(0.55), label,
                    size=12, color=GRAY, align=PP_ALIGN.LEFT)
        if i < n - 1:
            add_vrule(slide, x + seg_w - Inches(0.3), top, height, color=LIGHT_RULE, weight=0.75)


def bar_row(slide, x, y, w, label, value, max_value, bar_h=Inches(0.34),
            label_w=Inches(2.5), value_fmt="{:.3f}", unit="", color=ACCENT, value_bold=False):
    add_textbox(slide, x, y, label_w - Inches(0.15), bar_h + Inches(0.05), label,
                size=13, color=BLACK, anchor=MSO_ANCHOR.MIDDLE)
    track_x = x + label_w
    track_w = w - label_w - Inches(1.15)
    add_rule(slide, track_x, y + bar_h, track_w, color=LIGHT_RULE, weight=1.0)
    fill_w = max(Inches(0.04), track_w * (value / max_value))
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, track_x, y + Inches(0.03), fill_w, bar_h - Inches(0.06))
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()
    bar.shadow.inherit = False
    add_textbox(slide, track_x + track_w + Inches(0.1), y, Inches(1.0), bar_h + Inches(0.05),
                f"{value_fmt.format(value)}{unit}", size=14, bold=value_bold, color=BLACK,
                anchor=MSO_ANCHOR.MIDDLE)


def hero_stat(slide, top, big_text, caption, sub=None, size=58):
    add_textbox(slide, MARGIN, top, CONTENT_W, Inches(1.3), big_text,
                size=size, bold=True, color=ACCENT, align=PP_ALIGN.LEFT)
    add_rule(slide, MARGIN, top + Inches(1.3), Inches(2.2), color=ACCENT, weight=2.0)
    add_textbox(slide, MARGIN, top + Inches(1.42), CONTENT_W - Inches(1.0), Inches(0.6),
                caption, size=17, color=BLACK)
    if sub:
        add_textbox(slide, MARGIN, top + Inches(2.05), CONTENT_W - Inches(1.5), Inches(0.9),
                    sub, size=13, color=GRAY, italic=True)


slide = new_slide()
add_rule(slide, MARGIN, Inches(0.9), Inches(1.6), color=ACCENT, weight=2.5)
add_textbox(slide, MARGIN, Inches(1.05), CONTENT_W, Inches(0.35),
            "ICAIDES 2026  ·  UNIVERSITAS TARUMANAGARA  ·  12–13 NOV 2026",
            size=12, bold=True, color=ACCENT)
add_textbox(
    slide, MARGIN, Inches(2.1), CONTENT_W - Inches(1.0), Inches(2.2),
    "Reliable Structured-Decision Applicant Ranking with a Non-Autoregressive System One Model",
    size=34, bold=True, color=BLACK, line_spacing=1.05,
)
add_textbox(
    slide, MARGIN, Inches(4.5), CONTENT_W, Inches(0.5),
    "Jessie Angelica   ·   Hasanul Fahmi Zuhri   ·   Rusdianto Roestam   ·   Muhammad Arief",
    size=16, color=BLACK,
)
add_textbox(
    slide, MARGIN, Inches(4.95), CONTENT_W, Inches(0.45),
    "President University  ·  UNITAR International University  ·  BRIN",
    size=12.5, color=GRAY,
)
add_rule(slide, MARGIN, Inches(5.7), CONTENT_W, color=LIGHT_RULE, weight=1.0)
add_textbox(
    slide, MARGIN, Inches(5.9), CONTENT_W - Inches(2.0), Inches(0.9),
    "Can a model that only fills in structured answers rank applicants as reliably as one that writes free text — at a fraction of the cost?",
    size=16, italic=True, color=BLACK,
)

slide = new_slide()
header(slide, "Introduction", "Motivation & Gap")
two_column(
    slide, Inches(1.75),
    "Free-Text Assessment",
    [
        "Writes a paragraph verdict per applicant",
        "Can assert a strength the resume never supports",
        "No built-in confidence signal",
    ],
    "The Gap This Paper Fills",
    [
        "TypeSafe released Jev in September 2026",
        "The first “System One Model”",
        "No independent evaluation existed before this paper",
    ],
)
footer(slide)

slide = new_slide()
header(slide, "Introduction", "Objective & Three Contributions")
add_textbox(slide, MARGIN, Inches(1.7), CONTENT_W, Inches(0.5),
            "Jev is TypeSafe's model, not ours. Our contribution: deploying it reliably.",
            size=16, color=GRAY, italic=True)
contribs = [
    ("01", "Criteria-Grounded\nQuestion Design", "Concrete situational criteria, not bare ordinal labels"),
    ("02", "Cost & Latency\nEfficiency", "Direct sort replaces the iterative tournament"),
    ("03", "Repeat-to-Repeat\nReliability", "Stability tests for a single-call architecture"),
]
top = Inches(2.6)
col_w = CONTENT_W / 3
for i, (num, title, desc) in enumerate(contribs):
    x = MARGIN + i * col_w
    add_textbox(slide, x, top, col_w - Inches(0.4), Inches(0.9), num, size=44, bold=True, color=ACCENT)
    add_rule(slide, x, top + Inches(0.95), Inches(1.3), color=ACCENT, weight=1.5)
    add_textbox(slide, x, top + Inches(1.1), col_w - Inches(0.5), Inches(0.8), title, size=17, bold=True, color=BLACK)
    add_textbox(slide, x, top + Inches(2.0), col_w - Inches(0.5), Inches(0.9), desc, size=12.5, color=GRAY)
footer(slide)

slide = new_slide()
header(slide, "Background", "What Is a System One Model")
add_textbox(slide, MARGIN, Inches(1.75), CONTENT_W, Inches(0.5),
            "Non-autoregressive: answers typed questions in one parallel pass — no generated tokens.",
            size=17, color=BLACK)
primitives = [
    ("Noul", "yes / no + confidence", False),
    ("Choice", "multiple-choice", False),
    ("Score", "ordinal scale — used throughout this paper", True),
]
top = Inches(2.85)
col_w = CONTENT_W / 3
for i, (name, desc, selected) in enumerate(primitives):
    x = MARGIN + i * col_w
    add_textbox(slide, x, top, col_w - Inches(0.4), Inches(0.7), name,
                size=26, bold=True, color=ACCENT if selected else BLACK)
    add_rule(slide, x, top + Inches(0.78), Inches(1.6), color=ACCENT if selected else LIGHT_RULE,
             weight=2.0 if selected else 1.0)
    add_textbox(slide, x, top + Inches(0.95), col_w - Inches(0.5), Inches(0.9), desc,
                size=13.5, color=GRAY if not selected else BLACK)
footer(slide)

slide = new_slide()
header(slide, "Background", "Why It Works: Criteria Grounding & Calibration")
col_w = (CONTENT_W - Inches(1.0)) / 2
top = Inches(1.85)
add_textbox(slide, MARGIN, top, col_w, Inches(0.4), "BARE LABEL", size=12, bold=True, color=ACCENT)
add_textbox(slide, MARGIN, top + Inches(0.4), col_w, Inches(0.8), "“50”", size=30, bold=True, color=BLACK)
add_textbox(slide, MARGIN, top + Inches(1.3), col_w - Inches(0.3), Inches(1.0),
            "Judged alone, no neighboring levels to compare — no evidence for Jev to check.",
            size=14, color=GRAY)
right_x = MARGIN + col_w + Inches(1.0)
add_textbox(slide, right_x, top, col_w, Inches(0.4), "CONCRETE SITUATION", size=12, bold=True, color=ACCENT)
add_textbox(slide, right_x, top + Inches(0.4), col_w, Inches(0.8),
            "“Used substantively\nin a real role”", size=22, bold=True, color=BLACK, line_spacing=1.0)
add_textbox(slide, right_x, top + Inches(1.55), col_w - Inches(0.3), Inches(1.0),
            "Gives Jev resume evidence it can actually match against.",
            size=14, color=GRAY)
process_arrow(slide, MARGIN + col_w + Inches(0.15), top + Inches(0.55), size=22)
add_rule(slide, MARGIN, Inches(4.5), CONTENT_W, color=LIGHT_RULE, weight=1.0)
add_textbox(
    slide, MARGIN, Inches(4.7), CONTENT_W - Inches(1.0), Inches(1.0),
    "Trained via RLCD — rewarded for calibration, not human preference (RLHF) or a verifiable outcome (RLVR). Calibration ≠ correctness.",
    size=16, color=BLACK, italic=True,
)
footer(slide)

slide = new_slide()
header(slide, "Methodology", "Pipeline Architecture")
steps = [
    ("Extraction & Skill Classification", "Qwen2.5-14B extracts skills, classifies must-have vs. nice-to-have"),
    ("Shortlisting", "Cosine similarity ≥ 0.8 against job requirements, no Jev call"),
    ("Structured Assessment via Jev", "One parallel call per shortlisted pair — Contribution 1; each call also repeated 3× for reliability — Contribution 3"),
    ("Direct Sort", "Rank by overall score, no tournament — Contribution 2"),
    ("Final Ranking", "One ranked list per job profile"),
]
top = Inches(1.7)
row_h = Inches(0.88)
for i, (title, desc) in enumerate(steps):
    y = top + i * row_h
    process_step(slide, MARGIN, y, CONTENT_W, i + 1, title, desc, desc_size=12.5)
    if i < len(steps) - 1:
        add_vrule(slide, MARGIN + Inches(0.25), y + Inches(0.5), row_h - Inches(0.5), color=LIGHT_RULE, weight=1.0)
add_textbox(slide, MARGIN, top + len(steps) * row_h + Inches(0.05), CONTENT_W - Inches(1.0), Inches(0.4),
            "All three contributions live in step 3 and step 4 — extraction, shortlisting, and final ranking are unchanged from the prior system.",
            size=12.5, color=GRAY, italic=True)
footer(slide)

slide = new_slide()
header(slide, "Methodology — Contribution 1", "Criteria-Grounded Question Design")
two_column(
    slide, Inches(1.85),
    "Bare Ordinal Scale",
    ["0, 25, 50, 75, 100", "Each level judged alone", "No evidence to check against"],
    "Concrete Situational Criteria",
    ["“Skill only listed” vs. “used in a real role”",
     "Each level still judged alone", "Gives Jev resume evidence to match"],
)
add_rule(slide, MARGIN, Inches(6.35), CONTENT_W, color=LIGHT_RULE, weight=1.0)
add_textbox(slide, MARGIN, Inches(6.5), CONTENT_W, Inches(0.4),
            "Result ahead: per-requirement confidence 0.672 → 0.894", size=13.5, italic=True, color=ACCENT)
footer(slide)

slide = new_slide()
header(slide, "Methodology — Contribution 2", "Direct Sort Ranking")
two_column(
    slide, Inches(1.85),
    "Iterative Tournament",
    ["Monte Carlo knowledge-gradient + Plackett-Luce", "Round-by-round, waits on each iteration", "No parallelism"],
    "Direct Sort",
    ["One score per applicant", "Sort directly — no tournament", "Fully parallel"],
)
add_rule(slide, MARGIN, Inches(6.35), CONTENT_W, color=LIGHT_RULE, weight=1.0)
add_textbox(slide, MARGIN, Inches(6.5), CONTENT_W, Inches(0.4),
            "Result ahead: 273 vs. ≈2,061 model calls", size=13.5, italic=True, color=ACCENT)
footer(slide)

slide = new_slide()
header(slide, "Experimental Setup", "Experimental Setup")
metric_row(slide, Inches(1.9), [
    ("273", "shortlisted (job, applicant) pairs"),
    ("10", "job profiles"),
    ("500", "resumes — EraMatch v3.0, synthetic"),
    ("3×", "independent repeats per pair"),
])
add_rule(slide, MARGIN, Inches(4.35), CONTENT_W, color=LIGHT_RULE, weight=1.0)
add_textbox(
    slide, MARGIN, Inches(4.55), CONTENT_W - Inches(1.0), Inches(1.4),
    "One variable changed between systems: the assessment step. 46.1% of extracted requirements "
    "classified must-have. The must-have gate excludes 32 of 305 applicants (10.5%) a total-only "
    "baseline would have shortlisted.",
    size=15, color=BLACK,
)
footer(slide)

slide = new_slide()
header(slide, "Results", "Result: Criteria-Grounded Design Raises Confidence")
add_textbox(slide, MARGIN, Inches(1.75), CONTENT_W, Inches(0.35),
            "PER-REQUIREMENT CONFIDENCE", size=11.5, bold=True, color=ACCENT)
bar_row(slide, MARGIN, Inches(2.1), CONTENT_W, "Bare scale", 0.672, 1.0)
bar_row(slide, MARGIN, Inches(2.65), CONTENT_W, "Concrete criteria", 0.894, 1.0, value_bold=True)

add_textbox(slide, MARGIN, Inches(3.4), CONTENT_W, Inches(0.35),
            "POOLED SENIORITY / EDUCATION CONFIDENCE", size=11.5, bold=True, color=ACCENT)
bar_row(slide, MARGIN, Inches(3.75), CONTENT_W, "Bare scale", 0.635, 1.0)
bar_row(slide, MARGIN, Inches(4.3), CONTENT_W, "Concrete criteria", 0.766, 1.0, value_bold=True)

add_rule(slide, MARGIN, Inches(5.15), CONTENT_W, color=LIGHT_RULE, weight=1.0)
add_textbox(
    slide, MARGIN, Inches(5.35), CONTENT_W - Inches(1.0), Inches(1.1),
    "Wilcoxon signed-rank, all differences p < 0.001. Per-requirement: r = 0.89, 95% CI [0.204, 0.241] "
    "on the gain (n = 772). Effect holds for seniority, education, and pooled questions too.",
    size=13.5, color=GRAY, italic=True,
)
footer(slide)

slide = new_slide()
header(slide, "Results", "Result: Cost & Latency Efficiency")
col_w = (CONTENT_W - Inches(0.8)) / 2
top = Inches(1.85)
add_textbox(slide, MARGIN, top, col_w, Inches(0.35), "TOTAL MODEL CALLS", size=11.5, bold=True, color=ACCENT)
add_textbox(slide, MARGIN, top + Inches(0.35), col_w, Inches(0.9), "273 vs. ≈2,061",
            size=32, bold=True, color=BLACK)
add_textbox(slide, MARGIN, top + Inches(1.25), col_w - Inches(0.3), Inches(0.5),
            "well under half the prior system's calls", size=13, color=GRAY)

right_x = MARGIN + col_w + Inches(0.8)
add_vrule(slide, right_x - Inches(0.4), top, Inches(1.6), color=LIGHT_RULE, weight=1.0)
add_textbox(slide, right_x, top, col_w, Inches(0.35), "PER-CANDIDATE LATENCY", size=11.5, bold=True, color=ACCENT)
bar_row(slide, right_x, top + Inches(0.42), col_w, "This work", 16.5, 74.3, unit="s",
        value_fmt="{:.1f}", value_bold=True, label_w=Inches(1.6))
bar_row(slide, right_x, top + Inches(0.95), col_w, "Prior system", 74.3, 74.3, unit="s",
        value_fmt="{:.1f}", label_w=Inches(1.6))
add_textbox(slide, right_x, top + Inches(1.55), col_w - Inches(0.3), Inches(0.4),
            "live single-profile check", size=12, color=GRAY, italic=True)

add_rule(slide, MARGIN, Inches(4.1), CONTENT_W, color=LIGHT_RULE, weight=1.0)
add_textbox(
    slide, MARGIN, Inches(4.3), CONTENT_W - Inches(1.0), Inches(1.5),
    "Output priced at zero by design — Jev generates no tokens. Caveat: the same comparison table "
    "also reports Faithfulness 0.880 (prior) vs. calibrated confidence 0.894 (this work) — different "
    "constructs, groundedness vs. calibration, not on a shared scale. Not a claim of higher accuracy.",
    size=14, color=GRAY, italic=True,
)
footer(slide)

slide = new_slide()
header(slide, "Results", "Result: Reliability — Repeat-to-Repeat Stability")
hero_stat(
    slide, Inches(1.85),
    "ICC(1,1) = 0.998",
    "Intraclass correlation: the fraction of score variance across 3 repeats that is real applicant signal, not noise",
    sub="Range 0.995–1.000 across all 10 job profiles.",
    size=54,
)
add_rule(slide, MARGIN, Inches(4.55), CONTENT_W, color=LIGHT_RULE, weight=1.0)
add_textbox(slide, MARGIN, Inches(4.75), CONTENT_W, Inches(0.35),
            "RANKING STABILITY (KENDALL'S τ)", size=11.5, bold=True, color=ACCENT)
bar_row(slide, MARGIN, Inches(5.15), CONTENT_W, "This work", 0.965, 1.0, value_bold=True)
bar_row(slide, MARGIN, Inches(5.68), CONTENT_W, "Prior system", 0.957, 1.0)
add_textbox(
    slide, MARGIN, Inches(6.3), CONTENT_W - Inches(1.0), Inches(0.6),
    "At least matching, not a clear win — but no regression from a far cheaper architecture.",
    size=13, color=GRAY, italic=True,
)
footer(slide)

slide = new_slide()
header(slide, "Conclusion", "Conclusion")

add_textbox(slide, MARGIN, Inches(1.7), CONTENT_W, Inches(0.35),
            "OUTPERFORMS — COST & LATENCY", size=12, bold=True, color=ACCENT)
metric_row(slide, Inches(2.05), [
    ("273 vs. ≈2,061", "model calls"),
    ("16.5s vs. 74.3s", "per-applicant latency"),
    ("Removed by design", "free-text hallucination & schema risk"),
], num_size=22, height=Inches(0.95))

add_rule(slide, MARGIN, Inches(3.15), CONTENT_W, color=LIGHT_RULE, weight=1.0)
add_textbox(slide, MARGIN, Inches(3.3), CONTENT_W, Inches(0.35),
            "MATCHES — NOT A CLEAR WIN", size=12, bold=True, color=ACCENT)
add_textbox(slide, MARGIN, Inches(3.65), CONTENT_W - Inches(1.0), Inches(0.45),
            "Ranking stability τ = 0.965 vs. ≈0.957 — comparable, no regression from a much cheaper architecture.",
            size=14, color=BLACK)

add_rule(slide, MARGIN, Inches(4.35), CONTENT_W, color=LIGHT_RULE, weight=1.0)
add_textbox(slide, MARGIN, Inches(4.5), CONTENT_W, Inches(0.35),
            "LIMITATIONS & SCOPE", size=12, bold=True, color=ACCENT)
limitations = [
    "Calibration ≠ correctness — test against ground truth next",
    "Synthetic-resume corpus only — no fairness claim on real hiring data",
    "Screening aid, not a hire/reject decision — human review of evidence still required",
]
bullet_list(slide, MARGIN, Inches(4.9), CONTENT_W - Inches(1.0), limitations, size=14.5, gap=Pt(10))
footer(slide)

slide = new_slide()
add_rule(slide, MARGIN, Inches(2.9), Inches(1.6), color=ACCENT, weight=2.5)
add_textbox(slide, MARGIN, Inches(3.05), CONTENT_W, Inches(1.1),
            "Thank You", size=46, bold=True, color=BLACK)
add_textbox(slide, MARGIN, Inches(4.05), CONTENT_W, Inches(0.55),
            "Questions & Discussion", size=20, color=GRAY)
add_textbox(
    slide, MARGIN, Inches(5.3), CONTENT_W, Inches(0.9),
    "Jessie Angelica  ·  President University\nfahmi.zuhri@unitar.my",
    size=14, color=GRAY,
)

out_path = Path(__file__).parent / "icaides2026_jev.pptx"
prs.save(out_path)
print("Saved", out_path, "with", len(prs.slides), "slides")
