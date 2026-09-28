"""Generate Figures for paper2_jev.tex (default matplotlib/seaborn look).

Numbers are taken verbatim from the tables they replace in docs/paper2_jev.tex.
Unlike docs/presentation/generate_charts.py (monochrome, deck-only), these use
seaborn's default theme and default color palette -- no custom styling.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def score_ablation_line_chart(filename: str = "score_ablation_chart.png"):
    groups = ["Per-\nrequirement", "Seniority\n(years)", "Education", "Pooled\n(sen. + edu.)"]
    concrete = [0.894, 0.789, 0.759, 0.766]
    ordinal = [0.672, 0.536, 0.669, 0.635]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(groups, concrete, marker="o", label="Concrete situational criteria")
    ax.plot(groups, ordinal, marker="o", label="Bare ordinal (0--100)")

    for xi, values in enumerate((concrete, ordinal)):
        for x, y in zip(groups, values):
            ax.annotate(
                f"{y:.3f}",
                xy=(x, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    ax.set_title("Confidence: Concrete vs. Bare Ordinal Criteria")
    ax.set_ylabel("Mean confidence")
    ax.set_xlabel("Question group")
    ax.set_ylim(0.4, 1.0)
    ax.tick_params(axis="x", labelrotation=0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close(fig)


def ranking_stability_bar_chart(filename: str = "ranking_stability_chart.png"):
    profiles = [
        "IT security\nengineer",
        "Full-stack\nengineer",
        "Backend\nengineer",
        "Java\ndeveloper",
        "Data\nengineer",
        "Data scientist\nengineer",
        "DevOps\nengineer",
        "Cloud\nengineer",
        "UI/UX\ndesigner",
        "Frontend\nengineer",
    ]
    tau = [0.943, 0.948, 0.951, 0.952, 0.960, 0.965, 0.965, 0.969, 1.000, 1.000]

    fig, ax = plt.subplots(figsize=(12, 7.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars = ax.bar(profiles, tau, width=0.6)

    for bar, value in zip(bars, tau):
        ax.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            clip_on=False,
        )

    ax.set_title("Ranking Stability by Job Profile")
    ax.set_ylabel("Kendall's $\\tau$")
    ax.set_xlabel("Job Profile")
    ax.set_ylim(0.9, 1.0)
    ax.tick_params(axis="x", labelrotation=0, labelsize=10)
    for label in ax.get_xticklabels():
        label.set_ha("center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close(fig)


def _draw_box(ax, cx, cy, w, h, title, body, shape="rect"):
    """Draw one pipeline node (rectangle, parallelogram, or diamond) with a
    bold title near the top and a normal-weight body below it, generously
    padded so text never crowds the border."""
    if shape == "rect":
        x0, y0 = cx - w / 2, cy - h / 2
        ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, edgecolor="black", linewidth=1.3, zorder=2))
    elif shape == "parallelogram":
        skew = w * 0.18
        pts = [
            (cx - w / 2 + skew, cy + h / 2),
            (cx + w / 2 + skew, cy + h / 2),
            (cx + w / 2 - skew, cy - h / 2),
            (cx - w / 2 - skew, cy - h / 2),
        ]
        ax.add_patch(plt.Polygon(pts, closed=True, fill=False, edgecolor="black", linewidth=1.3, zorder=2))
    elif shape == "diamond":
        pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
        ax.add_patch(plt.Polygon(pts, closed=True, fill=False, edgecolor="black", linewidth=1.3, zorder=2))

    # Vertically center the whole title+body text block in the box (rather than
    # pinning the title to a fixed offset from the top), so line count doesn't
    # cause the title and body to collide -- this is what made the original
    # diagram's boxes feel cramped once it accumulated enough lines.
    title_lh, body_lh, gap = 0.42, 0.40, 0.35
    n_title = title.count("\n") + 1
    n_body = body.count("\n") + 1 if body else 0
    block_h = n_title * title_lh + (gap + n_body * body_lh if body else 0)
    title_y = cy + block_h / 2
    ax.text(cx, title_y, title, ha="center", va="top", fontsize=9.3, fontweight="bold", linespacing=1.5, zorder=3)
    if body:
        body_y = title_y - n_title * title_lh - gap
        ax.text(cx, body_y, body, ha="center", va="top", fontsize=8.6, linespacing=1.6, zorder=3)


def _straight_arrow(ax, xy_from, xy_to):
    ax.annotate(
        "",
        xy=xy_to,
        xytext=xy_from,
        arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.3, mutation_scale=14),
        zorder=1,
    )


def _elbow_arrow(ax, points):
    """Draw a right-angle multi-segment connector through `points`, with a
    single arrowhead on the final segment."""
    for (x0, y0), (x1, y1) in zip(points[:-2], points[1:-1]):
        ax.plot([x0, x1], [y0, y1], color="black", linewidth=1.3, zorder=1)
    _straight_arrow(ax, points[-2], points[-1])


def architecture_diagram(filename: str = "Overall Architecture of the Proposed Applicant Ranking Methodology.png"):
    """Redraw of the pipeline architecture diagram with taller nodes and
    generous internal/row padding (the original hand-made version packed
    5-6 lines of text into very short boxes). Content and pipeline logic
    are unchanged from docs/paper1.tex Section III, except the retry loop
    out of "Regenerate Assessment" is corrected to re-enter "3. Applicant
    Assessment" (the original diagram mistakenly routed it into box 2,
    contradicting the paper's own text)."""
    fig, ax = plt.subplots(figsize=(15.5, 11))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(-1.8, 17.8)
    ax.set_ylim(-7.0, 12.0)
    ax.axis("off")

    col_x = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0]
    box_w, box_h = 2.2, 5.2
    dia_w, dia_h = 2.8, 5.6
    row1_cy, row2_cy = 6.0, -3.4

    # --- Row 1 ---
    _draw_box(ax, col_x[0], row1_cy, box_w, box_h + 0.2,
              "Input", "Applicant Resumes/\nJob Profile", shape="parallelogram")
    _draw_box(ax, col_x[1], row1_cy, box_w, box_h,
              "1. Skill Extraction\n(LLM)",
              "Extract required skills &\nattribute from Job Profile\n\nExtract skills & attribute\nfrom Resumes")
    _draw_box(ax, col_x[2], row1_cy, box_w, box_h,
              "2. Semantic\nMatching &\nShortlisting",
              "Compute embedding &\ncosine similarity per\nskill pair\n\nShortlist top-N\nCandidates")
    _draw_box(ax, col_x[3], row1_cy, box_w, box_h,
              "3. Applicant\nAssessment (LLM)",
              "Generate strengths,\nweaknesses, and\nadditional skills\ngrounded in the resume")
    _draw_box(ax, col_x[4], row1_cy, dia_w, dia_h,
              "4.", "Does a\nweakness contradict\nan extracted\nskill?", shape="diamond")
    _draw_box(ax, col_x[5], row1_cy, box_w, box_h,
              "Regenerate\nAssessment", "(1 retry)")

    # --- Row 2 ---
    _draw_box(ax, col_x[0], row2_cy, box_w, box_h + 0.2,
              "Output", "Global ranking of\napplicants across\nall job profiles", shape="parallelogram")
    _draw_box(ax, col_x[1], row2_cy, box_w, box_h,
              "8. Final Applicant\nRanking",
              "Average utilities\nacross stability\nrepeats\n\nSort applicants from\nbest to worst by\nglobal utilities")
    _draw_box(ax, col_x[2], row2_cy, box_w, box_h,
              "7. Plackett-Luce\nAggregation",
              "Refit applicant\nutilities ($\\theta$) from\nevery ranking so far\n\nUpdate uncertainty\nestimate")
    _draw_box(ax, col_x[3], row2_cy, box_w, box_h,
              "6. Listwise\nTournament\nRanking (LLM)",
              "Rank the selected\nsubset\n\nProduce partial\nranking orders")
    _draw_box(ax, col_x[4], row2_cy, box_w, box_h,
              "5. MC-KG Subset\nSelection",
              "Estimate expected\ninformation gain per\ncandidate subset.\n\nSelect the most\ninformative subset for\nthe next comparisons")
    _draw_box(ax, col_x[5], row2_cy, box_w, box_h,
              "9. Evaluation\nMetric",
              "Faithfulness\nWeakness\nContradiction Rate\nKendall's $\\tau$\nconvergence")

    half_w, dia_half_w = box_w / 2, dia_w / 2
    row1_top, row1_bot = row1_cy + box_h / 2, row1_cy - box_h / 2
    dia_top, dia_bot = row1_cy + dia_h / 2, row1_cy - dia_h / 2
    row2_top, row2_bot = row2_cy + box_h / 2, row2_cy - box_h / 2

    # Row 1 forward flow: Input -> 1 -> 2 -> 3 -> 4
    for i in range(4):
        right_edge = col_x[i] + (dia_half_w if i == 4 else half_w)
        left_edge = col_x[i + 1] - (dia_half_w if i + 1 == 4 else half_w)
        _straight_arrow(ax, (right_edge, row1_cy), (left_edge, row1_cy))

    # 4 "Yes" -> Regenerate Assessment
    _straight_arrow(ax, (col_x[4] + dia_half_w, row1_cy), (col_x[5] - half_w, row1_cy))
    ax.text((col_x[4] + dia_half_w + col_x[5] - half_w) / 2, row1_cy + 0.35, "Yes", fontsize=8.6, ha="center")

    # 4 "No" -> down to 5
    _straight_arrow(ax, (col_x[4], dia_bot), (col_x[4], row2_top))
    ax.text(col_x[4] + 0.25, (dia_bot + row2_top) / 2, "No", fontsize=8.6, ha="left")

    # Regenerate Assessment -> loops back into 3. Applicant Assessment (retry)
    loop1_y = row1_top + 0.7
    _elbow_arrow(
        ax,
        [(col_x[5], row1_top), (col_x[5], loop1_y), (col_x[3] + 0.3, loop1_y), (col_x[3] + 0.3, row1_top)],
    )

    # 3. Applicant Assessment -> 9. Evaluation Metric (Faithfulness), routed above the retry loop
    loop2_y = row1_top + 1.6
    margin_x = col_x[5] + half_w + 0.85
    _elbow_arrow(
        ax,
        [
            (col_x[3] - 0.3, row1_top), (col_x[3] - 0.3, loop2_y),
            (margin_x, loop2_y), (margin_x, row2_cy + 0.6), (col_x[5] + half_w, row2_cy + 0.6),
        ],
    )

    # Row 2 flow: 5 -> 6 -> 7 -> 8 -> Output (right to left)
    for i in range(4, 0, -1):
        left_edge = col_x[i] - half_w
        right_edge = col_x[i - 1] + half_w
        _straight_arrow(ax, (left_edge, row2_cy), (right_edge, row2_cy))

    # 7. Plackett-Luce Aggregation -> 9. Evaluation Metric (Kendall's tau), via the row gap
    gap_y = row2_top + 0.8
    _elbow_arrow(
        ax,
        [(col_x[2], row2_top), (col_x[2], gap_y), (col_x[5], gap_y), (col_x[5], row2_top)],
    )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close(fig)


def main():
    score_ablation_line_chart()
    ranking_stability_bar_chart()
    architecture_diagram()
    print("Figures written to", FIGURES_DIR)


if __name__ == "__main__":
    main()
