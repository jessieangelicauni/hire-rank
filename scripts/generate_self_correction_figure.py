from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, RegularPolygon

OUT_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "figures" /
    "Hallucination-Aware Self-Correction Mechanism for Applicant Assessment.png"
)

_BOX_STYLE = dict(boxstyle="round,pad=0.35", linewidth=1.2)
_COLORS = {
    "generation": "#cfe2f3",
    "reference": "#fff2cc",
    "check": "#d9ead3",
    "correction": "#f4cccc",
    "accepted": "#d9ead3",
}


def _box(ax, xy, width, height, text, color, fontsize=9.5):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height, facecolor=color, edgecolor="black", zorder=2, **_BOX_STYLE,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, zorder=3, wrap=True)
    return (x, y, width, height)


def _arrow(ax, start, end, **kwargs):
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color="black",
                             zorder=2.5, **kwargs)
    ax.add_patch(arrow)


def build_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    gen = _box(ax, (0.3, 5.7), 2.8, 1.6,
               "Assessment Generation (LLM)\n\nGenerates strengths, weaknesses,\nand additional_skills grounded\nin the CV",
               _COLORS["generation"], fontsize=9)
    ref = _box(ax, (0.3, 2.6), 2.8, 1.6,
               "Candidate's Extracted\nSkill List\n\n(from the earlier skill-\nextraction stage -- not a\nfresh CV evidence retrieval)",
               _COLORS["reference"], fontsize=9)
    check = _box(ax, (4.0, 3.6), 3.0, 2.0,
                 "Weakness Contradiction\nCheck (code)\n\nLexical absence-claim heuristic\n+ embedding-similarity bridge\nfrom JD-skill wording to the\ncandidate's own skill wording",
                 _COLORS["check"], fontsize=9)

    diamond_center = (9.3, 4.6)
    diamond_radius = 1.3
    diamond = RegularPolygon(diamond_center, numVertices=4, radius=diamond_radius, orientation=0,
                              facecolor="white", edgecolor="black", zorder=2)
    ax.add_patch(diamond)
    ax.text(*diamond_center, "Any weakness\ncontradicts an\nextracted skill?", ha="center", va="center",
            fontsize=8.5, zorder=3)
    d_top = (diamond_center[0], diamond_center[1] + diamond_radius)
    d_bottom = (diamond_center[0], diamond_center[1] - diamond_radius)
    d_left = (diamond_center[0] - diamond_radius, diamond_center[1])
    d_right = (diamond_center[0] + diamond_radius, diamond_center[1])

    correction = _box(ax, (7.7, 6.7), 3.2, 1.1,
                       "Self-Correction: regenerate once with\ncorrective feedback naming the\ncontradicted skill(s)",
                       _COLORS["correction"], fontsize=8.5)
    dropped = _box(ax, (11.3, 3.85), 2.9, 1.4,
                    "Still contradicts after retry:\ndrop the contradicting\nweakness item(s)",
                    _COLORS["correction"], fontsize=8.5)
    accepted = _box(ax, (7.7, 1.0), 3.2, 1.5,
                     "Accepted Assessment\n\n(strengths and additional_skills\nare not live-checked here)",
                     _COLORS["accepted"], fontsize=8.5)

    _arrow(ax, (gen[0] + gen[2], gen[1] + gen[3] * 0.3), (check[0], check[1] + check[3] * 0.85))
    _arrow(ax, (ref[0] + ref[2], ref[1] + ref[3] * 0.7), (check[0], check[1] + check[3] * 0.15))

    _arrow(ax, (check[0] + check[2], check[1] + check[3] / 2), d_left)

    _arrow(ax, d_top, (correction[0] + correction[2] / 2, correction[1]))
    ax.text(diamond_center[0] + 0.15, d_top[1] + 0.25, "yes", fontsize=9)
    _arrow(ax, (correction[0], correction[1] + correction[3] * 0.2),
           (check[0] + check[2] * 0.7, check[1] + check[3]))

    _arrow(ax, d_right, (dropped[0], dropped[1] + dropped[3] / 2))
    ax.text(d_right[0] + 0.15, d_right[1] + 0.45,
            "still contradicts\nafter 1 retry", fontsize=8, ha="left")

    _arrow(ax, d_bottom, (accepted[0] + accepted[2] / 2, accepted[1] + accepted[3]))
    ax.text(diamond_center[0] + 0.15, (d_bottom[1] + accepted[1] + accepted[3]) / 2, "no", fontsize=9)

    _arrow(ax, (dropped[0], dropped[1] + dropped[3] * 0.15), (accepted[0] + accepted[2], accepted[1] + accepted[3] * 0.3))

    fig.tight_layout()
    return fig


def main() -> None:
    fig = build_figure()
    fig.savefig(OUT_PATH, dpi=200)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
