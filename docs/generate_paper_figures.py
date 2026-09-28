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

    fig, ax = plt.subplots(figsize=(12, 4.2))
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


def main():
    score_ablation_line_chart()
    ranking_stability_bar_chart()
    print("Figures written to", FIGURES_DIR)


if __name__ == "__main__":
    main()
