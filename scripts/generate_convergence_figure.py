"""Render Table III (ranking convergence by job profile) as a line-chart figure.

Reads the same underlying data the table's numbers come from -- tau_1 (single
unaggregated calls) from the no-must-have-gate run's evaluation report, and
tau_3 (three-call averages) from its multi-call-averaging validation -- and
plots them per job profile, using seaborn's default theme (no custom
palette/styling).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_convergence(run_id: str) -> list[tuple[str, int, float, float]]:
    run_dir = PROJECT_ROOT / "runs" / run_id / "evaluation"
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    mc = json.loads((run_dir / "multi_call_averaging_validation.json").read_text(encoding="utf-8"))

    tau1_by_jd = report["ranking_convergence"]["per_job_profile"]
    tau3_by_jd = mc["per_job_profile"]

    rows = [
        (jd_id, stats["n_candidates"], stats["mean_kendall_tau"], tau3_by_jd[jd_id]["mean_kendall_tau"])
        for jd_id, stats in tau1_by_jd.items()
    ]
    rows.sort(key=lambda r: r[2])
    return rows


def main(run_id: str, output_path: Path) -> None:
    rows = load_convergence(run_id)
    labels = [jd_id for jd_id, _, _, _ in rows]
    tau1 = [t1 for _, _, t1, _ in rows]
    tau3 = [t3 for _, _, _, t3 in rows]

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(labels, tau1, marker="o", label=r"$\tau_1$ (single call)")
    ax.plot(labels, tau3, marker="o", label=r"$\tau_3$ (3-call average)")

    wrapped_labels = [label.replace("-", "\n") for label in labels]

    ax.set_ylabel("Kendall-tau")
    ax.set_title("Ranking convergence by job profile")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(wrapped_labels, rotation=0, fontsize=11)
    ax.legend(loc="lower right")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="20260921-nomusthave-195128")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "docs" / "fig_convergence.png"))
    args = parser.parse_args()
    main(args.run_id, Path(args.output))
