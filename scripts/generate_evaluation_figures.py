"""Generates the two figures used in the IEEE-style evaluation report:

- Fig. 2, a within-repeat Kendall's Tau convergence trace per job role, computed
  from each role's primary tournament repeat (runs/<run-id>/<jd-id>/repeats.json)
  using the same convergence-trace math as candidate_ranking.evaluation.evaluation.
- Fig. 3, a Ragas Faithfulness bar chart per job role, from the exported
  console-web data and the run's Ragas faithfulness report.
"""
from __future__ import annotations

import math
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

matplotlib.style.use("default")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from candidate_ranking.ranking.plackett_luce import fit_utilities 

RUN_ID = "20260906-150825"
PL_PRIOR_VARIANCE = 1.0

_ACRONYMS = {"ui": "UI", "ux": "UX", "it": "IT"}


def _wrap_role_label(role_id: str) -> str:
    words = [_ACRONYMS.get(w, w.capitalize()) for w in role_id.split("-")]
    if len(words) == 1:
        return words[0]

    best_split, best_diff = 1, None
    for i in range(1, len(words)):
        diff = abs(len(" ".join(words[:i])) - len(" ".join(words[i:])))
        if best_diff is None or diff < best_diff:
            best_diff, best_split = diff, i
    return " ".join(words[:best_split]) + "\n" + " ".join(words[best_split:])


def compute_convergence_trace(iteration_history: list[dict], pl_prior_variance: float) -> list[dict]:
    if len(iteration_history) < 2:
        return []

    touched_sets: list[list[str]] = []
    seen: set[str] = set()
    for record in iteration_history:
        seen.update(record["subset_candidate_ids"])
        touched_sets.append(sorted(seen))

    utilities_at = []
    for i in range(len(iteration_history)):
        rankings_so_far = [r["ranking"] for r in iteration_history[: i + 1]]
        utilities_at.append(fit_utilities(rankings_so_far, touched_sets[i], pl_prior_variance))

    trace = []
    for i in range(1, len(iteration_history)):
        prev_ids, prev_u = touched_sets[i - 1], utilities_at[i - 1]
        curr_ids, curr_u = touched_sets[i], utilities_at[i]
        common = [cid for cid in prev_ids if cid in curr_ids]

        if len(common) < 2:
            tau = None
        else:
            prev_index = {cid: idx for idx, cid in enumerate(prev_ids)}
            curr_index = {cid: idx for idx, cid in enumerate(curr_ids)}
            prev_vals = [float(prev_u[prev_index[cid]]) for cid in common]
            curr_vals = [float(curr_u[curr_index[cid]]) for cid in common]
            result = kendalltau(prev_vals, curr_vals)
            tau = None if math.isnan(result.statistic) else float(result.statistic)

        trace.append({"iteration": iteration_history[i]["iteration"], "kendall_tau": tau})
    return trace


def _role_label(jd_id: str) -> str:
    return " ".join(_ACRONYMS.get(w, w.capitalize()) for w in jd_id.split("-"))


def _smooth(values: list[float], window: int = 7) -> list[float]:
    half = window // 2
    n = len(values)
    smoothed = []
    for i in range(n):
        window_slice = values[max(0, i - half):min(n, i + half + 1)]
        smoothed.append(sum(window_slice) / len(window_slice))
    return smoothed


def convergence_figure(run_dir: Path, jd_ids: list[str], out_path: Path) -> None:
    series = []
    for jd_id in jd_ids:
        repeats_path = run_dir / jd_id / "repeats.json"
        if not repeats_path.exists():
            continue
        repeats = json.loads(repeats_path.read_text())
        primary = next((r for r in repeats if r["repeat_index"] == 0 and r["status"] == "ok"), None)
        if primary is None:
            continue

        trace = compute_convergence_trace(primary["iteration_history"], PL_PRIOR_VARIANCE)
        points = [(p["iteration"], p["kendall_tau"]) for p in trace if p["kendall_tau"] is not None]
        if points:
            series.append({"jd_id": jd_id, "n": len(primary["candidate_ids"]), "points": points})

    series.sort(key=lambda s: s["n"])

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for s in series:
        xs = [p[0] for p in s["points"]]
        ys = _smooth([p[1] for p in s["points"]])
        ax.plot(xs, ys, linewidth=1.8, label=f"{_role_label(s['jd_id'])} (n={s['n']})")

    ax.set_title("Within-Repeat Rank Convergence by Job Role", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Tournament Iteration", fontsize=11, labelpad=8)
    ax.set_ylabel("Kendall's Tau (Rank Stability)", fontsize=11, labelpad=8)
    ax.set_ylim(0.8, 1.0)
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=2, loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _frange(start: float, stop: float, step: float) -> list[float]:
    n = round((stop - start) / step)
    return [start + i * step for i in range(n + 1)]


def bar_figure(labels: list[str], values: list[float], title: str, ylabel: str, out_path: Path) -> None:
    mean = sum(values) / len(values)

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    bars = ax.bar(labels, values, width=0.6, color="#4C72B0", zorder=3)
    ax.axhline(mean, color="black", linestyle="--", linewidth=1.2, zorder=2)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 0.008, f"{val:.3f}",
            ha="center", va="bottom", fontsize=8, zorder=4,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
        )

    floor = min(0.80, math.floor((min(values) - 0.02) * 20) / 20)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Role", fontsize=11, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    ax.set_ylim(floor, 1.01)
    ax.set_yticks([round(t, 2) for t in _frange(floor, 1.00, 0.05)])
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([_wrap_role_label(label) for label in labels], rotation=0, ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    run_dir = PROJECT_ROOT / "runs" / RUN_ID
    manifest = json.loads((run_dir / "manifest.json").read_text())
    convergence_figure(
        run_dir=run_dir,
        jd_ids=manifest["jd_ids"],
        out_path=PROJECT_ROOT / "docs/figures/table6_kendall_tau.png",
    )

    real_data = json.loads((PROJECT_ROOT / "console-web/src/data/real-data.json").read_text())
    kendall_rows = sorted(real_data["comparison"].values(), key=lambda v: v["kendallTau"])
    bar_figure(
        labels=[r["jdId"] for r in kendall_rows],
        values=[r["kendallTau"] for r in kendall_rows],
        title="Final Kendall's Tau by Job Role",
        ylabel="Mean Kendall's Tau",
        out_path=PROJECT_ROOT / "docs/figures/table7_kendall_tau_per_role.png",
    )

    ragas_report = json.loads((PROJECT_ROOT / f"runs/{RUN_ID}/ragas_faithfulness_report.json").read_text())
    per_jd = sorted(ragas_report["per_jd"].items(), key=lambda kv: kv[1])
    bar_figure(
        labels=[jd_id for jd_id, _ in per_jd],
        values=[score for _, score in per_jd],
        title="Faithfulness of Generated Assessments by Job Role",
        ylabel="Mean Faithfulness",
        out_path=PROJECT_ROOT / "docs/figures/table5_ragas_faithfulness.png",
    )


if __name__ == "__main__":
    main()
