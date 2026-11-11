"""
Charts and tables for the deck, from the results of retrieval_eval.py and answer_eval.py.

Usage:
    python -m eval.make_charts
Outputs (eval/results/):
    chart_retrieval.png       Hit@1 and MRR@10 for each retrieval stage
    chart_answer_quality.png  key-fact recall, citation accuracy and faithfulness by question type
    summary_tables.md         the same numbers as Markdown tables (paste into slides or notes)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "eval/results"

# Reference palette, light mode. Slots 1-3 pass all-pairs CVD checks; slot 3 (aqua) is below 3:1
# contrast, so every bar carries a visible value label.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True,
})

CATEGORY_LABELS = {"core": "Class questions", "factual": "Other facts", "paraphrase": "Paraphrases",
                   "followup": "Follow-ups", "out_of_scope": "Off-topic", "not_on_site": "Not on site"}


def retrieval_chart() -> list[dict] | None:
    path = RES / "retrieval_ablation.csv"
    if not path.exists():
        print("skip retrieval chart: run python -m eval.retrieval_eval first")
        return None
    rows = list(csv.DictReader(path.open()))
    names = [r["config"] for r in rows][::-1]                 # best (last) stage on top
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, metric, pct in ((axes[0], "Hit@1", True), (axes[1], "MRR@10", False)):
        vals = [float(r[metric]) for r in rows][::-1]
        bars = ax.barh(names, vals, height=0.55, color=SERIES[0])
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + 0.015, b.get_y() + b.get_height() / 2,
                    f"{v:.0%}" if pct else f"{v:.2f}", va="center", fontsize=10, color=INK)
        ax.set_xlim(0, 1.12)
        ax.set_title(metric, loc="left", fontsize=12, color=INK, pad=8)
        ax.tick_params(axis="y", length=0)
        if pct:
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    n = sum(1 for _ in json.loads((ROOT / "eval/testset.json").read_text()) if _.get("expected_urls"))
    fig.suptitle(f"Retrieval quality by pipeline stage ({n} questions)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RES / "chart_retrieval.png", dpi=200)
    plt.close(fig)
    return rows


def answer_chart() -> dict | None:
    path = RES / "answer_summary.json"
    if not path.exists():
        print("skip answer chart: run python -m eval.answer_eval first")
        return None
    summary = json.loads(path.read_text())
    cats = [c for c in ["core", "factual", "paraphrase", "followup"] if c in summary["by_category"]]
    metrics = [("key_fact_recall", "Key facts present"), ("citation_accuracy", "Cites the right page"),
               ("faithfulness", "Claims supported by sources")]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    h = 0.24
    gap = 0.02                                                   # surface gap between adjacent bars
    ys = range(len(cats))
    for j, (key, label) in enumerate(metrics):
        vals = [summary["by_category"][c].get(key) or 0 for c in cats]
        pos = [y + (j - 1) * (h + gap) for y in ys]     # axis is inverted: first metric on top, matching the legend
        bars = ax.barh(pos, vals, height=h, color=SERIES[j], label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2, f"{v:.0%}",
                    va="center", fontsize=9, color=INK)
    ax.set_yticks(list(ys), [f"{CATEGORY_LABELS[c]} (n={summary['by_category'][c]['n']})" for c in cats])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Answer quality by question type", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RES / "chart_answer_quality.png", dpi=200)
    plt.close(fig)
    return summary


def tables(ret: list[dict] | None, ans: dict | None) -> None:
    out = ["# Evaluation summary\n"]
    if ret:
        out += ["## Retrieval (page-level)\n", "| Configuration | Hit@1 | Hit@3 | Hit@5 | MRR@10 |", "|---|---|---|---|---|"]
        out += [f"| {r['config']} | {float(r['Hit@1']):.0%} | {float(r['Hit@3']):.0%} | {float(r['Hit@5']):.0%} | "
                f"{float(r['MRR@10']):.3f} |" for r in ret]
        out.append("")
    if ans:
        f = lambda v, p=True: "-" if v is None else (f"{v:.0%}" if p else f"{v:.2f}")
        out += ["## Answer quality\n",
                "| Question type | n | Key facts | Correct citation | Faithfulness | Relevance (1-5) | Correctness (1-5) | Declined correctly |",
                "|---|---|---|---|---|---|---|---|"]
        for c, a in list(ans["by_category"].items()) + [("Overall", ans["overall"])]:
            out.append(f"| {CATEGORY_LABELS.get(c, c)} | {a['n']} | {f(a['key_fact_recall'])} | {f(a['citation_accuracy'])} | "
                       f"{f(a['faithfulness'])} | {f(a['relevance_1to5'], False)} | {f(a['correctness_1to5'], False)} | "
                       f"{f(a['decline_accuracy'])} |")
        o = ans["overall"]
        out += ["", f"- Latency: median {o['latency_p50_s']}s, p95 {o['latency_p95_s']}s",
                f"- Answers rewritten by the hallucination check: {o['regenerated']}; safe fallbacks: {o['fallback']}"]
    (RES / "summary_tables.md").write_text("\n".join(out) + "\n")


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    ret, ans = retrieval_chart(), answer_chart()
    tables(ret, ans)
    print(f"Wrote charts and summary_tables.md to {RES}/")


if __name__ == "__main__":
    main()
