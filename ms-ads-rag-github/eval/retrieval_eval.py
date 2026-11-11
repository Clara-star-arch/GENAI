"""
Retrieval ablation: score each stage of the retriever on the eval set.
This produces the retrieval table for the deck.

Metrics (a result is relevant if its URL contains one of the question's expected_urls):
    Hit@1, Hit@3, Hit@5   share of questions with a relevant result in the top k
    MRR@10                mean of 1/rank of the first relevant result (0 if none in top 10)

These are page-level metrics. Step 9 adds chunk-level gold labels for a stricter score.

Usage:
    python -m eval.retrieval_eval                     # all configurations
    python -m eval.retrieval_eval --show-misses       # also list questions the full system misses
Outputs:
    eval/results/retrieval_ablation.csv    one row per configuration
    eval/results/retrieval_per_question.csv  first relevant rank per question per configuration
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rag.retriever import HybridRetriever

TESTSET = Path("eval/testset.json")
OUT = Path("eval/results")

CONFIGS = [  # name, search() flags
    ("Dense only",               dict(use_bm25=False, use_rerank=False, use_boost=False)),
    ("BM25 only",                dict(use_dense=False, use_rerank=False, use_boost=False)),
    ("Hybrid (RRF)",             dict(use_rerank=False, use_boost=False)),
    ("Hybrid + rerank",          dict(use_boost=False)),
    ("Hybrid + rerank + boost",  dict()),
]


def first_relevant_rank(hits, expected: list[str]) -> int | None:
    for rank, h in enumerate(hits, 1):
        if any(frag in h.metadata["url"] for frag in expected):
            return rank
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-misses", action="store_true")
    ap.add_argument("--core", action="store_true", help="only the 12 class questions")
    args = ap.parse_args()

    # Off-topic / not-on-site items have no correct page, so they are scored in answer_eval only.
    # Follow-ups are searched by their standalone form (the chain's rewrite is tested in answer_eval).
    tests = [t for t in json.loads(TESTSET.read_text()) if t.get("expected_urls")]
    if args.core:
        tests = [t for t in tests if t.get("category", "core") == "core"]
    for t in tests:
        t["search"] = t.get("standalone") or t["question"]
    r = HybridRetriever()
    OUT.mkdir(parents=True, exist_ok=True)

    summary, per_q = [], {t["id"]: {"id": t["id"], "question": t["question"]} for t in tests}
    for name, flags in CONFIGS:
        if "rerank" in name and r.reranker_name == "none":
            continue
        ranks = []
        for t in tests:
            hits = r.search(t["search"], k=10, **flags)
            rank = first_relevant_rank(hits, t["expected_urls"])
            ranks.append(rank)
            per_q[t["id"]][name] = rank if rank else "-"
        n = len(ranks)
        row = {
            "config": name,
            "Hit@1": sum(1 for x in ranks if x and x <= 1) / n,
            "Hit@3": sum(1 for x in ranks if x and x <= 3) / n,
            "Hit@5": sum(1 for x in ranks if x and x <= 5) / n,
            "MRR@10": sum(1 / x for x in ranks if x) / n,
        }
        summary.append(row)

    print(f"\nRetrieval ablation on {len(tests)} questions (reranker: {r.reranker_name})\n")
    print(f"{'Configuration':<26}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'MRR@10':>9}")
    for row in summary:
        print(f"{row['config']:<26}{row['Hit@1']:>8.0%}{row['Hit@3']:>8.0%}{row['Hit@5']:>8.0%}{row['MRR@10']:>9.3f}")

    suffix = "_core" if args.core else ""
    with (OUT / f"retrieval_ablation{suffix}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    cols = ["id", "question"] + [row["config"] for row in summary]
    with (OUT / f"retrieval_per_question{suffix}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(per_q.values())

    print("\nFirst relevant rank per question ('-' = not in top 10):")
    short = [row["config"] for row in summary]
    print(f"{'':5}" + "".join(f"{c[:12]:>14}" for c in short))
    for q in per_q.values():
        print(f"{q['id']:5}" + "".join(f"{str(q[c]):>14}" for c in short))

    if args.show_misses:
        best = summary[-1]["config"]
        for t in tests:
            rank = per_q[t["id"]][best]
            if rank == "-" or rank > 1:
                print(f"\n{t['id']} (rank {rank}) {t['question']}")
                for n, h in enumerate(r.search(t["search"], k=3), 1):
                    print(f"   {n}. score={h.score:+.2f} {h.metadata['heading_path'][:85]}  <- {h.metadata['url']}")

    print(f"\nSaved {OUT}/retrieval_ablation{suffix}.csv and retrieval_per_question{suffix}.csv")


if __name__ == "__main__":
    main()
