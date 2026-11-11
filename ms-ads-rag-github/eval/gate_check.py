"""
Calibrate the relevance gate (MIN_RELEVANCE + MIN_DENSE_SIM).

For every test-set question, prints the best reranker relevance and the best dense cosine
similarity, and whether the gate would decline it. Answerable questions should pass; the gate
may decline off-topic ones (the LLM prompt declines the rest).

    python -m eval.gate_check
"""
from __future__ import annotations

import json
from pathlib import Path

from rag import guardrails as G
from rag.retriever import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    items = json.loads((ROOT / "eval/testset.json").read_text(encoding="utf-8"))
    r = HybridRetriever()
    min_rel, min_sim = G.min_relevance(), G.min_dense_sim()
    print(f"MIN_RELEVANCE={min_rel}  MIN_DENSE_SIM={min_sim}  reranker={r.reranker_name}\n")
    print(f"{'id':<5}{'kind':<9}{'rerank':>8}{'dense':>8}  gate      question")
    worst_answerable, best_decline = 1.0, 0.0
    for it in items:
        if it.get("history"):
            continue                          # follow-ups depend on the rewrite step
        hits = r.search(it["question"], k=5)
        rel = max((h.relevance for h in hits), default=0.0)
        sim = max((h.debug.get("dense_sim") or 0.0 for h in hits), default=0.0)
        weak = r.reranker_name != "none" and rel < min_rel
        declined = not hits or (weak and sim < min_sim)
        kind = "answer" if it.get("expected_behavior", "answer") == "answer" else "decline"
        if kind == "answer":
            worst_answerable = min(worst_answerable, sim)
        else:
            best_decline = max(best_decline, sim)
        flag = "DECLINE" if declined else "pass"
        if kind == "answer" and declined:
            flag += " <-- wrong"
        print(f"{it['id']:<5}{kind:<9}{rel:>8.3f}{sim:>8.3f}  {flag:<9} {it['question'][:60]}")
    print(f"\nLowest dense similarity among answerable questions: {worst_answerable:.3f}")
    print(f"Highest dense similarity among decline questions:   {best_decline:.3f}")
    print("Pick MIN_DENSE_SIM between these two (closer to the first), set it in .env, re-run.")


if __name__ == "__main__":
    main()
