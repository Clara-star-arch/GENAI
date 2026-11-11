"""
Quick check after Step 4: run every eval question through dense search and BM25 separately
and show the top hits, so you can see whether retrieval works before building the hybrid
retriever (Step 5). It also gives you the "dense only" and "BM25 only" baselines for the deck.

A hit counts if its URL contains any of the question's expected_urls fragments.

Usage:
    python -m pipeline.smoke_search          # summary + top-3 per question
    python -m pipeline.smoke_search -k 5 -q  # quiet: summary table only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from pipeline.index import BM25_PATH, CHROMA_DIR, COLLECTION, META_PATH
from rag import embeddings as E
from rag.text import bm25_tokenize

TESTSET = Path("eval/testset.json")
CHUNKS = Path("data/processed/chunks.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    meta = json.loads(META_PATH.read_text())
    if meta["embedding_model"] != E.model_name():
        raise SystemExit(f"Index built with {meta['embedding_model']} but current model is "
                         f"{E.model_name()}. Re-run python -m pipeline.index")

    chunks = {c["id"]: c for c in map(json.loads, CHUNKS.read_text(encoding="utf-8").splitlines())}
    vs = Chroma(collection_name=COLLECTION, embedding_function=E.get_embeddings(),
                persist_directory=str(CHROMA_DIR))
    corpus = json.loads(BM25_PATH.read_text())
    bm25 = BM25Okapi(corpus["tokens"])

    tests = json.loads(TESTSET.read_text())
    tally = {"dense": 0, "bm25": 0}
    rows = []
    for t in tests:
        dense = [d.metadata for d in vs.similarity_search(t["question"], k=args.k)]
        scores = bm25.get_scores(bm25_tokenize(t["question"]))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:args.k]
        sparse = [chunks[corpus["ids"][i]]["metadata"] for i in top]

        hit = lambda hits: any(any(f in h["url"] for f in t["expected_urls"]) for h in hits)
        d_hit, s_hit = hit(dense), hit(sparse)
        tally["dense"] += d_hit
        tally["bm25"] += s_hit
        rows.append((t["id"], d_hit, s_hit))

        if not args.quiet:
            print(f"\n{t['id']}  {t['question']}")
            for name, hits, ok in (("dense", dense, d_hit), ("bm25 ", sparse, s_hit)):
                print(f"  {name} {'HIT ' if ok else 'MISS'}")
                for h in hits:
                    print(f"      - {h['heading_path'][:95]}")

    n = len(tests)
    print(f"\nHit@{args.k}   dense {tally['dense']}/{n}   bm25 {tally['bm25']}/{n}")
    misses = [qid for qid, d, s in rows if not (d or s)]
    if misses:
        print("Missed by both:", ", ".join(misses), "-> check the crawl/cleaning for these first")


if __name__ == "__main__":
    main()
