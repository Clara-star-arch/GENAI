"""
Step 4: embed chunks into Chroma and build the BM25 corpus.

Input:  data/processed/chunks.jsonl
Output: chroma_db/                     dense vector index (cosine)
        data/processed/bm25.json       tokenized corpus for keyword search
        data/processed/index_meta.json which embedding model built the index

Rebuilds from scratch every run, so re-running after re-crawling is always safe.

Usage:
    python -m pipeline.index
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag import embeddings as E
from rag.text import bm25_tokenize

CHUNKS = Path("data/processed/chunks.jsonl")
CHROMA_DIR = Path("chroma_db")
COLLECTION = "msads"
BM25_PATH = Path("data/processed/bm25.json")
META_PATH = Path("data/processed/index_meta.json")
BATCH = 100


def load_chunks() -> list[dict]:
    return [json.loads(l) for l in CHUNKS.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_dense(chunks: list[dict]) -> int:
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    vs = Chroma(
        collection_name=COLLECTION,
        embedding_function=E.get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        vs.add_documents(
            [Document(page_content=c["text"], metadata={**c["metadata"], "body": c["body"], "chunk_id": c["id"]})
             for c in batch],
            ids=[c["id"] for c in batch],
        )
        print(f"  embedded {min(i + BATCH, len(chunks))}/{len(chunks)}")
    return vs._collection.count()


def build_bm25(chunks: list[dict]) -> None:
    corpus = {"ids": [c["id"] for c in chunks],
              "tokens": [bm25_tokenize(c["text"]) for c in chunks]}
    BM25_PATH.write_text(json.dumps(corpus), encoding="utf-8")


def main() -> None:
    chunks = load_chunks()
    print(f"Indexing {len(chunks)} chunks with {E.provider()}:{E.model_name()}")
    t0 = time.time()
    n = build_dense(chunks)
    build_bm25(chunks)
    META_PATH.write_text(json.dumps({
        "embedding_provider": E.provider(),
        "embedding_model": E.model_name(),
        "collection": COLLECTION,
        "n_chunks": n,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
    assert n == len(chunks), f"Chroma has {n} vectors but there are {len(chunks)} chunks"
    print(f"Done in {time.time() - t0:.1f}s: {n} vectors in {CHROMA_DIR}/, BM25 corpus in {BM25_PATH}")


if __name__ == "__main__":
    main()
