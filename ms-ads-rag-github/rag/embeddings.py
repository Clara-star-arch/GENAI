"""
One place that decides the embedding model, so the indexer (Step 4) and the retriever (Step 5)
can never disagree. Pick a provider in .env:

    EMBEDDING_PROVIDER=openai   -> text-embedding-3-small (default when OPENAI_API_KEY is set)
    EMBEDDING_PROVIDER=hf       -> BAAI/bge-small-en-v1.5, runs locally, free
    EMBEDDING_PROVIDER=fake     -> deterministic hash vectors, ONLY for offline tests
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# bge models retrieve better when queries (not documents) carry this instruction.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def provider() -> str:
    p = os.getenv("EMBEDDING_PROVIDER")
    if p:
        return p.lower()
    return "openai" if os.getenv("OPENAI_API_KEY") else "hf"


def model_name() -> str:
    return {
        "openai": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "hf": os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        "fake": "fake-hash-256",
    }[provider()]


class _BgeEmbeddings(Embeddings):
    def __init__(self, name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def embed_query(self, text):
        return self.model.encode(BGE_QUERY_PREFIX + text, normalize_embeddings=True).tolist()


class _FakeEmbeddings(Embeddings):
    """Bag-of-words hashed into 256 dims. Lets the pipeline run with no model or network."""
    dim = 256

    def _vec(self, text: str):
        v = [0.0] * self.dim
        for w in text.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def get_embeddings() -> Embeddings:
    p = provider()
    if p == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model_name())
    if p == "hf":
        return _BgeEmbeddings(model_name())
    if p == "fake":
        return _FakeEmbeddings()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER={p!r}")
