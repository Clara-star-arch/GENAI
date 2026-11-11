"""
Step 5: hybrid retriever.

    query ──► dense (Chroma, top 20) ─┐
          └─► BM25 (top 20) ──────────┴─► Reciprocal Rank Fusion ─► cross-encoder rerank (top 20)
                                          ─► program soft boost ─► dedupe ─► top k

Why each stage:
  * Dense finds paraphrases ("cost" ~ "tuition"); BM25 nails exact terms ("TOEFL", "MBA/MS", "OPT").
  * RRF merges the two rankings without having to calibrate their very different score scales.
  * The cross-encoder reads query and chunk together, so it judges relevance far better than
    either first-stage method; it only runs on ~20 candidates, so it stays fast.
  * Program boost: if the question says "online" or "in-person", chunks tagged with that program
    move up and chunks tagged with the other program move down. It's a soft boost, not a filter,
    because many answers are tagged "both" and must stay reachable.

Config (.env):
    RERANKER=minilm   cross-encoder/ms-marco-MiniLM-L-6-v2 (default, ~90 MB, fast on CPU)
    RERANKER=bge      BAAI/bge-reranker-base (more accurate, ~1.1 GB; may not fit free hosting)
    RERANKER=none     skip reranking (RRF order)

Usage:
    from rag.retriever import HybridRetriever
    r = HybridRetriever()
    for hit in r.search("What is the TOEFL minimum?", k=5):
        print(hit.score, hit.metadata["heading_path"])

    python -m rag.retriever "What is the TOEFL minimum?"      # quick CLI check
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from rag import embeddings as E
from rag.text import bm25_tokenize

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

ROOT = Path(__file__).resolve().parents[1]          # project folder, wherever it's run from
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "msads"
CHUNKS_PATH = ROOT / "data/processed/chunks.jsonl"
BM25_PATH = ROOT / "data/processed/bm25.json"
META_PATH = ROOT / "data/processed/index_meta.json"

RERANKER_MODELS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge": "BAAI/bge-reranker-base",
}

ONLINE_Q = re.compile(r"\bonline\b|\bremote\b|\bpart[- ]time online\b", re.I)
INPERSON_Q = re.compile(r"\bin[- ]person\b|\bon[- ]campus\b|\bfull[- ]time\b|\bvisa\b|\bf-?1\b", re.I)
# Joint MBA/MS content (e.g. its own TOEFL minimum of 104) must not answer questions about the
# MS-ADS program itself, unless the question is about the joint degree.
JOINT_Q = re.compile(r"\bmba\b|\bbooth\b|\bjoint\b", re.I)
JOINT_SECTION = re.compile(r"\bMBA/MS\b|\bjoint MBA\b", re.I)


@dataclass
class Hit:
    id: str
    text: str                 # contextual-header text that was indexed (what the LLM reads)
    body: str                 # raw section text (what the UI shows)
    metadata: dict
    score: float = 0.0        # final score used for ordering (raw reranker logit + boost)
    relevance: float = 0.0    # reranker relevance squashed to [0, 1]; for display and grounding checks
    debug: dict = field(default_factory=dict)   # per-stage ranks/scores, for eval and the UI


def query_program(query: str) -> str | None:
    """'online' / 'in_person' if the question is clearly about one program, else None.
    'visa' counts as in-person intent only when 'online' is not mentioned
    ("does the online program sponsor visas?" is about the online program)."""
    on = bool(ONLINE_Q.search(query))
    ip = bool(INPERSON_Q.search(query))
    if on:
        return "online"
    if ip:
        return "in_person"
    return None


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class HybridRetriever:
    def __init__(self, reranker: str | None = None, n_candidates: int = 20,
                 boost: float = 1.0, penalty: float = 1.5):
        # boost/penalty are in reranker-logit units. Logits typically span about -10..+10, so a
        # boost of 1.0 decides between near-ties without overriding a clearly better answer.
        meta = json.loads(META_PATH.read_text())
        if meta["embedding_model"] != E.model_name():
            raise RuntimeError(f"Index built with {meta['embedding_model']} but the current embedding "
                               f"model is {E.model_name()}. Re-run: python -m pipeline.index")

        self.chunks: dict[str, dict] = {}
        for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c = json.loads(line)
                self.chunks[c["id"]] = c

        self.vs = Chroma(collection_name=COLLECTION, embedding_function=E.get_embeddings(),
                         persist_directory=str(CHROMA_DIR))
        corpus = json.loads(BM25_PATH.read_text())
        self.bm25_ids: list[str] = corpus["ids"]
        self.bm25 = BM25Okapi(corpus["tokens"])

        self.n_candidates = n_candidates
        self.boost, self.penalty = boost, penalty
        self.reranker_name = (reranker or os.getenv("RERANKER", "minilm")).lower()
        self._cross_encoder = None   # loaded lazily on first rerank

    # ---------- first stage ----------

    def dense(self, query: str, n: int) -> list[tuple[str, float]]:
        res = self.vs.similarity_search_with_score(query, k=n)      # cosine distance: lower = closer
        return [(d.metadata.get("chunk_id") or self._id_of(d), 1.0 - dist) for d, dist in res]

    def _id_of(self, doc) -> str:
        # Chroma returns the id on newer langchain versions; fall back to matching the text.
        if getattr(doc, "id", None):
            return doc.id
        for cid, c in self.chunks.items():
            if c["text"] == doc.page_content:
                return cid
        raise KeyError("chunk not found for dense result")

    def sparse(self, query: str, n: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(bm25_tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        return [(self.bm25_ids[i], float(scores[i])) for i in top if scores[i] > 0]

    # ---------- reranking ----------

    def _load_cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(RERANKER_MODELS[self.reranker_name], max_length=512)
        return self._cross_encoder

    def rerank_scores(self, query: str, ids: list[str]) -> dict[str, float]:
        """Raw cross-encoder logits for each candidate.
        Ranking uses the RAW logits: squashing them through a sigmoid first maps every strong
        candidate to ~1.000, which makes their order arbitrary and lets the boost dominate."""
        if self.reranker_name == "none" or not ids:
            return {}
        model = self._load_cross_encoder()
        raw = model.predict([(query, self.chunks[i]["text"]) for i in ids])
        return {i: float(s) for i, s in zip(ids, raw)}

    # ---------- full pipeline ----------

    def search(self, query: str, k: int = 5, use_bm25: bool = True, use_dense: bool = True,
               use_rerank: bool = True, use_boost: bool = True) -> list[Hit]:
        """Flags let the eval script switch stages off one at a time for the ablation."""
        n = self.n_candidates
        dense = self.dense(query, n) if use_dense else []
        sparse = self.sparse(query, n) if use_bm25 else []

        fused = rrf([[i for i, _ in dense], [i for i, _ in sparse]])
        candidates = sorted(fused, key=lambda i: -fused[i])[:n]

        rer = self.rerank_scores(query, candidates) if use_rerank else {}
        program = query_program(query) if use_boost else None
        joint_q = bool(JOINT_Q.search(query))

        dense_rank = {i: r for r, (i, _) in enumerate(dense, 1)}
        dense_sim = {i: s for i, s in dense}
        sparse_rank = {i: r for r, (i, _) in enumerate(sparse, 1)}

        hits = []
        for i in candidates:
            c = self.chunks[i]
            # Base score: reranker logit when available, otherwise RRF scaled to ~[0, 1].
            if rer:
                base, scale = rer[i], 1.0
            else:
                base, scale = fused[i] * 30, 0.1     # boost shrunk to match the smaller RRF scale
            adj = 0.0
            if program:
                tag = c["metadata"]["program_type"]
                if tag == program:
                    adj = self.boost * scale
                elif tag in ("online", "in_person"):
                    adj = -self.penalty * scale
            # Joint-program penalty: MBA/MS sections move down unless the question is about it.
            if use_boost and not joint_q and JOINT_SECTION.search(c["metadata"]["heading_path"]):
                adj -= self.penalty * scale
            hits.append(Hit(
                id=i, text=c["text"], body=c["body"], metadata=c["metadata"], score=base + adj,
                relevance=_sigmoid(rer[i]) if rer else min(1.0, fused[i] * 30),
                debug={"dense_rank": dense_rank.get(i), "bm25_rank": sparse_rank.get(i),
                       "dense_sim": round(dense_sim[i], 4) if i in dense_sim else None,
                       "rrf": round(fused[i], 5), "rerank_logit": round(rer[i], 3) if rer else None,
                       "program_adj": adj, "query_program": program, "joint_query": joint_q},
            ))
        hits.sort(key=lambda h: -h.score)
        if use_boost and not joint_q:
            # Joint-program sections never answer MS-ADS questions; even ranked low they confused
            # the generator and the groundedness judge (MBA/MS TOEFL 104 vs MS-ADS 102 / 5).
            hits = [h for h in hits if not JOINT_SECTION.search(h.metadata["heading_path"])]
        return self._dedupe(hits)[:k]

    @staticmethod
    def _dedupe(hits: list[Hit]) -> list[Hit]:
        """Keep one hit per section, and drop exact repeats of the same answer published on
        several pages (the site repeats FAQ text on other pages). Duplicates waste LLM context."""
        seen_sections, seen_bodies, out = set(), set(), []
        for h in hits:
            section = (h.metadata["url"], h.metadata["heading_path"])
            body = re.sub(r"\W+", " ", h.body).strip().lower()
            if section in seen_sections or body in seen_bodies:
                continue
            seen_sections.add(section)
            seen_bodies.add(body)
            out.append(h)
        return out

    # ---------- LangChain integration (used by the chain in Step 6) ----------

    def as_langchain(self, k: int = 5):
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever

        outer = self

        class _LC(BaseRetriever):
            def _get_relevant_documents(self, query: str, *,
                                        run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
                return [Document(page_content=h.text,
                                 metadata={**h.metadata, "score": h.score, "relevance": h.relevance,
                                           "body": h.body, "id": h.id})
                        for h in outer.search(query, k=k)]

        return _LC()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What are the minimum TOEFL and IELTS scores?"
    r = HybridRetriever()
    print(f"Q: {q}   (reranker={r.reranker_name}, program={query_program(q)})\n")
    for n, h in enumerate(r.search(q, k=5), 1):
        d = h.debug
        print(f"{n}. score={h.score:+.2f}  relevance={h.relevance:.3f}  {h.metadata['heading_path'][:80]}")
        print(f"     dense#{d['dense_rank']}  bm25#{d['bm25_rank']}  logit={d['rerank_logit']}  "
              f"boost={d['program_adj']:+.2f}  [{h.metadata['program_type']}]  {h.metadata['url']}")
