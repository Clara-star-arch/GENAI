"""
Step 6: the RAG chain (LangChain LCEL), with the Step 7 guardrails wired in.

    question ─► redact PII ─► moderation ─► greeting? ─► rewrite follow-up using history
             ─► hybrid retrieve (top 5) ─► relevance gate ─► generate answer with [n] citations
             ─► groundedness check ─► (one rewrite if needed) ─► safe fallback ─► Response

Usage:
    from rag.chain import RAGChain
    bot = RAGChain()
    r = bot.answer("What is the TOEFL minimum?")
    r2 = bot.answer("What about IELTS?", history=[("What is the TOEFL minimum?", r.answer)])
    print(r2.answer, [s.metadata["url"] for s in r2.cited_sources])
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag import guardrails as G
from rag import prompts as P
from rag.llm import get_chat_model, model_name
from rag.retriever import Hit, HybridRetriever

CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
MAX_HISTORY_TURNS = 3


@dataclass
class Response:
    answer: str
    sources: list[Hit] = field(default_factory=list)      # everything shown to the LLM, numbered 1..k
    cited: list[int] = field(default_factory=list)        # source numbers the answer cites
    question: str = ""                                    # as typed, after PII redaction
    search_query: str = ""                                # after follow-up rewriting
    flags: dict = field(default_factory=dict)             # guardrail outcomes, for the UI and logs
    timings: dict = field(default_factory=dict)

    @property
    def cited_sources(self) -> list[Hit]:
        return [self.sources[n - 1] for n in self.cited if 0 < n <= len(self.sources)]

    def to_dict(self) -> dict:
        return {
            "answer": self.answer, "question": self.question, "search_query": self.search_query,
            "cited": self.cited, "flags": self.flags, "timings": self.timings,
            "sources": [{"n": i + 1, "id": h.id, "score": h.score, "relevance": h.relevance,
                         "body": h.body, "metadata": h.metadata, "debug": h.debug}
                        for i, h in enumerate(self.sources)],
        }


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for n, h in enumerate(hits, 1):
        m = h.metadata
        # The section's own heading goes into the content as a title line: some answers live only
        # in headings (e.g. the two scholarship names), and the model reads metadata lines less.
        heading = m["heading_path"].split(" > ")[-1].strip()
        blocks.append(f"[{n}] Page: {m['title']}\nSection: {m['heading_path']}\nURL: {m['url']}\n"
                      f"Content:\n## {heading}\n{h.body}")
    return "\n\n".join(blocks)


def extract_citations(answer: str, n_sources: int) -> tuple[str, list[int]]:
    """Find [n] / [n, m] markers; drop markers that point at sources that don't exist."""
    cited: list[int] = []

    def fix(m):
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
        valid = [x for x in nums if 1 <= x <= n_sources]
        for x in valid:
            if x not in cited:
                cited.append(x)
        return f"[{', '.join(map(str, valid))}]" if valid else ""

    cleaned = CITE_RE.sub(fix, answer)
    return re.sub(r"[ \t]+([.,;:])", r"\1", cleaned).strip(), cited


class RAGChain:
    def __init__(self, retriever: HybridRetriever | None = None, k: int = 5):
        self.retriever = retriever or HybridRetriever()
        self.k = k
        self.answer_chain = (
            ChatPromptTemplate.from_messages([("system", P.SYSTEM_PROMPT), ("human", P.ANSWER_HUMAN)])
            | get_chat_model("answer") | StrOutputParser()
        )
        self.condense_chain = (
            ChatPromptTemplate.from_messages([("system", P.CONDENSE_SYSTEM), ("human", P.CONDENSE_HUMAN)])
            | get_chat_model("condense") | StrOutputParser()
        )
        self.judge_chain = (
            ChatPromptTemplate.from_messages([("system", P.JUDGE_SYSTEM), ("human", P.JUDGE_HUMAN)])
            | get_chat_model("judge") | StrOutputParser()
        )

    # ---------- steps ----------

    def rewrite(self, question: str, history: list[tuple[str, str]]) -> str:
        if not history:
            return question
        turns = history[-MAX_HISTORY_TURNS:]
        hist = "\n".join(f"User: {u}\nAssistant: {a[:400]}" for u, a in turns)
        out = self.condense_chain.invoke({"history": hist, "question": question}).strip()
        return out.strip('"') or question

    def generate(self, question: str, hits: list[Hit], note: str = "") -> str:
        return self.answer_chain.invoke({
            "context": format_context(hits),
            "question": question + note,
            "today": date.today().strftime("%B %d, %Y"),
        })

    def judge(self, answer: str, hits: list[Hit]) -> tuple[bool, list[str]]:
        raw = self.judge_chain.invoke({"context": format_context(hits), "answer": answer})
        return G.parse_judge(raw)

    # ---------- full pipeline ----------

    def answer(self, question: str, history: list[tuple[str, str]] | None = None) -> Response:
        t0 = time.time()
        history = history or []
        q, pii = G.redact_pii(question.strip())
        flags = {"pii_redacted": pii, "moderation_flagged": False, "injection_suspected": False,
                 "low_relevance": False, "grounded": None, "regenerated": False, "fallback": False}
        resp = Response(answer="", question=q, flags=flags)

        flagged, cats = G.moderate(q)
        if flagged:
            flags.update(moderation_flagged=True, moderation_categories=cats)
            resp.answer = P.MODERATION_REPLY
            return self._finish(resp, t0)

        flags["injection_suspected"] = G.detect_injection(q)

        if G.is_greeting(q):
            resp.answer = P.GREETING_REPLY
            return self._finish(resp, t0)

        resp.search_query = self.rewrite(q, history)
        t1 = time.time()
        hits = self.retriever.search(resp.search_query, k=self.k)
        resp.sources = hits
        resp.timings["retrieve_s"] = round(time.time() - t1, 2)

        # Relevance gate: decline without an LLM call only when the reranker AND the embedding
        # similarity both say the best source is weak. Borderline cases go to the LLM, whose
        # prompt still says "I don't know" when the sources don't answer the question.
        top = max((h.relevance for h in hits), default=0.0)
        top_sim = max((h.debug.get("dense_sim") or 0.0 for h in hits), default=0.0)
        flags["top_relevance"] = round(top, 4)
        flags["top_dense_sim"] = round(top_sim, 4)
        weak_rerank = self.retriever.reranker_name != "none" and top < G.min_relevance()
        if not hits or (weak_rerank and top_sim < G.min_dense_sim()):
            flags["low_relevance"] = True
            resp.answer = P.NO_INFO_REPLY
            return self._finish(resp, t0)

        t2 = time.time()
        draft = self.generate(resp.search_query, hits)
        resp.timings["generate_s"] = round(time.time() - t2, 2)

        if G.setting("GROUNDING_CHECK", "on") == "on":
            t3 = time.time()
            ok, unsupported = self.judge(draft, hits)
            if not ok:
                flags["regenerated"] = True
                flags["unsupported_claims"] = unsupported[:5]
                note = P.REGENERATE_NOTE.format(unsupported="\n".join(f"- {u}" for u in unsupported[:5]))
                draft = self.generate(resp.search_query, hits, note=note)
                ok, unsupported = self.judge(draft, hits)
            flags["grounded"] = ok
            resp.timings["judge_s"] = round(time.time() - t3, 2)
            if not ok:
                flags["fallback"] = True
                seen, links = set(), []
                for h in hits[:3]:
                    u = h.metadata["url"]
                    if u not in seen:
                        seen.add(u)
                        links.append(f"- [{h.metadata['title']}]({u})")
                resp.answer = P.UNVERIFIED_REPLY.format(links="\n".join(links))
                return self._finish(resp, t0)

        resp.answer, resp.cited = extract_citations(draft, len(hits))
        return self._finish(resp, t0)

    def _finish(self, resp: Response, t0: float) -> Response:
        resp.timings["total_s"] = round(time.time() - t0, 2)
        G.log_event("interactions", {
            "question": resp.question, "search_query": resp.search_query, "answer": resp.answer,
            "cited_urls": [h.metadata["url"] for h in resp.cited_sources],
            "flags": resp.flags, "timings": resp.timings, "model": model_name("answer"),
        })
        return resp
