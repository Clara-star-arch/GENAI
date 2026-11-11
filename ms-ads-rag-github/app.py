"""
Step 8: Streamlit chat UI for the MS-ADS Assistant.

Run locally (from the project folder):
    streamlit run app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

# On Streamlit Cloud, keys live in st.secrets; copy them into the environment before importing rag.
try:
    for k, v in st.secrets.items():
        if isinstance(v, (str, int, float)):
            os.environ.setdefault(k, str(v))
except Exception:  # noqa: BLE001  (no secrets file locally is fine)
    pass

from rag import guardrails as G          # noqa: E402
from rag.chain import RAGChain           # noqa: E402

st.set_page_config(page_title="MS-ADS Assistant", layout="centered")

SUGGESTED = [
    "What is the tuition for the program?",
    "What scholarships are available?",
    "What are the TOEFL and IELTS minimums?",
    "What are the deadlines for the in-person program?",
    "What is the capstone project?",
    "How many courses do I need to graduate?",
    "Is the program STEM/OPT eligible?",
    "How do I apply to the MBA/MS program?",
]

FLAG_NOTES = {
    "pii_redacted": "Personal details in your question were removed before processing.",
    "regenerated": "The first draft included an unsupported statement, so it was rewritten.",
    "fallback": "I couldn't verify an answer from the website, so I'm pointing you to the pages.",
    "low_relevance": "Nothing on the program website matched this closely.",
    "moderation_flagged": "This request was declined by the content filter.",
}


@st.cache_resource(show_spinner="Loading the knowledge base…")
def load_bot() -> RAGChain:
    # chroma_db/ is not committed to git: on a fresh deploy, build it once from the chunks
    # (296 embeddings, a few seconds, well under a cent).
    if not (ROOT / "chroma_db" / "chroma.sqlite3").exists():
        from pipeline import index
        index.main()
    bot = RAGChain()
    bot.retriever.search("warm up", k=1)     # load the reranker now, not on the first question
    return bot


def init_state() -> None:
    st.session_state.setdefault("messages", [])     # {"role", "content", "resp"?}
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("feedback_logged", set())


# ---------- rendering ----------

def render_sources(resp: dict, idx: int, show_debug: bool) -> None:
    sources = resp["sources"]
    if not sources:
        return
    cited = resp["cited"] or []
    shown = [s for s in sources if s["n"] in cited] or sources[:3]
    label = f"Sources ({len(shown)})" if cited else "Most relevant pages"
    with st.expander(label):
        for s in shown:
            m = s["metadata"]
            st.markdown(f"**[{s['n']}] {m['heading']}**  \n"
                        f"[{m['title']}]({m['url']}) · {m['heading_path']}")
            st.progress(min(1.0, max(0.0, s["relevance"])), text=f"relevance {s['relevance']:.2f}")
            body = s["body"]
            st.caption(body[:500] + ("…" if len(body) > 500 else ""))
        if show_debug:
            st.markdown("**Retrieval details** (all candidates sent to the model)")
            st.dataframe([{
                "#": s["n"], "cited": "✓" if s["n"] in cited else "",
                "section": s["metadata"]["heading_path"][-60:],
                "program": s["metadata"]["program_type"],
                "dense rank": s["debug"].get("dense_rank"), "BM25 rank": s["debug"].get("bm25_rank"),
                "rerank logit": s["debug"].get("rerank_logit"), "boost": s["debug"].get("program_adj"),
                "relevance": round(s["relevance"], 3),
            } for s in sources], hide_index=True, width="stretch")
            if resp["search_query"] and resp["search_query"] != resp["question"]:
                st.caption(f"Follow-up rewritten as: *{resp['search_query']}*")
            st.caption(f"Timings: {resp['timings']}")


def render_flags(resp: dict) -> None:
    notes = [text for key, text in FLAG_NOTES.items() if resp["flags"].get(key)]
    for n in notes:
        st.caption(n)


def render_feedback(resp: dict, idx: int) -> None:
    choice = st.feedback("thumbs", key=f"fb_{idx}")
    if choice is not None and idx not in st.session_state.feedback_logged:
        st.session_state.feedback_logged.add(idx)
        G.log_event("feedback", {"question": resp["question"], "answer": resp["answer"],
                                 "rating": "up" if choice == 1 else "down",
                                 "cited_urls": [s["metadata"]["url"] for s in resp["sources"]
                                                if s["n"] in (resp["cited"] or [])]})
        st.toast("Thanks for the feedback!")


def render_message(msg: dict, idx: int, show_debug: bool) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "resp" in msg:
            render_flags(msg["resp"])
            render_sources(msg["resp"], idx, show_debug)
            render_feedback(msg["resp"], idx)


# ---------- page ----------

def main() -> None:
    init_state()

    with st.sidebar:
        st.markdown("### Try asking")
        for q in SUGGESTED:
            if st.button(q, width="stretch", key=f"sugg_{q}"):
                st.session_state.pending = q
        st.divider()
        show_debug = st.toggle("Show retrieval details", value=False,
                               help="Ranks and scores from each retrieval stage")
        if st.button("Clear conversation", width="stretch"):
            st.session_state.messages = []
            st.session_state.feedback_logged = set()
            st.rerun()
        st.divider()
        st.caption("Answers come only from the MS in Applied Data Science pages on "
                   "datascience.uchicago.edu. Details such as tuition and deadlines can change; "
                   "confirm with the admissions team before making decisions. "
                   "Please don't share personal information.")

    st.title("MS-ADS Assistant")
    st.caption("Ask about admissions, deadlines, tuition, courses and the in-person and online "
               "programs of UChicago's MS in Applied Data Science.")

    try:
        bot = load_bot()
    except Exception as e:  # noqa: BLE001
        st.error("The knowledge base isn't built yet or couldn't be loaded. Run `bash run.sh` first.\n\n"
                 f"Details: {e}")
        st.stop()

    for i, msg in enumerate(st.session_state.messages):
        render_message(msg, i, show_debug)

    typed = st.chat_input("Ask a question about the MS-ADS program")
    question = typed or st.session_state.pending
    st.session_state.pending = None
    if not question:
        return

    history = [(m["resp"]["question"], m["content"]) for m in st.session_state.messages
               if m["role"] == "assistant" and "resp" in m]
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the program website…"):
            try:
                resp = bot.answer(question, history).to_dict()
            except Exception as e:  # noqa: BLE001
                G.log_event("errors", {"error": str(e)[:500]})
                st.error("Something went wrong while answering. Please try again.")
                return

    # Store the redacted question, so personal details never persist in the chat history.
    st.session_state.messages.append({"role": "user", "content": resp["question"]})
    st.session_state.messages.append({"role": "assistant", "content": resp["answer"], "resp": resp})
    st.rerun()


main()
