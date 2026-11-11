# MS-ADS RAG Chatbot

A RAG chatbot that answers questions about UChicago's MS in Applied Data Science, built from the program website.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # add OPENAI_API_KEY (needed from Step 4)
```

## Step 1: Crawl
```bash
python -m scraper.crawl
```
- Raw HTML goes to `data/raw/`, and one row per page goes to `data/raw/manifest.jsonl`.
- Scope is set in `scraper/config.py`:
  - **follow:** MS-ADS pages, in-person and online program pages, and tuition. These are crawled and their links are explored.
  - **leaf:** English-language scores, the PSD fee waiver, and Booth MBA/MS. These are fetched but their links are not followed.
  - Events and news pages are excluded.
- If pages are flagged `THIN` (JS-rendered), run `pip install playwright && playwright install chromium`, then `python -m scraper.crawl --playwright`.

## Step 2: Clean and structure
```bash
python -m pipeline.clean          # -> data/processed/pages.jsonl + preview.md
python -m pipeline.verify_facts   # are the answers to all eval questions in the data?
```
What the cleaner does:
1. Finds the main content and strips the header, nav, sidebar, breadcrumbs, footer, and share widgets.
2. Walks the content in order and splits it into sections at every heading. FAQ accordion questions become their own sections, so each answer stays with its question.
3. Converts tables to Markdown and keeps line breaks (addresses) and nested-list indentation.
4. Appends link URLs to link text, so answers can cite pages such as the advising portal.
5. Removes lines that repeat across 40% or more of pages (leftover boilerplate).
6. Tags every page with a `category` and every section with a `program_type` (`online`, `in_person`, or `both`). The section-level tag uses the heading path, so an "Online Program" FAQ group is tagged `online` even on a shared FAQ page.

**Open `preview.md` and skim it.** It's the fastest way to catch extraction problems before chunking.

## Step 3: Chunk
```bash
python -m pipeline.chunk                     # -> data/processed/chunks.jsonl
python -m pipeline.chunk --max 300 --overlap 40   # try other sizes for the ablation
```
1. **One section becomes one chunk** when it fits within 400 tokens. Each FAQ question and its answer is always its own chunk.
2. **Long sections are split at sentence boundaries** (or table rows) and packed back up to 400 tokens, with about 50 tokens of overlap. A split inside a table repeats the table's header row, so dates stay attached to their labels.
3. **Tiny sections** (an intro line under a heading) are merged into the next related section.
4. **Contextual headers.** Every chunk's indexed text begins with `Source: <page > heading path>` and `Program: <type>`. The raw text is kept separately in `body` so the UI can show it.

## Step 4: Index
```bash
python -m pipeline.index          # -> chroma_db/ (dense, cosine) + data/processed/bm25.json
python -m pipeline.smoke_search   # eval questions vs dense and vs BM25, top hits + Hit@k
```
- The embedding model is chosen in `.env` (`EMBEDDING_PROVIDER=openai` or `hf`) and read by `rag/embeddings.py`. The indexer and the retriever both use that file, so they can't disagree. `index_meta.json` records which model built the index, and smoke search refuses to run if it doesn't match.
- BM25 tokenization (`rag/text.py`) keeps compounds such as `mba/ms` and `stem/opt` whole and also indexes their parts.
- The index is rebuilt from scratch on every run, so re-crawl, then re-run clean, chunk, and index.
- The smoke-search Hit@k is **page-level** (the right URL is in the top k). It's good for spotting missing data, but it's lenient. The strict chunk-level metric comes in Step 9.
- Offline testing with no API key or model download: `EMBEDDING_PROVIDER=fake`. Never use it for the real index.

## Step 5: Hybrid retriever
```bash
python -m rag.retriever "What are the TOEFL and IELTS minimums?"   # top 5 with per-stage debug info
python -m eval.retrieval_eval --show-misses                          # ablation table for the deck
```
Pipeline (`rag/retriever.py`):
1. **Dense** (Chroma, top 20) and **BM25** (top 20) run in parallel.
2. **Reciprocal Rank Fusion** (k=60) merges the two rankings, so the very different score scales never need calibrating.
3. **Cross-encoder rerank** of the top 20. `RERANKER=minilm` is the default (fast, small enough for free hosting); `bge` is more accurate; `none` skips this stage. The model downloads once on first use.
4. **Program soft boost.** If the question says online or in-person (or mentions visas), chunks tagged with that program get +0.10 and chunks tagged with the other program get -0.15. Chunks tagged `both` are unaffected. It's a boost, not a filter, so shared answers are never hidden.
5. **De-duplication.** One hit per section, and the same answer repeated on several pages is dropped.

Every hit carries `debug` info (dense rank, BM25 rank, rerank score, boost), which the UI can show as "why this source". `retriever.as_langchain(k=5)` returns a LangChain `BaseRetriever` for the chain in Step 6.

`eval/retrieval_eval.py` scores five configurations (dense only, BM25 only, hybrid, + rerank, + boost) on Hit@1/3/5 and MRR@10, and writes CSVs to `eval/results/`.

## Step 6: LLM chain
```bash
python -m rag.chat                        # terminal chat with memory ('reset' clears it)
python -m rag.chat -q "What is the tuition?"
```
`rag/chain.py` (LangChain LCEL) does the following:
- **Rewrites follow-ups.** "What about online?" is turned into a standalone question using the last 3 turns before searching.
- **Retrieves** the top 5 sources with the hybrid retriever.
- **Answers with `gpt-4o-mini`** under a grounded system prompt (`rag/prompts.py`). The prompt requires the model to answer only from the sources, cite every fact as `[n]`, copy figures exactly, separate in-person from online answers, say "I don't know" rather than guess, and stay on topic.
- **Checks citations.** Citations to sources that don't exist are removed. Each response carries the cited sources, guardrail flags, and timings.

## Step 7: Responsible AI guardrails (`rag/guardrails.py`)
| Guardrail | What it does |
|---|---|
| PII redaction | Emails, phone numbers, SSNs, and card numbers (Luhn-checked) become `[EMAIL]` and so on before the LLM, logs, or chat history see them |
| Moderation | OpenAI moderation endpoint; flagged requests get a polite refusal |
| Relevance gate | If the best source's reranker relevance is below `MIN_RELEVANCE`, the bot answers "I couldn't find that" instead of guessing |
| Hallucination check | A second model checks every statement against the sources. One rewrite is tried with the unsupported claims listed; if it still fails, the bot falls back to links to the relevant pages |
| Prompt injection | Attempts such as "ignore previous instructions" are flagged and logged; the system prompt treats source text as data |
| Scope | Off-topic questions are declined; greetings get a canned introduction (no LLM call) |
| Logging | `logs/interactions.jsonl` (redacted question, answer, cited URLs, flags, timings) and `logs/feedback.jsonl` |

## Step 8: UI
```bash
streamlit run app.py
```
- Chat with memory.
- Suggested questions in the sidebar.
- A **Sources** panel under each answer, showing the cited sections, a link to each page, a relevance bar, and a snippet.
- Guardrail notes (for example, "personal details were removed").
- Thumbs up/down feedback, logged.
- A **Show retrieval details** toggle, which displays each candidate's dense rank, BM25 rank, reranker logit, and program boost, plus the rewritten follow-up and timings.

## Step 9: Evaluation
```bash
python -m eval.retrieval_eval           # retrieval ablation on all 35 answerable questions
python -m eval.retrieval_eval --core    # the same on the 12 class questions only
python -m eval.answer_eval              # end-to-end answer quality (~4 min, a few cents)
python -m eval.make_charts              # PNG charts + Markdown tables for the deck
# or all of it:  bash run.sh --eval
```
**The test set** (`eval/testset.json`, 42 items):

| Category | n | What it tests |
|---|---|---|
| core | 12 | the class questions, with answers updated to the current site |
| factual | 13 | other facts across the FAQ, curriculum, capstone, careers, and the program pages |
| paraphrase | 6 | casual or reworded versions of the core questions |
| followup | 4 | two-turn conversations ("what about IELTS?") |
| out_of_scope | 5 | off-topic questions, another school's tuition, and a prompt-injection attempt |
| not_on_site | 2 | plausible questions the site doesn't answer, which the bot should decline |

**Answer metrics** (`eval/answer_eval.py`, which runs the same chain as the app, guardrails included):
- **Key-fact recall.** Deterministic: the share of each item's `must_include` patterns (for example `104` and `Suite 2800`) found in the answer.
- **Citation accuracy.** The answer cites a source from one of the expected pages.
- **Faithfulness.** An LLM grader splits the answer into claims and checks each one against the retrieved sources; the score is the share supported.
- **Relevance and correctness (1–5).** LLM-graded against the question and the reference answer.
- **Decline accuracy.** Off-topic and not-on-site items must be declined without leaking the system prompt.
- **Latency** (median and p95), plus how often the hallucination check rewrote an answer or fell back.

**Outputs** go to `eval/results/`:
- `answers_review.md`, which shows every answer with its sources and scores. Read this, and use it to pick examples for the deck.
- `chart_retrieval.png` and `chart_answer_quality.png`.
- `summary_tables.md`.
- CSV and JSON detail files.

## Offline testing
Set `EMBEDDING_PROVIDER=fake LLM_PROVIDER=fake MODERATION=off RERANKER=none` to run the whole stack without API keys or model downloads. The answers are canned, so use this only to check that the code runs.

## Full rebuild
```bash
bash run.sh           # clean -> verify -> chunk -> index -> smoke search -> retrieval eval
bash run.sh --crawl   # same, re-crawling the site first
```

## Layout
```
scraper/   config.py (crawl scope), crawl.py
pipeline/  clean.py, verify_facts.py, chunk.py, index.py, smoke_search.py
rag/       embeddings.py, text.py, retriever.py, llm.py, prompts.py, guardrails.py, chain.py, chat.py
app.py     Streamlit UI
eval/      testset.json, retrieval_eval.py, results/
tests/fixtures/  sample HTML for testing the pipeline offline
```

## Changes after the first evaluation (Nov 10)
Tuned against the class evaluation set:
- **Joint-program penalty** (`rag/retriever.py`): sections about the joint MBA/MS degree move down unless the question mentions MBA, Booth or joint. Its FAQ lists a TOEFL minimum of 104, which was confusing answers about the MS-ADS minimum (102 on the prior scale, or 5 on the current scale).
- **Two-signal relevance gate** (`rag/chain.py`, `rag/guardrails.py`): the bot declines without an LLM call only when the reranker relevance is below `MIN_RELEVANCE` **and** the best dense similarity is below `MIN_DENSE_SIM`. The MiniLM reranker alone had scored the "Meet your Admissions Counselor" section very low for the advising-appointment question. Calibrate with `python -m eval.gate_check`.
- **Prompts** (`rag/prompts.py`): copy figures with their qualifiers; prefer the MS-ADS pages over joint-program pages when figures disagree; list names that appear in section headings (scholarships); give the link for actions (appointments); follow-up rewriting keeps the previous question's topic, with examples; the groundedness judge accepts figures copied with their qualifiers and statements backed by any one source.
- **Test set** (`eval/testset.json`): TOEFL reference is 102/5 (the site's MS-ADS value), and the advising links are the two counselors' current scheduling links.
