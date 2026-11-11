"""
Step 3: turn cleaned sections into retrieval chunks.

Input:  data/processed/pages.jsonl   (from pipeline/clean.py)
Output: data/processed/chunks.jsonl  one chunk per line

Strategy
1. One section = one chunk when it fits (most FAQ answers, short program sections).
2. Long sections are split on line boundaries up to MAX_TOKENS, with OVERLAP_TOKENS carried over.
   A split inside a Markdown table repeats the table's header row, so dates stay attached to labels.
3. Tiny non-FAQ sections (an intro line under a heading) are merged into the following section
   when it shares a parent heading, so retrieval doesn't return near-empty chunks.
4. Each chunk's embedded text starts with a contextual header:
       Source: FAQs > Application Process > Is there an application fee waiver?
       Program: both
   The page/heading path tells the embedding model what the text is about. This is the single
   cheapest retrieval improvement; keep it as an ablation for the deck.

Usage:
    python -m pipeline.chunk                # defaults
    python -m pipeline.chunk --max 300 --overlap 40
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

IN_PATH = Path("data/processed/pages.jsonl")
OUT_PATH = Path("data/processed/chunks.jsonl")

MAX_TOKENS = 400
OVERLAP_TOKENS = 50
MIN_TOKENS = 40          # sections below this are merge candidates
FAQ_LEVEL = 5            # accordion questions (see clean.py) are never merged


# ---------- token counting ----------

def _make_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s)), "tiktoken"
    except Exception:  # noqa: BLE001  (no network for the encoding file, not installed, ...)
        return lambda s: int(len(s.split()) * 1.3) + 1, "word-estimate"


count_tokens, TOKENIZER = _make_counter()


# ---------- splitting ----------

SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def _hard_split(sent: str, max_tokens: int) -> list[str]:
    """Last resort for a single 'sentence' longer than max_tokens: cut on words."""
    words, out, cur = sent.split(), [], []
    for w in words:
        if cur and count_tokens(" ".join(cur + [w])) > max_tokens:
            out.append(" ".join(cur))
            cur = []
        cur.append(w)
    if cur:
        out.append(" ".join(cur))
    return out


def is_separator(ln: str) -> bool:          # the |---|---| row of a Markdown table
    return is_table_line(ln) and set(ln.replace("|", "").strip()) <= {"-", ":", " "}


def to_units(text: str, max_tokens: int) -> list[dict]:
    """Break text into packable units: one per sentence (prose) or per row (tables).
    Each unit remembers the separator that preceded it, so re-joining restores the
    original formatting, and which table header it belongs to."""
    lines = text.split("\n")
    units: list[dict] = []
    hdr: str | None = None
    for i, ln in enumerate(lines):
        if is_table_line(ln):
            if i == 0 or not is_table_line(lines[i - 1]):          # first row of a new table
                has_sep = i + 1 < len(lines) and is_separator(lines[i + 1])
                hdr = "\n".join(lines[i:i + 2] if has_sep else lines[i:i + 1])
            units.append({"text": ln, "sep": "\n", "hdr": hdr})
            continue
        hdr = None
        sents = [p for s in SENT_RE.split(ln) for p in _hard_split(s, max_tokens)]
        for j, s in enumerate(sents):
            units.append({"text": s, "sep": "\n" if j == 0 else " ", "hdr": None})
    for u in units:
        u["tokens"] = count_tokens(u["text"])
    return units


def _join(units: list[dict]) -> str:
    return units[0]["text"] + "".join(u["sep"] + u["text"] for u in units[1:]) if units else ""


def split_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    units = to_units(text, max_tokens)
    if sum(u["tokens"] for u in units) <= max_tokens:
        return [text]

    chunks: list[str] = []
    cur: list[dict] = []
    cur_tokens = 0
    for u in units:
        if cur and cur_tokens + u["tokens"] > max_tokens:
            chunks.append(_join(cur))
            # carry trailing units totalling <= overlap tokens into the next chunk
            carry: list[dict] = []
            for prev in reversed(cur):
                if sum(c["tokens"] for c in carry) + prev["tokens"] > overlap:
                    break
                carry.insert(0, prev)
            # continuing inside a table: start with its header, then the carried rows of that table
            if u["hdr"]:
                hdr_lines = set(u["hdr"].split("\n"))
                rows = [c for c in carry if c["hdr"] == u["hdr"] and c["text"] not in hdr_lines]
                carry = [{"text": u["hdr"], "sep": "\n", "hdr": u["hdr"],
                          "tokens": count_tokens(u["hdr"])}] + rows
            cur = carry
            cur_tokens = sum(c["tokens"] for c in cur)
        cur.append(u)
        cur_tokens += u["tokens"]
    if cur:
        chunks.append(_join(cur))
    return chunks


# ---------- merging tiny sections ----------

def parent_path(path: str) -> str:
    return path.rsplit(" > ", 1)[0] if " > " in path else path


def is_qa(s: dict) -> bool:
    return bool(s.get("accordion")) or s["level"] == FAQ_LEVEL


def merge_tiny(sections: list[dict], max_tokens: int) -> list[dict]:
    """Merge a tiny, non-FAQ section into the NEXT section when the next one is its
    child or sibling and the result still fits."""
    out: list[dict] = []
    pending: dict | None = None
    for s in sections:
        s = dict(s)
        if pending:
            related = (s["heading_path"].startswith(pending["heading_path"])
                       or parent_path(s["heading_path"]) == parent_path(pending["heading_path"]))
            if related and not is_qa(s) and \
                    count_tokens(pending["text"] + s["text"]) <= max_tokens:
                s["text"] = f"{pending['heading']}: {pending['text']}\n{s['text']}"
            else:
                out.append(pending)
            pending = None
        if not is_qa(s) and count_tokens(s["text"]) < MIN_TOKENS:
            pending = s
        else:
            out.append(s)
    if pending:
        out.append(pending)
    return out


# ---------- driver ----------

def contextual_header(heading_path: str, program_type: str) -> str:
    return f"Source: {heading_path}\nProgram: {program_type}\n\n"


def chunk_id(url: str, heading_path: str, idx: int) -> str:
    return hashlib.sha1(f"{url}|{heading_path}|{idx}".encode()).hexdigest()[:12]


def build_chunks(pages: list[dict], max_tokens: int, overlap: int) -> list[dict]:
    chunks = []
    for p in pages:
        for si, s in enumerate(merge_tiny(p["sections"], max_tokens)):
            header = contextual_header(s["heading_path"], s["program_type"])
            budget = max_tokens - count_tokens(header)
            for i, piece in enumerate(split_text(s["text"], budget, overlap)):
                chunks.append({
                    # section index included: two sections on a page can share a heading path
                    "id": chunk_id(p["url"], f"{si}|{s['heading_path']}", i),
                    "text": header + piece,          # what gets embedded + BM25-indexed
                    "body": piece,                   # what the UI shows as the source snippet
                    "metadata": {
                        "url": p["url"],
                        "title": p["title"],
                        "category": p["category"],
                        "program_type": s["program_type"],
                        "heading": s["heading"],
                        "heading_path": s["heading_path"],
                        "chunk_index": i,
                    },
                })
    for c in chunks:
        c["metadata"]["n_tokens"] = count_tokens(c["text"])
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=MAX_TOKENS)
    ap.add_argument("--overlap", type=int, default=OVERLAP_TOKENS)
    args = ap.parse_args()

    pages = [json.loads(l) for l in IN_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    chunks = build_chunks(pages, args.max, args.overlap)

    ids = [c["id"] for c in chunks]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate chunk ids: {sorted(dupes)[:5]}"
    paths = [(c["metadata"]["url"], c["metadata"]["heading_path"]) for c in chunks if c["metadata"]["chunk_index"] == 0]
    repeated = {p for p in paths if paths.count(p) > 1}
    if repeated:
        print(f"note: {len(repeated)} heading path(s) appear more than once on a page, e.g. {next(iter(repeated))[1][-80:]!r}")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    sizes = [c["metadata"]["n_tokens"] for c in chunks]
    print(f"{len(chunks)} chunks from {len(pages)} pages  (tokenizer: {TOKENIZER})")
    print(f"tokens/chunk  min={min(sizes)}  median={int(statistics.median(sizes))}  max={max(sizes)}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
