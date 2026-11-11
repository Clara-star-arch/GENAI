"""Tokenizer for BM25. Shared by the indexer and the retriever so both tokenize identically."""
import re

STOPWORDS = set("""
a an the and or of to in on for with at by from is are was were be been being do does did
i me my we our you your it its this that these those what which who whom how when where why
can could will would should may might must shall about as into than then there their them
""".split())

# Keep slash/dot compounds such as "mba/ms", "stem/opt", "u.s." as single tokens, plus their parts.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[/.\-][a-z0-9]+)*")


def bm25_tokenize(text: str) -> list[str]:
    out = []
    for tok in TOKEN_RE.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        out.append(tok)
        if any(ch in tok for ch in "/.-"):
            out.extend(p for p in re.split(r"[/.\-]", tok) if p and p not in STOPWORDS)
    return out
