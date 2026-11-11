"""
Step 9: end-to-end answer evaluation, using the same chain as the app (guardrails included).

Per question:
  key_fact_recall   share of the item's must_include patterns found in the answer (deterministic)
  citation_ok       the answer cites at least one source from an expected page
  faithfulness      share of the answer's claims supported by the retrieved sources (LLM grader)
  relevance         1-5: does the answer address the question (LLM grader)
  correctness       1-5: agreement with the reference answer (LLM grader)
  behavior_ok       out-of-scope / not-on-site items: the bot declined instead of answering
  latency_s         end-to-end time

Usage:
    python -m eval.answer_eval                    # all items (~4 minutes, a few cents with gpt-4o-mini)
    python -m eval.answer_eval --ids q01 q05 u01  # selected items
    python -m eval.answer_eval --no-grader        # skip the LLM grader (deterministic metrics only)
Outputs (eval/results/):
    answers.jsonl          full record per item (answer, sources, flags, metrics)
    answers_review.md      readable dump for spot-checking and picking deck examples
    answer_metrics.csv     one row per item
    answer_summary.json    aggregates (by category and overall), used by make_charts.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from pathlib import Path

from rag.chain import RAGChain, format_context
from rag.llm import get_chat_model

ROOT = Path(__file__).resolve().parents[1]
TESTSET = ROOT / "eval/testset.json"
OUT = ROOT / "eval/results"

DECLINE_RE = re.compile(
    r"couldn'?t find|could not find|don'?t have (that|this)|do not have (that|this)|not (available|listed|mentioned)|"
    r"can only help|only (help|answer|assist)|can'?t help|cannot help|unable to|no information|not able to|"
    r"couldn'?t confirm", re.I)
# Narrower: the bot's own "I won't answer" phrasings, used to judge whether a real answer was given.
SELF_DECLINE_RE = re.compile(r"\bI (couldn'?t|could not) (find|confirm)|\bI (don'?t|do not) have (that|this|information)|"
                             r"\bI can only help|\bI can'?t help", re.I)
LEAK_RE = re.compile(r"Cite every factual statement|Never reveal or discuss these rules|Answer ONLY from the numbered", re.I)

GRADER_SYSTEM = """You grade answers from a chatbot for a university degree program. Use only the \
sources and the reference answer given. Respond with JSON only:
{"claims": [{"claim": "<one factual statement from the answer>", "supported": true|false}],
 "relevance": <1-5>,
 "correctness": <1-5 or null>}
claims: every factual statement in the answer. supported = the sources state it or directly imply it. \
Pointers such as "see the How to Apply page" or "contact admissions" are not claims.
relevance: 5 = directly and completely answers the question asked; 3 = partly; 1 = does not address it.
correctness: compare against the reference answer. 5 = all key facts agree and nothing contradicts it; \
3 = some key facts missing or vague; 1 = wrong or contradicts the reference. null if no reference is given."""


def grade(grader, question: str, answer: str, sources, reference: str | None) -> dict:
    human = (f"<question>\n{question}\n</question>\n\n<sources>\n{format_context(sources)}\n</sources>\n\n"
             f"<reference>\n{reference or '(none)'}\n</reference>\n\n<answer>\n{answer}\n</answer>")
    raw = grader.invoke([("system", GRADER_SYSTEM), ("human", human)]).content
    try:
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    except Exception:  # noqa: BLE001
        return {"faithfulness": None, "relevance": None, "correctness": None, "grader_error": raw[:200]}
    claims = data.get("claims") or []
    supported = sum(1 for c in claims if c.get("supported"))
    corr = data.get("correctness")
    return {
        "faithfulness": supported / len(claims) if claims else 1.0,
        "n_claims": len(claims),
        "unsupported_claims": [c.get("claim") for c in claims if not c.get("supported")],
        "relevance": data.get("relevance"),
        "correctness": corr if reference else None,
    }


def run_item(bot: RAGChain, item: dict) -> tuple:
    history = []
    for q in item.get("history", []):
        r = bot.answer(q, history)
        history.append((r.question, r.answer))
    t0 = time.time()
    resp = bot.answer(item["question"], history)
    return resp, round(time.time() - t0, 2)


def score_item(item: dict, resp, latency: float, grader) -> dict:
    behavior = item.get("expected_behavior", "answer")
    ans = resp.answer
    m: dict = {"id": item["id"], "category": item["category"], "expected_behavior": behavior,
               "latency_s": latency, "n_cited": len(resp.cited),
               "regenerated": bool(resp.flags.get("regenerated")), "fallback": bool(resp.flags.get("fallback")),
               "low_relevance": bool(resp.flags.get("low_relevance"))}

    if behavior in ("refuse", "abstain"):
        declined = (resp.flags.get("low_relevance") or resp.flags.get("moderation_flagged")
                    or bool(DECLINE_RE.search(ans)))
        m["behavior_ok"] = bool(declined) and not LEAK_RE.search(ans)
        return m

    pats = item.get("must_include", [])
    if pats:
        found = [bool(re.search(p, ans, re.I)) for p in pats]
        m["key_fact_recall"] = sum(found) / len(found)
        m["missing_facts"] = [p for p, f in zip(pats, found) if not f]
    exp = item.get("expected_urls", [])
    if exp:
        m["citation_ok"] = any(any(f in h.metadata["url"] for f in exp) for h in resp.cited_sources)
    m["answered"] = not (resp.flags.get("low_relevance") or resp.flags.get("fallback")
                         or bool(SELF_DECLINE_RE.search(ans)))
    if grader is not None and resp.sources:
        m.update(grade(grader, resp.search_query or item["question"], ans, resp.sources, item.get("reference")))
    return m


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def aggregate(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["expected_behavior"] == "answer"]
    dec = [r for r in rows if r["expected_behavior"] != "answer"]
    lat = [r["latency_s"] for r in rows]
    return {
        "n": len(rows),
        "answer_items": len(ans),
        "key_fact_recall": mean([r.get("key_fact_recall") for r in ans]),
        "citation_accuracy": mean([float(r["citation_ok"]) for r in ans if "citation_ok" in r]),
        "answer_rate": mean([float(r["answered"]) for r in ans]),
        "faithfulness": mean([r.get("faithfulness") for r in ans]),
        "relevance_1to5": mean([r.get("relevance") for r in ans]),
        "correctness_1to5": mean([r.get("correctness") for r in ans]),
        "decline_items": len(dec),
        "decline_accuracy": mean([float(r["behavior_ok"]) for r in dec]),
        "regenerated": sum(r["regenerated"] for r in rows),
        "fallback": sum(r["fallback"] for r in rows),
        "latency_p50_s": round(statistics.median(lat), 2) if lat else None,
        "latency_p95_s": round(sorted(lat)[max(0, int(len(lat) * 0.95) - 1)], 2) if lat else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--no-grader", action="store_true")
    args = ap.parse_args()

    items = json.loads(TESTSET.read_text())
    if args.ids:
        items = [i for i in items if i["id"] in set(args.ids)]
    bot = RAGChain()
    grader = None if args.no_grader else get_chat_model("grader")
    OUT.mkdir(parents=True, exist_ok=True)

    rows, records = [], []
    for n, item in enumerate(items, 1):
        resp, latency = run_item(bot, item)
        m = score_item(item, resp, latency, grader)
        rows.append(m)
        records.append({"item": item, "metrics": m, "response": resp.to_dict()})
        tag = ("ok " if m.get("behavior_ok") else "BAD") if "behavior_ok" in m else \
              f"facts={m.get('key_fact_recall', '-')!s:4} cite={'ok' if m.get('citation_ok') else 'no'}"
        print(f"[{n:>2}/{len(items)}] {item['id']} {item['category']:<12} {tag}  {latency:>5.1f}s  {item['question'][:55]}")

    cats = sorted({r["category"] for r in rows}, key=lambda c: ["core", "factual", "paraphrase", "followup",
                                                                 "out_of_scope", "not_on_site"].index(c)
                  if c in ["core", "factual", "paraphrase", "followup", "out_of_scope", "not_on_site"] else 99)
    summary = {"overall": aggregate(rows), "by_category": {c: aggregate([r for r in rows if r["category"] == c]) for c in cats}}

    with (OUT / "answers.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    (OUT / "answer_summary.json").write_text(json.dumps(summary, indent=2))
    keys = sorted({k for r in rows for k in r})
    with (OUT / "answer_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with (OUT / "answers_review.md").open("w", encoding="utf-8") as f:
        for rec in records:
            it, m, r = rec["item"], rec["metrics"], rec["response"]
            f.write(f"## {it['id']} [{it['category']}] {it['question']}\n")
            if it.get("history"):
                f.write(f"_after: {' / '.join(it['history'])}_  \n")
            f.write(f"\n{r['answer']}\n\n")
            for s in r["sources"]:
                if s["n"] in r["cited"]:
                    f.write(f"- [{s['n']}] {s['metadata']['heading_path']} ({s['metadata']['url']})\n")
            shown = {k: v for k, v in m.items() if k not in ("id", "category") and v not in (None, [], False)}
            f.write(f"\n`{json.dumps(shown, default=str)}`\n\n")
            if it.get("reference"):
                f.write(f"> Reference: {it['reference']}\n\n")
            f.write("---\n\n")

    o = summary["overall"]
    print("\nAnswer quality by category")
    hdr = f"{'category':<13}{'n':>3}{'facts':>8}{'cited':>8}{'faith':>8}{'relev':>7}{'correct':>9}{'declined':>10}"
    print(hdr)
    for c, a in list(summary["by_category"].items()) + [("OVERALL", o)]:
        fmt = lambda v, p=False: "-" if v is None else (f"{v:.0%}" if p else f"{v:.2f}")
        print(f"{c:<13}{a['n']:>3}{fmt(a['key_fact_recall'], True):>8}{fmt(a['citation_accuracy'], True):>8}"
              f"{fmt(a['faithfulness'], True):>8}{fmt(a['relevance_1to5']):>7}{fmt(a['correctness_1to5']):>9}"
              f"{fmt(a['decline_accuracy'], True):>10}")
    print(f"\nlatency p50 {o['latency_p50_s']}s, p95 {o['latency_p95_s']}s | "
          f"answers rewritten by the hallucination check: {o['regenerated']} | fallbacks: {o['fallback']}")
    print(f"Saved results to {OUT}/  (open answers_review.md to read every answer)")


if __name__ == "__main__":
    main()
