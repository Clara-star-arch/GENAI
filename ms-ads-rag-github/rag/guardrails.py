"""
Step 7: Responsible AI guardrails.

    user message
      -> redact_pii          emails, phone numbers, SSNs, card numbers never reach the LLM or the logs
      -> moderate            OpenAI moderation endpoint; harmful requests are refused
      -> detect_injection    "ignore your instructions" style attempts are flagged and logged
    retrieval
      -> relevance gate      if the best source is weak, answer "I don't know" instead of guessing
    generation
      -> check_grounded      a second model checks every statement against the sources;
                             one rewrite is tried, then a safe fallback with links

Config (.env):
    MODERATION=on|off          default on (needs OPENAI_API_KEY)
    GROUNDING_CHECK=on|off     default on (one extra cheap LLM call per answer)
    MIN_RELEVANCE=0.05         minimum reranker relevance of the best source (0-1)
    MIN_DENSE_SIM=0.40         ...unless the best dense cosine similarity is at least this
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def setting(name: str, default: str) -> str:
    return os.getenv(name, default).strip().lower()


# ---------- PII redaction ----------

def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
        alt = not alt
    return total % 10 == 0


PII_PATTERNS = [  # order matters: most specific first
    ("SSN",   re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CARD",  re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)")),
]


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Returns (redacted text, list of PII types found)."""
    found: list[str] = []

    def sub(label, rx, s):
        def repl(m):
            if label == "CARD" and not _luhn_ok(re.sub(r"\D", "", m.group(0))):
                return m.group(0)                      # 16 digits that fail Luhn: not a card
            found.append(label)
            return f"[{label}]"
        return rx.sub(repl, s)

    for label, rx in PII_PATTERNS:
        text = sub(label, rx, text)
    return text, sorted(set(found))


# ---------- moderation ----------

def moderate(text: str) -> tuple[bool, list[str]]:
    """(flagged, categories). Fails open: if the API is unavailable, the message is allowed
    (the system prompt still restricts the assistant to program topics)."""
    if setting("MODERATION", "on") != "on" or not os.getenv("OPENAI_API_KEY"):
        return False, []
    try:
        from openai import OpenAI
        res = OpenAI().moderations.create(model="omni-moderation-latest", input=text).results[0]
        cats = [k for k, v in res.categories.model_dump().items() if v]
        return bool(res.flagged), cats
    except Exception as e:  # noqa: BLE001
        log_event("moderation_error", {"error": str(e)[:200]})
        return False, []


# ---------- prompt injection ----------

INJECTION_RE = re.compile(
    r"ignore (all |any |the )?(previous|prior|above|your) (instructions|rules|prompt)|"
    r"(reveal|show|print|repeat) (your|the) (system )?(prompt|instructions|rules)|"
    r"you are now|act as (?!a (prospective|current) student)|jailbreak|developer mode",
    re.I,
)


def detect_injection(text: str) -> bool:
    return bool(INJECTION_RE.search(text))


# ---------- small talk ----------

GREETING_RE = re.compile(r"^\s*(hi|hello|hey|hiya|good (morning|afternoon|evening)|thanks|thank you|"
                         r"what can you do|who are you|help)\b[\s!.?]*$", re.I)


def is_greeting(text: str) -> bool:
    return bool(GREETING_RE.match(text))


# ---------- relevance gate ----------

def min_relevance() -> float:
    return float(os.getenv("MIN_RELEVANCE", "0.05"))


def min_dense_sim() -> float:
    """Second signal for the relevance gate: cosine similarity between the question and the best
    dense hit. The small MiniLM reranker sometimes scores a good source very low (e.g. "advising
    appointment" vs the "Meet your Admissions Counselor" section), so the gate only declines when
    BOTH the reranker and the embedding similarity are weak. Calibrate with eval/gate_check.py."""
    return float(os.getenv("MIN_DENSE_SIM", "0.40"))


# ---------- groundedness ----------

def parse_judge(raw: str) -> tuple[bool, list[str]]:
    """Parse the judge's JSON. Unparseable output counts as grounded (fail open), because a
    malfunctioning judge should not block every answer; the event is logged."""
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0) if m else raw)
        unsupported = [str(u) for u in data.get("unsupported", []) if str(u).strip()]
        return bool(data.get("grounded", True)), unsupported
    except Exception:  # noqa: BLE001
        log_event("judge_parse_error", {"raw": raw[:300]})
        return True, []


# ---------- logging ----------

def log_event(kind: str, payload: dict) -> None:
    """Append one JSON line to logs/<kind>.jsonl. Callers pass already-redacted text only."""
    try:
        LOG_DIR.mkdir(exist_ok=True)
        with (LOG_DIR / f"{kind}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **payload}, ensure_ascii=False) + "\n")
    except OSError:
        pass   # read-only hosting: logging is best-effort
