"""
Chat model factory. Configure in .env:

    LLM_PROVIDER=openai     (default)
    LLM_MODEL=gpt-4o-mini   answer model
    JUDGE_MODEL=gpt-4o-mini groundedness checker and follow-up rewriter
    LLM_PROVIDER=fake       canned outputs, ONLY for offline tests (no API key needed)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "openai").lower()


def model_name(role: str = "answer") -> str:
    if provider() == "fake":
        return "fake"
    if role == "answer":
        return os.getenv("LLM_MODEL", "gpt-4o-mini")
    return os.getenv("JUDGE_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))


def get_chat_model(role: str = "answer") -> Runnable:
    """role: 'answer' | 'condense' | 'judge' | 'grader' (evaluation only)."""
    if provider() == "fake":
        return RunnableLambda(lambda prompt: AIMessage(content=_fake_reply(role, prompt)))
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=model_name(role), temperature=0, timeout=30, max_retries=2)
    if role in ("judge", "grader"):
        return llm.bind(response_format={"type": "json_object"})
    return llm


def _fake_reply(role: str, prompt) -> str:
    """Deterministic stand-in used by tests: answers with the first line of source [1]."""
    text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
    if role == "judge":
        return json.dumps({"grounded": "UNSUPPORTED" not in text, "unsupported": []})
    if role == "grader":
        return json.dumps({"claims": [{"claim": "fake", "supported": True}], "relevance": 4, "correctness": 4})
    if role == "condense":
        m = re.search(r"Latest message:\s*(.+?)\n", text)
        return (m.group(1) if m else text).strip() + " (rewritten)"
    m = re.search(r"\[1\][^\n]*\nURL:[^\n]*\n(.+?)(?:\n|$)", text)
    return (m.group(1).strip() if m else "I don't have that information.") + " [1]"
