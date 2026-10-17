"""Preflight: verify your environment before the model-backed chapters (Part II+).

Part I needs none of this. But the moment a real model enters (Chapter 10), a working
run depends on things the book cannot check for you: the right Python, the pinned
package set, a reachable LM Studio, and — the part that bites hardest — whether your
local model actually supports the handful of capabilities the chapters use.

Run this once, after `pip install -r requirements.txt` and starting LM Studio. It checks,
in order:

  1. Python version
  2. the pinned package versions (warns on drift)
  3. LM Studio reachability + loaded models
  4. six capabilities the book relies on: chat, the thinking switch (Qwen3.8 is a
     hybrid model; the code turns its thinking off), tool-calling, json_schema
     structured output, streaming, and embeddings

Each line prints PASS / WARN / FAIL / SKIP with a one-line hint on what to change
(a different model, a different method, or a missing package) when something is off. The
process exits non-zero if any required check fails, so CI can gate on it.

Run:  python code/preflight.py
"""

import importlib.metadata as md
import json
import sys
import urllib.request

LM_BASE = "http://localhost:1234/v1"
CHAT_MODEL = "qwen/qwen3.8-27b"          # or "openai/gpt-oss-20b"
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"

# A subset of the pins that most often drift; the full set is in requirements.txt.
EXPECTED = {
    "langgraph": "1.2.14",
    "langchain-core": "1.6.9",
    "langchain-openai": "1.7.0",
}

results = []   # (name, status, detail); status in {PASS, WARN, FAIL, SKIP}


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    print(f"[{status:4}] {name}" + (f"  — {detail}" if detail else ""))


# --- 1. Python ---------------------------------------------------------------
def check_python() -> None:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 11)
    record("Python >= 3.11", "PASS" if ok else "FAIL",
           f"found {v.major}.{v.minor}.{v.micro}"
           + ("" if ok else "; the book uses 3.11 syntax (X | Y unions, etc.)"))


# --- 2. pinned packages ------------------------------------------------------
def check_versions() -> None:
    for pkg, want in EXPECTED.items():
        try:
            have = md.version(pkg)
        except md.PackageNotFoundError:
            record(pkg, "FAIL", f"not installed (want =={want}); pip install -r requirements.txt")
            continue
        if have == want:
            record(f"{pkg}=={want}", "PASS")
        else:
            record(pkg, "WARN",
                   f"have {have}, book pinned {want} — install from requirements.txt to match the set")


# --- 3+. LM Studio + capabilities (need langchain-openai) --------------------
def _chat():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(base_url=LM_BASE, api_key="lm-studio", model=CHAT_MODEL,
                      temperature=0, reasoning_effort="none")


def _short(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {str(exc)[:90]}"


def check_server() -> bool:
    try:
        with urllib.request.urlopen(f"{LM_BASE}/models", timeout=5) as r:
            ids = [m["id"] for m in json.load(r).get("data", [])]
        record("LM Studio reachable", "PASS", f"loaded: {ids or '(none loaded)'}")
        return True
    except Exception as exc:
        record("LM Studio reachable", "FAIL",
               f"{LM_BASE} unreachable — start it with `lms server start` ({exc.__class__.__name__})")
        return False


def check_chat() -> None:
    try:
        out = _chat().invoke("Reply with the single word: ready").content.strip()
        record("chat completion", "PASS", f"model said {out[:30]!r}")
    except Exception as exc:
        record("chat completion", "FAIL", _short(exc) + " — try a different CHAT_MODEL")


def check_thinking_switch() -> None:
    """Qwen3.8 thinks before every answer unless told not to. The code says
    reasoning_effort="none" -- which LM Studio honours only for a model installed
    from its catalogue (`lms get qwen/qwen3.8-27b`); a Hugging Face download of
    the same weights has no switch, and every call then pays for thinking it
    never shows (and json_schema output comes back empty)."""
    try:
        with urllib.request.urlopen(LM_BASE.replace("/v1", "/api/v1/models"), timeout=5) as r:
            models = {m["key"]: m for m in json.load(r).get("models", [])}
        caps = (models.get(CHAT_MODEL) or {}).get("capabilities", {})
        if "reasoning" in caps:
            record("thinking switch", "PASS", f"{CHAT_MODEL} exposes reasoning {caps['reasoning'].get('allowed_options')}")
        elif CHAT_MODEL in models:
            record("thinking switch", "WARN",
                   f"{CHAT_MODEL} has no reasoning setting — install it from the catalogue: lms get qwen/qwen3.8-27b")
        else:
            record("thinking switch", "WARN", f"{CHAT_MODEL} not found in /api/v1/models — lms get qwen/qwen3.8-27b")
    except Exception as exc:
        record("thinking switch", "WARN", _short(exc))


def check_tool_calling() -> None:
    try:
        from langchain_core.tools import tool

        @tool
        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        msg = _chat().bind_tools([add]).invoke("What is 2 plus 3? Use the add tool.")
        if getattr(msg, "tool_calls", None):
            record("tool-calling", "PASS", "model requested the tool")
        else:
            record("tool-calling", "WARN",
                   "model answered without calling the tool — some local models under-use tools (Ch11)")
    except Exception as exc:
        record("tool-calling", "FAIL",
               _short(exc) + " — some servers reject object-valued tool_choice; try another model")


def check_json_schema() -> None:
    try:
        from typing import Literal

        from pydantic import BaseModel, Field

        class Ans(BaseModel):
            verdict: Literal["yes", "no"] = Field(description="the answer")

        out = _chat().with_structured_output(Ans, method="json_schema").invoke(
            "Is water wet? Answer yes or no.")
        record("json_schema structured output", "PASS", f"parsed verdict={out.verdict!r}")
    except Exception as exc:
        record("json_schema structured output", "FAIL",
               _short(exc) + " — LM Studio needs method='json_schema' (not json_mode); keep the schema small")


def check_streaming() -> None:
    try:
        chunks = [c for c in _chat().stream("Count out loud: one two three") if c.content]
        record("streaming", "PASS" if chunks else "WARN", f"{len(chunks)} token chunk(s)")
    except Exception as exc:
        record("streaming", "FAIL", _short(exc))


def check_embeddings() -> None:
    try:
        from langchain_openai import OpenAIEmbeddings

        emb = OpenAIEmbeddings(base_url=LM_BASE, api_key="lm-studio", model=EMBED_MODEL,
                               check_embedding_ctx_length=False)
        vec = emb.embed_query("hello")
        record("embeddings", "PASS", f"dim {len(vec)}")
    except Exception as exc:
        record("embeddings", "FAIL",
               _short(exc) + f" — load an embedding model ({EMBED_MODEL}) in LM Studio")


CAPABILITIES = [
    ("chat completion", check_chat),
    ("thinking switch", check_thinking_switch),
    ("tool-calling", check_tool_calling),
    ("json_schema structured output", check_json_schema),
    ("streaming", check_streaming),
    ("embeddings", check_embeddings),
]


def summarize() -> None:
    print("=" * 60)
    fails = [n for n, s, _ in results if s == "FAIL"]
    softs = [n for n, s, _ in results if s in ("WARN", "SKIP")]
    if fails:
        print(f"{len(fails)} check(s) FAILED: {', '.join(fails)}")
        print("Fix these before the model-backed chapters — the hints above say how.")
        sys.exit(1)
    tail = f" ({len(softs)} warning/skip — read them, they may still bite)" if softs else ""
    print("All required checks passed." + tail)
    sys.exit(0)


def main() -> None:
    print("Preflight — LangGraph in Small Pieces (needed from Part II on)")
    print("=" * 60)
    check_python()
    check_versions()

    try:
        import langchain_openai  # noqa: F401
    except ImportError:
        record("langchain-openai importable", "SKIP",
               "not installed — capability probes skipped; pip install -r requirements.txt")
        summarize()
        return

    if not check_server():
        for name, _ in CAPABILITIES:
            record(name, "SKIP", "LM Studio not reachable")
        summarize()
        return

    for _, fn in CAPABILITIES:
        fn()
    summarize()


if __name__ == "__main__":
    main()
