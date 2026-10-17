"""Chapter 0 — Setup.

A smoke test, not a lesson: run it once after installing, to confirm your
environment can run LangGraph before Chapter 1. It prints the library versions,
builds and runs a one-node graph, and does an OPTIONAL probe for a local LLM
(only needed in Part II — a failure there is fine).

NOTE for anyone diffing the book's output blocks against this file: Chapter 0's
block is the ONE deliberate exception to "output blocks are literal stdout". It
shows a fresh Part I machine — langchain-openai not installed, no model server —
which is what the reader sees on their first run and what the prose says it is.
On a fully set-up machine this file prints versions and a list of loaded models
instead. That divergence is intended, not drift.

Run:  python code/ch00_setup.py
"""

from importlib.metadata import version, PackageNotFoundError


def show_versions() -> None:
    # Version drift is the #1 reason a beginner's code "won't run" — so print
    # the versions first, before anything else can go wrong.
    for pkg in ("langgraph", "langchain-core", "langchain-openai"):
        try:
            # version() reads what pip installed; it does not import the package
            print(f"  {pkg:<18} {version(pkg)}")
        except PackageNotFoundError:
            print(f"  {pkg:<18} (not installed)")


def build_and_run() -> str:
    # A one-node graph — proof that StateGraph -> compile -> invoke works end to
    # end. This is exactly the machine Chapter 1 takes apart; here we only check
    # that it turns over, without explaining a thing.
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END

    class State(TypedDict):
        ok: str

    def ping(state: State) -> dict:
        return {"ok": "LangGraph is working"}

    builder = StateGraph(State)
    builder.add_node("ping", ping)
    builder.add_edge(START, "ping")
    builder.add_edge("ping", END)
    graph = builder.compile()

    return graph.invoke({"ok": ""})["ok"]


def check_lm_studio(base_url: str = "http://localhost:1234/v1") -> None:
    # Optional: probe the local LM Studio server used in Part II. Not reachable
    # is NOT a failure — Part I never touches a real model.
    try:
        import json
        import urllib.request

        # A health check with the standard library, so setup needs no packages:
        # ask the server what models it has, and give up after 2 seconds.
        req = urllib.request.Request(f"{base_url}/models")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.load(resp)
        ids = [m.get("id", "?") for m in data.get("data", [])]
        print(f"  LM Studio is up; loaded models: {ids or '(none)'}")
    except Exception:
        print("  LM Studio not reachable (Part I doesn't need it; start it in Part II)")


if __name__ == "__main__":
    print("== library versions ==")
    show_versions()

    print("\n== minimal-graph smoke test ==")
    result = build_and_run()
    assert result == "LangGraph is working", f"unexpected: {result!r}"
    print(f"  {result}")

    print("\n== local LLM (optional) ==")
    check_lm_studio()

    print("\nRequired checks passed — on to Chapter 1.")
