"""Chapter 5 Challenge — Silver: data-driven tool priority.

The chapter's agent hard-codes its fallback with an if/elif ladder: try the docs,
then the web on a miss. Add a third tool and you edit the ladder again. This
generalizes the ladder into *data*: a priority list of tools the agent walks in
order, trying the next only when the current one missed. Reordering or adding a
tool is now editing a list — the agent logic never changes.

Each tool writes its own result field, so "already tried" = "my field is present"
(possibly empty) and "found something" = "my field is non-empty" — the same
present-but-empty distinction the chapter draws.

Run:  python code/answers/ch05_tool_priority.py   (see answers/README.md)
"""

from typing import NotRequired, TypedDict

from langgraph.graph import StateGraph, START, END

# (route name, result field, node name) — walked top to bottom. Add/reorder here.
# Names match the chapter: the docs tool routes via "search", the web tool writes
# "web_results" (the faq tool is the new tier this challenge adds).
TOOLS = [
    ("search", "docs", "search_docs"),
    ("faq", "faq", "search_faq"),
    ("web", "web_results", "web_search"),
]

DOCS_KB = {"kafka": ["Kafka Manual: batch consumers"]}
FAQ_KB = {"reset password": ["FAQ: use the reset link"]}


class State(TypedDict):
    question: str
    route: NotRequired[str]
    docs: NotRequired[list[str]]
    faq: NotRequired[list[str]]
    web_results: NotRequired[list[str]]
    answer: NotRequired[str]


def agent_node(state: State) -> dict:
    # Found something already? Go answer.
    if any(state.get(field) for _, field, _ in TOOLS):
        return {"route": "generate"}
    # Otherwise walk the priority list and pick the first untried tool.
    for route, field, _ in TOOLS:
        if field not in state:          # not tried yet
            return {"route": route}
    return {"route": "generate"}        # everything tried, all missed


def _lookup(kb: dict, question: str) -> list[str]:
    q = question.lower()
    hits: list[str] = []
    for topic, docs in kb.items():
        if topic in q:
            hits.extend(docs)
    return hits


def search_docs_node(state: State) -> dict:
    return {"docs": _lookup(DOCS_KB, state["question"])}


def search_faq_node(state: State) -> dict:
    return {"faq": _lookup(FAQ_KB, state["question"])}


def web_search_node(state: State) -> dict:
    return {"web_results": [f"[web] thread about '{state['question']}'"]}   # web always finds something


def generate_node(state: State) -> dict:
    sources = state.get("docs", []) + state.get("faq", []) + state.get("web_results", [])
    if sources:
        return {"answer": "Answer from: " + "; ".join(sources)}
    return {"answer": "Sorry, nothing found."}


def route_after_agent(state: State) -> str:
    return state["route"]


def build():
    b = StateGraph(State)
    b.add_node("agent", agent_node)
    b.add_node("search_docs", search_docs_node)
    b.add_node("search_faq", search_faq_node)
    b.add_node("web_search", web_search_node)
    b.add_node("generate", generate_node)
    b.add_edge(START, "agent")
    b.add_conditional_edges("agent", route_after_agent, {
        "search": "search_docs", "faq": "search_faq",
        "web": "web_search", "generate": "generate",
    })
    for _, _, node in TOOLS:
        b.add_edge(node, "agent")       # every tool loops back to the agent
    b.add_edge("generate", END)
    return b.compile()


graph = build()


def _tools_used(result: dict) -> list[str]:
    # A tool that ran leaves its field present (possibly empty), so key presence —
    # not truthiness — is what "attempted" means. Reporting truthiness would hide
    # the tiers that ran and missed; this reports the real walk order.
    return [field for _, field, _ in TOOLS if field in result]


# --- tests -------------------------------------------------------------------
def test_docs_hit_stops_at_first_tier():
    result = graph.invoke({"question": "How do I configure kafka?"},
                          {"recursion_limit": 15})
    assert _tools_used(result) == ["docs"]           # hit at tier 1; faq/web never ran
    assert "Kafka Manual" in result["answer"]


def test_faq_only_walks_past_docs():
    result = graph.invoke({"question": "How do I reset password?"},
                          {"recursion_limit": 15})
    assert _tools_used(result) == ["docs", "faq"]    # docs ran and missed, faq hit, web skipped
    assert "reset link" in result["answer"]


def test_unknown_falls_through_to_web():
    result = graph.invoke({"question": "obscure thing"}, {"recursion_limit": 15})
    # docs + faq both ran and missed, web caught it — the full walk is visible now
    assert _tools_used(result) == ["docs", "faq", "web_results"]


if __name__ == "__main__":
    for q in ["How do I configure kafka?", "How do I reset password?", "obscure thing"]:
        r = graph.invoke({"question": q}, {"recursion_limit": 15})
        print(f"{q!r:34} -> used {_tools_used(r)} | {r['answer']}")

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
