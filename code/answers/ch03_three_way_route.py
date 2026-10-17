"""Chapter 3 Challenge — Bronze: add a third branch to the conditional edge.

The chapter routes two ways: "search" -> search_docs, "answer" -> generate.
Add a third: a "clarify" branch for questions too short to act on (empty, or a
couple of characters). The agent gains one more possible decision, the mapping
table gains one more entry, and a new `clarify` node handles it — and the two
branches that already worked are untouched. That's decision-in-state paying off.

  "How do I configure X?" -> search  -> search_docs -> generate -> END
  "Thanks a lot!"         -> answer  ->               generate -> END
  ""  /  "Hi"             -> clarify -> clarify (ask to rephrase) -> END

Run:  python code/answers/ch03_three_way_route.py   (see answers/README.md)
"""

from typing import NotRequired, TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    question: str
    route: NotRequired[str]
    docs: NotRequired[list[str]]
    answer: NotRequired[str]


def agent_node(state: State) -> dict:
    q = state["question"].strip()
    if len(q) < 3:                                    # empty or too short to act on
        route = "clarify"
    elif any(k in q.lower() for k in ("how", "why", "configure", "error", "install")):
        route = "search"
    else:
        route = "answer"
    return {"route": route}


def search_docs_node(state: State) -> dict:
    return {"docs": [f"Config Manual: {state['question']}"]}


def generate_node(state: State) -> dict:
    docs = state.get("docs", [])                      # defensive read (Chapter 2 gotcha)
    if docs:
        return {"answer": "(from the docs) " + "; ".join(docs)}
    return {"answer": f"(direct) I can help with: {state['question']}"}


def clarify_node(state: State) -> dict:
    # The new branch: too little to go on, so ask for more instead of guessing.
    return {"answer": "Could you rephrase with a bit more detail?"}


def route_after_agent(state: State) -> str:
    return state["route"]


def build():
    b = StateGraph(State)
    b.add_node("agent", agent_node)
    b.add_node("search_docs", search_docs_node)
    b.add_node("generate", generate_node)
    b.add_node("clarify", clarify_node)

    b.add_edge(START, "agent")
    b.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "search": "search_docs",
            "answer": "generate",
            "clarify": "clarify",      # the third destination
        },
    )
    b.add_edge("search_docs", "generate")
    b.add_edge("generate", END)
    b.add_edge("clarify", END)         # clarify is its own dead-end exit
    return b.compile()


graph = build()


# --- tests -------------------------------------------------------------------
def test_search_branch():
    result = graph.invoke({"question": "How do I configure Kafka?"})
    assert result["route"] == "search"
    assert result["answer"].startswith("(from the docs)")


def test_answer_branch():
    result = graph.invoke({"question": "Thanks a lot!"})
    assert result["route"] == "answer"
    assert result["answer"].startswith("(direct)")


def test_clarify_branch():
    result = graph.invoke({"question": ""})
    assert result["route"] == "clarify"
    assert "rephrase" in result["answer"]
    assert "docs" not in result        # neither search_docs nor generate ran


if __name__ == "__main__":
    for q in ["How do I configure Kafka?", "Thanks a lot!", ""]:
        r = graph.invoke({"question": q})
        print(f"{q!r:28} -> {r['route']:8} | {r['answer']}")

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
