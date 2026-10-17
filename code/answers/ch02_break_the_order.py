"""Chapter 2 Challenge — Silver: edges encode data dependencies.

The chapter's Gotcha warns that a bad edge order fails only at RUN time, because
`TypedDict` doesn't check dependencies for you. This proves it both ways, with the
*same* two node functions and only the edges moved:

  broken:  START -> generate -> search_docs -> END
           `generate` reads state["docs"] before search_docs ever wrote it
           -> KeyError, and notice WHEN: not at compile(), not at invoke() entry,
              but the instant generate reads a field nobody wrote yet.

  fixed:   START -> search_docs -> generate -> END
           now `docs` exists when generate reads it -> a real answer.

Run:  python code/answers/ch02_break_the_order.py   (see answers/README.md)
"""

from typing import NotRequired, TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    question: str
    docs: NotRequired[list[str]]
    answer: NotRequired[str]


def search_docs_node(state: State) -> dict:
    return {"docs": [f"doc for {state['question']}"]}


def generate_node(state: State) -> dict:
    docs = state["docs"]          # a plain read: no .get() fallback, on purpose
    return {"answer": f"answer from {len(docs)} docs"}


def build(order: str):
    b = StateGraph(State)
    b.add_node("search_docs", search_docs_node)
    b.add_node("generate", generate_node)
    if order == "broken":
        b.add_edge(START, "generate")            # generate first — before docs exist
        b.add_edge("generate", "search_docs")
        b.add_edge("search_docs", END)
    else:
        b.add_edge(START, "search_docs")         # correct dependency order
        b.add_edge("search_docs", "generate")
        b.add_edge("generate", END)
    return b.compile()


broken_graph = build("broken")
fixed_graph = build("fixed")


# --- tests -------------------------------------------------------------------
def test_broken_order_raises_keyerror():
    try:
        broken_graph.invoke({"question": "x"})
    except KeyError as e:
        assert "docs" in str(e)
    else:
        raise AssertionError("expected KeyError('docs') from the bad edge order")


def test_fixed_order_produces_answer():
    result = fixed_graph.invoke({"question": "x"})
    assert result["answer"] == "answer from 1 docs"


if __name__ == "__main__":
    try:
        broken_graph.invoke({"question": "x"})
    except KeyError as e:
        print("broken order -> KeyError:", e)
    print("fixed order  ->", fixed_graph.invoke({"question": "x"}))

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
