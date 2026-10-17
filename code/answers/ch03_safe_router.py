"""Chapter 3 Challenge — Silver: a routing function must return a known key.

If the routing function returns a string that isn't a key in the conditional
edge's mapping, LangGraph raises `KeyError` at run time — the router "chose" a
door that doesn't exist. This shows the crash, then makes the router robust by
folding any unrecognized decision back to a safe default ("answer").

Why it matters: the moment a REAL LLM writes `route` (Part II), it can emit
"Search", "look_it_up", or anything else. A brittle router turns that into a 500;
a one-line guard turns it into a graceful fallback.

Run:  python code/answers/ch03_safe_router.py   (see answers/README.md)
"""

from typing import NotRequired, TypedDict

from langgraph.graph import StateGraph, START, END

VALID_ROUTES = ("search", "answer")


class State(TypedDict):
    question: str
    route: NotRequired[str]
    answer: NotRequired[str]


def make_agent(forced_route: str):
    def agent_node(state: State) -> dict:
        # Simulate a model that may emit an unexpected label.
        return {"route": forced_route}
    return agent_node


def search_docs_node(state: State) -> dict:
    return {"answer": "searched::"}


def generate_node(state: State) -> dict:
    # Append rather than overwrite, so we can SEE whether search_docs ran first.
    return {"answer": state.get("answer", "") + "generated"}


def brittle_router(state: State) -> str:
    return state["route"]                                  # whatever the agent wrote


def safe_router(state: State) -> str:
    route = state["route"]
    return route if route in VALID_ROUTES else "answer"    # unknown -> safe default


def build(router, forced_route: str):
    b = StateGraph(State)
    b.add_node("agent", make_agent(forced_route))
    b.add_node("search_docs", search_docs_node)
    b.add_node("generate", generate_node)
    b.add_edge(START, "agent")
    b.add_conditional_edges(
        "agent", router,
        {"search": "search_docs", "answer": "generate"},
    )
    b.add_edge("search_docs", "generate")
    b.add_edge("generate", END)
    return b.compile()


# --- tests -------------------------------------------------------------------
def test_brittle_router_crashes_on_unknown_route():
    graph = build(brittle_router, forced_route="Search")   # capital S: not a key
    try:
        graph.invoke({"question": "x"})
    except KeyError as e:
        assert "Search" in str(e)
    else:
        raise AssertionError("expected KeyError for an unmapped route")


def test_safe_router_falls_back():
    graph = build(safe_router, forced_route="Search")      # unknown -> "answer"
    result = graph.invoke({"question": "x"})
    assert result["answer"] == "generated"                 # went straight to generate


def test_safe_router_still_honors_valid_route():
    graph = build(safe_router, forced_route="search")      # known -> search_docs first
    result = graph.invoke({"question": "x"})
    assert result["answer"] == "searched::generated"       # proves search_docs ran


if __name__ == "__main__":
    try:
        build(brittle_router, "Search").invoke({"question": "x"})
    except KeyError as e:
        print("brittle router, route='Search' -> KeyError:", e)
    print("safe router,   route='Search' ->",
          build(safe_router, "Search").invoke({"question": "x"})["answer"])
    print("safe router,   route='search' ->",
          build(safe_router, "search").invoke({"question": "x"})["answer"])

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
