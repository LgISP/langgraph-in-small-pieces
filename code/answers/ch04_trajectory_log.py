"""Chapter 4 Challenge — Bronze: make the loop's path observable.

The chapter counts *how many* passes the loop took (`iterations`). This records
*which route* it took each pass, so after the run you can read the whole
trajectory: ["search", "web", "generate"].

The trick is read-and-append — `{"trajectory": state.get("trajectory", []) + [route]}`.
No reducer is needed (unlike Chapter 8) because the agent runs once per superstep,
sequentially: each pass reads the current list and writes a longer one. The
defensive `state.get(..., [])` is the same first-visit lesson as the chapter.

Run:  python code/answers/ch04_trajectory_log.py   (see answers/README.md)
"""

from typing import NotRequired, TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    question: str
    route: NotRequired[str]
    docs: NotRequired[list[str]]
    web_results: NotRequired[list[str]]
    iterations: NotRequired[int]
    answer: NotRequired[str]
    trajectory: NotRequired[list[str]]   # NEW: the route taken on each pass


def agent_node(state: State) -> dict:
    docs = state.get("docs", [])
    web = state.get("web_results", [])
    iterations = state.get("iterations", 0)
    if not docs:
        route = "search"
    elif not web:
        route = "web"
    else:
        route = "generate"
    # Append this pass's route to the running trajectory (read-and-append).
    trajectory = state.get("trajectory", []) + [route]
    return {"route": route, "iterations": iterations + 1, "trajectory": trajectory}


def search_docs_node(state: State) -> dict:
    return {"docs": ["Config Manual §3", "FAQ entry"]}


def web_search_node(state: State) -> dict:
    return {"web_results": ["Blog post", "Stack Overflow answer"]}


def generate_node(state: State) -> dict:
    docs = state.get("docs", [])
    web = state.get("web_results", [])
    return {"answer": f"Answer built from {len(docs)} docs and {len(web)} web hits."}


def route_after_agent(state: State) -> str:
    return state["route"]


def build():
    b = StateGraph(State)
    b.add_node("agent", agent_node)
    b.add_node("search_docs", search_docs_node)
    b.add_node("web_search", web_search_node)
    b.add_node("generate", generate_node)
    b.add_edge(START, "agent")
    b.add_conditional_edges("agent", route_after_agent,
                            {"search": "search_docs", "web": "web_search",
                             "generate": "generate"})
    b.add_edge("search_docs", "agent")
    b.add_edge("web_search", "agent")
    b.add_edge("generate", END)
    return b.compile()


graph = build()


# --- tests -------------------------------------------------------------------
def test_trajectory_records_every_pass():
    result = graph.invoke({"question": "How do I configure Kafka?"})
    assert result["trajectory"] == ["search", "web", "generate"]


def test_trajectory_length_matches_pass_count():
    result = graph.invoke({"question": "anything"})
    # The last route is always "generate" (the only exit).
    assert result["trajectory"][-1] == "generate"
    assert len(result["trajectory"]) == 3


if __name__ == "__main__":
    result = graph.invoke({"question": "How do I configure Kafka?"})
    print("trajectory:", result["trajectory"])

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
