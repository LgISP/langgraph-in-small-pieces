"""Chapter 6 Challenge — Silver: resume continues, it doesn't restart.

The chapter says resume picks up from the parked checkpoint, so completed nodes
don't re-run — "recovery granularity is one node." This *proves* it with counters.
The two expensive nodes count their own calls. We crash at `generate`, resume on
the same thread, and assert each expensive node ran exactly once *in total*
across the crash and the resume — not twice — and that `iterations` never reset.

Run:  python code/answers/ch06_resume_no_redo.py   (see answers/README.md)
"""

from typing import NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

CALLS = {"search_docs": 0, "web_search": 0}
SIMULATE_OUTAGE = True


class State(TypedDict):
    question: str
    route: NotRequired[str]
    docs: NotRequired[list[str]]
    web_results: NotRequired[list[str]]
    iterations: NotRequired[int]
    answer: NotRequired[str]


def agent_node(state: State) -> dict:
    docs = state.get("docs", [])
    web = state.get("web_results", [])
    route = "search" if not docs else "web" if not web else "generate"
    return {"route": route, "iterations": state.get("iterations", 0) + 1}


def search_docs_node(state: State) -> dict:
    CALLS["search_docs"] += 1                 # expensive — count every real call
    return {"docs": ["doc 1", "doc 2"]}


def web_search_node(state: State) -> dict:
    CALLS["web_search"] += 1
    return {"web_results": ["web A"]}


def generate_node(state: State) -> dict:
    if SIMULATE_OUTAGE:
        raise RuntimeError("downstream model timed out")
    return {"answer": f"answer from {len(state.get('docs', []))} docs "
                      f"+ {len(state.get('web_results', []))} web"}


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
    return b.compile(checkpointer=InMemorySaver())


graph = build()
_run_seq = 0


def run_crash_then_resume():
    # A fresh thread_id each call: a completed thread's saved state would
    # otherwise short-circuit the next run (agent would see docs/web already there).
    global SIMULATE_OUTAGE, _run_seq
    _run_seq += 1
    cfg = {"configurable": {"thread_id": f"crash-resume-{_run_seq}"}}
    CALLS["search_docs"] = CALLS["web_search"] = 0
    SIMULATE_OUTAGE = True
    try:
        graph.invoke({"question": "How do I configure Kafka?"}, config=cfg)
    except RuntimeError:
        pass                                  # crashed at generate, as staged
    SIMULATE_OUTAGE = False
    return graph.invoke(None, config=cfg)     # invoke(None) = resume the saved run


# --- tests -------------------------------------------------------------------
def test_expensive_nodes_run_once_across_crash_and_resume():
    run_crash_then_resume()
    assert CALLS["search_docs"] == 1          # NOT 2 — resume didn't redo it
    assert CALLS["web_search"] == 1


def test_resume_completes_and_iterations_are_preserved():
    result = run_crash_then_resume()
    assert result["answer"] == "answer from 2 docs + 1 web"
    assert result["iterations"] == 3          # never reset to 0 on resume


if __name__ == "__main__":
    result = run_crash_then_resume()
    print("calls after crash+resume:", CALLS, "(each expensive node ran once)")
    print("iterations:", result["iterations"], "| answer:", result["answer"])

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
