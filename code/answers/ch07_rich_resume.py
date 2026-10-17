"""Chapter 7 Challenge — Bronze: resume with a decision, not just a yes/no.

The chapter resumes with `Command(resume="yes")` — a bare string. But whatever
you pass to `Command(resume=...)` becomes the return value of `interrupt()`, and
it can be any (serializable) value. Here the human resumes with a *structured*
decision — approve, plus a narrowed scope — and the executing node acts on it.
The interrupt/resume plumbing is identical; only the payload got richer.

Run:  python code/answers/ch07_rich_resume.py   (see answers/README.md)
"""

from typing import NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command


class State(TypedDict):
    approved: NotRequired[bool]
    scope: NotRequired[str]
    result: NotRequired[str]


def plan_node(state: State) -> dict:
    return {}


def human_approval_node(state: State) -> dict:
    decision = interrupt({"action": "delete knowledge base entries",
                          "ask": "approve? optionally narrow the scope"})
    # `decision` is whatever Command(resume=...) carried — here, a dict.
    return {"approved": decision["approved"], "scope": decision.get("scope", "all")}


def execute_node(state: State) -> dict:
    if not state["approved"]:
        return {"result": "cancelled"}
    return {"result": f"deleted: {state['scope']}"}


def build():
    b = StateGraph(State)
    b.add_node("plan", plan_node)
    b.add_node("human_approval", human_approval_node)
    b.add_node("execute", execute_node)
    b.add_edge(START, "plan")
    b.add_edge("plan", "human_approval")
    b.add_edge("human_approval", "execute")
    b.add_edge("execute", END)
    return b.compile(checkpointer=InMemorySaver())


graph = build()


def run(thread_id: str, decision: dict) -> dict:
    cfg = {"configurable": {"thread_id": thread_id}}
    paused = graph.invoke({}, config=cfg)
    assert "__interrupt__" in paused          # it stopped to ask
    return graph.invoke(Command(resume=decision), config=cfg)


# --- tests -------------------------------------------------------------------
def test_structured_approve_with_scope():
    final = run("t1", {"approved": True, "scope": "only stale entries"})
    assert final["result"] == "deleted: only stale entries"


def test_reject_carries_through():
    final = run("t2", {"approved": False})
    assert final["result"] == "cancelled"


def test_approve_defaults_scope_when_omitted():
    final = run("t3", {"approved": True})
    assert final["result"] == "deleted: all"


if __name__ == "__main__":
    print(run("demo-scope", {"approved": True, "scope": "only stale entries"})["result"])
    print(run("demo-reject", {"approved": False})["result"])

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
