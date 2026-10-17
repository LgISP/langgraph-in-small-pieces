"""Chapter 7 Challenge — Silver: two serial gates.

The Deep Dive describes serial gates: two `interrupt()` calls in *different*
nodes, one after the other. Each pauses and resumes on its own
`Command(resume=…)`, carrying only its decision — you don't re-supply the earlier
answer, because the checkpoint already remembers it. This builds exactly that: a
plan must be approved, and *then* the execution must be approved, as two separate
gates on one thread.

Run:  python code/answers/ch07_serial_gates.py   (see answers/README.md)
"""

from typing import NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command


class State(TypedDict):
    plan_ok: NotRequired[bool]
    exec_ok: NotRequired[bool]
    result: NotRequired[str]


def approve_plan_node(state: State) -> dict:
    decision = interrupt({"gate": "plan", "ask": "approve the plan?"})
    return {"plan_ok": decision == "yes"}


def approve_exec_node(state: State) -> dict:
    # Reached only if the plan gate passed (see routing below).
    decision = interrupt({"gate": "execute", "ask": "approve the execution?"})
    return {"exec_ok": decision == "yes"}


def execute_node(state: State) -> dict:
    if state.get("plan_ok") and state.get("exec_ok"):
        return {"result": "executed"}
    return {"result": "cancelled"}


def after_plan(state: State) -> str:
    # If the plan was rejected, skip the second gate entirely.
    return "approve_exec" if state.get("plan_ok") else "execute"


def build():
    b = StateGraph(State)
    b.add_node("approve_plan", approve_plan_node)
    b.add_node("approve_exec", approve_exec_node)
    b.add_node("execute", execute_node)
    b.add_edge(START, "approve_plan")
    b.add_conditional_edges("approve_plan", after_plan,
                            {"approve_exec": "approve_exec", "execute": "execute"})
    b.add_edge("approve_exec", "execute")
    b.add_edge("execute", END)
    return b.compile(checkpointer=InMemorySaver())


graph = build()


def run(thread_id: str, plan_answer: str, exec_answer: str | None) -> dict:
    cfg = {"configurable": {"thread_id": thread_id}}
    graph.invoke({}, config=cfg)                       # runs to the FIRST gate
    state = graph.invoke(Command(resume=plan_answer), config=cfg)  # answer gate 1
    # If the plan was rejected, the graph skipped gate 2 and already finished.
    if "__interrupt__" in state:
        state = graph.invoke(Command(resume=exec_answer), config=cfg)  # answer gate 2
    return state


# --- tests -------------------------------------------------------------------
def test_both_yes_executes():
    assert run("g1", "yes", "yes")["result"] == "executed"


def test_second_gate_no_cancels():
    # Note: we never re-supply the plan's "yes" — the checkpoint remembers it.
    assert run("g2", "yes", "no")["result"] == "cancelled"


def test_first_gate_no_skips_second_gate():
    final = run("g3", "no", None)
    assert final["result"] == "cancelled"
    assert final.get("exec_ok") is None          # gate 2 never ran


if __name__ == "__main__":
    print("yes,yes ->", run("d1", "yes", "yes")["result"])
    print("yes,no  ->", run("d2", "yes", "no")["result"])
    print("no      ->", run("d3", "no", None)["result"])

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
