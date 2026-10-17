"""Chapter 6 Challenge — Bronze: thread_id is a filing label, and it isolates.

The chapter calls `thread_id` a filing label: everything on one thread is saved
together, separate from every other thread. This proves the "separate" half. One
compiled graph, two different questions on two different `thread_id`s — and the
two timelines never touch: each thread's snapshots and final state are its own.

Run:  python code/answers/ch06_thread_isolation.py   (see answers/README.md)
"""

from typing import NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    question: str
    answer: NotRequired[str]


def answer_node(state: State) -> dict:
    return {"answer": f"ans:{state['question']}"}


def build():
    b = StateGraph(State)
    b.add_node("answer", answer_node)
    b.add_edge(START, "answer")
    b.add_edge("answer", END)
    return b.compile(checkpointer=InMemorySaver())   # durable → threads persist


graph = build()

cfg_a = {"configurable": {"thread_id": "thread-A"}}
cfg_b = {"configurable": {"thread_id": "thread-B"}}


def _seed():
    # Populate both threads (idempotent: same question each time).
    graph.invoke({"question": "alpha"}, config=cfg_a)
    graph.invoke({"question": "beta"}, config=cfg_b)


# --- tests -------------------------------------------------------------------
def test_threads_hold_their_own_answer():
    a = graph.invoke({"question": "alpha"}, config=cfg_a)
    b = graph.invoke({"question": "beta"}, config=cfg_b)
    assert a["answer"] == "ans:alpha"
    assert b["answer"] == "ans:beta"


def test_saved_state_does_not_leak_across_threads():
    _seed()
    # Each thread's latest snapshot still reflects only itself.
    assert graph.get_state(cfg_a).values["question"] == "alpha"
    assert graph.get_state(cfg_b).values["question"] == "beta"


def test_each_thread_has_its_own_history():
    _seed()
    hist_a = list(graph.get_state_history(cfg_a))
    hist_b = list(graph.get_state_history(cfg_b))
    # No snapshot on A ever mentions beta, and vice versa (the pristine pre-input
    # snapshot has no question at all, so we forbid cross-contamination rather
    # than require every snapshot to carry the value).
    assert not any(s.values.get("question") == "beta" for s in hist_a)
    assert not any(s.values.get("question") == "alpha" for s in hist_b)
    assert any(s.values.get("question") == "alpha" for s in hist_a)
    assert any(s.values.get("question") == "beta" for s in hist_b)


if __name__ == "__main__":
    print("thread-A ->", graph.invoke({"question": "alpha"}, config=cfg_a)["answer"])
    print("thread-B ->", graph.invoke({"question": "beta"}, config=cfg_b)["answer"])
    print("A latest question:", graph.get_state(cfg_a).values["question"])
    print("B latest question:", graph.get_state(cfg_b).values["question"])

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
