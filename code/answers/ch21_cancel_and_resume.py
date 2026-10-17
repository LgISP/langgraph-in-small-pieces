"""Chapter 21 Challenge - Gold: cancel a run at its next await, then finish it.

An async run can be stopped from outside: `asyncio.wait_for(graph.ainvoke(...),
timeout)` raises `TimeoutError` and the task is cancelled at its next `await`.
What that leaves behind depends on the checkpointer. With `AsyncSqliteSaver`
every superstep that finished before the cancel is on disk, so the same thread
can be picked up and run to the end -- and the steps already done are not run
again. This is Chapter 6's resume, reached through the async door, and the
reason the Gotcha insists on the `Async` saver: the sync one cannot even be
opened under `ainvoke`.

Three nodes, each waiting on something slow. The timeout lands in the middle of
the second. The tests check what was saved, what was not, and that resuming
does not repeat the first.

Run:  python code/answers/ch21_cancel_and_resume.py   (see answers/README.md)
"""

import asyncio
import os
import tempfile
from typing import TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

WAIT = 0.3
runs: dict[str, int] = {"fetch": 0, "research": 0, "generate": 0}   # how often each node body ran


class State(TypedDict):
    question: str
    docs: str
    notes: str
    answer: str


async def fetch(state: State) -> dict:
    runs["fetch"] += 1
    await asyncio.sleep(WAIT)
    return {"docs": "docs for " + state["question"]}


async def research(state: State) -> dict:
    runs["research"] += 1
    await asyncio.sleep(WAIT)                 # the timeout lands while this await is open
    return {"notes": "notes from " + state["docs"]}


async def generate(state: State) -> dict:
    runs["generate"] += 1
    await asyncio.sleep(WAIT)
    return {"answer": "answer from " + state["notes"]}


def build(checkpointer):
    b = StateGraph(State)
    b.add_node("fetch", fetch)
    b.add_node("research", research)
    b.add_node("generate", generate)
    b.add_edge(START, "fetch")
    b.add_edge("fetch", "research")
    b.add_edge("research", "generate")
    b.add_edge("generate", END)
    return b.compile(checkpointer=checkpointer)


async def cancel_then_resume(db_path: str) -> dict:
    """Cancel one run midway, inspect the checkpoint, finish it, and report."""
    for k in runs:
        runs[k] = 0
    config = {"configurable": {"thread_id": "gold"}}
    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:     # the async saver, as a context manager
        graph = build(saver)
        cancelled = False
        try:
            await asyncio.wait_for(graph.ainvoke({"question": "q"}, config), timeout=1.5 * WAIT)
        except TimeoutError:
            cancelled = True
        saved = await graph.aget_state(config)                            # what the cancel left on disk
        left_on_disk = {k: v for k, v in saved.values.items() if k != "question"}
        next_up = tuple(saved.next)
        final = await graph.ainvoke(None, config)                          # None: resume, do not restart
    return {"cancelled": cancelled, "left_on_disk": left_on_disk, "next_up": next_up,
            "final": final, "runs": dict(runs)}


def fresh_db() -> str:
    return os.path.join(tempfile.mkdtemp(), "gold.sqlite")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_the_timeout_cancels_the_run():
    r = asyncio.run(cancel_then_resume(fresh_db()))
    assert r["cancelled"] is True


def test_finished_steps_are_on_disk_and_the_interrupted_one_is_not():
    r = asyncio.run(cancel_then_resume(fresh_db()))
    assert r["left_on_disk"] == {"docs": "docs for q"}      # fetch finished; research did not
    assert r["next_up"] == ("research",)                   # and that is where the thread resumes


def test_resume_finishes_without_repeating_what_was_done():
    r = asyncio.run(cancel_then_resume(fresh_db()))
    assert r["final"]["answer"] == "answer from notes from docs for q"
    assert r["runs"]["fetch"] == 1                         # never re-run
    assert r["runs"]["research"] == 2                      # the cancelled attempt, then the real one
    assert r["runs"]["generate"] == 1


if __name__ == "__main__":
    r = asyncio.run(cancel_then_resume(fresh_db()))
    print(f"cancelled at {1.5 * WAIT:.2f}s : {r['cancelled']}")
    print(f"left on disk           : {r['left_on_disk']}   next up: {r['next_up']}")
    print(f"after resume           : {r['final']['answer']!r}")
    print(f"node bodies ran        : {r['runs']}")
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
