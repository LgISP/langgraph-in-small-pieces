"""Chapter 21 - what `max_concurrency` caps, and what it does not.

A reader's question, and the one this file answers by running it: a node fans
out to seven tasks and the run is started with `{"max_concurrency": 4}`. Does
that split the fan-out into two supersteps, four tasks and then three?

It does not. Seven tasks under a cap of four are still ONE superstep: the same
checkpoints are written, superstep 0 still holds seven tasks, and every one of
the seven reads the same input state - the three that start last see none of
the first four's writes. The cap moves how many run at once, and the wall
clock. Nothing else.

Each task logs one line as it starts - the time, its thread, its number, and
how much of `done` it can see - so the waves are readable straight off the
timestamps: with no cap, seven lines at 0.0s; with a cap of four, four lines
at 0.0s and three at 0.4s.

There is no model here and nothing to install beyond the book's requirements:
every "call" is an `asyncio.sleep`, so the timings are stable and the whole
file runs in about two seconds. `timed` is the chapter's own helper, imported
from ch21_async.py rather than written again.

Run:  python code/ch21_max_concurrency.py
"""

import asyncio
import operator
import threading
import time
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ch21_async import timed      # `with timed("label") as t:` prints the label and t.seconds

FANOUT = 7          # tasks in the one superstep
CAP = 4             # max_concurrency for the capped run
WORK = 0.40         # how long one task waits

T0 = 0.0            # start of the current run; the log's times are relative to it


class State(TypedDict):
    done: Annotated[list[int], operator.add]
    saw: Annotated[list[int], operator.add]      # what each task saw of `done` when it started


async def work(state: dict) -> dict:
    n, saw = state["n"], len(state["done"])
    print(f"      {time.perf_counter() - T0:4.1f}s  {threading.current_thread().name:<11}"
          f"  work {n}   saw len(done)={saw}")
    await asyncio.sleep(WORK)
    return {"done": [n], "saw": [saw]}


def build():
    """START -> seven `work` tasks -> END. Chapter 8's Send fan-out and nothing
    else: the seven tasks are the whole superstep."""
    g = StateGraph(State)
    g.add_node("work", work)
    g.add_conditional_edges(
        START,
        lambda s: [Send("work", {"n": i, "done": s["done"]}) for i in range(FANOUT)],
        ["work"],
    )
    g.add_edge("work", END)
    return g.compile(checkpointer=InMemorySaver())


async def one_run(cap: int | None) -> tuple[float, int, int, list[int]]:
    global T0
    graph = build()
    config: dict = {"configurable": {"thread_id": f"cap-{cap}"}}
    if cap is not None:
        config["max_concurrency"] = cap               # <- the key under test

    print(f"\n   {f'max_concurrency={cap}' if cap else 'max_concurrency unset'}")
    T0 = time.perf_counter()
    with timed("   wall clock") as run:               # the chapter's timer, from ch21_async.py
        out = await graph.ainvoke({"done": [], "saw": []}, config)

    history = list(graph.get_state_history(config))
    fan_step = next(s for s in history if s.metadata and s.metadata["step"] == 0)
    print(f"      tasks in superstep 0 : {len(fan_step.next)}   {fan_step.next}")
    print(f"      checkpoints written  : {len(history)}")
    return run.seconds, len(fan_step.next), len(history), out["saw"]


async def main() -> None:
    print(f"\n===== {FANOUT} fan-out tasks, with and without a cap of {CAP} =====")
    free_wall, free_tasks, free_ckpts, free_saw = await one_run(None)
    cap_wall, cap_tasks, cap_ckpts, cap_saw = await one_run(CAP)

    assert free_tasks == cap_tasks == FANOUT          # one superstep, seven tasks, either way
    assert free_ckpts == cap_ckpts                    # the cap costs no extra superstep
    assert free_saw == cap_saw == [0] * FANOUT        # nobody saw a neighbour's write
    assert cap_wall > free_wall * 1.5                 # it is paid for in wall clock: two waves


if __name__ == "__main__":
    asyncio.run(main())
