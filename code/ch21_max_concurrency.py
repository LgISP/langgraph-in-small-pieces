"""Chapter 21 - what `max_concurrency` caps, and what it does not.

A reader's question, and the one this file answers by running it: a node fans
out to seven tasks and the run is started with `{"max_concurrency": 4}`. Does
that split the fan-out into two supersteps, four tasks and then three?

It does not. Two claims, each measured:

  1. the superstep is not split   seven tasks, cap of four, still ONE superstep:
                                  the same checkpoints are written, and every
                                  one of the seven reads the same input state
  2. the cap stops at the node    one node that gathers five calls runs all five
                                  at once under a cap of one - LangGraph counts
                                  tasks, not awaits

There is no model here and nothing to install beyond the book's requirements:
every "call" is an `asyncio.sleep`, so the timings are stable and the whole
file runs in about two seconds.

Run:  python code/ch21_max_concurrency.py
"""

import asyncio
import operator
import time
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

FANOUT = 7          # tasks in the one superstep
CAP = 4             # max_concurrency for the capped run
WORK = 0.40         # how long one task waits


class State(TypedDict):
    done: Annotated[list[int], operator.add]
    saw: Annotated[list[int], operator.add]      # what each task saw of `done` when it started


class Gauge:
    """Counts tasks that are inside `work` at the same moment."""

    def __init__(self) -> None:
        self.live = 0
        self.peak = 0
        self.starts: list[float] = []
        self.t0 = time.perf_counter()

    def enter(self) -> None:
        self.live += 1
        self.peak = max(self.peak, self.live)
        self.starts.append(time.perf_counter() - self.t0)

    def leave(self) -> None:
        self.live -= 1


def build(gauge: Gauge):
    """fan -> seven `work` tasks -> join. The shape is Chapter 8's Send fan-out."""

    async def fan(state: State) -> dict:
        return {}

    async def work(state: dict) -> dict:
        gauge.enter()
        await asyncio.sleep(WORK)
        gauge.leave()
        # `state["done"]` here is what this task was handed at the start of the
        # superstep -- not what the tasks beside it have since produced.
        return {"done": [state["n"]], "saw": [len(state["done"])]}

    async def join(state: State) -> dict:
        return {}

    g = StateGraph(State)
    g.add_node("fan", fan)
    g.add_node("work", work)
    g.add_node("join", join)
    g.add_edge(START, "fan")
    g.add_conditional_edges(
        "fan",
        lambda s: [Send("work", {"n": i, "done": s["done"]}) for i in range(FANOUT)],
        ["work"],
    )
    g.add_edge("work", "join")
    g.add_edge("join", END)
    return g.compile(checkpointer=InMemorySaver())


async def one_run(cap: int | None) -> tuple[int, float, int, list[int]]:
    gauge = Gauge()
    graph = build(gauge)
    config: dict = {"configurable": {"thread_id": f"cap-{cap}"}}
    if cap is not None:
        config["max_concurrency"] = cap               # <- the key under test

    t0 = time.perf_counter()
    out = await graph.ainvoke({"done": [], "saw": []}, config)
    wall = time.perf_counter() - t0

    history = list(graph.get_state_history(config))
    steps = {s.metadata["step"]: s for s in history if s.metadata}
    fan_step = steps[1]                               # the superstep that holds the fan-out

    label = f"max_concurrency={cap}" if cap else "max_concurrency unset"
    print(f"\n   {label}")
    print(f"     tasks in flight at the peak : {gauge.peak}")
    print(f"     wall clock                  : {wall:4.1f}s   (one task waits {WORK}s)")
    print(f"     checkpoints written         : {len(history)}")
    print(f"     tasks in superstep 1        : {len(fan_step.next)}  {fan_step.next[:3]}...")
    print(f"     each task saw len(done)     : {out['saw']}")
    print(f"     task start times            : {[round(s, 2) for s in gauge.starts]}")
    return gauge.peak, wall, len(history), out["saw"]


async def superstep_is_not_split() -> None:
    print(f"\n===== 1. {FANOUT} fan-out tasks, with and without a cap of {CAP} =====")
    free_peak, free_wall, free_ckpts, free_saw = await one_run(None)
    cap_peak, cap_wall, cap_ckpts, cap_saw = await one_run(CAP)

    assert free_peak == FANOUT, free_peak
    assert cap_peak == CAP, cap_peak                  # the cap is obeyed
    assert free_ckpts == cap_ckpts                    # and costs no extra superstep
    assert free_saw == cap_saw == [0] * FANOUT        # nobody saw a neighbour's write
    assert cap_wall > free_wall                       # it is paid for in wall clock

    print(
        f"\n   the cap changed how many ran at once ({free_peak} -> {cap_peak}) and the wall clock"
        f" ({free_wall:.1f}s -> {cap_wall:.1f}s)."
    )
    print(
        f"   it did not change the graph: {free_ckpts} checkpoints either way, {FANOUT} tasks in"
        " superstep 1 either way,"
    )
    print("   and every task read the same input state -- len(done) == 0 for all of them.")
    print("   four ran, three waited, and all seven landed together at the end of the step.")


async def the_cap_stops_at_the_node() -> None:
    print("\n===== 2. the same cap, around a node that gathers =====")
    gauge = Gauge()

    async def one_call() -> None:
        gauge.enter()
        await asyncio.sleep(WORK)
        gauge.leave()

    async def gathering(state: State) -> dict:
        await asyncio.gather(*(one_call() for _ in range(5)))     # five awaits, one task
        return {"done": [0]}

    g = StateGraph(State)
    g.add_node("gathering", gathering)
    g.add_edge(START, "gathering")
    g.add_edge("gathering", END)

    await g.compile().ainvoke({"done": [], "saw": []}, {"max_concurrency": 1})

    assert gauge.peak == 5, gauge.peak
    print(f"\n   max_concurrency=1, and five calls ran at once inside the node (peak {gauge.peak}).")
    print("   LangGraph counts tasks; what a task does with the loop is the node's own business.")
    print("   the cap for those five is a semaphore, in the node -- Section 7.")


async def main() -> None:
    await superstep_is_not_split()
    await the_cap_stops_at_the_node()
    print("\n   both claims asserted, not just printed.")


if __name__ == "__main__":
    asyncio.run(main())
