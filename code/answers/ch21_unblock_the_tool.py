"""Chapter 21 Challenge - Silver: a blocking tool in an async fan-out, and a cap.

One of the three tools is a legacy client: a plain function that blocks in
`time.sleep` the way a synchronous HTTP library blocks in a socket read. The
rule is that you may not rewrite it. Called from inside an async node it
stalls the event loop, and the fan-out that was supposed to overlap three waits
runs them one after another -- for every run on the loop, not just this one.

The fix is one line: `asyncio.to_thread(blocking_tool, ...)` runs the legacy
call on a worker thread and hands the loop an awaitable, so the node's
`gather` overlaps it with the others. The second half of the challenge is a
ceiling: an `asyncio.Semaphore` so that no matter how many runs are in flight,
at most N tool calls are out at once. The tests count the peak.

Run:  python code/answers/ch21_unblock_the_tool.py   (see answers/README.md)
"""

import asyncio
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

WAIT = 0.3
CAP = 2


async def search_docs(q: str) -> str:
    await asyncio.sleep(WAIT)
    return f"docs: {q}"


async def search_web(q: str) -> str:
    await asyncio.sleep(WAIT)
    return f"web: {q}"


def legacy_tickets(q: str) -> str:
    """The tool you may not rewrite: it blocks, like a sync HTTP client."""
    time.sleep(WAIT)
    return f"tickets: {q}"


class State(TypedDict):
    question: str
    sources: list[str]
    seconds: float


async def research_stalled(state: State) -> dict:
    """The bug. `legacy_tickets` runs on the loop and nothing else can."""
    t0 = time.perf_counter()
    q = state["question"]
    sources = await asyncio.gather(search_docs(q), search_web(q),
                                   asyncio.sleep(0, legacy_tickets(q)))   # evaluated right here, blocking
    return {"sources": list(sources), "seconds": time.perf_counter() - t0}


async def research_unblocked(state: State) -> dict:
    """The fix: the blocking call goes to a thread; the loop gets an awaitable."""
    t0 = time.perf_counter()
    q = state["question"]
    sources = await asyncio.gather(search_docs(q), search_web(q),
                                   asyncio.to_thread(legacy_tickets, q))
    return {"sources": list(sources), "seconds": time.perf_counter() - t0}


# --- the ceiling -------------------------------------------------------------
gate = asyncio.Semaphore(CAP)
in_flight = 0
peak = 0


async def capped(coro):
    """Run one tool call under the gate; count how many are out at once."""
    global in_flight, peak
    async with gate:
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            return await coro
        finally:
            in_flight -= 1


async def research_capped(state: State) -> dict:
    t0 = time.perf_counter()
    q = state["question"]
    sources = await asyncio.gather(capped(search_docs(q)), capped(search_web(q)),
                                   capped(asyncio.to_thread(legacy_tickets, q)))
    return {"sources": list(sources), "seconds": time.perf_counter() - t0}


def build(research):
    b = StateGraph(State)
    b.add_node("research", research)
    b.add_edge(START, "research")
    b.add_edge("research", END)
    return b.compile()


async def many(research, n: int = 4) -> tuple[list[dict], float]:
    """`n` runs at once on one loop: the shape of a server with n callers."""
    g = build(research)
    t0 = time.perf_counter()
    results = await asyncio.gather(*(g.ainvoke({"question": f"q{i}"}) for i in range(n)))
    return list(results), time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_the_blocking_tool_serialises_every_run_on_the_loop():
    _, wall = asyncio.run(many(research_stalled, n=4))
    assert wall >= 4 * WAIT * 0.9            # four runs, and each one's blocking call stalls the others


def test_to_thread_gives_the_overlap_back():
    _, stalled = asyncio.run(many(research_stalled, n=4))
    results, wall = asyncio.run(many(research_unblocked, n=4))
    assert all(len(r["sources"]) == 3 for r in results)
    assert wall < 2 * WAIT                   # four runs x three tools, about one wait's time
    assert stalled / wall > 2


def test_the_cap_holds_under_load():
    global in_flight, peak, gate
    in_flight, peak = 0, 0

    async def run():
        global gate
        gate = asyncio.Semaphore(CAP)        # a semaphore stays bound to the first loop it waited on
        return await many(research_capped, n=4)

    results, _ = asyncio.run(run())
    assert all(len(r["sources"]) == 3 for r in results)
    assert peak == CAP                       # twelve calls wanted to start; never more than CAP were out


if __name__ == "__main__":
    _, stalled = asyncio.run(many(research_stalled))
    _, fixed = asyncio.run(many(research_unblocked))
    print(f"4 runs, blocking tool on the loop   {stalled:.2f}s")
    print(f"4 runs, blocking tool on a thread   {fixed:.2f}s   ({stalled / fixed:.1f}x)")

    async def run():
        global gate
        gate = asyncio.Semaphore(CAP)
        return await many(research_capped)

    _, capped_wall = asyncio.run(run())
    print(f"4 runs, cap of {CAP} tool calls        {capped_wall:.2f}s   peak in flight = {peak}")
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
