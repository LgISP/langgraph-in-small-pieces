"""Chapter 21 Challenge - Bronze: one node, three waits, one await.

The chapter's bill in miniature. A node calls three tools that each wait on
something outside the process; written the obvious way, the three waits add
up. `asyncio.gather` hands all three to the event loop, and the node's time
becomes the slowest tool rather than the sum.

No model here. The tools sleep on the loop for a fixed time, so the ratio the
tests check is stable on any machine: the chapter's rule is to compare runs
against each other and never against a number.

Run:  python code/answers/ch21_serial_to_concurrent.py   (see answers/README.md)
"""

import asyncio
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

WAIT = 0.3   # seconds each tool spends waiting; the ratio is what matters


async def search_docs(question: str) -> str:
    await asyncio.sleep(WAIT)
    return f"docs: notes on {question}"


async def search_web(question: str) -> str:
    await asyncio.sleep(WAIT)
    return f"web: a page about {question}"


async def search_tickets(question: str) -> str:
    await asyncio.sleep(WAIT)
    return f"tickets: an issue mentioning {question}"


class State(TypedDict):
    question: str
    sources: list[str]
    seconds: float


async def research_serial(state: State) -> dict:
    """The obvious node: each await finishes before the next starts."""
    t0 = time.perf_counter()
    sources = [await search_docs(state["question"]),
               await search_web(state["question"]),
               await search_tickets(state["question"])]
    return {"sources": sources, "seconds": time.perf_counter() - t0}


async def research_concurrent(state: State) -> dict:
    """The same three calls, all in flight at once."""
    t0 = time.perf_counter()
    q = state["question"]
    sources = await asyncio.gather(search_docs(q), search_web(q), search_tickets(q))
    return {"sources": list(sources), "seconds": time.perf_counter() - t0}


def build(research):
    b = StateGraph(State)
    b.add_node("research", research)
    b.add_edge(START, "research")
    b.add_edge("research", END)
    return b.compile()


async def compare(question: str = "Kafka batch consumers") -> tuple[dict, dict]:
    serial = await build(research_serial).ainvoke({"question": question})
    concurrent = await build(research_concurrent).ainvoke({"question": question})
    return serial, concurrent


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_both_nodes_return_the_same_sources():
    serial, concurrent = asyncio.run(compare())
    assert serial["sources"] == concurrent["sources"]
    assert len(serial["sources"]) == 3


def test_serial_node_pays_for_every_wait():
    serial, _ = asyncio.run(compare())
    assert serial["seconds"] >= 3 * WAIT * 0.9


def test_concurrent_node_pays_for_the_slowest_wait_only():
    serial, concurrent = asyncio.run(compare())
    assert concurrent["seconds"] < 2 * WAIT          # three waits, about one wait's time
    assert serial["seconds"] / concurrent["seconds"] > 2   # the ratio, not the number


if __name__ == "__main__":
    serial, concurrent = asyncio.run(compare())
    print(f"serial      {serial['seconds']:.2f}s   {serial['sources']}")
    print(f"concurrent  {concurrent['seconds']:.2f}s   ({serial['seconds'] / concurrent['seconds']:.1f}x)")
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
