"""Chapter 22 - Observability: build your own tracer (no LangSmith).

Observability means answering "who ran, when, for how long, and what did it
produce?" for every step of a run. Tools like LangSmith do this by recording
Graph -> Node -> LLM -> Retriever -> Tool -> Token -> Latency, then drawing you
a waterfall - conceptually OpenTelemetry + Jaeger + APM, but aimed at AI.

This file does NOT call LangSmith. Instead it builds the smallest possible
tracer, so you can see what observability actually is: you instrument each node
by recording, right BEFORE and right AFTER it runs, who it was, when it started,
how long it took, and which state keys it wrote. Collect those records and you
have a "trace" (the call chain). LangSmith does exactly this, only it (1)
instruments automatically so you write no decorator, (2) reaches inside the
LLM / retriever / tool calls too, and (3) stores it in the cloud behind a
visual waterfall UI.

Run:  python code/ch22_observability.py
"""

import functools
import time
from typing import TypedDict

from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph

# ===========================================================================
# The tracer itself: one logbook plus one decorator.
# ===========================================================================
TRACE = []

# perf_counter is monotonic; subtracting _T0 makes every stamp "ms since start"
_T0 = time.perf_counter()


def _thread_id():
    """The run this node is part of, or None when the function is called outside
    a graph run -- a test exercising the node directly, say. The record must not
    fail the node over a missing id."""
    try:
        return get_config().get("configurable", {}).get("thread_id")
    except RuntimeError:
        return None


def traced(fn):
    """Wrap a node so it records its own start, end, duration, and output keys.

    This is instrumentation: without touching the node's internal logic, we
    splice a little "record this" code in front of it and behind it.
    """
    @functools.wraps(fn)
    def wrapper(state):
        start = time.perf_counter()
        result = error = None
        try:
            result = fn(state)
            return result
        except BaseException as exc:      # noqa: BLE001 -- re-raised in the same breath
            # Note it and re-raise it untouched. An except that logged and returned
            # would leave the graph believing the node succeeded.
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            # finally, so a step that RAISED is still a step that happened. Append
            # after the call and the one step you most want to see -- the one that
            # blew up -- is the only one missing from the trace.
            TRACE.append({
                "node": fn.__name__,
                "thread_id": _thread_id(),
                "start_ms": (start - _T0) * 1000,
                "dur_ms": (time.perf_counter() - start) * 1000,
                "output_keys": list(result.keys()) if result else [],
                "error": error,
            })
    return wrapper


# ===========================================================================
# The graph. It is the familiar agent/search/generate loop, except every node
# is wrapped in @traced, and each carries a different sleep to fake real work.
# ===========================================================================
class State(TypedDict):
    question: str
    route: str
    docs: list
    web_results: list
    iterations: int
    answer: str


@traced
def agent_node(state: State) -> dict:
    time.sleep(0.05)
    docs = state.get("docs", [])
    web = state.get("web_results", [])
    route = "search" if not docs else ("web" if not web else "generate")
    return {"route": route, "iterations": state.get("iterations", 0) + 1}


@traced
def search_docs_node(state: State) -> dict:
    time.sleep(0.30)
    return {"docs": ["doc-1", "doc-2"]}


@traced
def web_search_node(state: State) -> dict:
    time.sleep(0.50)
    return {"web_results": ["page-A"]}


@traced
def generate_node(state: State) -> dict:
    time.sleep(0.40)
    return {"answer": "the final answer"}


def route_after_agent(state: State) -> str:
    return state["route"]


builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("search_docs", search_docs_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent", route_after_agent,
    {"search": "search_docs", "web": "web_search", "generate": "generate"},
)
builder.add_edge("search_docs", "agent")
builder.add_edge("web_search", "agent")
builder.add_edge("generate", END)
graph = builder.compile()


# ===========================================================================
# Draw the logbook as a waterfall - the same idea as the LangSmith / Jaeger UI.
# ===========================================================================
def print_waterfall():
    print("\n" + "=" * 70)
    print("TRACE waterfall (each node: when it started, how long it ran)")
    print("=" * 70)
    total = max(t["start_ms"] + t["dur_ms"] for t in TRACE)
    scale = 40 / total

    for i, t in enumerate(TRACE):
        indent = int(t["start_ms"] * scale)
        barlen = max(1, int(t["dur_ms"] * scale))    # never zero, or a fast node vanishes
        bar = " " * indent + "#" * barlen
        print(
            f"{i:>2} {t['node']:<16} {bar}  "
            f"{t['dur_ms']:>6.1f}ms  wrote {t['output_keys']}"
        )

    print("-" * 70)
    print(f"Total: {total:.1f}ms")
    slowest = max(TRACE, key=lambda t: t["dur_ms"])
    print(
        f"Slowest node: {slowest['node']} ({slowest['dur_ms']:.1f}ms) "
        f"<- optimize this one first"
    )


if __name__ == "__main__":
    print("Running the graph (every node is recorded automatically by @traced)...")
    graph.invoke({"question": "How do I configure Kafka?"},
                 config={"configurable": {"thread_id": "run-1"}})
    print_waterfall()
