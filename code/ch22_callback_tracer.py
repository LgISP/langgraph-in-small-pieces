"""Chapter 22 - The runtime's own hook, and what the decorator cannot see.

`@traced` is the right mental model: observation is a side channel, spliced around
a node without the node knowing. It is also the wrong LAYER, and the difference
shows up the moment a node does more than one thing.

A decorator wraps the function you wrote. Inside that function a model gets called
and a tool runs -- and both land in the wrapper as one duration. LangChain's
callback system hooks the runtime instead, so it sees each of those (`on_llm_start`,
`on_tool_start`) as its own event, tagged with the node it happened inside.

Retries are a different layer, and easy to get wrong: a node-level `RetryPolicy`
re-runs the WHOLE node, so both the decorator and the callback see each attempt as
a fresh node execution (N wrapper calls, N `on_chain_start`s) -- not a single
`on_retry` event. In LangGraph 1.2.14 node retries do not call
`BaseCallbackHandler.on_retry` at all; that hook fires for an explicit Runnable
`.with_retry()`, not for a node's `RetryPolicy`.

Run:  python code/ch22_callback_tracer.py   (Part 2 needs the local model)
"""

import time
from collections import defaultdict

from langchain_core.callbacks import BaseCallbackHandler

import ch22_observability as ch21


class GraphTracer(BaseCallbackHandler):
    """The chapter's logbook, filled by the runtime instead of by a decorator.

    Two things had to be learned by running it, and neither is guessable:

    1. **The node's name is not in `serialized`.** `serialized.get("name")` comes
       back empty for every event; the name lives in `metadata["langgraph_node"]`,
       next to `langgraph_step`.

    2. **Not every chain event is a node.** A six-visit run emits TEN
       `on_chain_start` calls: the six node steps, one for the graph itself, and
       three for `route_after_agent`. A conditional edge is a runnable too, and
       the callback layer wraps it -- which is why only `agent`, the one node with
       `add_conditional_edges`, is affected; the three nodes wired with a plain
       `add_edge` emit nothing extra. (Decoration is NOT the cause: all four node
       functions here are @traced.) Each router event also carries the metadata of
       the node it routes OUT of, so unfiltered, `agent` shows up six times rather
       than three. Filter on the `graph:step:` tag and you get exactly the six the
       decorator recorded, in the same order. Skip the filter and the node with
       the conditional edge silently counts double -- the failure mode of a tracer
       nobody checks against a known-good one.
    """

    def __init__(self):
        self.steps: list[dict] = []
        self.inner: list[tuple[str, str]] = []
        self._start: dict = {}

    @staticmethod
    def _node(kwargs) -> str | None:
        return (kwargs.get("metadata") or {}).get("langgraph_node")

    @staticmethod
    def _is_node(kwargs) -> bool:
        return any(t.startswith("graph:step:") for t in (kwargs.get("tags") or []))

    def on_chain_start(self, serialized, inputs, **kwargs):
        if self._is_node(kwargs):
            thread = (kwargs.get("metadata") or {}).get("thread_id")
            self._start[kwargs.get("run_id")] = (self._node(kwargs), thread, time.perf_counter())

    def on_chain_end(self, outputs, **kwargs):
        got = self._start.pop(kwargs.get("run_id"), None)
        if got:
            node, thread, start = got
            self.steps.append({
                "node": node,
                "thread_id": thread,
                "dur_ms": (time.perf_counter() - start) * 1000,
                "output_keys": sorted(outputs) if isinstance(outputs, dict) else [],
            })

    def on_chain_error(self, error, **kwargs):
        # A failing node fires on_chain_error, not on_chain_end. Without this, the
        # step would never be recorded and `_start` would leak the entry -- the same
        # failure blind spot the decorator avoids with a `finally`. Record the error
        # and clear the pending start, so a crash still leaves a trace.
        got = self._start.pop(kwargs.get("run_id"), None)
        if got:
            node, thread, start = got
            self.steps.append({
                "node": node,
                "thread_id": thread,
                "dur_ms": (time.perf_counter() - start) * 1000,
                "error": type(error).__name__,
            })

    # --- the half a decorator cannot reach -----------------------------------
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.inner.append(("llm", self._node(kwargs)))

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.inner.append(("tool", self._node(kwargs)))

    def on_retry(self, retry_state, **kwargs):
        # NOTE: node-level RetryPolicy does NOT call this in LangGraph 1.2.14 -- a
        # node retry re-runs the whole node and reappears as another on_chain_start.
        # This fires only for an explicit Runnable `.with_retry()`. Kept for that case.
        self.inner.append(("retry", self._node(kwargs)))


def part1_same_steps_as_the_decorator():
    """The callback tracer is a drop-in at the node level -- proved, not asserted."""
    tracer = GraphTracer()
    ch21.TRACE.clear()
    ch21.graph.invoke({"question": "How do I configure Kafka?"},
                      config={"callbacks": [tracer]})

    from_decorator = [t["node"].removesuffix("_node") for t in ch21.TRACE]
    from_callbacks = [s["node"] for s in tracer.steps]
    print("decorator:", from_decorator)
    print("callbacks:", from_callbacks)
    print("identical:", from_decorator == from_callbacks)
    return from_decorator, from_callbacks


def _build_model_graph():
    """A small SELF-CONTAINED graph whose two nodes each call the model. Kept local
    (not an import of a later RAG chapter) so this demo depends only on the local
    model, not on anything built after this chapter."""
    from typing import TypedDict
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, START, END

    llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio",
                     model="qwen/qwen3.8-27b", temperature=0, reasoning_effort="none")

    class S(TypedDict):
        question: str
        rewritten: str
        answer: str

    def rewrite_node(state: S) -> dict:
        out = llm.invoke(f"Rewrite this as a precise search query: {state['question']}")
        return {"rewritten": out.content.strip()}

    def answer_node(state: S) -> dict:
        out = llm.invoke(f"Answer concisely: {state['rewritten']}")
        return {"answer": out.content.strip()}

    b = StateGraph(S)
    b.add_node("rewrite", rewrite_node)
    b.add_node("answer", answer_node)
    b.add_edge(START, "rewrite")
    b.add_edge("rewrite", "answer")
    b.add_edge("answer", END)
    return b.compile()


def part2_what_the_decorator_cannot_see():
    """A graph whose nodes call a model. The decorator sees two node durations; the
    callbacks see the model call inside each, tagged with its node."""
    tracer = GraphTracer()
    _build_model_graph().invoke(
        {"question": "How do I configure Spring Boot Kafka batch consumers?"},
        config={"callbacks": [tracer]})

    print("\nnodes :", [s["node"] for s in tracer.steps])
    print("inside:", tracer.inner)
    per_node = defaultdict(int)
    for kind, node in tracer.inner:
        per_node[node] += 1
    print("model calls per node:", dict(per_node))
    print("\nThe decorator would report two node durations. The callbacks show a")
    print("model call inside each -- and only one of the two tracers can tell you")
    print("where the time actually went.")
    return tracer


if __name__ == "__main__":
    print("=" * 68)
    print("Part 1 - the same six steps, from the runtime instead of a wrapper")
    print("=" * 68)
    part1_same_steps_as_the_decorator()

    print("\n" + "=" * 68)
    print("Part 2 - inside the nodes (needs the local model)")
    print("=" * 68)
    part2_what_the_decorator_cannot_see()
