"""Chapter 8 — Parallel & Reducers.

To do work in parallel you *fan out*: one node hands off to several
downstream nodes that run together in the same superstep. But when two
of those nodes write the *same* state key, their writes collide. This
file walks the whole arc, in order:

    (a) the bug     — two parallel branches write `sources` →
                      InvalidUpdateError
    (b) the fix     — the identical graph, but `sources` carries a
                      reducer (operator.add), so parallel writes
                      accumulate instead of overwriting
    (c) the router  — a conditional edge returns a LIST of node names,
                      fanning out to a runtime-chosen subset of tools
    (d) the scale   — a Send fan-out splits a compound question into
                      sub-questions, retrieves each in a parallel
                      worker, and a reducer merges the findings before
                      a synthesizer writes one answer

The shape to remember: Send schedules the workers; the reducer
combines their updates; the next node runs when its trigger/join
fires (a plain in-edge, not a separate barrier stage).

Parallel nodes stay silent (interleaved prints would not be
reproducible), and every list we PRINT is sorted first -- choosing a
display order explicitly rather than inheriting whatever the merge
produced. An application should not depend on the stored order.

Run:  python code/ch08_parallel_reducers.py
"""

import operator
from typing import Annotated, NotRequired, TypedDict

from langgraph.errors import InvalidUpdateError
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# ===========================================================================
# (a) + (b) — the SAME fan-out graph, built twice. The only difference is
# whether `sources` carries a reducer. That one annotation is the fix.
# ===========================================================================

class BuggyState(TypedDict):
    question: str
    sources: NotRequired[list[str]]              # no reducer → writes collide
    answer: NotRequired[str]


class MergedState(TypedDict):
    question: str
    # operator.add is the function itself, not a call: LangGraph invokes it
    sources: Annotated[list[str], operator.add]  # reducer → writes accumulate
    answer: NotRequired[str]


def agent_node(state: dict) -> dict:
    # Kicks off the fan-out to both tools; writes nothing itself.
    return {}


def search_docs_node(state: dict) -> dict:
    # A parallel worker: silent on purpose. It only returns its own
    # contribution; it neither reads nor merges the other branch.
    return {"sources": ["doc 1", "doc 2"]}


def web_search_node(state: dict) -> dict:
    return {"sources": ["web A", "web B"]}


def generate_node(state: dict) -> dict:
    return {"answer": f"answer from {len(state['sources'])} sources"}


def build_fanout(state_cls: type):
    builder = StateGraph(state_cls)
    builder.add_node("agent", agent_node)
    builder.add_node("search_docs", search_docs_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "agent")
    builder.add_edge("agent", "search_docs")
    builder.add_edge("agent", "web_search")
    # `generate` runs once only because both branches finish in the same
    # superstep; a plain in-edge is a trigger, not a wait. See build_router.
    builder.add_edge("search_docs", "generate")
    builder.add_edge("web_search", "generate")
    builder.add_edge("generate", END)
    return builder.compile()


# ===========================================================================
# (c) — a parallel router. A conditional edge whose routing function returns
# a LIST of node names fans out to every name in the list at once.
# ===========================================================================

class RouterState(TypedDict):
    question: str
    plan: NotRequired[list[str]]
    sources: Annotated[list[str], operator.add]


def decide_node(state: RouterState) -> dict:
    # Decide which tools this question needs, and record the plan. Stand-in
    # for an LLM: keyword checks pick doc search, web search, or both.
    q = state["question"].lower()
    plan: list[str] = []
    if any(k in q for k in ("configure", "error", "how")):
        plan.append("doc_search")
    if any(k in q for k in ("latest", "news", "2026")):
        plan.append("web_search")
    if not plan:                       # nothing matched → run both
        plan = ["doc_search", "web_search"]
    print(f"[decide] plan: {plan}")
    return {"plan": plan}


def route_parallel(state: RouterState) -> list[str]:
    # Return a LIST → fan out to every node in it (a single string would
    # take just one edge, the Ch3 case). Every possible target must be
    # declared in the path_map list on add_conditional_edges.
    return state["plan"]


def doc_search_node(state: RouterState) -> dict:
    return {"sources": [f"[doc] about '{state['question']}'"]}


def web_search_node2(state: RouterState) -> dict:
    return {"sources": [f"[web] about '{state['question']}'"]}


def merge_node(state: RouterState) -> dict:
    # Sort to choose the display order.
    sources = sorted(state["sources"])
    noun = "source" if len(sources) == 1 else "sources"
    print(f"[merge] joined {len(sources)} {noun}: {sources}")
    return {}


def build_router():
    builder = StateGraph(RouterState)
    builder.add_node("decide", decide_node)
    builder.add_node("doc_search", doc_search_node)
    builder.add_node("web_search", web_search_node2)
    # defer=True: run `merge` once nothing else is pending. A list-form join,
    # add_edge(["doc_search", "web_search"], "merge"), would wait for BOTH —
    # and this plan may fire only one of them.
    builder.add_node("merge", merge_node, defer=True)

    builder.add_edge(START, "decide")
    # A conditional edge returning a LIST fans out. The third argument lists
    # every node the router may target.
    builder.add_conditional_edges(
        "decide", route_parallel, ["doc_search", "web_search"]
    )
    builder.add_edge("doc_search", "merge")
    builder.add_edge("web_search", "merge")
    builder.add_edge("merge", END)
    return builder.compile()


# ===========================================================================
# (d) — Send fan-out: split a compound question, retrieve each part in
# parallel, reduce the findings, then synthesize one answer. Unlike (c),
# the number of workers is not known until runtime.
# ===========================================================================

class MapState(TypedDict):
    question: str
    sub_questions: NotRequired[list[str]]
    findings: Annotated[list[str], operator.add]
    answer: NotRequired[str]


class RetrieveState(TypedDict):
    # The payload each Send delivers to one worker — not the whole graph
    # state, just what that worker needs.
    index: int
    sub_question: str


def split_node(state: MapState) -> dict:
    # Orchestrator: break a compound question into atomic sub-questions.
    # Splitting on " and " is a stand-in for the LLM decomposition of
    # Part III; the shape is what matters.
    raw = state["question"].rstrip("?")
    parts = [p.strip() for p in raw.split(" and ") if p.strip()]
    print(f"[split] {len(parts)} sub-questions:")
    for i, p in enumerate(parts, 1):
        print(f"        {i}. {p}")
    return {"sub_questions": parts}


def fan_out(state: MapState) -> list[Send]:
    # A conditional edge that returns a LIST of Send. That list is the map:
    # one parallel `retrieve` worker per sub-question, sized at runtime.
    # Each Send carries its own payload.
    return [
        Send("retrieve", {"index": i, "sub_question": sq})
        for i, sq in enumerate(state["sub_questions"], 1)
    ]


def retrieve_node(state: RetrieveState) -> dict:
    # A parallel worker: silent. It writes ONE finding; the reducer on
    # `findings` accumulates the writes from every worker.
    sq = state["sub_question"]
    return {"findings": [f"{state['index']}. [doc] setup notes for '{sq}'"]}


def synthesize_node(state: MapState) -> dict:
    # Sort to choose the display order.
    findings = sorted(state["findings"])
    print(f"[synthesize] merged {len(findings)} findings")
    body = "\n".join(f"  - {f}" for f in findings)
    return {"answer": f"Here is what I found:\n{body}"}


def build_mapreduce():
    builder = StateGraph(MapState)
    builder.add_node("split", split_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("synthesize", synthesize_node)

    builder.add_edge(START, "split")
    # The Send targets ("retrieve") must be declared in the path_map list,
    # or the compiled graph loses the edge.
    builder.add_conditional_edges("split", fan_out, ["retrieve"])
    # All Send workers are spawned in one superstep, so `synthesize` runs once.
    builder.add_edge("retrieve", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()


if __name__ == "__main__":
    print("=" * 62)
    print("(a) two parallel branches write 'sources', no reducer")
    print("=" * 62)
    buggy = build_fanout(BuggyState)
    try:
        buggy.invoke({"question": "How do I set retries?"})
        print("no error (unexpected)")
    except InvalidUpdateError as e:
        # Both branches wrote `sources` in one superstep; the plain
        # (LastValue) channel refuses to pick a winner.
        print("caught InvalidUpdateError:")
        for line in str(e).splitlines():
            print(f"  {line}")

    print()
    print("=" * 62)
    print("(b) same graph + Annotated[list, operator.add] reducer")
    print("=" * 62)
    merged = build_fanout(MergedState)
    result = merged.invoke({"question": "How do I set retries?"})
    sources = sorted(result["sources"])   # an explicit display order
    print(f"merged {len(sources)} sources: {sources}")

    print()
    print("=" * 62)
    print("(c) parallel router: conditional edge returns a LIST of names")
    print("=" * 62)
    router = build_router()
    for q in ("How do I configure the latest release?",   # plan: both branches
              "How do I configure retries?"):             # plan: one branch
        router.invoke({"question": q})

    print()
    print("=" * 62)
    print("(d) Send fan-out: split -> retrieve x N -> synthesize")
    print("=" * 62)
    mapreduce = build_mapreduce()
    q = "How do I configure retries and enable TLS and rotate credentials?"
    print(f"question: {q}")
    result = mapreduce.invoke({"question": q})
    print()
    print(result["answer"])
