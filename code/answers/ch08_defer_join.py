"""Chapter 8 Challenge — Bronze: a join names *who triggers*, not a barrier.

The Gotcha in the chapter showed that a plain edge from each branch to `merge`
runs `merge` *twice* — once as each branch lands — because a join names who
triggers the node, not a barrier that waits for everyone. There are two ways to
declare a real join:

    add_edge(["doc_search", "rerank"], "merge")   # wait for these two
    add_node("merge", merge_node, defer=True)     # wait for the run to drain

Both fix the double firing. They are not interchangeable, and the second half of
this file shows why: point the same two declarations at a *router* that picks its
branches at runtime, and the list form waits for a branch that was never
scheduled — so `merge` never runs at all, with no error to notice.

The double firing only shows up when the branches are *uneven*: as in the
chapter's Gotcha, `web_search` is followed by a `rerank` step, so it finishes a
superstep later than `doc_search`. With plain edges, `doc_search` triggers `merge`
in one superstep and `rerank` triggers it again in the next.

Run:  ../.venv1x/bin/python code/answers/ch08_defer_join.py   (see answers/README.md)
"""

import operator
from typing import Annotated, NotRequired, TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    sources: Annotated[list[str], operator.add]  # parallel writes accumulate
    merge_runs: Annotated[int, operator.add]     # counts how often merge fires
    plan: NotRequired[list[str]]                 # used by the router half


def doc_search(state: dict) -> dict:
    return {"sources": ["doc"]}


def web_search(state: dict) -> dict:
    return {"sources": ["web"]}


def rerank(state: dict) -> dict:
    # An extra step on the web branch, so it finishes one superstep later.
    return {"sources": ["reranked"]}


def merge_node(state: dict) -> dict:
    # Every time merge runs, add 1 to the counter.
    return {"merge_runs": 1}


# --- part 1: a FIXED fan-out with uneven branches ----------------------------
def build_fixed(join: str):
    """The chapter's Gotcha shape, wired three ways."""
    b = StateGraph(State)
    b.add_node("doc_search", doc_search)
    b.add_node("web_search", web_search)
    b.add_node("rerank", rerank)
    b.add_node("merge", merge_node, defer=(join == "defer"))

    b.add_edge(START, "doc_search")
    b.add_edge(START, "web_search")               # fan out to both branches
    b.add_edge("web_search", "rerank")            # uneven: web branch is longer

    if join == "list":
        b.add_edge(["doc_search", "rerank"], "merge")
    else:                                          # "plain" and "defer"
        b.add_edge("doc_search", "merge")
        b.add_edge("rerank", "merge")

    b.add_edge("merge", END)
    return b.compile()


# --- part 2: the same joins under a RUNTIME-CHOSEN subset --------------------
def build_router(join: str):
    """`decide` fans out to whatever is in `plan` — here, one branch of two."""
    b = StateGraph(State)
    b.add_node("decide", lambda s: {})
    b.add_node("doc_search", doc_search)
    b.add_node("web_search", web_search)
    b.add_node("merge", merge_node, defer=(join == "defer"))

    b.add_edge(START, "decide")
    b.add_conditional_edges("decide", lambda s: s["plan"],
                            ["doc_search", "web_search"])
    if join == "list":
        b.add_edge(["doc_search", "web_search"], "merge")
    else:
        b.add_edge("doc_search", "merge")
        b.add_edge("web_search", "merge")

    b.add_edge("merge", END)
    return b.compile()


def _runs(graph, **extra) -> int:
    return graph.invoke({"sources": [], "merge_runs": 0, **extra})["merge_runs"]


# --- tests -------------------------------------------------------------------
def test_plain_edges_fire_merge_twice():
    # A plain edge from each branch triggers merge once per branch, and the
    # branches land in different supersteps.
    assert _runs(build_fixed("plain")) == 2


def test_both_declarations_fix_the_fixed_fan_out():
    assert _runs(build_fixed("list")) == 1
    assert _runs(build_fixed("defer")) == 1


def test_list_join_never_fires_under_a_partial_plan():
    # The list form waits for `web_search` too — and it was never scheduled.
    # No error, no join: the run simply ends after doc_search.
    assert _runs(build_router("list"), plan=["doc_search"]) == 0


def test_defer_still_fires_under_a_partial_plan():
    assert _runs(build_router("defer"), plan=["doc_search"]) == 1


def test_every_source_survives_either_way():
    for join in ("plain", "list", "defer"):
        final = build_fixed(join).invoke({"sources": [], "merge_runs": 0})
        assert set(final["sources"]) == {"doc", "web", "reranked"}


if __name__ == "__main__":
    print("fixed fan-out, uneven branches")
    for join in ("plain", "list", "defer"):
        print(f"  {join:6s} -> merge ran {_runs(build_fixed(join))} time(s)")

    print("router whose plan fires ONE branch of two")
    for join in ("list", "defer"):
        runs = _runs(build_router(join), plan=["doc_search"])
        print(f"  {join:6s} -> merge ran {runs} time(s)"
              f"{'   <- silently skipped' if runs == 0 else ''}")

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    print()
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
