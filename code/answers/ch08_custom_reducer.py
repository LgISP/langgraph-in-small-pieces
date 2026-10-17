"""Chapter 8 Challenge — Silver: a reducer is any two-argument merge function.

The chapter uses `operator.add` to make parallel writes accumulate. But a reducer
is just a function `(existing, incoming) -> merged` — you can write your own. Two
parallel retrievers often return *overlapping* sources; `operator.add` would keep
the duplicates. A custom union reducer merges them while dropping repeats, so the
final list is deduplicated no matter what order the workers finished in.

Run:  python code/answers/ch08_custom_reducer.py   (see answers/README.md)
"""

import operator
from typing import Annotated, NotRequired, TypedDict

from langgraph.graph import StateGraph, START, END


def union(existing: list[str], incoming: list[str]) -> list[str]:
    """Reducer: concatenate, but keep first occurrence only (order-preserving)."""
    merged = list(existing)
    for item in incoming:
        if item not in merged:
            merged.append(item)
    return merged


class DedupState(TypedDict):
    question: str
    sources: Annotated[list[str], union]        # custom reducer, not operator.add
    answer: NotRequired[str]


class PlainAddState(TypedDict):
    question: str
    sources: Annotated[list[str], operator.add]  # for contrast: keeps duplicates
    answer: NotRequired[str]


# Two branches that both surface "shared doc" — an overlap on purpose.
def docs_node(state: dict) -> dict:
    return {"sources": ["shared doc", "doc-only"]}


def web_node(state: dict) -> dict:
    return {"sources": ["shared doc", "web-only"]}


def generate_node(state: dict) -> dict:
    return {"answer": f"{len(state['sources'])} sources"}


def build(state_type):
    b = StateGraph(state_type)
    b.add_node("agent", lambda s: {})
    b.add_node("docs", docs_node)
    b.add_node("web", web_node)
    b.add_node("generate", generate_node)
    b.add_edge(START, "agent")
    b.add_edge("agent", "docs")                 # two plain edges out = fan-out
    b.add_edge("agent", "web")
    b.add_edge("docs", "generate")
    b.add_edge("web", "generate")
    b.add_edge("generate", END)
    return b.compile()


dedup_graph = build(DedupState)
plain_graph = build(PlainAddState)


# --- tests -------------------------------------------------------------------
def test_union_reducer_deduplicates():
    result = dedup_graph.invoke({"question": "q"})
    assert sorted(result["sources"]) == ["doc-only", "shared doc", "web-only"]
    assert result["sources"].count("shared doc") == 1


def test_operator_add_keeps_the_duplicate():
    # Contrast: the built-in reducer concatenates, so "shared doc" appears twice.
    result = plain_graph.invoke({"question": "q"})
    assert result["sources"].count("shared doc") == 2


if __name__ == "__main__":
    print("union reducer  ->", sorted(dedup_graph.invoke({"question": "q"})["sources"]))
    print("operator.add   ->", sorted(plain_graph.invoke({"question": "q"})["sources"]))

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
