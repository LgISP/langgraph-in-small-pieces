"""Chapter 9 Challenge — Bronze: write the adapter node for mismatched schemas.

The chapter's Gotcha shows a parent that calls the question `query` and results
`sources`, so the subgraph's `question`/`docs` don't auto-map — and the Try-it
makes you *watch it break*. This is the fix, made runnable: a small wrapper node
maps parent → child on the way in, invokes the subgraph like any compiled graph,
and maps child → parent on the way out. The subgraph itself is unchanged; the
wrapper is a translation layer at the seam.

Run:  python code/answers/ch09_schema_adapter.py   (see answers/README.md)
"""

from typing import NotRequired, TypedDict

from langgraph.graph import StateGraph, START, END


# --- the subgraph (speaks question/docs), verbatim from the chapter -----------
class RagState(TypedDict):
    question: str                   # input, shared with the parent
    rewritten: NotRequired[str]     # internal only: the rewritten query
    docs: NotRequired[list[str]]    # output, shared with the parent


def query_rewrite_node(state: RagState) -> dict:
    q = state["question"]
    rewritten = f"{q} (rewritten: + synonyms/context)"
    print(f"      [rag/rewrite] rewrote query: {rewritten}")
    return {"rewritten": rewritten}


def vector_search_node(state: RagState) -> dict:
    rewritten = state["rewritten"]
    print(f"      [rag/search] searching with: {rewritten}")
    return {"docs": ["coarse doc 1", "coarse doc 2", "coarse doc 3"]}


def rerank_node(state: RagState) -> dict:
    docs = state["docs"]
    top = docs[:2]                  # pretend the reranker keeps the top 2
    print(f"      [rag/rerank] reranked, kept top-{len(top)}")
    return {"docs": top}


rag_builder = StateGraph(RagState)
rag_builder.add_node("rewrite", query_rewrite_node)
rag_builder.add_node("search", vector_search_node)
rag_builder.add_node("rerank", rerank_node)
rag_builder.add_edge(START, "rewrite")
rag_builder.add_edge("rewrite", "search")
rag_builder.add_edge("search", "rerank")
rag_builder.add_edge("rerank", END)

rag_subgraph = rag_builder.compile()   # the compiled subgraph


# --- the parent, whose schema does NOT match (query/sources) ------------------
# This is the ONLY thing the challenge changes relative to the chapter: a parent
# schema that calls the question `query` and the docs `sources`, plus the adapter
# node that translates at the seam.
class MismatchState(TypedDict):
    query: str                          # the question, under a different name
    sources: NotRequired[list[str]]     # the docs, under a different name
    answer: NotRequired[str]


def rag_adapter_node(state: MismatchState) -> dict:
    # Map parent -> child in, invoke, map child -> parent out.
    child_out = rag_subgraph.invoke({"question": state["query"]})
    return {"sources": child_out["docs"]}


def generate_node(state: MismatchState) -> dict:
    return {"answer": f"answer from {len(state.get('sources', []))} sources"}


parent_builder = StateGraph(MismatchState)
parent_builder.add_node("rag", rag_adapter_node)   # the WRAPPER, not the raw subgraph
parent_builder.add_node("generate", generate_node)
parent_builder.add_edge(START, "rag")
parent_builder.add_edge("rag", "generate")
parent_builder.add_edge("generate", END)

parent = parent_builder.compile()


# --- tests -------------------------------------------------------------------
def test_adapter_bridges_the_name_mismatch():
    result = parent.invoke({"query": "How do I configure Kafka?"})
    assert result["sources"] == ["coarse doc 1", "coarse doc 2"]  # reranked top-2 crossed back
    assert result["answer"] == "answer from 2 sources"


def test_subgraph_private_field_does_not_leak():
    result = parent.invoke({"query": "anything"})
    assert "rewritten" not in result                    # stayed inside the subgraph
    assert "question" not in result and "docs" not in result   # child keys didn't leak


if __name__ == "__main__":
    result = parent.invoke({"query": "How do I configure Kafka?"})
    print("parent final state keys:", sorted(result))
    print("sources:", result["sources"])
    print("answer:", result["answer"])

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
