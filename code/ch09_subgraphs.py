"""Chapter 9 — Subgraphs.

A whole compiled graph can be used as a single node inside another graph. We
take the doc-bot's RAG flow — rewrite the query, search, rerank — and package
it as its own self-contained subgraph, then drop that whole graph into a parent
as one node:

    parent:  START → agent → rag → generate → END
                              └── the RAG subgraph runs its 3 steps in here

The parent doesn't know `rag` has three steps inside. It hands in a question and
gets back docs; the subgraph does its work in its own state space and returns
just a conclusion.

Run:  python code/ch09_subgraphs.py
"""

from typing import NotRequired, TypedDict

from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# Part 1 — build the RAG subgraph (its own complete StateGraph).
# ---------------------------------------------------------------------------
# The subgraph has its own State. For simplicity it shares the `question` and
# `docs` field names with the parent; as long as those shared fields line up,
# data flows in and out. `rewritten` is private to the subgraph.
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


# Build and compile the subgraph — note it is a complete, standalone graph.
rag_builder = StateGraph(RagState)
rag_builder.add_node("rewrite", query_rewrite_node)
rag_builder.add_node("search", vector_search_node)
rag_builder.add_node("rerank", rerank_node)
rag_builder.add_edge(START, "rewrite")
rag_builder.add_edge("rewrite", "search")
rag_builder.add_edge("search", "rerank")
rag_builder.add_edge("rerank", END)

rag_subgraph = rag_builder.compile()   # the compiled subgraph


# ---------------------------------------------------------------------------
# Part 2 — the parent graph. Use rag_subgraph as one ordinary node.
# ---------------------------------------------------------------------------
class MainState(TypedDict):
    question: str
    docs: NotRequired[list[str]]    # shared: subgraph writes, parent reads
    answer: NotRequired[str]


def agent_node(state: MainState) -> dict:
    print("[agent] decided to do RAG")
    return {}


def generate_node(state: MainState) -> dict:
    docs = state["docs"]
    print(f"[generate] generating an answer from {len(docs)} reranked doc(s)")
    return {"answer": f"final answer (used {len(docs)} curated docs): {docs}"}


main_builder = StateGraph(MainState)
main_builder.add_node("agent", agent_node)

# The crux: pass the compiled subgraph straight to add_node. The second
# argument is not a function — it's a whole graph. The parent calls it like an
# ordinary node (State in → State update out) and never sees the three steps
# (rewrite / search / rerank) inside.
main_builder.add_node("rag", rag_subgraph)

main_builder.add_node("generate", generate_node)
main_builder.add_edge(START, "agent")
main_builder.add_edge("agent", "rag")
main_builder.add_edge("rag", "generate")
main_builder.add_edge("generate", END)

main_graph = main_builder.compile()


if __name__ == "__main__":
    print("========== parent graph run ==========")
    question = "How do I configure batch consumers in Kafka?"
    result = main_graph.invoke({"question": question})
    print("\n=== final answer ===")
    print(result["answer"])
    # `rewritten` never crossed back: it is not in the parent's state.
    print("\nparent state keys:", sorted(result))
