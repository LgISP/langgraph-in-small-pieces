"""Chapter 9 Challenge - Bronze: see inside a subgraph node.

The parent graph treats `rag` as one node, so the parent's final state only shows
the shared output keys. Streaming with `subgraphs=True` opens the box: inner
updates arrive with the subgraph namespace, and `get_graph(xray=1)` expands the
subgraph in the Mermaid diagram.

Run:  python code/answers/ch09_see_inside_node.py   (see answers/README.md)
"""

from pathlib import Path
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph


# --- the subgraph: rewrite -> search -> rerank ------------------------------
class RagState(TypedDict):
    question: str
    rewritten: NotRequired[str]
    docs: NotRequired[list[str]]


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
    top = docs[:2]
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

rag_subgraph = rag_builder.compile()


# --- the parent: agent -> rag(subgraph) -> generate -------------------------
class MainState(TypedDict):
    question: str
    docs: NotRequired[list[str]]
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
main_builder.add_node("rag", rag_subgraph)
main_builder.add_node("generate", generate_node)
main_builder.add_edge(START, "agent")
main_builder.add_edge("agent", "rag")
main_builder.add_edge("rag", "generate")
main_builder.add_edge("generate", END)

main_graph = main_builder.compile()


QUESTION = "How do I configure batch consumers in Kafka?"


def collect_updates() -> list[tuple[tuple[str, ...], dict]]:
    return list(
        main_graph.stream(
            {"question": QUESTION},
            stream_mode="updates",
            subgraphs=True,
        )
    )


def format_namespace(ns: tuple[str, ...]) -> str:
    return "()" if not ns else repr(ns)


def print_updates(updates: list[tuple[tuple[str, ...], dict]]) -> None:
    print("\n=== stream_mode='updates', subgraphs=True ===")
    for namespace, update in updates:
        print(f"{format_namespace(namespace):<48} {list(update)}")


def write_xray_mermaid() -> Path:
    mermaid = main_graph.get_graph(xray=1).draw_mermaid()
    out = Path(__file__).with_name("ch09_see_inside_node_xray.mmd")
    out.write_text(mermaid, encoding="utf-8")
    return out


def print_xray_mermaid(path: Path) -> None:
    print("\n=== get_graph(xray=1).draw_mermaid() ===")
    print(f"wrote full Mermaid source to: {path}")
    print("Render it by pasting that file into https://mermaid.live,")
    print("or with Mermaid CLI: mmdc -i "
          f"{path.name} -o ch09_see_inside_node_xray.svg")
    print("\nThe escaped ids like rag\\3arewrite are normal Mermaid source;")
    print("the rendered diagram shows a readable subgraph named 'rag'.")


def test_stream_exposes_inner_steps():
    updates = collect_updates()
    outer_nodes = [list(update)[0] for namespace, update in updates if namespace == ()]
    inner_nodes = [list(update)[0] for namespace, update in updates if namespace != ()]

    assert outer_nodes == ["agent", "rag", "generate"]
    assert inner_nodes == ["rewrite", "search", "rerank"]
    assert all(namespace[0].startswith("rag:") for namespace, _ in updates if namespace)


def test_xray_mermaid_expands_subgraph():
    mermaid = main_graph.get_graph(xray=1).draw_mermaid()
    assert "subgraph rag" in mermaid
    assert "rag\\3arewrite(rewrite)" in mermaid
    assert "rag\\3asearch(search)" in mermaid
    assert "rag\\3arerank(rerank)" in mermaid


if __name__ == "__main__":
    updates = collect_updates()
    print_updates(updates)
    mermaid_path = write_xray_mermaid()
    print_xray_mermaid(mermaid_path)

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
