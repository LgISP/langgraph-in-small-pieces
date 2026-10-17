"""Chapter 2 — A Line of Nodes.

The single node from Chapter 1, stretched into a three-node line:
    START → agent → search_docs → generate → END
Nodes never call each other to run the next step; the edges decide the order,
and the state accumulates as it flows down the line.

Run:  python code/ch02_linear_pipeline.py
"""

from typing import NotRequired, TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    question: str                 # input, set by the caller
    docs: NotRequired[list[str]]  # written by search_docs
    answer: NotRequired[str]      # written by generate


def agent_node(state: State) -> dict:
    print("[agent] got the question; passing it on")
    return {}   # an empty update: changes nothing


def search_docs_node(state: State) -> dict:
    question = state["question"]
    print("[search_docs] searching for:", question)

    fake_docs = [
        f"Config Manual, ch.3: recommended setup for “{question}”",
        f"FAQ: common pitfalls and fixes for “{question}”",
    ]
    return {"docs": fake_docs}   # write docs; say nothing about what's next


def generate_node(state: State) -> dict:
    docs = state["docs"]
    print("[generate] writing an answer from", len(docs), "documents")

    body = "\n".join(f"  - {d}" for d in docs)
    answer = f"Based on the retrieved material:\n{body}"
    return {"answer": answer}


builder = StateGraph(State)

builder.add_node("agent", agent_node)
builder.add_node("search_docs", search_docs_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "agent")
builder.add_edge("agent", "search_docs")
builder.add_edge("search_docs", "generate")
builder.add_edge("generate", END)

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke({"question": "How do I configure Kafka in Spring Boot?"})
    print("\n=== final answer ===")
    print(result["answer"])
