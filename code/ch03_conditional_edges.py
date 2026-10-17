"""Chapter 3 — The Agent Decides.

The empty `agent` node from Chapter 2 gets a job: it looks at the question and
writes a *decision* to state. A routing function reads that decision, and a
conditional edge turns it into the next node — so the graph branches:

    "How do I configure …?"  → agent → search_docs → generate → END
    "Hello there!"           → agent →               generate → END

The agent still doesn't jump anywhere itself; it only writes `route`.

Run:  python code/ch03_conditional_edges.py
"""

from typing import NotRequired, TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    question: str                  # input, set by the caller
    route: NotRequired[str]        # written by agent: "search" or "answer"
    docs: NotRequired[list[str]]   # written by search_docs (only on the search branch)
    answer: NotRequired[str]       # written by generate


def agent_node(state: State) -> dict:
    question = state["question"]

    # Stand-in for an LLM: if the question looks like it needs the docs, search.
    keywords = ("how", "why", "configure", "error", "set up", "install")
    need_search = any(kw in question.lower() for kw in keywords)

    route = "search" if need_search else "answer"
    print(f"[agent] decided: {route}  (question: {question})")

    # Only the decision is written. Where we go next is the router's job.
    return {"route": route}


def search_docs_node(state: State) -> dict:
    question = state["question"]
    print("[search_docs] retrieving…")
    docs = [
        f"Config Manual: setup notes for “{question}”",
        f"FAQ: “{question}”",
    ]
    return {"docs": docs}


def generate_node(state: State) -> dict:
    # absent on the "answer" branch: nothing wrote it
    docs = state.get("docs", [])
    if docs:
        body = "\n".join(f"  - {d}" for d in docs)
        answer = f"(from the docs)\n{body}"
    else:
        answer = f"(answered directly) I can help with: {state['question']}"
    print("[generate] writing the answer")
    return {"answer": answer}


# The routing function is NOT a node. It reads state and returns a string; the
# runtime looks that string up in the conditional edge's mapping table.
def route_after_agent(state: State) -> str:
    return state["route"]


builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("search_docs", search_docs_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "agent")

# The conditional edge: from `agent`, call route_after_agent, then map its
# returned string to a destination node.
builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "search": "search_docs",   # decision "search" → run search_docs
        "answer": "generate",      # decision "answer" → skip straight to generate
    },
)

builder.add_edge("search_docs", "generate")
builder.add_edge("generate", END)

graph = builder.compile()


if __name__ == "__main__":
    for q in ["How do I configure Kafka in Spring Boot?", "Hello there!"]:
        print(f"\n===== question: {q} =====")
        result = graph.invoke({"question": q})
        print("answer:", result["answer"])
