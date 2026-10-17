"""Chapter 5 — Tool Nodes.

A tool is just a node. `search_docs` and `web_search` are ordinary nodes with a
clear contract — read some state, do the work, write results back — and the
agent routes to whichever it needs. This builds directly on Chapter 4's loop:
the agent acts, looks at what came back, and decides what to reach for next.

    "…Kafka…"   is in the knowledge base → search_docs → generate
    "…Python…"  is not, so it falls back → web_search  → generate

`search_docs` now *actually looks something up* in a small keyed knowledge base,
so a hit returns realistic docs and a miss returns an empty list. On a miss the
agent falls back to `web_search`. (Both are still stubs — a real vector search
arrives in Part III — but the contract is the shape it will keep.)

Run:  python code/ch05_tool_nodes.py
"""

from typing import NotRequired, TypedDict

from langgraph.graph import StateGraph, START, END


# A stand-in for an internal documentation store: topic -> some docs. A real
# vector search replaces this in Part III; the node's contract won't change.
KNOWLEDGE_BASE: dict[str, list[str]] = {
    "kafka": [
        "Kafka Manual: use spring.kafka.listener.type=batch "
        "to enable batch consumption.",
        "Kafka FAQ: batch consumers need max-poll-records tuning.",
    ],
    "redis": [
        "Redis Manual: set the connection pool via spring.data.redis.*.",
    ],
    # No "python" topic: the second question below will miss.
}


class State(TypedDict):
    question: str                       # input, set by the caller
    route: NotRequired[str]             # written by agent: search|web|generate
    docs: NotRequired[list[str]]        # written by search_docs (may be empty)
    web_results: NotRequired[list[str]] # written by web_search
    iterations: NotRequired[int]        # loop counter the agent keeps (Ch4)
    answer: NotRequired[str]            # written by generate


# Tool node #1. Contract: reads `question`, writes `docs`.
def search_docs_node(state: State) -> dict:
    question = state["question"].lower()   # lowercase for matching
    print("[search_docs] searching the knowledge base:", state["question"])

    hits: list[str] = []
    # Any topic whose keyword appears in the question contributes its docs.
    for topic, docs in KNOWLEDGE_BASE.items():
        if topic in question:
            hits.extend(docs)

    print(f"[search_docs] {len(hits)} hit(s)")
    return {"docs": hits}   # may be empty


# Tool node #2. Contract: reads `question`, writes `web_results`.
def web_search_node(state: State) -> dict:
    question = state["question"]
    print("[web_search] falling back to a web search:", question)
    # Pretend the open web always turns up something.
    results = [f"[web] community thread about “{question}”"]
    return {"web_results": results}


# The agent decides by looking at what the tools have returned so far. Same
# discipline as Chapter 3: it writes a `route` and stops — it never calls a
# tool itself. What's new since Chapter 4 is that the decision now depends on a
# tool's *result*: search first, and only fall back to the web on a miss.
def agent_node(state: State) -> dict:
    docs = state.get("docs", [])
    web = state.get("web_results", [])
    searched_docs = "docs" in state          # have we tried the local docs yet?
    searched_web = "web_results" in state    # ...and have we tried the web yet?

    if not searched_docs:
        route = "search"              # haven't searched locally yet
    elif len(docs) == 0 and not searched_web:
        route = "web"                 # local search missed — fall back to web
    else:
        route = "generate"            # have material (hit/web) -> answer

    print(f"[agent] docs={len(docs)} web={len(web)} -> {route}")
    return {"route": route, "iterations": state.get("iterations", 0) + 1}


def generate_node(state: State) -> dict:
    # A tool that didn't run leaves its field absent.
    docs = state.get("docs", [])
    web = state.get("web_results", [])
    sources = docs + web
    if sources:
        body = "\n".join(f"  - {s}" for s in sources)
        answer = f"Answer (from {len(sources)} source(s)):\n{body}"
    else:
        answer = "Sorry, I couldn't find anything relevant."
    return {"answer": answer}


# Not a node: reads state, returns a key into the mapping below.
def route_after_agent(state: State) -> str:
    return state["route"]


# The graph is Chapter 4's loop, unchanged: one conditional edge picks the next
# tool, and two back-edges return to the agent so it can reassess.
builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("search_docs", search_docs_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "search": "search_docs",   # try the internal docs
        "web": "web_search",       # fall back to the open web
        "generate": "generate",    # enough material — answer
    },
)
builder.add_edge("search_docs", "agent")
builder.add_edge("web_search", "agent")
builder.add_edge("generate", END)

graph = builder.compile()


if __name__ == "__main__":
    # Two questions take the two paths:
    #   "Kafka"  is in the knowledge base → local hit, no web needed
    #   "Python" is not                   → local miss → fall back to web
    for q in [
        "How do I configure Kafka batch consumers?",
        "How do I write a Python decorator?",
    ]:
        print(f"\n===== question: {q} =====")
        # recursion_limit is Chapter 4's safety net — still on, though this
        # graph settles in a couple of hops.
        result = graph.invoke({"question": q}, config={"recursion_limit": 10})
        print("--- answer ---")
        print(result["answer"])
