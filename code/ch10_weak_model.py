"""Chapter 10 — A real model, and how a weak one fails.

This is the Chapter 5 doc-bot with ONE change: the `agent` node no longer decides
with an `if / elif` ladder — it asks a real model. The model reads the state,
follows the ordered rules in the prompt, and replies with one word (SEARCH / WEB /
GENERATE) that we parse into a route. Everything else — the node names, the routes,
the compiled graph topology, the two stub tools — is Chapter 5's, unchanged.

That makes this file a controlled experiment. Swap the model on the MODEL line and
nothing else, then run it again:

    capable  (qwen/qwen3.8-27b) follows the rules, settles in a few hops
    weak     (qwen2.5-vl-7b-instruct) ignores the state and loops until the fuse

Run the capable model:  python code/ch10_weak_model.py qwen/qwen3.8-27b
Run the weak model:     python code/ch10_weak_model.py          (weak is default)
"""

import sys
from typing import NotRequired, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError


# Two models served locally by LM Studio. The weak one is the default so that a
# bare run reproduces the failure; pass the capable id as an argument to contrast.
WEAK_MODEL = "qwen2.5-vl-7b-instruct"      # a 7B that ignores the state and loops
CAPABLE_MODEL = "qwen/qwen3.8-27b"  # follows instructions, terminates
MODEL = sys.argv[1] if len(sys.argv) > 1 else WEAK_MODEL

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",                   # the local server ignores the key
    model=MODEL,                           # the one line we swap to change models
    temperature=0,                         # 0 = as reproducible as a real model gets
    reasoning_effort="none",               # Qwen3.8 thinks by default; off for now
)


class State(TypedDict):
    question: str                       # input, set by the caller
    route: NotRequired[str]             # written by agent: search|web|generate
    docs: NotRequired[list[str]]        # written by search_docs (may be empty)
    web_results: NotRequired[list[str]] # written by web_search
    iterations: NotRequired[int]        # loop counter the agent keeps (Ch4)
    answer: NotRequired[str]            # written by generate


KNOWLEDGE_BASE: dict[str, list[str]] = {
    "kafka": [
        "Kafka Manual: use spring.kafka.listener.type=batch "
        "to enable batch consumption.",
        "Kafka FAQ: batch consumers need max-poll-records tuning.",
    ],
    "redis": [
        "Redis Manual: set the connection pool via spring.data.redis.*.",
    ],
}


def excerpt(text: str, width: int = 140) -> str:
    """First line of a model answer, cut at a word boundary, with '...' if cut.

    Answers are long and often contain code fences; the trace only needs
    enough to see WHAT the model decided to say, not the whole essay.
    """
    first = text.strip().splitlines()[0] if text.strip() else ""
    if len(first) <= width and first == text.strip():
        return first
    cut = first[:width].rsplit(" ", 1)[0] if len(first) > width else first
    return cut.rstrip(" ,;:") + " ..."


# The agent, now backed by a real model. Same job as Chapter 5 — read the state,
# write a route — but the decision comes from the model's free text, which we
# parse into one of the three routes.
def agent_node(state: State) -> dict:
    docs = state.get("docs", [])
    web = state.get("web_results", [])
    searched_docs = "docs" in state
    searched_web = "web_results" in state

    prompt = f"""
        You are the router for a documentation question-answering agent.
        Look at the current state and decide the single next action.

        Question: {state['question']}
        Local documents found so far: {len(docs)}
        Web results so far: {len(web)}
        Already searched the local docs: {searched_docs}
        Already searched the web: {searched_web}

        Apply these rules top to bottom and act on the FIRST that matches; once
        one matches, ignore the rules below it.
        1. If you ALREADY have at least one source (local documents found > 0 OR
           web results > 0), you have enough -> GENERATE. Do not search again,
           even if the web has not been searched: one source is enough.
        2. else if the local docs have not been searched yet -> SEARCH
        3. else if the web has not been searched yet -> WEB
        4. otherwise -> GENERATE

        Reply with exactly one word: SEARCH, WEB, or GENERATE. Nothing else."""

    reply = llm.invoke(prompt).content.strip().upper()

    # Brittle free-text parsing: hunt for a keyword in whatever the model said.
    if "SEARCH" in reply:
        route = "search"
    elif "WEB" in reply:
        route = "web"
    else:
        route = "generate"

    print(f"[agent] model said {reply!r} -> {route}")
    return {"route": route, "iterations": state.get("iterations", 0) + 1}


def search_docs_node(state: State) -> dict:
    question = state["question"].lower()
    hits: list[str] = []
    for topic, docs in KNOWLEDGE_BASE.items():
        if topic in question:
            hits.extend(docs)
    print(f"[search_docs] {len(hits)} hit(s)")
    return {"docs": hits}


def web_search_node(state: State) -> dict:
    print("[web_search] falling back to a web search")
    return {"web_results": [f"[web] community thread about “{state['question']}”"]}


def generate_node(state: State) -> dict:
    sources = state.get("docs", []) + state.get("web_results", [])
    sources_text = "\n".join(f"- {s}" for s in sources) or "(no sources)"
    prompt = f"""
        Answer the user's question concisely, using only the sources
        below. Do not invent anything.

        Question: {state['question']}

        Sources:
        {sources_text}

        Answer:"""
    answer = llm.invoke(prompt).content.strip()
    print("[generate] the model wrote an answer")
    return {"answer": answer}


def route_after_agent(state: State) -> str:
    return state["route"]


# The graph is Chapter 5's, unchanged — same nodes, same edges, same router.
builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("search_docs", search_docs_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {"search": "search_docs", "web": "web_search", "generate": "generate"},
)
builder.add_edge("search_docs", "agent")
builder.add_edge("web_search", "agent")
builder.add_edge("generate", END)
graph = builder.compile()


if __name__ == "__main__":
    print(f"===== model: {MODEL} =====")
    question = "How do I configure Kafka batch consumers?"
    print(f"question: {question}")
    try:
        # recursion_limit is Chapter 4's fuse. A capable model never gets near it;
        # a weak one that loops will trip it instead of hanging your process.
        result = graph.invoke({"question": question}, {"recursion_limit": 8})
        print("--- answer ---")
        print(excerpt(result["answer"]))   # first line, trimmed: answers run long
    except GraphRecursionError:
        print("!! GraphRecursionError: the agent looped until the fuse blew")
