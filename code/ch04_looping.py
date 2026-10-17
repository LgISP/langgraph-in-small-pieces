"""Chapter 4 — Looping Until Enough.

The graph learns to go in circles. A tool node no longer flows *forward* to
the next step — it flows *back* to the agent, which looks at the new state and
decides whether to gather more or stop:

    agent → search_docs → agent → web_search → agent → generate → END

Each pass, the agent re-reads state and picks again: search the docs, search
the web, or (once it has enough) generate. Only `generate` leaves for END.
A back-edge is the whole trick — and `recursion_limit` is the fuse that keeps a
loop that never says "enough" from running forever.

Run:  python code/ch04_looping.py
"""

from typing import NotRequired, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, START, END
from langgraph.managed import RemainingSteps


class State(TypedDict):
    question: str                       # input, set by the caller
    route: NotRequired[str]             # written by agent each pass
    docs: NotRequired[list[str]]        # written by search_docs
    web_results: NotRequired[list[str]]  # written by web_search
    iterations: NotRequired[int]        # written by agent: passes so far
    answer: NotRequired[str]            # written by generate


def agent_node(state: State) -> dict:
    # Not in state on the first pass.
    docs = state.get("docs", [])
    web = state.get("web_results", [])
    iterations = state.get("iterations", 0)

    # Stand-in for an LLM deciding "do I have enough to answer?":
    #   nothing yet      → search the docs
    #   docs but no web  → search the web
    #   both in hand     → enough; go generate
    if not docs:
        route = "search"
    elif not web:
        route = "web"
    else:
        route = "generate"

    print(
        f"[agent] pass {iterations + 1}: "
        f"docs={len(docs)} web={len(web)} → {route}"
    )
    # Record the decision AND that we went around once more.
    return {"route": route, "iterations": iterations + 1}


def search_docs_node(state: State) -> dict:
    print("[search_docs] retrieving from the docs…")
    return {"docs": ["Config Manual §3", "FAQ entry"]}


def web_search_node(state: State) -> dict:
    print("[web_search] searching the web…")
    return {"web_results": ["Blog post", "Stack Overflow answer"]}


def generate_node(state: State) -> dict:
    docs = state.get("docs", [])
    web = state.get("web_results", [])
    print(f"[generate] composing from {len(docs)} docs + {len(web)} web hits")
    answer = f"Answer built from {len(docs)} docs and {len(web)} web hits."
    return {"answer": answer}


# The routing function is unchanged in spirit from Chapter 3: read the decision
# the agent wrote, return it as a string. Now it can point back to a tool.
def route_after_agent(state: State) -> str:
    return state["route"]


builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("search_docs", search_docs_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "agent")

# The agent's conditional edge now has three destinations.
builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "search": "search_docs",
        "web": "web_search",
        "generate": "generate",
    },
)

# The back-edges: a tool finishes and returns to `agent`, which decides again.
# This is the loop — control flows backward, not onward.
builder.add_edge("search_docs", "agent")
builder.add_edge("web_search", "agent")

# Only generate reaches the exit.
builder.add_edge("generate", END)

graph = builder.compile()


# ===========================================================================
# The runtime's own counter: RemainingSteps
# ===========================================================================
# `iterations` above is a fuse this file counts by hand, and it counts something
# slightly different from the thing that actually trips: `recursion_limit` is in
# SUPERSTEPS, and one pass through agent -> tool -> agent is several. Declaring a
# field as `RemainingSteps` asks the runtime for the real number, so the agent can
# wrap up on the budget that is going to end the run rather than on a proxy for it.


class BudgetState(TypedDict):
    question: str
    route: NotRequired[str]
    answer: NotRequired[str]
    remaining_steps: RemainingSteps     # managed by the runtime -- read, never write


def budget_agent_node(state: BudgetState) -> dict:
    if state["remaining_steps"] < 4:
        print(f"  [agent] {state['remaining_steps']} left -> wrapping up while I can")
        return {"route": "generate"}
    print(f"  [agent] {state['remaining_steps']} left -> keep looking")
    return {"route": "search"}


def budget_search_node(state: BudgetState) -> dict:
    return {}                           # finds nothing, so the agent loops


def budget_generate_node(state: BudgetState) -> dict:
    return {"answer": "answered before the fuse blew"}


def demo_remaining_steps() -> None:
    print("\n===== RemainingSteps: the agent reads the fuse instead of guessing =====")
    b = StateGraph(BudgetState)
    b.add_node("agent", budget_agent_node)
    b.add_node("search", budget_search_node)
    b.add_node("generate", budget_generate_node)
    b.add_edge(START, "agent")
    b.add_conditional_edges("agent", lambda s: s["route"],
                            {"search": "search", "generate": "generate"})
    b.add_edge("search", "agent")
    b.add_edge("generate", END)
    graph = b.compile()

    # Same graph, same loop, no GraphRecursionError: the agent sees the end coming.
    result = graph.invoke({"question": "anything"}, {"recursion_limit": 10})
    print("answer:", result["answer"])
    print("-> the first superstep of a recursion_limit=10 run reads 9")


if __name__ == "__main__":
    question = "How do I configure Kafka in Spring Boot?"

    # First, set the fuse too short. This healthy loop needs a handful of
    # steps to finish; capping it below that makes the runtime blow the fuse.
    print("===== run 1: recursion_limit=4 (too low) =====")
    try:
        graph.invoke({"question": question}, {"recursion_limit": 4})
    except GraphRecursionError:
        print("!! GraphRecursionError: hit the step limit before finishing")

    # Now give it room. The agent reaches "enough" and routes to generate.
    print("\n===== run 2: recursion_limit=10 (enough) =====")
    result = graph.invoke({"question": question}, {"recursion_limit": 10})
    print("\nanswer:", result["answer"])
    print("passes:", result["iterations"])

    demo_remaining_steps()
