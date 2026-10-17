"""Chapter 4 Challenge — Silver: a budget with graceful degradation.

The chapter's Try-it gives a runaway agent a crude self-brake: `if iterations >= 3:
route = "generate"`. That stops the loop, but silently — the caller can't tell a
budget-exhausted answer from a satisfied one. This does it properly: when the
agent hits its pass budget, it *flags* that it gave up and `generate` produces a
clearly-labelled partial answer instead of pretending everything is fine.

Two runs show the contrast the challenge asks for. A question the stub docs cover
lets the agent finish on its own terms; a question they don't sends it round until
the budget stops it, and only then is `gave_up` set.

Contrast with `recursion_limit` (the chapter's fuse): the fuse *crashes* the run;
a budget lets the agent *stop itself cleanly* and still return something useful.

Run:  python code/answers/ch04_budget_degrade.py   (see answers/README.md)
"""

from typing import NotRequired, TypedDict

from langgraph.graph import StateGraph, START, END

MAX_PASSES = 3
TOPIC = "Kafka"          # the one subject this stub corpus can answer


class State(TypedDict):
    question: str
    route: NotRequired[str]
    docs: NotRequired[list[str]]
    iterations: NotRequired[int]
    gave_up: NotRequired[bool]
    answer: NotRequired[str]


def agent_node(state: State) -> dict:
    iterations = state.get("iterations", 0) + 1
    docs = state.get("docs", [])

    # Satisfied the moment a search comes back on-topic...
    if any(TOPIC in d for d in docs):
        return {"route": "generate", "iterations": iterations}
    # ...otherwise keep searching. Without a budget this loops until the fuse
    # blows; the budget below is the clean brake.
    if iterations >= MAX_PASSES:
        # Out of budget: stop, but record that we did so under duress.
        return {"route": "generate", "iterations": iterations, "gave_up": True}
    return {"route": "search", "iterations": iterations}


def search_docs_node(state: State) -> dict:
    # A one-topic stub corpus: questions about Kafka get a real hit, everything
    # else gets a miss the agent will keep chasing.
    docs = state.get("docs", [])
    if TOPIC.lower() in state["question"].lower():
        found = f"{TOPIC} Manual: brokers, partitions, retention."
    else:
        found = f"doc #{len(docs) + 1} (nothing on topic)"
    return {"docs": docs + [found]}


def generate_node(state: State) -> dict:
    docs = state.get("docs", [])
    if state.get("gave_up"):
        answer = (f"Partial answer (stopped after {state['iterations']} passes, "
                  f"budget reached) from {len(docs)} docs.")
    else:
        answer = f"Answer from {len(docs)} docs."
    return {"answer": answer}


def route_after_agent(state: State) -> str:
    return state["route"]


def build():
    b = StateGraph(State)
    b.add_node("agent", agent_node)
    b.add_node("search_docs", search_docs_node)
    b.add_node("generate", generate_node)
    b.add_edge(START, "agent")
    b.add_conditional_edges("agent", route_after_agent,
                            {"search": "search_docs", "generate": "generate"})
    b.add_edge("search_docs", "agent")
    b.add_edge("generate", END)
    return b.compile()


graph = build()


# --- tests -------------------------------------------------------------------
def test_stops_at_budget_without_crashing():
    # No recursion_limit is passed on purpose: the budget must stop the loop
    # on its own. (The built-in default is far too large to act as a fuse —
    # 10007 supersteps on LangGraph 1.2.11 — see Chapter 4.)
    result = graph.invoke({"question": "unanswerable"})
    assert result["iterations"] == MAX_PASSES   # the pass that trips the budget
    assert result["gave_up"] is True


def test_degraded_answer_is_labelled():
    result = graph.invoke({"question": "unanswerable"})
    assert "Partial answer" in result["answer"]
    assert "budget reached" in result["answer"]


def test_satisfied_run_is_not_labelled():
    # The other half of the challenge: a question the docs cover ends on the
    # agent's own terms, inside the budget, with no gave_up flag to raise.
    result = graph.invoke({"question": "How do I configure Kafka?"})
    assert not result.get("gave_up")
    assert "Partial answer" not in result["answer"]
    assert result["iterations"] < MAX_PASSES


if __name__ == "__main__":
    for question in ("unanswerable", "How do I configure Kafka?"):
        result = graph.invoke({"question": question})
        print(f"===== question: {question} =====")
        print("  iterations:", result["iterations"],
              "| gave_up:", result.get("gave_up", False))
        print("  answer:", result["answer"])

    _tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print("PASS", t.__name__)
    print(f"\nall {len(_tests)} tests passed")
