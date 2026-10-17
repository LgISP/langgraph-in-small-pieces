"""Chapter 8, section "Command, the other dynamic router" -- runnable ground truth.

Four things the section claims, each shown by running it:
  (a) a Command[Literal[...]] annotation registers the edges -> draw_mermaid() shows them
  (b) without the annotation the graph still runs, but the drawing is WRONG (arrow to END)
  (c) goto=["a", "b"] runs both nodes in the same superstep -- Command can fan out
  (d) goto=[Send(...), ...] composes: update state AND fan out in one return

Run:  python code/ch08_command_router.py
"""
import operator
from typing import Annotated, Literal, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send


class State(TypedDict):
    question: str
    route: NotRequired[str]
    sources: Annotated[list[str], operator.add]   # reducer: parallel writes accumulate


def search_docs_node(state: State) -> dict:
    return {"sources": ["doc 1", "doc 2"]}


def web_search_node(state: State) -> dict:
    return {"sources": ["web A", "web B"]}


# (a) annotated: the Literal is read at add_node time and registers the edges
def command_agent_node(state: State) -> Command[Literal["search_docs", "web_search"]]:
    target = "search_docs" if "deploy" in state["question"] else "web_search"
    return Command(
        update={"route": target},
        goto=target,
    )


# (b) same body, no annotation: runs the same, draws differently
def unannotated_agent_node(state: State) -> Command:
    target = "search_docs" if "deploy" in state["question"] else "web_search"
    return Command(update={"route": target}, goto=target)


# (c) a LIST of node names: both run in the same superstep
def fan_out_agent_node(state: State) -> Command[Literal["search_docs", "web_search"]]:
    return Command(update={"route": "both"}, goto=["search_docs", "web_search"])


def build(agent_fn) -> StateGraph:
    builder = StateGraph(State)
    builder.add_node("agent", agent_fn)
    builder.add_node("search_docs", search_docs_node)
    builder.add_node("web_search", web_search_node)
    builder.add_edge(START, "agent")
    builder.add_edge("search_docs", END)
    builder.add_edge("web_search", END)
    return builder.compile()


def edges_of(graph) -> list[str]:
    """Just the edge lines of draw_mermaid(), so the difference is easy to see."""
    return [line.strip() for line in graph.get_graph().draw_mermaid().splitlines()
            if "-->" in line or "-.->" in line]


def steps_of(graph, state: dict) -> dict[int, list[str]]:
    """Which nodes ran in which superstep (stream_mode='debug' reports every task)."""
    steps: dict[int, list[str]] = {}
    for event in graph.stream(state, stream_mode="debug"):
        if event["type"] == "task":
            steps.setdefault(event["step"], []).append(event["payload"]["name"])
    return steps


def banner(title: str) -> None:
    print("\n" + "=" * 62 + f"\n{title}\n" + "=" * 62)


if __name__ == "__main__":
    q = {"question": "How do I deploy?", "sources": []}

    banner("(a) Command[Literal[...]] annotation: edges are registered")
    g = build(command_agent_node)
    for line in edges_of(g):
        print("  ", line)
    print("result:", g.invoke(q))

    banner("(b) no annotation: same run, WRONG drawing (arrow straight to END)")
    g = build(unannotated_agent_node)
    for line in edges_of(g):
        print("  ", line)
    print("result:", g.invoke(q))

    banner("(c) goto=[...]: a Command can fan out -- both run in one superstep")
    g = build(fan_out_agent_node)
    print("supersteps:", steps_of(g, q))
    print("result:", g.invoke(q))

    banner("(d) goto=[Send(...)]: update state AND fan out in one return")

    class MapState(TypedDict):
        question: str
        findings: Annotated[list[str], operator.add]

    class Worker(TypedDict):
        sub_question: str

    def split_node(state: MapState) -> Command[Literal["retrieve"]]:
        subs = [s.strip() for s in state["question"].split(" and ")]
        return Command(
            update={"findings": [f"split into {len(subs)}"]},            # the update...
            goto=[Send("retrieve", {"sub_question": s}) for s in subs],  # ...and the fan-out
        )

    def retrieve_node(payload: Worker) -> dict:
        return {"findings": [f"[doc] {payload['sub_question']}"]}

    b = StateGraph(MapState)
    b.add_node("split", split_node)
    b.add_node("retrieve", retrieve_node)
    b.add_edge(START, "split")
    b.add_edge("retrieve", END)
    g = b.compile()
    print("supersteps:", steps_of(g, {"question": "configure retries and enable TLS", "findings": []}))
    print("result:", g.invoke({"question": "configure retries and enable TLS", "findings": []}))
